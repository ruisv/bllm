// bllm::Grammar — GBNF grammar-constrained decoding, at parity with llama.cpp's.
//
// A grammar turns generation into a walk over a pushdown automaton: at every step only the
// tokens whose text keeps the grammar satisfiable may be sampled, so the reply is
// STRUCTURALLY guaranteed (valid JSON, a choice from a fixed list, a phone number) instead
// of merely asked for in the prompt.
//
// Two things make this file bigger than "match a regex":
//
//  - The automaton is nondeterministic. `root ::= "a" | "ab"` is two live positions after
//    "a", so the state is a SET of stacks, each stack a path through the rule elements.
//    This is llama.cpp's representation (element arrays + stacks of pointers into them),
//    ported, because it is the shape that makes rule references and recursion cheap.
//  - Tokens are not characters. A BPE token is an arbitrary byte string that can END IN THE
//    MIDDLE of a UTF-8 sequence (Qwen splits many CJK characters across tokens), so the
//    matcher carries a partial-code-point state and can answer "could this half character
//    still become something the grammar accepts?" — see matchPartial().
//
// No hobot/tokenizer dependency: this is pure string/logic and builds on the Mac. The
// vocab-side glue (token id -> bytes) is decodeByteLevelToken() at the bottom, and the
// sampler-facing constraint is GrammarConstraint.
#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bllm/native_sampler.h"   // TokenConstraint

namespace bllm {

namespace grammar_detail {

// One element of a rule. A rule is a flat sequence of these, with Alt separating
// alternates and End closing the last one.
enum class ET : uint8_t {
  End,           // end of an alternate
  Alt,           // "|"
  RuleRef,       // value = rule id
  Char,          // value = code point; start of a positive char set
  CharNot,       // value = code point; start of a negated char set ([^...])
  CharRngUpper,  // value = upper bound of a range whose lower bound is the previous element
  CharAlt,       // value = another member of the set that started with Char/CharNot
  CharAny,       // "." — any single code point
};

struct Elem {
  ET type;
  uint32_t value = 0;
};

using Rule = std::vector<Elem>;

inline bool isEndOfSequence(const Elem* e) {
  return e->type == ET::End || e->type == ET::Alt;
}

// --- UTF-8 ------------------------------------------------------------------
// A code point half-read: `value` holds the bits so far, `n_remain` how many continuation
// bytes are still missing (0 = nothing pending, <0 = the bytes were not valid UTF-8).
struct PartialUtf8 {
  uint32_t value = 0;
  int n_remain = 0;
};

inline int utf8Len(uint8_t first) {
  if (first < 0x80) return 1;
  if ((first >> 5) == 0x6) return 2;
  if ((first >> 4) == 0xE) return 3;
  if ((first >> 3) == 0x1E) return 4;
  return 0;                                  // continuation byte or invalid lead
}

// Decode `src` into code points, continuing from a half-read one and reporting whatever is
// left half-read at the end. This is why a token that stops mid-character is not an error.
inline std::pair<std::vector<uint32_t>, PartialUtf8> decodeUtf8(const std::string& src,
                                                                PartialUtf8 start) {
  std::vector<uint32_t> out;
  uint32_t value = start.value;
  int n_remain = start.n_remain;
  size_t i = 0;
  // Finish the code point carried over from the previous token first.
  while (i < src.size() && n_remain > 0) {
    const uint8_t next = (uint8_t)src[i];
    if ((next >> 6) != 2) return {out, {0, -1}};      // not a continuation byte
    value = (value << 6) + (next & 0x3F);
    ++i;
    if (--n_remain == 0) out.push_back(value);
  }
  while (i < src.size()) {
    const uint8_t first = (uint8_t)src[i];
    const int len = utf8Len(first);
    if (len == 0) return {out, {0, -1}};
    static const uint8_t kMask[5] = {0, 0x7F, 0x1F, 0x0F, 0x07};
    value = first & kMask[len];
    n_remain = len - 1;
    ++i;
    while (i < src.size() && n_remain > 0) {
      const uint8_t next = (uint8_t)src[i];
      if ((next >> 6) != 2) return {out, {0, -1}};
      value = (value << 6) + (next & 0x3F);
      ++i;
      --n_remain;
    }
    if (n_remain == 0) out.push_back(value);
  }
  return {out, {value, n_remain}};
}

}  // namespace grammar_detail

// A parsed GBNF grammar plus the position(s) it is currently in.
//
// GBNF (llama.cpp's):
//   root   ::= "yes" | "no"
//   object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? ws "}"
//   digit  ::= [0-9]
// with `"..."` literals (\n \r \t \\ \" \xHH \uHHHH escapes), `[a-z]` / `[^a-z]` classes,
// `.` for any character, `(...)` groups, the `*` `+` `?` operators, `|` alternation and
// `#` comments.
class Grammar {
 public:
  using Elem = grammar_detail::Elem;
  using Rule = grammar_detail::Rule;
  using Stack = std::vector<const Elem*>;
  using Stacks = std::vector<Stack>;

  // Parse `gbnf` and position the grammar at the start of `root_rule`.
  explicit Grammar(const std::string& gbnf, const std::string& root_rule = "root") {
    parse(gbnf);
    auto it = names_.find(root_rule);
    if (it == names_.end())
      throw std::runtime_error("[grammar] no rule named '" + root_rule + "'");
    root_ = it->second;
    // A reference to a name that is never defined leaves an EMPTY rule behind (ruleId
    // creates the slot on first mention), which the matcher would walk off the end of.
    // Catch it here, with the name, instead of crashing mid-generation.
    for (size_t i = 0; i < rules_.size(); ++i)
      for (const auto& e : rules_[i])
        if (e.type == grammar_detail::ET::RuleRef && !defined_[e.value])
          throw std::runtime_error("[grammar] rule '" + nameOf(i) + "' references '" +
                                   nameOf(e.value) + "', which is never defined");
    reset();
  }

  // Back to the start of the root rule (a fresh generation).
  void reset() {
    partial_ = {};
    stacks_.clear();
    Stack s;
    const Rule& r = rules_[root_];
    size_t i = 0;
    while (true) {                              // one starting stack per root alternate
      Stack alt;
      if (!grammar_detail::isEndOfSequence(&r[i])) alt.push_back(&r[i]);
      advanceStack(alt, stacks_);
      while (!grammar_detail::isEndOfSequence(&r[i])) ++i;
      if (r[i].type == grammar_detail::ET::Alt) ++i; else break;
    }
  }

  // Is the grammar satisfied here — i.e. may generation stop (EOS) now?
  bool can_end() const {
    for (const auto& s : stacks_) if (s.empty()) return true;
    return false;
  }

  // Is the grammar still live (any position at all)?
  bool alive() const { return !stacks_.empty(); }

  // Satisfied AND unable to consume anything more — the text is finished, only EOS fits.
  bool complete() const {
    for (const auto& s : stacks_) if (!s.empty()) return false;
    return !stacks_.empty();
  }

  // Would `text` keep the grammar satisfiable? Does not change the state.
  bool accepts(const std::string& text) const {
    if (complete()) return false;      // fast path: nothing can follow a finished grammar
    Stacks st = stacks_;
    grammar_detail::PartialUtf8 p = partial_;
    return feed(text, st, p);
  }

  // Commit `text`. Throws if it does not fit — callers should have asked accepts() first.
  void accept(const std::string& text) {
    if (!feed(text, stacks_, partial_))
      throw std::runtime_error("[grammar] token '" + text + "' is not allowed here");
  }

 private:
  // Push every position reachable from `stack` without consuming input: a rule reference
  // expands into one stack per alternate of the referenced rule, recursively.
  void advanceStack(const Stack& stack, Stacks& out) const {
    using grammar_detail::ET;
    // Dedup: the same position is often reachable by several paths (any rule that can be
    // entered two ways), and without this a recursive rule multiplies the stack set on
    // every step until the matcher crawls.
    for (const auto& s : out) if (s == stack) return;
    if (stack.empty()) {
      out.push_back(stack);                    // an accepting position
      return;
    }
    const Elem* pos = stack.back();
    switch (pos->type) {
      case ET::RuleRef: {
        const Rule& sub = rules_[pos->value];
        size_t i = 0;
        while (true) {
          Stack next(stack.begin(), stack.end() - 1);          // pop the reference
          if (!grammar_detail::isEndOfSequence(pos + 1))
            next.push_back(pos + 1);                           // what follows it
          if (!grammar_detail::isEndOfSequence(&sub[i]))
            next.push_back(&sub[i]);                           // this alternate
          advanceStack(next, out);
          while (!grammar_detail::isEndOfSequence(&sub[i])) ++i;
          if (sub[i].type == ET::Alt) ++i; else break;
        }
        break;
      }
      case ET::Char:
      case ET::CharNot:
      case ET::CharAny:
        out.push_back(stack);                  // waiting on input
        break;
      default:
        throw std::runtime_error("[grammar] internal: unexpected element on a stack");
    }
  }

  // Does `cp` match the char set starting at `pos`? Returns the element after the set.
  static std::pair<bool, const Elem*> matchChar(const Elem* pos, uint32_t cp) {
    using grammar_detail::ET;
    const bool positive = pos->type != ET::CharNot;
    bool found = false;
    do {
      if (pos->type == ET::CharAny) {
        found = true;
        pos += 1;
      } else if (pos[1].type == ET::CharRngUpper) {
        found = found || (pos->value <= cp && cp <= pos[1].value);   // [a-z]
        pos += 2;
      } else {
        found = found || pos->value == cp;
        pos += 1;
      }
    } while (pos->type == ET::CharAlt);
    return {found == positive, pos};
  }

  // Could a code point that is only half read still match the set at `pos`? The bits so far
  // pin the code point to a range; the set matches if the range overlaps it. Conservative
  // by construction: it may say yes to a character that turns out not to match, which costs
  // one wasted candidate, never a wrong acceptance (the full check runs when it completes).
  static bool matchPartial(const Elem* pos, grammar_detail::PartialUtf8 p) {
    using grammar_detail::ET;
    const bool positive = pos->type != ET::CharNot;
    if (p.n_remain < 0) return false;                       // invalid UTF-8
    uint32_t low = p.value << (p.n_remain * 6);
    const uint32_t high = low | ((1u << (p.n_remain * 6)) - 1);
    if (low == 0) {                                          // no bits yet: use the minimum
      if (p.n_remain == 1) low = 1 << 7;                     // a code point of this length
      else if (p.n_remain == 2) low = 1 << 11;
      else if (p.n_remain == 3) low = 1 << 16;
    }
    do {
      if (pos->type == ET::CharAny) return positive;
      if (pos[1].type == ET::CharRngUpper) {
        if (pos->value <= high && low <= pos[1].value) return positive;
        pos += 2;
      } else {
        if (low <= pos->value && pos->value <= high) return positive;
        pos += 1;
      }
    } while (pos->type == ET::CharAlt);
    return !positive;
  }

  // Run `text` through `stacks`/`partial`, in place. False = the grammar rejects it, and
  // the arguments are then meaningless (callers pass copies when only asking).
  bool feed(const std::string& text, Stacks& stacks, grammar_detail::PartialUtf8& p) const {
    if (stacks.empty()) return false;
    auto [cps, rest] = grammar_detail::decodeUtf8(text, p);
    if (rest.n_remain < 0) return false;                    // not valid UTF-8 at all
    for (const uint32_t cp : cps) {
      Stacks next;
      for (const auto& s : stacks) {
        if (s.empty()) continue;                            // finished: consumes nothing
        auto [ok, after] = matchChar(s.back(), cp);
        if (!ok) continue;
        Stack ns(s.begin(), s.end() - 1);
        if (!grammar_detail::isEndOfSequence(after)) ns.push_back(after);
        advanceStack(ns, next);
      }
      if (next.empty()) return false;
      stacks = std::move(next);
    }
    p = rest;
    if (p.n_remain > 0) {
      // The text ended mid-character. Keep only the positions where the half-read code
      // point can still land somewhere valid.
      Stacks live;
      for (const auto& s : stacks)
        if (!s.empty() && matchPartial(s.back(), p)) live.push_back(s);
      if (live.empty()) return false;
      stacks = std::move(live);
    }
    return true;
  }

  // --- parser ---------------------------------------------------------------
  // GBNF text -> rules_. Sub-rules are synthesised for groups and for the *, + and ?
  // operators, exactly as llama.cpp does: `X*` becomes `sub ::= X sub |` (right-recursive),
  // which is why repetition needs no dedicated element type.
  void parse(const std::string& src) {
    const char* p = src.c_str();
    p = skipSpace(p, true);
    while (*p) {
      parseRule(p);
      p = skipSpace(p, true);
    }
    if (rules_.empty()) throw std::runtime_error("[grammar] no rules");
  }

  static bool isWordChar(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
           c == '-' || c == '_';
  }

  // Skip spaces and comments. `newlines` = also skip line breaks (between rules).
  static const char* skipSpace(const char* p, bool newlines) {
    while (*p == ' ' || *p == '\t' || *p == '#' || (newlines && (*p == '\n' || *p == '\r'))) {
      if (*p == '#') { while (*p && *p != '\n') ++p; }
      else ++p;
    }
    return p;
  }

  uint32_t ruleId(const std::string& name) {
    auto it = names_.find(name);
    if (it != names_.end()) return it->second;
    const uint32_t id = (uint32_t)rules_.size();
    names_[name] = id;
    rules_.emplace_back();
    defined_.push_back(false);
    return id;
  }

  std::string nameOf(size_t id) const {
    for (const auto& kv : names_) if (kv.second == id) return kv.first;
    return "<" + std::to_string(id) + ">";
  }

  void parseRule(const char*& p) {
    const char* start = p;
    while (isWordChar(*p)) ++p;
    if (p == start) throw std::runtime_error("[grammar] expected a rule name");
    const std::string name(start, p);
    p = skipSpace(p, false);
    if (!(p[0] == ':' && p[1] == ':' && p[2] == '='))
      throw std::runtime_error("[grammar] expected '::=' after rule '" + name + "'");
    p += 3;
    Rule r;
    parseAlternates(p, name, r);
    const uint32_t id = ruleId(name);
    rules_[id] = std::move(r);
    defined_[id] = true;
    p = skipSpace(p, false);
    if (*p && *p != '\n' && *p != '\r')
      throw std::runtime_error("[grammar] unexpected text after rule '" + name + "'");
  }

  void parseAlternates(const char*& p, const std::string& name, Rule& out) {
    parseSequence(p, name, out);
    p = skipSpace(p, false);
    while (*p == '|') {
      out.push_back({grammar_detail::ET::Alt, 0});
      ++p;
      p = skipSpace(p, true);
      parseSequence(p, name, out);
      p = skipSpace(p, false);
    }
    out.push_back({grammar_detail::ET::End, 0});
  }

  void parseSequence(const char*& p, const std::string& name, Rule& out) {
    using grammar_detail::ET;
    p = skipSpace(p, false);
    while (true) {
      const size_t sym_start = out.size();      // where the symbol we may repeat begins
      if (*p == '"') {                          // literal: one Char element per code point
        ++p;
        while (*p != '"') {
          if (!*p) throw std::runtime_error("[grammar] unterminated string in '" + name + "'");
          out.push_back({ET::Char, parseChar(p)});
        }
        ++p;
      } else if (*p == '[') {                   // char class
        ++p;
        ET first = ET::Char;
        if (*p == '^') { ++p; first = ET::CharNot; }
        bool started = false;
        while (*p != ']') {
          if (!*p) throw std::runtime_error("[grammar] unterminated [...] in '" + name + "'");
          const uint32_t c = parseChar(p);
          out.push_back({started ? ET::CharAlt : first, c});
          started = true;
          if (*p == '-' && p[1] != ']') {       // a-z
            ++p;
            out.push_back({ET::CharRngUpper, parseChar(p)});
          }
        }
        if (!started) throw std::runtime_error("[grammar] empty [] in '" + name + "'");
        ++p;
      } else if (*p == '.') {
        ++p;
        out.push_back({ET::CharAny, 0});
      } else if (*p == '(') {                   // group -> its own rule
        ++p;
        p = skipSpace(p, true);
        const std::string sub = subName(name);
        Rule r;
        parseAlternates(p, sub, r);
        const uint32_t id = ruleId(sub);
        rules_[id] = std::move(r);
        defined_[id] = true;
        out.push_back({ET::RuleRef, id});
        p = skipSpace(p, true);
        if (*p != ')') throw std::runtime_error("[grammar] expected ')' in '" + name + "'");
        ++p;
      } else if (isWordChar(*p)) {              // reference to another rule
        const char* s = p;
        while (isWordChar(*p)) ++p;
        out.push_back({ET::RuleRef, ruleId(std::string(s, p))});
      } else {
        break;                                  // end of this sequence
      }
      p = skipSpace(p, false);
      if (*p == '*' || *p == '+' || *p == '?') {
        applyRepeat(*p, out, sym_start, name);
        ++p;
        p = skipSpace(p, false);
      }
    }
  }

  // Rewrite the symbol at [from, end) as a reference to a fresh recursive rule.
  //   X*  ->  sub ::= X sub |          (zero or more)
  //   X+  ->  sub ::= X sub | X        (one or more)
  //   X?  ->  sub ::= X |              (optional)
  void applyRepeat(char op, Rule& out, size_t from, const std::string& name) {
    using grammar_detail::ET;
    if (from == out.size())
      throw std::runtime_error("[grammar] '" + std::string(1, op) +
                               "' with nothing before it in '" + name + "'");
    const Rule sym(out.begin() + from, out.end());
    const std::string sub = subName(name);
    const uint32_t id = ruleId(sub);
    Rule r = sym;
    if (op == '*' || op == '+') r.push_back({ET::RuleRef, id});   // recurse
    r.push_back({ET::Alt, 0});
    if (op == '+') r.insert(r.end(), sym.begin(), sym.end());     // ... | X
    r.push_back({ET::End, 0});
    rules_[id] = std::move(r);
    defined_[id] = true;
    out.resize(from);
    out.push_back({ET::RuleRef, id});
  }

  std::string subName(const std::string& parent) {
    return parent + "_" + std::to_string(next_sub_++);
  }

  // One character of a literal or class, resolving escapes.
  static uint32_t parseChar(const char*& p) {
    if (*p == '\\') {
      ++p;
      switch (*p) {
        case 'n': ++p; return '\n';
        case 'r': ++p; return '\r';
        case 't': ++p; return '\t';
        case '\\': ++p; return '\\';
        case '"': ++p; return '"';
        case '\'': ++p; return '\'';
        case '[': ++p; return '[';
        case ']': ++p; return ']';
        case 'x': return parseHex(++p, 2);
        case 'u': return parseHex(++p, 4);
        case 'U': return parseHex(++p, 8);
        default:
          throw std::runtime_error("[grammar] unknown escape \\" + std::string(1, *p));
      }
    }
    // A literal is written in UTF-8; take a whole code point, so "北" is one element.
    const int len = grammar_detail::utf8Len((uint8_t)*p);
    if (len == 0) throw std::runtime_error("[grammar] invalid UTF-8 in the grammar");
    auto [cps, rest] = grammar_detail::decodeUtf8(std::string(p, len), {});
    if (cps.size() != 1 || rest.n_remain != 0)
      throw std::runtime_error("[grammar] invalid UTF-8 in the grammar");
    p += len;
    return cps[0];
  }

  static uint32_t parseHex(const char*& p, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; ++i, ++p) {
      const char c = *p;
      v <<= 4;
      if (c >= '0' && c <= '9') v += c - '0';
      else if (c >= 'a' && c <= 'f') v += c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v += c - 'A' + 10;
      else throw std::runtime_error("[grammar] expected " + std::to_string(n) +
                                    " hex digits in an escape");
    }
    return v;
  }

  std::vector<Rule> rules_;
  std::vector<bool> defined_;              // a name can be referenced before it is defined
  std::map<std::string, uint32_t> names_;
  uint32_t root_ = 0;
  int next_sub_ = 0;
  Stacks stacks_;
  grammar_detail::PartialUtf8 partial_;
};

// --- vocabulary glue --------------------------------------------------------

// The bytes a token contributes, from its raw tokenizer form.
//
// Byte-level BPE (GPT-2's, which Qwen and Phi both use) does not store token text: it
// stores each byte remapped to a printable code point, so a space is "Ġ" and 0xE5 is "å".
// Decoding the id instead would lose exactly the tokens that matter here — one that holds
// half a CJK character comes back as U+FFFD, and the grammar would see a replacement
// character rather than the bytes it needs to match. So invert the mapping instead.
//
// Returns false for a token that is not byte-level text at all (`<|im_end|>` and friends):
// those are control tokens and must never be offered to a grammar.
inline uint8_t parseHexByte(const char* p) {
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::runtime_error("[grammar] bad hex byte");
  };
  return (uint8_t)((hex(p[0]) << 4) | hex(p[1]));
}

inline bool decodeByteLevelToken(const std::string& tok, std::string* out) {
  static const std::unordered_map<uint32_t, uint8_t> kInverse = [] {
    std::unordered_map<uint32_t, uint8_t> m;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
      const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                             (b >= 0xAE && b <= 0xFF);
      m[printable ? (uint32_t)b : (uint32_t)(256 + extra++)] = (uint8_t)b;
    }
    return m;
  }();
  out->clear();
  // SentencePiece-style byte fallback, e.g. "<0x0A>" — rare here, cheap to honour.
  if (tok.size() == 6 && tok.compare(0, 3, "<0x") == 0 && tok[5] == '>') {
    try { out->push_back((char)parseHexByte(tok.c_str() + 3)); } catch (...) { return false; }
    return true;
  }
  auto [cps, rest] = grammar_detail::decodeUtf8(tok, {});
  if (rest.n_remain != 0) return false;
  for (const uint32_t cp : cps) {
    auto it = kInverse.find(cp);
    if (it == kInverse.end()) return false;         // not in the byte alphabet
    out->push_back((char)it->second);
  }
  return true;
}

// Does this raw token look like a control token rather than text?
//
// A grammar must never be able to match `<|im_end|>` as if it were the literal characters
// "<|im_end|>" — that would let a constrained reply smuggle in a chat marker. The byte-level
// mapping cannot tell them apart (both are printable ASCII), and the tokenizer API exposes
// no "is special" flag, so this recognises the `<|...|>` convention that the ChatML family
// (Qwen, Phi, DeepSeek) uses for all of them. Callers should ALSO blank the ids they know
// from the model config — this is the backstop, not the primary defence.
//
// The cost of being wrong: a model whose ordinary vocabulary contained a literal `<|x|>`
// string would lose it inside grammars. No tokenizer in this family has one.
inline bool isControlTokenText(const std::string& tok) {
  return tok.size() >= 4 && tok.compare(0, 2, "<|") == 0 &&
         tok.compare(tok.size() - 2, 2, "|>") == 0;
}

// The sampler-facing view of a grammar: which token ids may come next.
//
// Holds the token->bytes table (built once per session) so the per-step test is a string
// walk and not a tokenizer call. Control tokens are never allowed: EOS is the engine's
// business, and letting a grammar match `<|im_end|>` as literal text would be nonsense.
class GrammarConstraint : public TokenConstraint {
 public:
  // `text[id]` = the bytes of token `id`, empty for control/opaque tokens.
  GrammarConstraint(Grammar grammar, std::vector<std::string> text)
      : g_(std::move(grammar)), text_(std::move(text)) {}

  bool allows(int id) const override {
    if (id < 0 || id >= (int)text_.size() || text_[id].empty()) return false;
    return g_.accepts(text_[id]);
  }
  void accept(int id) override {
    if (id >= 0 && id < (int)text_.size() && !text_[id].empty()) g_.accept(text_[id]);
  }
  bool can_end() const override { return g_.can_end(); }

  void reset() { g_.reset(); }
  Grammar& grammar() { return g_; }

 private:
  Grammar g_;
  std::vector<std::string> text_;
};

}  // namespace bllm
