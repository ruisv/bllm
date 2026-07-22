// Unit test for bllm::Grammar — GBNF parsing + the constrained-decoding matcher, and its
// integration with bllm::Sampler. Pure logic (no hobot, no tokenizer), so it builds and
// runs on the Mac:
//
//   c++ -std=c++17 -I include tests/test_native_grammar.cc -o /tmp/tg && /tmp/tg
#include "bllm/native_grammar.h"

#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
  } while (0)

// Feed a whole string one piece at a time; true if every piece was accepted.
static bool walk(bllm::Grammar& g, const std::vector<std::string>& pieces) {
  for (const auto& s : pieces) {
    if (!g.accepts(s)) return false;
    g.accept(s);
  }
  return true;
}

// Does the grammar accept exactly this string, ending satisfied?
static bool matches(const std::string& gbnf, const std::string& text) {
  bllm::Grammar g(gbnf);
  if (!g.accepts(text)) return false;
  g.accept(text);
  return g.can_end();
}

int main() {
  // --- literals and alternation --------------------------------------------
  {
    const std::string gbnf = "root ::= \"yes\" | \"no\"\n";
    CHECK(matches(gbnf, "yes"));
    CHECK(matches(gbnf, "no"));
    CHECK(!matches(gbnf, "maybe"));
    bllm::Grammar g(gbnf);
    CHECK(!g.can_end());                       // "" is not a complete answer
    CHECK(g.accepts("y"));
    g.accept("y");
    CHECK(!g.can_end());                       // mid-word
    CHECK(!g.accepts("n"));                    // the "no" branch died at "y"
    g.accept("es");
    CHECK(g.can_end() && g.complete());
    CHECK(!g.accepts("!"));                    // finished: nothing may follow
  }

  // A prefix shared by two alternates keeps both alive (the automaton is a SET of stacks).
  {
    bllm::Grammar g("root ::= \"ab\" | \"ac\"\n");
    g.accept("a");
    CHECK(g.accepts("b") && g.accepts("c"));
    g.accept("c");
    CHECK(g.can_end());
  }

  // --- char classes, ranges, negation, "." ---------------------------------
  {
    CHECK(matches("root ::= [a-z]\n", "q"));
    CHECK(!matches("root ::= [a-z]\n", "Q"));
    CHECK(matches("root ::= [a-zA-Z0-9_]\n", "_"));
    CHECK(matches("root ::= [^0-9]\n", "x"));
    CHECK(!matches("root ::= [^0-9]\n", "7"));
    CHECK(matches("root ::= .\n", "\xE5\x8C\x97"));      // "北" — one code point, not 3 bytes
    CHECK(matches("root ::= [\\x41-\\x43]\n", "B"));      // hex escapes
    CHECK(matches("root ::= \"\\n\"\n", "\n"));
  }

  // --- operators ------------------------------------------------------------
  {
    CHECK(matches("root ::= [0-9]*\n", ""));
    CHECK(matches("root ::= [0-9]*\n", "12345"));
    CHECK(!matches("root ::= [0-9]+\n", ""));
    CHECK(matches("root ::= [0-9]+\n", "7"));
    CHECK(matches("root ::= \"a\"? \"b\"\n", "b"));
    CHECK(matches("root ::= \"a\"? \"b\"\n", "ab"));
    CHECK(!matches("root ::= \"a\"? \"b\"\n", "aab"));
    // A group repeats as a unit.
    CHECK(matches("root ::= (\"ab\")+\n", "ababab"));
    CHECK(!matches("root ::= (\"ab\")+\n", "aba"));
  }

  // --- rule references and recursion ---------------------------------------
  {
    const std::string gbnf =
        "root  ::= list\n"
        "list  ::= \"[\" (item (\",\" item)*)? \"]\"\n"
        "item  ::= [0-9]+\n";
    CHECK(matches(gbnf, "[]"));
    CHECK(matches(gbnf, "[1]"));
    CHECK(matches(gbnf, "[1,22,333]"));
    CHECK(!matches(gbnf, "[1,]"));
    CHECK(!matches(gbnf, "[1"));               // unterminated: valid so far, not complete
    bllm::Grammar g(gbnf);
    g.accept("[1");
    CHECK(!g.can_end());
  }

  // Nested recursion: the classic JSON-ish value.
  {
    const std::string gbnf =
        "root   ::= value\n"
        "value  ::= object | array | number | \"true\" | \"false\" | \"null\"\n"
        "object ::= \"{\" (pair (\",\" pair)*)? \"}\"\n"
        "pair   ::= \"\\\"\" [a-z]+ \"\\\"\" \":\" value\n"
        "array  ::= \"[\" (value (\",\" value)*)? \"]\"\n"
        "number ::= \"-\"? [0-9]+\n";
    CHECK(matches(gbnf, "{\"a\":1}"));
    CHECK(matches(gbnf, "{\"a\":{\"b\":[1,-2,true]},\"c\":null}"));
    CHECK(!matches(gbnf, "{\"a\":1"));
    CHECK(!matches(gbnf, "{a:1}"));
  }

  // --- tokens are not characters -------------------------------------------
  // A BPE token is an arbitrary byte string, and a multi-byte character is routinely split
  // across two of them. The matcher must hold the half-read code point between tokens.
  {
    bllm::Grammar g("root ::= \"北京\"\n");
    const std::string bj = "\xE5\x8C\x97\xE4\xBA\xAC";
    CHECK(walk(g, {bj.substr(0, 1), bj.substr(1, 3), bj.substr(4)}));   // split mid-character
    CHECK(g.can_end());
  }
  {
    // And a half character that cannot possibly complete into something allowed is rejected
    // right there, not after the fact: [a-z] can never be reached by a 3-byte lead.
    bllm::Grammar g("root ::= [a-z]+\n");
    CHECK(!g.accepts("\xE5"));
    // while the same lead is fine when the grammar wants a CJK range
    bllm::Grammar cjk("root ::= [\\u4e00-\\u9fff]+\n");
    CHECK(cjk.accepts("\xE5"));
  }

  // --- left recursion is rejected, not crashed on --------------------------
  // A rule that reaches itself without consuming input expands forever in advanceStack.
  // Grammars arrive over HTTP, so this was a remote SIGSEGV (measured: exit 139) before the
  // constructor started checking. Every shape here used to take the process down.
  {
    auto rejects = [](const std::string& gbnf) {
      try { bllm::Grammar g(gbnf); } catch (const std::exception&) { return true; }
      return false;
    };
    CHECK(rejects("root ::= root\n"));                        // direct
    CHECK(rejects("root ::= a\na ::= b\nb ::= a\n"));          // indirect
    CHECK(rejects("root ::= \"a\" | root\n"));                 // via an alternate
    CHECK(rejects("root ::= opt root\nopt ::= \"x\"?\n"));     // past a nullable prefix
    CHECK(rejects("root ::= x*\nx ::= \"a\"?\n"));             // a nullable body under *

    // ... while right recursion — what *, + and ? actually expand into — stays legal.
    auto accepts_grammar = [](const std::string& gbnf) {
      try { bllm::Grammar g(gbnf); } catch (const std::exception&) { return false; }
      return true;
    };
    CHECK(accepts_grammar("root ::= item*\nitem ::= \"a\" | \"ab\"\n"));
    CHECK(accepts_grammar("root ::= [0-9]+\n"));
    CHECK(accepts_grammar("root ::= (\"a\" (\"b\")*)+ \"c\"\n"));
    CHECK(accepts_grammar("root ::= v\nv ::= o | \"1\"\no ::= \"{\" v \"}\"\n"));  // nested
  }

  // A Grammar's stacks point INTO its own rule storage, so it must move, never copy — a
  // copy would keep pointing at the source and dangle when the source dies. The copy
  // constructor is deleted, which makes that a compile error; moving must still work,
  // because that is how GrammarConstraint takes ownership.
  {
    static_assert(!std::is_copy_constructible<bllm::Grammar>::value,
                  "Grammar must not be copyable: its stacks alias its own rules");
    static_assert(std::is_move_constructible<bllm::Grammar>::value, "moving must work");
    bllm::Grammar a("root ::= \"ab\"\n");
    bllm::Grammar b = std::move(a);
    CHECK(b.accepts("a"));
    b.accept("a");
    CHECK(b.accepts("b") && !b.can_end());
  }

  // --- errors are reported, not guessed ------------------------------------
  {
    auto rejects = [](const std::string& gbnf) {
      try { bllm::Grammar g(gbnf); } catch (const std::exception&) { return true; }
      return false;
    };
    CHECK(rejects("root = \"a\"\n"));                  // "=" is not "::="
    CHECK(rejects("nope ::= \"a\"\n"));                // no root rule
    CHECK(rejects("root ::= undefined_rule\n"));       // dangling reference
    CHECK(rejects("root ::= \"unterminated\n"));
    CHECK(rejects("root ::= [a-z\n"));
    CHECK(rejects("root ::= *\n"));                    // operator with nothing to repeat
    CHECK(rejects("root ::= \"a\" \\q\n"));            // unknown escape
  }

  // --- byte-level token text ------------------------------------------------
  {
    std::string out;
    CHECK(bllm::decodeByteLevelToken("hello", &out) && out == "hello");
    CHECK(bllm::decodeByteLevelToken("\xC4\xA0the", &out) && out == " the");  // "Ġthe"
    // "北" byte-level: 0xE5 0x8C 0x97 -> "åĮĹ"
    CHECK(bllm::decodeByteLevelToken("\xC3\xA5\xC4\xAE\xC4\xB9", &out) &&
          out == "\xE5\x8C\x97");
    CHECK(bllm::decodeByteLevelToken("<0x0A>", &out) && out == "\n");
    // A control token IS printable ASCII, so the byte mapping cannot reject it — the
    // separate check is what keeps a grammar from matching "<|im_end|>" as literal text.
    CHECK(bllm::decodeByteLevelToken("<|im_end|>", &out));
    CHECK(bllm::isControlTokenText("<|im_end|>"));
    CHECK(bllm::isControlTokenText("<|endoftext|>"));
    CHECK(!bllm::isControlTokenText("hello"));
    CHECK(!bllm::isControlTokenText("<div>"));
  }

  // --- sampler integration --------------------------------------------------
  // The constraint has to survive the sampler's candidate truncation: the cut is taken on
  // raw logits, and the tokens a grammar allows may all be below it.
  {
    // A toy vocab: id -> text. 0 is eos.
    std::vector<std::string> vocab(2000);
    for (size_t i = 0; i < vocab.size(); ++i) vocab[i] = "x";        // filler
    vocab[0] = "";                                                    // eos / control
    vocab[1500] = "yes";
    vocab[1501] = "no";

    auto make = [&](const char* gbnf) {
      return bllm::GrammarConstraint(bllm::Grammar(gbnf), vocab);
    };
    auto constraint = make("root ::= \"yes\" | \"no\"\n");

    bllm::NativeSamplingParams p;
    p.constraint = &constraint;
    p.eos = {0};
    // Every filler token outranks the two allowed ones, and there are more fillers than the
    // 512-candidate budget — so this only works if the vocab is searched when the cut comes
    // back empty.
    std::vector<float> lg(vocab.size(), 5.0f);
    lg[1500] = -20.0f;
    lg[1501] = -21.0f;
    bllm::Sampler s(p);
    const int t = s.pick([&](int i) { return lg[i]; }, (int)lg.size());
    CHECK(t == 1500);                            // the higher-logit allowed token
    s.record(t);
    // "yes" is complete: only eos may follow.
    const int t2 = s.pick([&](int i) { return lg[i]; }, (int)lg.size());
    CHECK(t2 == 0);
  }
  {
    // A grammar that can end but can also continue must leave eos available, or generation
    // could never stop.
    std::vector<std::string> vocab = {"", "1", "2", "a"};
    bllm::GrammarConstraint constraint(bllm::Grammar("root ::= [0-9]+\n"), vocab);
    bllm::NativeSamplingParams p;
    p.constraint = &constraint;
    p.eos = {0};
    bllm::Sampler s(p);
    std::vector<float> lg = {-1.0f, 5.0f, 0.0f, 9.0f};    // "a" leads, but is not allowed
    const int t = s.pick([&](int i) { return lg[i]; }, 4);
    CHECK(t == 1);
    s.record(t);
    lg[0] = 100.0f;                                        // now eos leads
    CHECK(s.pick([&](int i) { return lg[i]; }, 4) == 0);   // allowed: the grammar is satisfied
  }

  {
    // With nothing left to allow and no eos to fall back on, the sampler says "stop" (-1)
    // instead of picking a token the grammar forbids.
    std::vector<std::string> vocab = {"a", "b"};
    bllm::GrammarConstraint constraint(bllm::Grammar("root ::= \"a\"\n"), vocab);
    bllm::NativeSamplingParams p;
    p.constraint = &constraint;
    p.eos = {};
    bllm::Sampler s(p);
    std::vector<float> lg = {1.0f, 2.0f};
    CHECK(s.pick([&](int i) { return lg[i]; }, 2) == 0);
    s.record(0);
    CHECK(s.pick([&](int i) { return lg[i]; }, 2) == -1);
  }

  // A deeply recursive grammar must not make the stack set explode: without dedup in
  // advanceStack this walk slows to a crawl as the alternates multiply.
  {
    bllm::Grammar g("root ::= item*\nitem ::= \"a\" | \"ab\" | \"abc\"\n");
    for (int i = 0; i < 200; ++i) { CHECK(g.accepts("a")); g.accept("a"); }
    CHECK(g.can_end());
  }

  if (failures == 0) std::printf("all native-grammar tests passed\n");
  return failures == 0 ? 0 : 1;
}
