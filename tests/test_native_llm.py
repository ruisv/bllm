"""The native text runtime: the dense and hybrid engines, no libxlm.

Point BLLM_TEST_NATIVE_DIR at a model directory (`model.json`); everything skips
otherwise. Run pytest on the board with the `BLLM_TEST_*` env vars set (see CONTRIBUTING.md).

**One big model per process.** A session pins its weights in the BPU's ION carveout
until it is destroyed — a hybrid session plus a 448 px omni session at once exhausted
the carveout and reset the board. Hence the module-scoped fixture, and hence the VLM
tests living in their own module: pytest finalises this one before that one starts.
"""

import gc
import os

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")

NATIVE_DIR = os.environ.get("BLLM_TEST_NATIVE_DIR")
OMNI_DIR = os.environ.get("BLLM_TEST_OMNI_DIR")


@pytest.fixture(scope="module")
def llm():
    if not (bllm._HAVE_NATIVE and NATIVE_DIR):
        pytest.skip("set BLLM_TEST_NATIVE_DIR")
    session = bllm.NativeSession(NATIVE_DIR)
    yield session
    del session
    gc.collect()          # release the ION carveout before the next module loads a model


def test_chat_and_stream(llm):
    llm.reset()
    assert llm.arch in ("dense", "hybrid")
    assert "巴黎" in llm.chat("法国的首都是哪里？只答城市名。", max_new=16)
    llm.reset()
    chunks = list(llm.stream_chat("用一句话介绍北京。", max_new=32))
    assert chunks and "".join(chunks)
    assert llm.last_decode_tps >= 0


def test_stop_string_trims_reply_and_stream(llm):
    """A stop string ends the turn, and appears in neither the return value nor the
    streamed chunks — its own prefix is held back until it completes."""
    llm.reset()
    # Count 1..9 on separate lines, then stop at "5". The reply must end before it, and
    # the stop string spans the "\n" / "5" token boundary, so this exercises the holdback.
    chunks = []
    reply = llm.chat("从1数到9，每个数字单独一行，只输出数字。", max_new=60,
                     on_text=chunks.append, stop=["5"])
    assert "5" not in reply, reply
    assert "1" in reply and "4" in reply, reply
    assert "5" not in "".join(chunks), chunks           # nothing past the stop leaked
    assert "".join(chunks) == reply                     # stream and return agree


def test_stop_string_absent_returns_full_reply(llm):
    """A stop string that never occurs must not truncate anything."""
    llm.reset()
    reply = llm.chat("只回答一个字：好", max_new=8, stop=["\n\n\n<<never>>"])
    assert reply.strip()


def test_sampling_knobs_at_libxlm_parity(llm):
    """The full libxlm knob set drives the native sampler: greedy is deterministic, and
    min_p / typ_p / top-k/p with repeat+frequency+presence penalties still generate
    coherently (unit-level correctness is in tests/test_native_sampler.cc)."""
    llm.reset()
    llm.set_sampling()                                    # all defaults => greedy
    a = llm.chat("法国的首都是哪里？只答城市名。", max_new=8); llm.reset()
    b = llm.chat("法国的首都是哪里？只答城市名。", max_new=8); llm.reset()
    assert a == b and "巴黎" in a                          # greedy is reproducible

    llm.set_sampling(temp=0.7, top_p=0.9, top_k=40, min_p=0.05, typ_p=0.95,
                     penalty_last_n=64, rep_pen=1.1, penalty_freq=0.5, penalty_present=0.5,
                     seed=1)
    r = llm.chat("用一句话介绍北京。", max_new=40)
    assert r.strip() and len(r) > 4                        # coherent, no crash
    llm.reset()
    llm.set_sampling()                                     # restore greedy for later tests


def test_temperature_varies_unless_a_seed_is_given(llm):
    """The default seed must be SEED_RANDOM, not a fixed number.

    A fixed default silently disabled temperature: the sampler is constructed per
    generation, so it replayed the identical random stream every turn and the same
    question came back byte-identical no matter how high the temperature. Measured
    before the fix: three temperature=0.8 requests, three identical replies.
    """
    q = "讲一个很短的笑话。"

    llm.set_sampling(temp=0.9)                 # default seed => fresh draw per generation
    replies = set()
    for _ in range(4):
        llm.reset()
        replies.add(llm.chat(q, max_new=28))
    assert len(replies) > 1, f"temperature had no effect — all {len(replies)} replies identical"

    llm.set_sampling(temp=0.9, seed=99)        # explicit seed => exactly reproducible
    llm.reset(); a = llm.chat(q, max_new=28)
    llm.reset(); b = llm.chat(q, max_new=28)
    assert a == b, "an explicit seed did not reproduce"

    llm.reset()
    llm.set_sampling()                         # restore greedy for later tests


def test_logit_bias_bans_and_forces_tokens(llm):
    """logit_bias is the one knob that can push a token UP, past the candidate cut the
    sampler takes on raw logits — so this exercises more than the arithmetic (the unit
    coverage for the truncation invariant is in tests/test_native_sampler.cc)."""
    q = "法国的首都是哪里？只答城市名。"
    llm.reset(); llm.set_sampling()
    plain = llm.chat(q, max_new=8)
    assert "巴黎" in plain

    banned = llm.encode("巴黎")                        # every token the answer is made of
    llm.reset()
    llm.set_sampling(logit_bias={t: -100.0 for t in banned})
    assert "巴黎" not in llm.chat(q, max_new=8)

    # And the other direction: +100 on a token the answer would never contain forces it.
    forced = llm.encode("香蕉")
    llm.reset()
    llm.set_sampling(logit_bias={forced[0]: 100.0})
    assert llm.decode([forced[0]]) in llm.chat(q, max_new=8)

    llm.reset()
    llm.set_sampling()                                 # clears the bias for later tests
    assert "巴黎" in llm.chat(q, max_new=8)


def test_logprobs_report_the_real_distribution(llm):
    """logprobs are an exact full-vocab log-softmax of the raw logits, one entry per
    generated token — so they are real probabilities (<= 0, alternatives summing to <= 1)
    and they line up with the reply."""
    import math

    llm.reset()
    llm.set_sampling(logprobs=3)
    reply = llm.chat("法国的首都是哪里？只答城市名。", max_new=8)
    lp = llm.last_logprobs()
    assert lp, "logprobs=3 produced no report"
    assert "".join(llm.decode([e["id"]]) for e in lp) == reply   # one entry per reply token

    for e in lp:
        assert e["logprob"] <= 0.0
        assert len(e["top"]) == 3
        alts = [p for _, p in e["top"]]
        assert alts == sorted(alts, reverse=True)                # most likely first
        assert sum(math.exp(p) for p in alts) <= 1.0 + 1e-4      # a slice of one distribution
        # greedy picked the argmax, and the report is of the same (unadjusted) logits
        assert e["id"] == e["top"][0][0]
        assert e["logprob"] == pytest.approx(e["top"][0][1], abs=1e-5)

    llm.reset()
    llm.set_sampling()                                   # logprobs off again
    llm.chat("法国的首都是哪里？只答城市名。", max_new=8)
    assert llm.last_logprobs() == [], "logprobs kept being computed after being turned off"


def test_logprobs_stop_at_the_stop_string(llm):
    """A stop string cuts the reply at a byte offset, but the tokens that matched it were
    still generated — the report must cover the returned text and nothing beyond it."""
    llm.reset()
    llm.set_sampling(logprobs=0)
    reply = llm.chat("从1数到9，每个数字单独一行，只输出数字。", max_new=60, stop=["5"])
    lp = llm.last_logprobs()
    assert "".join(llm.decode([e["id"]]) for e in lp) == reply
    llm.reset()
    llm.set_sampling()


def test_grammar_constrains_the_reply(llm):
    """GBNF-constrained decoding: the reply is structurally guaranteed, not requested.

    The grammar has to win against what the model wants to say — including against the
    sampler's 512-candidate cut, since the allowed tokens are often nowhere near the top
    (the unit-level proof of that is in tests/test_native_grammar.cc)."""
    llm.reset()
    llm.set_grammar('root ::= "yes" | "no"\n')
    assert llm.chat("Is the sky blue? Answer in English.", max_new=16) in ("yes", "no")
    llm.reset()
    # Same grammar, second turn: it must restart at the root rather than stay finished.
    assert llm.chat("Is 2 + 2 equal to 5?", max_new=16) in ("yes", "no")

    llm.reset()
    llm.clear_grammar()
    free = llm.chat("Say hello in one short sentence.", max_new=24)
    assert free.strip() and free not in ("yes", "no")


def test_grammar_produces_json_that_parses(llm):
    """The point of the JSON Schema path: json.loads() on the reply cannot fail."""
    import json

    schema = {"type": "object",
              "properties": {"name": {"type": "string"},
                             "is_capital": {"type": "boolean"},
                             "country": {"type": "string"}},
              "required": ["name", "is_capital", "country"]}
    llm.reset()
    llm.set_grammar(bllm.json_grammar(schema))
    obj = json.loads(llm.chat("Describe the city Paris as JSON.", max_new=80))
    assert set(obj) == {"name", "is_capital", "country"}
    assert isinstance(obj["is_capital"], bool) and isinstance(obj["name"], str)

    llm.reset()
    llm.set_grammar(bllm.json_grammar({"enum": ["positive", "negative", "neutral"]}))
    assert json.loads(llm.chat("Sentiment of: I love this!", max_new=16)) in (
        "positive", "negative", "neutral")
    llm.reset()
    llm.clear_grammar()


def test_a_broken_grammar_is_rejected_at_set_time(llm):
    """Not mid-generation, and with the reason in the message."""
    with pytest.raises(RuntimeError, match=r"\[grammar\]"):
        llm.set_grammar("root ::= undefined_thing\n")
    with pytest.raises(RuntimeError, match=r"\[grammar\]"):
        llm.set_grammar('nope ::= "a"\n')          # no root rule
    assert not llm.has_grammar
    llm.reset()
    assert llm.chat("法国的首都是哪里？只答城市名。", max_new=8)   # still usable


def test_prompt_cache_round_trips_bit_identically(llm, tmp_path):
    """save_state/load_state (libxlm's path_prompt_cache): feeding a prefix then saving,
    reloading, and continuing must reproduce the in-one-go greedy continuation exactly."""
    llm.reset(); llm.set_sampling()                        # greedy → deterministic
    r1 = llm.generate("北京是", max_new=24)
    llm.reset()
    llm.generate("北京是", max_new=0)                       # feed the prefix, generate nothing
    path = str(tmp_path / "state.bin")
    llm.save_state(path)
    llm.reset()
    llm.load_state(path)
    r2 = llm.generate("", max_new=24)                      # continue from the restored state
    assert r1 == r2 and r1
    llm.reset()


def test_generate_async_returns_a_future(llm):
    """chat_async submits on a background thread and hands back a Future (libxlm's
    xlm_infer_async intent): it returns before generation finishes, result() blocks."""
    from concurrent.futures import Future
    llm.reset()
    fut = llm.chat_async("法国的首都是哪里？只答城市名。", max_new=8)
    assert isinstance(fut, Future)
    assert "巴黎" in fut.result(timeout=60)
    llm.reset()


def test_perplexity_ranks_coherent_below_scrambled(llm):
    """Teacher-forced perplexity (libxlm's xlm_ppl on the native path): a fluent sentence
    is more predictable than its shuffled words, so it must score lower."""
    llm.reset()
    coherent = llm.perplexity("北京是中华人民共和国的首都，是全国的政治、文化和国际交往中心。")
    scrambled = llm.perplexity("首都 是 交往 北京 文化 的 政治 中华人民共和国 中心 国际。")
    assert coherent > 1.0
    assert scrambled > coherent
    llm.reset()


def test_multi_turn_recalls_context(llm):
    llm.reset()
    llm.chat("请记住数字 4271。", max_new=24)
    assert "4271" in llm.chat("刚才让你记的数字是多少？", max_new=24)


def test_context_overflow_raises_rather_than_degrading(llm):
    """Rolling past cache_len evicts the earliest tokens — which act as attention
    sinks — and turns these full-attention models into gibberish. So the engine
    refuses. This pins behaviour, not performance."""
    llm.reset()
    huge = "北京是中国的首都，上海是最大的城市。" * 400
    with pytest.raises(RuntimeError, match="context overflow"):
        llm.chat(huge, max_new=4)
    llm.reset()


@pytest.mark.skipif(not bllm._HAVE_NATIVE, reason="native runtime not built")
def test_native_session_rejects_an_omni_model_dir():
    """Loading a multimodal model dir with the text session must say so, not fail
    deep inside hbDNN with "Model not exists: prefill"."""
    if not OMNI_DIR:
        pytest.skip("set BLLM_TEST_OMNI_DIR")
    with pytest.raises(RuntimeError, match="omni"):
        bllm.NativeSession(OMNI_DIR)
