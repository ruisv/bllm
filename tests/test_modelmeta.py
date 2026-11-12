"""bllm._modelmeta — the Blocker-A synthesizer that turns a bare .hbm + tokenizer dir into a
native model package. Pure metadata work, so this runs off-board (no model load)."""

import json

import pytest

bllm = pytest.importorskip("bllm", reason="bllm module not built/on PYTHONPATH")
mm = pytest.importorskip("bllm._modelmeta")


def _tokenizer(tmp_path, *, chatml=True, added=True, gen_eos=None):
    """Write a minimal tokenizer.json (+ optional generation_config.json) and return its dir."""
    d = tmp_path / "tok"
    d.mkdir()
    at = []
    if added:
        at = [{"id": 151643, "content": "<|endoftext|>", "special": True}]
        if chatml:
            at += [{"id": 151644, "content": "<|im_start|>", "special": True},
                   {"id": 151645, "content": "<|im_end|>", "special": True}]
    vocab = {"a": 0, "b": 1}
    if not added:                         # legacy sentencepiece style
        vocab["</s>"] = 2
    (d / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": at}))
    if gen_eos is not None:
        (d / "generation_config.json").write_text(json.dumps({"eos_token_id": gen_eos}))
    return d


def test_dense_chatml_from_tokenizer_specials(tmp_path):
    tok = _tokenizer(tmp_path, chatml=True)
    hbm = tmp_path / "m.hbm"; hbm.write_bytes(b"\x00")
    cfg = mm.dense_model_json(hbm, tok)
    assert cfg["arch"] == "dense" and cfg["backend"] == "native"
    assert cfg["chat"] == {"format": "chatml", "im_start": 151644, "im_end": 151645,
                           "system": "You are a helpful assistant."}
    # No generation_config → declared specials in EOS_CANDIDATES order: im_end (the turn
    # stop) first, then endoftext.
    assert cfg["eos"] == [151645, 151643]


def test_generation_config_eos_wins(tmp_path):
    tok = _tokenizer(tmp_path, chatml=True, gen_eos=[151645, 151643])
    hbm = tmp_path / "m.hbm"; hbm.write_bytes(b"\x00")
    cfg = mm.dense_model_json(hbm, tok)
    assert cfg["eos"] == [151645, 151643]   # HF's list, verbatim


def test_no_chatml_markers_falls_back_to_raw(tmp_path):
    tok = _tokenizer(tmp_path, chatml=False)
    hbm = tmp_path / "m.hbm"; hbm.write_bytes(b"\x00")
    cfg = mm.dense_model_json(hbm, tok)
    assert cfg["chat"] == {"format": "none"}


def test_sentencepiece_eos_from_vocab(tmp_path):
    tok = _tokenizer(tmp_path, chatml=False, added=False)   # no added_tokens (InternLM2-like)
    hbm = tmp_path / "m.hbm"; hbm.write_bytes(b"\x00")
    cfg = mm.dense_model_json(hbm, tok)
    assert cfg["eos"] == [2]              # </s> from the vocabulary


def test_native_dir_for_materializes_symlinks(tmp_path):
    tok = _tokenizer(tmp_path, chatml=True)
    hbm = tmp_path / "m.hbm"; hbm.write_bytes(b"\x01\x02")
    out = mm.native_dir_for(hbm, tok, cache_root=tmp_path / "cache")
    assert (out / "model.json").is_file()
    assert (out / "model.hbm").resolve() == hbm.resolve()
    assert (out / "tokenizer.json").resolve() == (tok / "tokenizer.json").resolve()
    cfg = json.loads((out / "model.json").read_text())
    assert cfg["arch"] == "dense"
    # Loading the same pair again is idempotent (same cache key).
    assert mm.native_dir_for(hbm, tok, cache_root=tmp_path / "cache") == out


def test_native_dir_for_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        mm.native_dir_for(tmp_path / "nope.hbm", tmp_path)


# ── official D-Robotics LLM 2.0 packages ────────────────────────────────────────────────

_Q3 = "Qwen3-0.6B_language_chunk_256_cache_2048_w8_nash-m_corenum_1_1.hbm"


def _desc(*, template="{{ '<|im_start|>' }}", gen_eos=(151645, 151643), extra_added=(),
          horizon=True, add_bos=False):
    added = [{"id": 151643, "content": "<|endoftext|>", "special": True},
             {"id": 151644, "content": "<|im_start|>", "special": True},
             {"id": 151645, "content": "<|im_end|>", "special": True}]
    added += [{"id": i, "content": c, "special": True} for c, i in extra_added]
    d = {"config": {"architectures": ["Qwen3ForCausalLM"]},
         "tokenizer": {"model": {"vocab": {"a": 0}}, "added_tokens": added},
         "tokenizer_config": {"chat_template": template, "eos_token": "<|im_end|>",
                              "add_bos_token": add_bos, "bos_token": "<|endoftext|>"}}
    if gen_eos is not None:
        d["generation_config"] = {"eos_token_id": list(gen_eos)}
    if horizon:
        d["horizon"] = {"prefill_chunk_size": 256, "decode_cache_len": 2048, "vocab_size": 151936}
    return d


def _package(tmp_path, *, vision=False, name=_Q3):
    p = tmp_path / "Qwen3-0.6B_w8"
    p.mkdir()
    (p / name).write_bytes(b"\x00")
    (p / "Qwen3-0.6B_embed_tokens.bin").write_bytes(b"\x00")
    if vision:
        (p / "Qwen3-0.6B_vision_448x448_w8_nash-m_corenum_1.hbm").write_bytes(b"\x00")
    return p


def test_official_model_json_chatml_name_eos():
    cfg = mm.official_model_json(_desc(), _Q3)
    assert cfg["arch"] == "dense" and cfg["embed"] == "embed_tokens.bin"
    assert cfg["chat"]["format"] == "chatml" and cfg["chat"]["im_start"] == 151644
    assert cfg["eos"] == [151645, 151643]                 # generation_config wins
    assert cfg["cache_len"] == 2048
    # ctx, quant and board are all in the name — they are what tells two packages apart
    assert cfg["name"] == "Qwen3-0.6B · ctx2048 · w8 · RDK S100P (official LLM 2.0)"


def test_official_model_json_eos_falls_back_to_tokenizer():
    cfg = mm.official_model_json(_desc(gen_eos=None), _Q3)
    assert cfg["eos"][0] == 151645                         # tokenizer_config eos_token


_PHI4 = ("{% for message in messages %}{{ '<|' + message['role'] + '|>' + message['content'] "
         "+ '<|end|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}"
         "{% else %}{{ eos_token }}{% endif %}")


def test_official_model_json_phi_family_built_by_concatenation():
    # Phi-4-mini's real template never spells "<|user|>": the markers are concatenated.
    phi = [("<|user|>", 200021), ("<|assistant|>", 200019), ("<|system|>", 200022),
           ("<|end|>", 200020)]
    cfg = mm.official_model_json(_desc(template=_PHI4, extra_added=phi), _Q3)
    assert cfg["chat"] == {"format": "phi", "r_user": 200021, "r_assistant": 200019,
                           "r_end": 200020, "system": "", "r_system": 200022,
                           "think_on": "", "think_off": "", "thinking_default": False}


# Generation-prompt tails as they appear in the jinja SOURCE (literal backslash-n).
_GEN = "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
_QWEN3 = (_GEN + "{%- if enable_thinking is defined and enable_thinking is false %}"
          "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}")
_MINICPM5 = (_GEN + "{%- if enable_thinking is defined %}{%- if enable_thinking is false %}"
             "{{- '<think>\\n\\n</think>\\n\\n' }}{%- elif enable_thinking is true %}"
             "{{- '<think>\\n' }}{%- endif %}{%- endif %}{%- endif %}")
_THINKING_ONLY = "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}"


@pytest.mark.parametrize("template,on,off,default", [
    (_QWEN3, "", "<think>\n\n</think>\n\n", False),          # thinks unprompted when on
    (_MINICPM5, "<think>\n", "<think>\n\n</think>\n\n", False),  # opens the block when on
    (_THINKING_ONLY, "<think>\n", "<think>\n", True),           # cannot be switched off
    (_GEN + "{%- endif %}", "", "", False),                       # no think mode at all
])
def test_official_thinking_follows_the_template(template, on, off, default):
    ch = mm.official_model_json(_desc(template=template), _Q3)["chat"]
    assert (ch["think_on"], ch["think_off"], ch["thinking_default"]) == (on, off, default)


def test_official_thinking_switch_without_a_closed_block_is_refused():
    with pytest.raises(ValueError, match="enable_thinking"):
        mm.official_model_json(_desc(template=_GEN + "{% if enable_thinking %}x{% endif %}"), _Q3)


def test_official_default_system_is_the_templates_own():
    q25 = ("{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a "
           "helpful assistant.<|im_end|>\\n' }}" + _GEN + "{%- endif %}")
    ch = mm.official_model_json(_desc(template=q25), _Q3)["chat"]
    assert ch["system"] == "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    assert mm.official_model_json(_desc(template=_QWEN3), _Q3)["chat"]["system"] == ""
    assert mm.official_model_json(_desc(template=_QWEN3), _Q3,
                                  system="Be brief.")["chat"]["system"] == "Be brief."


def test_official_bos_when_the_template_always_emits_one():
    # MiniCPM5: add_bos_token is False, yet the template opens with {{- bos_token }}.
    d = _desc(template="{{- bos_token }}" + _QWEN3, extra_added=[("<s>", 0)])
    d["tokenizer_config"]["bos_token"] = "<s>"
    assert mm.official_model_json(d, _Q3)["chat"]["bos"] == 0


def test_official_model_json_rejects_unknown_template_and_non_llm2():
    with pytest.raises(ValueError, match="neither ChatML nor Phi"):
        mm.official_model_json(_desc(template="[INST] {{ m }} [/INST]"), _Q3)
    with pytest.raises(ValueError, match="horizon"):
        mm.official_model_json(_desc(horizon=False), _Q3)


def test_official_model_json_bos_only_when_the_template_adds_one():
    assert "bos" not in mm.official_model_json(_desc(), _Q3)["chat"]
    assert mm.official_model_json(_desc(add_bos=True), _Q3)["chat"]["bos"] == 151643


def test_find_official_package(tmp_path):
    p = _package(tmp_path)
    pkg = mm.find_official_package(p)
    assert pkg["language"].name == _Q3 and pkg["vision"] is None
    (p / "model.json").write_text("{}")                    # ours, not theirs
    assert mm.find_official_package(p) is None
    assert mm.find_official_package(tmp_path) is None      # nothing there


def test_find_official_package_rejects_two_models(tmp_path):
    p = _package(tmp_path)
    (p / "Draft_language_chunk_256_cache_2048_w8_nash-m_corenum_1_1.hbm").write_bytes(b"\x00")
    with pytest.raises(ValueError, match="more than one model"):
        mm.find_official_package(p)


def test_official_dir_for_synthesizes_once_then_hits_cache(tmp_path):
    p = _package(tmp_path)
    calls = []

    def read(path):
        calls.append(path)
        return json.dumps(_desc())

    d1 = mm.official_dir_for(p, cache_root=tmp_path / "cache", read_desc=read)
    cfg = json.loads((d1 / "model.json").read_text())
    assert cfg["chat"]["format"] == "chatml"
    tok = json.loads((d1 / "tokenizer.json").read_text())
    assert tok["added_tokens"][1]["content"] == "<|im_start|>"
    assert (d1 / "model.hbm").resolve() == (p / _Q3).resolve()
    assert (d1 / "embed_tokens.bin").is_symlink()
    d2 = mm.official_dir_for(p, cache_root=tmp_path / "cache", read_desc=read)
    assert d2 == d1 and len(calls) == 1                    # second load reads nothing


def test_official_dir_for_vlm_needs_vision_false(tmp_path):
    p = _package(tmp_path, vision=True)
    read = lambda path: json.dumps(_desc())
    with pytest.raises(NotImplementedError, match="vision=False"):
        mm.official_dir_for(p, cache_root=tmp_path / "cache", read_desc=read)
    d = mm.official_dir_for(p, vision=False, cache_root=tmp_path / "cache", read_desc=read)
    assert json.loads((d / "model.json").read_text())["arch"] == "dense"
