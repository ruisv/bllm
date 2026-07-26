#!/usr/bin/env python3
"""OpenAI-compatible HTTP server for the BLLM native runtime.

A FastAPI facade over ``serving.Engine``, which owns the model on a single worker
thread (one BPU graph serves one generation at a time) and keeps a prefix cache of
engine snapshots so a multi-turn conversation does not re-prefill its history.

State model: the OpenAI protocol is stateless — the client re-sends the full
``messages[]`` every call — so there is no server-side session id. Instead the whole
rendered prompt is tokenized and matched against cached snapshots by exact token
prefix; a hit restores that state and prefills only the new tail. Matching cannot be
wrong (see ``serving/cache.py``): a mismatch just means a shorter match or a full
prefill.

Config via env:
    BLLM_MODEL          model directory (with model.json) OR a bare .hbm     [required]
    BLLM_TOKENIZER_DIR  tokenizer dir, only when BLLM_MODEL is a bare .hbm
    BLLM_BACKEND        load backend, default "auto"
    BLLM_MAX_NEW        default max_new tokens when a request omits it (1024)
    BLLM_API_KEY        if set, require "Authorization: Bearer <key>"
    BLLM_BPU_PRIORITY   optional int, set_bpu_priority() at startup
    BLLM_MAX_QUEUE      queued requests before returning 429 (8)
    BLLM_CACHE_MAX_MB   prefix-cache budget in MB, or "auto" (default) to size it from
                        MemAvailable at startup; 0 disables the cache
    BLLM_CACHE_TTL_S    drop a cached prefix unused for this long (1800)
    BLLM_CACHE_MAX_ENTRIES  hard entry cap alongside the byte budget (64)
    BLLM_CORS_ORIGINS   comma-separated allowed origins (default "*")

Run:  uvicorn server:app --host 0.0.0.0 --port 8866   (single worker only)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Iterator, Optional

import bllm
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from serving.engine import Cancelled, Engine, QueueFull

# --- ChatML rendering -------------------------------------------------------
IM_END = "<|im_end|>"


def parse_content(content) -> tuple[str, list[str]]:
    """A message's ``content`` -> (concatenated text, image_url values).

    OpenAI content is either a plain string or an array of typed parts. An
    ``image_url`` part's ``url`` is either a ``data:image/...;base64,...`` inline
    image or an ``http(s)://`` reference — both are handled by ``_load_image_url``
    at the point they are actually decoded, not here.

    Raises HTTPException(400) on a content part type that is neither ``text`` nor
    ``image_url`` — silently dropping it would have the model answer confidently
    about content it never received, which is worse than an error.
    """
    if isinstance(content, str) or content is None:
        return (content or ""), []
    if not isinstance(content, list):
        raise HTTPException(status_code=400, detail="message content must be a string or an array")
    texts: list[str] = []
    images: list[str] = []
    for p in content:
        if not isinstance(p, dict):
            raise HTTPException(status_code=400, detail="content parts must be objects")
        kind = p.get("type")
        if kind == "text":
            texts.append(p.get("text", ""))
        elif kind == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            if not isinstance(url, str) or not url:
                raise HTTPException(status_code=400, detail="image_url.url must be a non-empty string")
            images.append(url)
        else:
            raise HTTPException(status_code=400,
                                detail=f"unsupported content part type {kind!r}; "
                                       f"only 'text' and 'image_url' are handled")
    return "".join(texts), images


def render_chatml(messages: list[dict], enable_thinking: bool) -> tuple[str, str]:
    """Render an OpenAI ``messages[]`` array into a ChatML prompt, split into
    ``(history, opener)``. Model-agnostic across the Qwen/DeepSeek ChatML family.
    Text-only: the caller has already checked no message carries an image_url part
    (those route through the VLM path instead — see ``_vlm_turns_from``).

    The split is what makes the prefix cache work for chat. The opener —
    ``<|im_start|>assistant\n`` plus, when thinking is off, an empty ``<think>``
    block — is specific to THIS request: next turn the client sends the assistant's
    reply back as content, so that same position renders as
    ``<|im_start|>assistant\n{reply}<|im_end|>\n``. A snapshot taken past the split
    is therefore never a prefix of the next turn's prompt and can never be reused.
    Only `history` is stable across turns, so only `history` is cacheable.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        text, _ = parse_content(m.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{text}{IM_END}\n")
    opener = "<|im_start|>assistant\n"
    if not enable_thinking:  # Qwen3.5: suppress the thinking block
        opener += "<think>\n\n</think>\n\n"
    return "".join(parts), opener


def _vlm_turns_from(messages: list[dict]) -> list[dict]:
    """``messages[]`` -> a list of ``{"role", "text", "images"}`` dicts for the VLM
    serving path (``Engine.submit_vlm``). System messages are dropped: the native
    VLM session bakes its system prompt from ``model.json`` at load time and has no
    per-request override (unlike the text path, which renders one from ChatML).
    The last entry must be a user turn — that is the one which gets answered.
    """
    turns = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            continue
        text, images = parse_content(m.get("content", ""))
        if images and role != "user":
            raise HTTPException(status_code=400, detail=f"image_url is only valid on a user message")
        turns.append({"role": role, "text": text, "images": images})
    if not turns or turns[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="the last message must be from the user")
    return turns


engine: Engine  # set on startup


# --- FastAPI app ------------------------------------------------------------
app = FastAPI(title="bllm-serve", version="2")

# CORS so browser-based OpenAI-compatible clients (chatbox-lite, Open WebUI, …)
# can hit the API cross-origin. Default open (it's a test/edge service); narrow
# via BLLM_CORS_ORIGINS="https://a.com,https://b.com".
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("BLLM_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global engine
    engine = Engine()


@app.on_event("shutdown")
def _shutdown() -> None:
    engine.shutdown()


def _check_auth(authorization: Optional[str]) -> None:
    key = os.environ.get("BLLM_API_KEY")
    if not key:
        return
    if authorization != f"Bearer {key}":
        raise HTTPException(status_code=401, detail="invalid api key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": engine.name, "queue_depth": engine.queue_depth}


@app.get("/metrics")
def metrics(authorization: Optional[str] = Header(None)) -> dict:
    """Cache hit rate, token reuse, queue depth — the numbers that say whether the
    prefix cache is actually earning its memory on this workload."""
    _check_auth(authorization)
    return engine.metrics()


@app.get("/v1/models")
def models(authorization: Optional[str] = Header(None)) -> dict:
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": engine.name, "object": "model", "created": int(time.time()),
                  "owned_by": "bllm"}],
    }


def _num(body: dict, key: str, default: float):
    """Read a numeric field, treating only null/absent as "not given".

    Not ``body.get(k, d) or d``: that idiom folds an explicit 0 into the default, so
    ``temperature: 0`` (greedy — a thing clients ask for deliberately) was
    indistinguishable from omitting it, and ``top_p: 0`` silently became 1.0.
    """
    v = body.get(key)
    return default if v is None else v


def _sampling_from(body: dict) -> dict:
    """Map OpenAI sampling fields onto the engine's.

    presence_penalty/frequency_penalty are OpenAI's own and are honoured directly —
    they were previously dropped on the floor while only the non-standard
    `repetition_penalty` was read, so a client setting them saw no effect at all.
    OpenAI defines both on [-2, 2] with 0 = off, which is exactly what the engine's
    penalty_present/penalty_freq mean, so they pass through unscaled.
    """
    return {
        "temp": float(_num(body, "temperature", 0.0)),
        "top_p": float(_num(body, "top_p", 1.0)),
        "top_k": int(_num(body, "top_k", 0)),
        "min_p": float(_num(body, "min_p", 0.0)),
        "rep_pen": float(_num(body, "repetition_penalty", 1.0)),
        "penalty_present": float(_num(body, "presence_penalty", 0.0)),
        "penalty_freq": float(_num(body, "frequency_penalty", 0.0)),
        # No seed from the client => vary per request. It used to default to a fixed
        # 1234, which pinned the RNG on EVERY request: three identical temperature=0.8
        # requests came back byte-identical, so temperature did nothing. An explicit
        # seed still reproduces exactly, which is what OpenAI's `seed` promises.
        "seed": int(_num(body, "seed", bllm.SEED_RANDOM)),
        "logit_bias": _logit_bias_from(body),
        "logprobs": _logprobs_from(body),
    }


def _logprobs_from(body: dict) -> int:
    """OpenAI's `logprobs` (bool) + `top_logprobs` (0..20) collapsed onto the engine's one
    integer: -1 = off, N >= 0 = report each token's logprob plus N alternatives.

    top_logprobs without logprobs=true is an error in the OpenAI API, and here it would
    silently cost two extra full-vocab passes per token for a field nobody reads back, so
    it is rejected rather than quietly honoured or quietly dropped."""
    want = body.get("logprobs")
    top = body.get("top_logprobs")
    if top is not None and not want:
        raise HTTPException(status_code=400,
                            detail="top_logprobs requires logprobs=true")
    if not want:
        return -1
    if top is None:
        return 0
    if not isinstance(top, int) or isinstance(top, bool) or not 0 <= top <= 20:
        raise HTTPException(status_code=400,
                            detail="top_logprobs must be an integer in [0, 20]")
    return top


def _logit_bias_from(body: dict) -> dict:
    """OpenAI `logit_bias`: {"<token id>": bias} with bias on [-100, 100].

    The keys arrive as strings (JSON object keys always do) and the engine wants ints.
    Out-of-range values are rejected rather than clamped: a client that sends 1000 is
    working from a different scale, and silently reinterpreting it would hide that.
    """
    raw = body.get("logit_bias")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="logit_bias must be an object")
    out: dict[int, float] = {}
    for k, v in raw.items():
        try:
            tid, bias = int(k), float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400,
                                detail=f"logit_bias: bad entry {k!r}: {v!r}")
        if not -100.0 <= bias <= 100.0:
            raise HTTPException(status_code=400,
                                detail=f"logit_bias[{tid}]={bias} is outside [-100, 100]")
        out[tid] = bias
    return out


def _grammar_from(body: dict) -> str:
    """GBNF for this request, from `response_format` or a raw `grammar`, or "".

    Structured output has two entrances and they must not disagree, so asking for both is
    an error rather than a precedence rule nobody can remember:

      response_format: {"type": "json_object"}                  any valid JSON
      response_format: {"type": "json_schema",
                        "json_schema": {"schema": {...}}}       that shape exactly
      grammar: "root ::= ..."                                   raw GBNF (llama.cpp's field)
    """
    rf = body.get("response_format")
    raw = body.get("grammar")
    if rf is not None and raw:
        raise HTTPException(status_code=400,
                            detail="use either response_format or grammar, not both")
    if raw:
        if not isinstance(raw, str):
            raise HTTPException(status_code=400, detail="grammar must be a GBNF string")
        return raw
    if rf is None:
        return ""
    if not isinstance(rf, dict):
        raise HTTPException(status_code=400, detail="response_format must be an object")
    kind = rf.get("type", "text")
    if kind == "text":
        return ""
    if kind == "json_object":
        return bllm.json_grammar()
    if kind == "json_schema":
        spec = rf.get("json_schema") or {}
        schema = spec.get("schema") if isinstance(spec, dict) else None
        if not isinstance(schema, dict):
            raise HTTPException(status_code=400,
                                detail="response_format.json_schema.schema must be an object")
        try:
            return bllm.json_grammar(schema)
        except ValueError as exc:               # an unsupported or malformed schema
            raise HTTPException(status_code=400, detail=f"json_schema: {exc}")
        except RecursionError:                  # nesting deeper than the converter recurses
            raise HTTPException(status_code=400, detail="json_schema: nested too deeply")
    raise HTTPException(status_code=400, detail=f"unsupported response_format type {kind!r}")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request,
                           authorization: Optional[str] = Header(None)) -> Any:
    _check_auth(authorization)
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages[] required")

    max_new = int(body.get("max_tokens") or body.get("max_completion_tokens")
                  or engine.default_max_new)
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    stop = [s for s in dict.fromkeys(stop) if s != IM_END]
    created = int(time.time())
    cid = f"chatcmpl-{created}"

    grammar = _grammar_from(body)
    if grammar and not engine.supports_grammar:
        # Refuse rather than answer unconstrained: the client asked for a guarantee, and
        # silently downgrading it to "the prompt says please" is the failure mode
        # structured output exists to remove.
        raise HTTPException(status_code=400,
                            detail="this bllm runtime has no grammar support; "
                                   "install a newer bllm to use response_format/grammar")

    if engine.is_vlm:
        turns = _vlm_turns_from(messages)
        try:
            job = engine.submit_vlm(turns, max_new, stop, _sampling_from(body), grammar=grammar)
        except QueueFull as exc:
            return JSONResponse(status_code=429, content=_error_body(exc, "rate_limit_exceeded"))
    else:
        if any(parse_content(m.get("content", ""))[1] for m in messages):
            raise HTTPException(
                status_code=400,
                detail="this model has no vision tower; image_url requires a model directory "
                       "built with a --visual tower (see docs/MODELS.zh.md)")
        enable_thinking = bool(body.get("enable_thinking", False))
        history, opener = render_chatml(messages, enable_thinking)
        prompt = history + opener
        ids = engine.encode(prompt)
        # How much of `ids` is stable history, and so eligible for caching. Encoding the
        # two parts separately could in principle tokenize differently at the seam, so
        # verify it really is a prefix rather than assuming — the seam is a special-token
        # boundary, but a wrong split would poison the cache, and checking is cheap.
        hist_ids = engine.encode(history) if history else []
        cacheable = len(hist_ids) if ids[:len(hist_ids)] == hist_ids else 0
        try:
            job = engine.submit(prompt, ids, max_new, stop, _sampling_from(body),
                                cacheable=cacheable, grammar=grammar)
        except QueueFull as exc:
            return JSONResponse(status_code=429, content=_error_body(exc, "rate_limit_exceeded"))

    if bool(body.get("stream", False)):
        return StreamingResponse(_sse(request, cid, created, job),
                                 media_type="text/event-stream")

    loop = asyncio.get_running_loop()
    try:
        # Drain to completion on a thread; the worker owns the engine either way.
        await loop.run_in_executor(None, lambda: [None for _ in engine.stream(job)])
        res = await loop.run_in_executor(None, lambda: engine.wait(job))
    except Exception as exc:  # return a proper OpenAI error, not a bare 500
        status, payload = _error_payload(exc)
        return JSONResponse(status_code=status, content=payload)

    choice: dict = {"index": 0, "message": {"role": "assistant", "content": res.text},
                    "finish_reason": res.finish_reason}
    if res.logprobs is not None:
        choice["logprobs"] = {"content": res.logprobs}
    return JSONResponse({
        "id": cid, "object": "chat.completion", "created": created, "model": engine.name,
        "choices": [choice],
        "usage": _usage(res),
    })


def _usage(res) -> dict:
    return {
        "prompt_tokens": res.n_prompt,
        "completion_tokens": res.n_completion,
        "total_tokens": res.n_prompt + res.n_completion,
        # OpenAI's standard field — clients surface it directly, so the cache's effect
        # is visible without reading /metrics.
        "prompt_tokens_details": {"cached_tokens": res.n_cached},
    }


# --- streaming --------------------------------------------------------------
def _error_payload(exc: BaseException) -> tuple[int, dict]:
    """(http_status, OpenAI-style error body) for an exception."""
    msg = str(exc)
    if isinstance(exc, QueueFull):
        return 429, _error_body(exc, "rate_limit_exceeded")
    if isinstance(exc, Cancelled):
        return 499, _error_body(exc, "request_cancelled")
    if "context overflow" in msg:
        return 400, _error_body(exc, "context_length_exceeded")
    # A grammar that does not parse is the client's text, not our bug — the engine only
    # sees it when the job runs, so the 400 has to be recovered here.
    if "[grammar]" in msg:
        return 400, _error_body(exc, "invalid_grammar")
    return 500, _error_body(exc, None)


def _error_body(exc: BaseException, code: Optional[str]) -> dict:
    return {"error": {"message": str(exc), "type": exc.__class__.__name__, "code": code}}


async def _sse(request: Request, cid: str, created: int, job):
    """Bridge the worker's deltas to OpenAI SSE chunks.

    On disconnect this cancels the job and keeps draining until the worker says it is
    done. Returning early would hand the engine back while a generation is still
    running against it — the bug this rewrite exists to kill.
    """
    loop = asyncio.get_running_loop()
    gen = engine.stream(job)
    yield _chunk(cid, created, {"role": "assistant"})
    try:
        while True:
            try:
                piece = await loop.run_in_executor(None, lambda: next(gen, _DONE))
            except Exception as exc:
                yield _error_event(exc)
                yield "data: [DONE]\n\n"
                return
            if piece is _DONE:
                break
            if await request.is_disconnected():
                engine.cancel(job)
                # keep draining: the worker must finish before the next job starts
                continue
            yield _chunk(cid, created, {"content": piece})
    finally:
        if not job.done.is_set():
            engine.cancel(job)
        await loop.run_in_executor(None, lambda: job.done.wait(30))

    res = job.result
    if job.error is not None:
        yield _error_event(job.error)
    else:
        # logprobs ride on the final chunk, all of them at once, rather than per delta: a
        # delta is a decoded text slice (a stop-string holdback can merge or defer tokens),
        # so there is no honest per-chunk alignment to publish. A client that concatenates
        # every chunk's logprobs.content — the normal way to consume them — still ends up
        # with exactly the right array, just at the end of the stream.
        yield _chunk(cid, created, {}, finish_reason=res.finish_reason, usage=_usage(res),
                     logprobs=res.logprobs)
    yield "data: [DONE]\n\n"


_DONE = object()


def _error_event(exc: BaseException) -> str:
    _, body = _error_payload(exc)
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def _chunk(cid: str, created: int, delta: dict, finish_reason: Optional[str] = None,
           usage: Optional[dict] = None, logprobs: Optional[list] = None) -> str:
    choice: dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
    obj: dict[str, Any] = {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": engine.name,
        "choices": [choice],
    }
    if usage is not None:
        obj["usage"] = usage
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
