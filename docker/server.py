#!/usr/bin/env python3
"""OpenAI-compatible HTTP server for the BLLM native runtime.

A thin FastAPI facade over ``bllm.NativeSession`` that speaks the OpenAI
``/v1/chat/completions`` protocol (streaming + non-streaming). One board holds a
single BPU prefill/decode graph + KV-cache, so the engine is a **singleton** and
requests are **serialized** (one generation in flight at a time) behind a lock.

State model: the OpenAI protocol is stateless — the client re-sends the full
``messages[]`` every call — so each request ``reset()``s the engine and replays
the whole conversation as one ChatML prompt (no server-side session memory).

Config via env:
    BLLM_MODEL          model directory (with model.json) OR a bare .hbm     [required]
    BLLM_TOKENIZER_DIR  tokenizer dir, only when BLLM_MODEL is a bare .hbm
    BLLM_BACKEND        load backend, default "auto"
    BLLM_MAX_NEW        default max_new tokens when a request omits it (400)
    BLLM_API_KEY        if set, require "Authorization: Bearer <key>"
    BLLM_BPU_PRIORITY   optional int, set_bpu_priority() at startup

Run:  uvicorn server:app --host 0.0.0.0 --port 8866   (single worker only)
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from typing import Any, Iterator, Optional

import bllm
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# --- ChatML rendering -------------------------------------------------------
IM_END = "<|im_end|>"


def render_chatml(messages: list[dict], enable_thinking: bool) -> str:
    """Render an OpenAI ``messages[]`` array into a ChatML prompt ending with an
    open assistant turn. Model-agnostic across the Qwen/DeepSeek ChatML family."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):  # OpenAI array content -> concat text parts
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        parts.append(f"<|im_start|>{role}\n{content}{IM_END}\n")
    parts.append("<|im_start|>assistant\n")
    if not enable_thinking:  # Qwen3.5: suppress the thinking block
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts)


# --- engine singleton -------------------------------------------------------
class Engine:
    """Serialized wrapper around a single NativeSession."""

    def __init__(self) -> None:
        model = os.environ.get("BLLM_MODEL")
        if not model:
            raise RuntimeError("BLLM_MODEL is not set (model dir with model.json, or a .hbm)")
        kwargs: dict[str, Any] = {"backend": os.environ.get("BLLM_BACKEND", "auto")}
        tok = os.environ.get("BLLM_TOKENIZER_DIR")
        if tok:
            kwargs["tokenizer_dir"] = tok
        self.sess = bllm.load(model, **kwargs)
        prio = os.environ.get("BLLM_BPU_PRIORITY")
        if prio:  # empty string (unset env in compose) counts as no priority
            self.sess.set_bpu_priority(int(prio))
        self.name = getattr(self.sess, "name", None) or os.path.basename(model.rstrip("/"))
        self.default_max_new = int(os.environ.get("BLLM_MAX_NEW", "400"))
        self.lock = asyncio.Lock()  # one generation in flight at a time

    def _stream_sync(self, prompt: str, max_new: int, stop: list[str],
                     temperature: float, top_p: float, top_k: int,
                     rep_pen: float, seed: int) -> Iterator[str]:
        """Blocking generator — must run in a worker thread."""
        self.sess.reset()
        self.sess.set_sampling(temp=temperature, top_p=top_p, top_k=top_k,
                               rep_pen=rep_pen, seed=seed)
        yield from self.sess.stream(prompt, max_new=max_new, stop=stop)


engine: Engine  # set on startup


# --- FastAPI app ------------------------------------------------------------
app = FastAPI(title="bllm-serve", version="1")

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


def _check_auth(authorization: Optional[str]) -> None:
    key = os.environ.get("BLLM_API_KEY")
    if not key:
        return
    if authorization != f"Bearer {key}":
        raise HTTPException(status_code=401, detail="invalid api key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": engine.name}


@app.get("/v1/models")
def models(authorization: Optional[str] = Header(None)) -> dict:
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": engine.name, "object": "model", "created": int(time.time()),
                  "owned_by": "bllm"}],
    }


def _sampling_from(body: dict) -> dict:
    return {
        "temperature": float(body.get("temperature", 0.0) or 0.0),
        "top_p": float(body.get("top_p", 1.0) or 1.0),
        "top_k": int(body.get("top_k", 0) or 0),
        "rep_pen": float(body.get("repetition_penalty", 1.0) or 1.0),
        "seed": int(body.get("seed", 1234) or 1234),
    }


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
    stop = list(dict.fromkeys([IM_END, *stop]))  # always stop on <|im_end|>
    enable_thinking = bool(body.get("enable_thinking", False))
    prompt = render_chatml(messages, enable_thinking)
    samp = _sampling_from(body)
    created = int(time.time())
    cid = f"chatcmpl-{created}"
    stream = bool(body.get("stream", False))

    def gen() -> Iterator[str]:
        return engine._stream_sync(prompt, max_new, [s for s in stop if s != IM_END] or [],
                                   **samp)

    if stream:
        return StreamingResponse(_sse(cid, created, gen),
                                 media_type="text/event-stream")

    # non-streaming: accumulate under the lock
    async with engine.lock:
        text, ntok = await _collect(gen)
    completion = {
        "id": cid, "object": "chat.completion", "created": created, "model": engine.name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": _approx_tok(prompt), "completion_tokens": ntok,
                  "total_tokens": _approx_tok(prompt) + ntok},
    }
    return JSONResponse(completion)


# --- streaming/thread bridge ------------------------------------------------
async def _collect(gen_factory) -> tuple[str, int]:
    loop = asyncio.get_running_loop()

    def run() -> tuple[str, int]:
        pieces = list(gen_factory())
        return "".join(pieces), len(pieces)

    return await loop.run_in_executor(None, run)


async def _sse(cid: str, created: int, gen_factory):
    """Drive the blocking generator in a thread; emit OpenAI SSE chunks."""
    async with engine.lock:
        loop = asyncio.get_running_loop()
        q: "queue.Queue[Optional[str]]" = queue.Queue()

        def pump() -> None:
            try:
                for piece in gen_factory():
                    q.put(piece)
            finally:
                q.put(None)  # sentinel

        threading.Thread(target=pump, daemon=True).start()

        # role delta first (OpenAI convention)
        yield _chunk(cid, created, {"role": "assistant"})
        while True:
            piece = await loop.run_in_executor(None, q.get)
            if piece is None:
                break
            yield _chunk(cid, created, {"content": piece})
        yield _chunk(cid, created, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"


def _chunk(cid: str, created: int, delta: dict, finish_reason: Optional[str] = None) -> str:
    import json
    obj = {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": engine.name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _approx_tok(text: str) -> int:
    # best-effort: no python tokenizer at this layer; ~4 chars/token
    return max(1, len(text) // 4)
