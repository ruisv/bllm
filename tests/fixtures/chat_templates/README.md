# Chat templates (test fixtures)

The original Hugging Face chat templates `tests/test_chatfmt.py` renders with jinja2 as the
reference for `bllm._chatfmt`. Each file is the `chat_template` of the model repo below,
unmodified. All of them are Apache-2.0, from Qwen.

| file | source repo |
|---|---|
| `qwen3.jinja` | `Qwen/Qwen3-1.7B` (the same template across the Qwen3 sizes) |
| `qwen3-thinking-2507.jinja` | `Qwen/Qwen3-4B-Thinking-2507` |
| `qwen3-instruct-2507.jinja` | `Qwen/Qwen3-4B-Instruct-2507` |
| `qwen2.5.jinja` | `Qwen/Qwen2.5-7B-Instruct` (the same template as the other Qwen2.5 Instruct sizes) |
| `qwen3-vl.jinja` | `Qwen/Qwen3-VL-2B-Instruct` |
| `qwen3.5.jinja` | `Qwen/Qwen3.5-0.8B` |

The official D-Robotics LLM 2.0 packages of these models embed the same templates, character
for character (checked on Qwen3-1.7B, Qwen3-4B-Instruct-2507 / Thinking-2507, Qwen2.5-3B and
Qwen3-VL-2B). A template updated upstream is a reason to re-check, not to edit these copies.
