// BLLM — BPU LLM runtime for RDK S100. Public umbrella header.
//
// BLLM wraps the D-Robotics OE-LLM runtime (libxlm.so) behind a task-style
// C++ API, mirroring the bcdl / ccdl family. Autoregressive LLM decode runs on
// libxlm's prefill/decode dual-graph + KV-cache — NOT bcdl's hbDNNInferV2 path.
//
// Include this to pull in the whole public surface:
//
//   #include "bllm/bllm.h"
//   bllm::SessionOptions o;
//   o.model_path = "Qwen2.5_1.5B_Instruct_1024.hbm";
//   o.tokenizer_dir = "Qwen2.5_1.5B_Instruct_config/";
//   o.chat_template_path = "Qwen2.5_1.5B_Instruct_config/Qwen2.5_1.5B_Instruct.jinja";
//   bllm::LlmSession llm(o);                 // model_type auto-detected as Qwen2.5
//   std::string reply = llm.generate("你好", [](std::string_view t){ std::cout << t; });

#ifndef BLLM_BLLM_H_
#define BLLM_BLLM_H_

#include "bllm/common.h"
#include "bllm/llm_session.h"
#include "bllm/omni_session.h"
#include "bllm/types.h"
#include "bllm/vlm_session.h"

#endif  // BLLM_BLLM_H_
