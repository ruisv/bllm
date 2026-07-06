cmake_minimum_required(VERSION 3.22)
project(bllm LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Umbrella interface target (header-only for now) ──────────────────────────
# The real library (bllm::LlmSession wrapping libxlm.so) lands in M1. Until the
# OE-LLM SDK path is wired into the build, this exposes only the public headers.
add_library(bllm INTERFACE)
target_include_directories(bllm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# ── OE-LLM SDK (board-only) ──────────────────────────────────────────────────
# TODO(M1): locate the OE-LLM runtime (libxlm.so + xlm.h) from the SDK, e.g.
#   set(OELLM_SDK "$ENV{HOME}/llm_s100/D-Robotics_LLM_S100_1.0.0_SDK/oellm_runtime")
#   target_include_directories(bllm INTERFACE ${OELLM_SDK}/include)
#   target_link_libraries(bllm INTERFACE ${OELLM_SDK}/lib/libxlm.so)
# then build src/ into a real (non-INTERFACE) bllm target and add examples/.

# add_subdirectory(examples)   # enable once the M1 wrapper exists
