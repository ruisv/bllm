# Native-runtime demos. bllm::native carries the include dirs, hobot::dnn, the tokenizer
# and the fp16 flag; each is gated on what it actually needs.
if(BLLM_HAVE_NATIVE)
  add_executable(bllm_selfengine selfengine.cc)          # token ids in/out
  target_link_libraries(bllm_selfengine PRIVATE bllm::native)

  add_executable(bllm_qwen35 qwen35_native.cc)           # hybrid Gated-DeltaNet
  target_link_libraries(bllm_qwen35 PRIVATE bllm::native)
endif()

if(BLLM_HAVE_NATIVE AND BLLM_HAVE_TOKENIZERS)
  add_executable(bllm_tokenizer_probe tokenizer_probe.cc)
  target_link_libraries(bllm_tokenizer_probe PRIVATE bllm::native)

  # Unified native session (bllm::NativeLlm): model dir in, streaming chat out.
  add_executable(bllm_chat_native chat_native.cc)
  target_link_libraries(bllm_chat_native PRIVATE bllm::native)

  # Text embeddings (bllm::NativeEmbedder): one encoder graph, last-token pooling.
  add_executable(bllm_embed_native embed_native.cc)
  target_link_libraries(bllm_embed_native PRIVATE bllm::native)

  # Native VLM (bllm::NativeVlm): image/audio/video + text -> streamed reply.
  # File decoding is vendored (stb_image + dr_wav); video frame sampling
  # (video_io.h) shells out to ffmpeg and is example-only, never in the library.
  add_executable(bllm_vlm_native vlm_native.cc)
  target_link_libraries(bllm_vlm_native PRIVATE bllm::native)
  target_include_directories(bllm_vlm_native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/vendor/stb" "${CMAKE_SOURCE_DIR}/vendor/dr_libs")

  # Vision tower alone (bllm::VisionTower): scores a freshly compiled tower against
  # the HF fp32 reference, with the host preprocessing either in or out of the path.
  add_executable(bllm_vision_probe vision_probe.cc)
  target_link_libraries(bllm_vision_probe PRIVATE bllm::native)
  target_include_directories(bllm_vision_probe PRIVATE "${CMAKE_SOURCE_DIR}/vendor/stb")
endif()
