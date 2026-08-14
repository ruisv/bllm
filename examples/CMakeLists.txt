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

  # Any graph at all, no shape conventions assumed: feed f32 files, dump f32
  # outputs. This is what bisects a partially-compiled graph against its host
  # reference when the shaped probes above would refuse to load it.
  add_executable(bllm_graph_probe graph_probe.cc)
  target_link_libraries(bllm_graph_probe PRIVATE bllm::native)

  # Being-H0.5's four-way shared-KV replay (bllm::BeingHStack), fed pre-built
  # stream embeddings so the graph plumbing is tested on its own. That port is
  # still in progress and its sources are not published yet, so this builds only
  # where they are present rather than breaking a checkout that lacks them.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/beingh_probe.cc")
    add_executable(bllm_beingh_probe beingh_probe.cc)
    target_link_libraries(bllm_beingh_probe PRIVATE bllm::native)
    target_include_directories(bllm_beingh_probe PRIVATE "${CMAKE_SOURCE_DIR}/vendor/stb")
  endif()

  # pi0.5 end to end (bllm::Pi05Stack): SigLIP -> PaliGemma prefill -> the
  # flow-matching loop, scored against the host golden for the same observation.
  add_executable(bllm_pi05_probe pi05_probe.cc)
  target_link_libraries(bllm_pi05_probe PRIVATE bllm::native)

  # The same stack held resident behind a socket, so a simulator can score the
  # policy without paying 9 s of model load per inference.
  add_executable(bllm_pi05_server pi05_server.cc)
  target_link_libraries(bllm_pi05_server PRIVATE bllm::native)

  # π0.5 as a policy from a model package — images and a sentence in, actions
  # out, with the tokenizer, embedding table and normalisation carried by the
  # package rather than by host-side openpi.
  add_executable(bllm_pi05_policy pi05_policy.cc)
  target_link_libraries(bllm_pi05_policy PRIVATE bllm::native)
  target_include_directories(bllm_pi05_policy PRIVATE "${CMAKE_SOURCE_DIR}/vendor/stb")
endif()
