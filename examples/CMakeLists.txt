add_executable(bllm_chat chat.cc)
target_link_libraries(bllm_chat PRIVATE bllm::bllm)
# libxlm.so has undefined refs to its (runtime-loaded) sibling libs.
target_link_options(bllm_chat PRIVATE -Wl,-unresolved-symbols=ignore-in-shared-libs)
