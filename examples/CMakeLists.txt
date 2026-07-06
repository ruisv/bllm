# Each executable links libxlm.so, which has undefined refs to its
# (runtime-loaded) sibling libs — relax the unresolved-symbol check.
add_executable(bllm_chat chat.cc)
target_link_libraries(bllm_chat PRIVATE bllm::bllm)
target_link_options(bllm_chat PRIVATE -Wl,-unresolved-symbols=ignore-in-shared-libs)

add_executable(bllm_omni omni.cc)
target_link_libraries(bllm_omni PRIVATE bllm::bllm)
target_link_options(bllm_omni PRIVATE -Wl,-unresolved-symbols=ignore-in-shared-libs)
