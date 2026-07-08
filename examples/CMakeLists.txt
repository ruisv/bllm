# Each executable links libxlm.so, which has undefined refs to its
# (runtime-loaded) sibling libs — relax the unresolved-symbol check.
add_executable(bllm_chat chat.cc)
target_link_libraries(bllm_chat PRIVATE bllm::bllm)
target_link_options(bllm_chat PRIVATE -Wl,-unresolved-symbols=ignore-in-shared-libs)

add_executable(bllm_omni omni.cc)
target_link_libraries(bllm_omni PRIVATE bllm::bllm)
target_link_options(bllm_omni PRIVATE -Wl,-unresolved-symbols=ignore-in-shared-libs)

add_executable(bllm_vlm vlm.cc)
target_link_libraries(bllm_vlm PRIVATE bllm::bllm)
target_link_options(bllm_vlm PRIVATE -Wl,-unresolved-symbols=ignore-in-shared-libs)

# The NATIVE engine — drives a .hbm directly on hbDNN/hbUCP, no libxlm. It links
# only the generic hobot runtime, so it is gated on FindHobot rather than the
# OE-LLM SDK.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(Hobot QUIET)
if(Hobot_FOUND)
  add_executable(bllm_selfengine selfengine.cc)
  target_link_libraries(bllm_selfengine PRIVATE hobot::dnn)
  target_include_directories(bllm_selfengine PRIVATE "${CMAKE_SOURCE_DIR}/include")
  message(STATUS "bllm_selfengine: native hbDNN engine (no libxlm)")
else()
  message(STATUS "bllm_selfengine: skipped (hobot runtime not found)")
endif()
