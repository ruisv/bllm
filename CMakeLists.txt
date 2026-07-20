cmake_minimum_required(VERSION 3.22)
project(bllm LANGUAGES CXX VERSION 0.1.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BLLM_BUILD_EXAMPLES "Build the example programs" ON)
option(BLLM_BUILD_PYTHON   "Build the nanobind Python bindings" OFF)
option(BLLM_INSTALL_PYTHON "Install the Python package into site-packages (for conda/pip)" OFF)

# BLLM is the self-built native engine driving `.hbm` graphs straight on hbDNN/hbUCP —
# the same runtime bcdl/vision uses — plus, for string I/O, the C++ tokenizer. The old
# libxlm (OE-LLM SDK) backend was removed after native proved a validated superset (tag
# `v-libxlm-final`, the design notes); nothing here probes for the SDK anymore.
#
# Off-board (a Mac checkout) the runtime isn't found: we still configure, so editors and
# `cmake --build` on the docs/tests keep working.
find_package(Hobot QUIET)       # hbDNN / hbUCP / HBRT4
find_package(Tokenizers QUIET)  # HuggingFace `tokenizers` (Rust) via tokenizers-cpp

# ── native backend ───────────────────────────────────────────────────────────
# The engines are header-only, but `model.json` parsing is not, so this is a real
# compiled target. Downstream links ONE thing and gets the include dirs, the hobot
# runtime, the tokenizer and the fp16 ISA flag.
set(BLLM_HAVE_NATIVE OFF)
set(BLLM_HAVE_TOKENIZERS OFF)
if(Hobot_FOUND)
  find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
  if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(STATUS "BLLM: nlohmann/json.hpp not found — native backend skipped")
  else()
    add_library(bllm_native src/model_config.cc)
    target_include_directories(bllm_native PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_include_directories(bllm_native PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(bllm_native PUBLIC hobot::dnn)
    # NativeHybridEngine embeds an fp16 embedding table and converts with __fp16.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      target_compile_options(bllm_native PUBLIC -march=armv8.2-a+fp16)
    endif()
    if(Tokenizers_FOUND)
      target_link_libraries(bllm_native PUBLIC tokenizers::tokenizers)
      target_compile_definitions(bllm_native PUBLIC BLLM_HAVE_TOKENIZERS)
      set(BLLM_HAVE_TOKENIZERS ON)
    else()
      message(STATUS "BLLM: tokenizers-cpp not found — native backend speaks token ids "
                     "only (no NativeLlm/NativeVlm; set -DTOKENIZERS_ROOT)")
    endif()
    # ALIAS is build-tree only; EXPORT_NAME is what makes `bllm::native` resolve
    # for a downstream find_package(bllm).
    set_target_properties(bllm_native PROPERTIES EXPORT_NAME native)
    add_library(bllm::native ALIAS bllm_native)
    set(BLLM_HAVE_NATIVE ON)
    message(STATUS "BLLM: native hbDNN backend enabled (tokenizer: ${BLLM_HAVE_TOKENIZERS})")
  endif()
endif()

if(NOT BLLM_HAVE_NATIVE)
  message(WARNING "BLLM: the native backend isn't buildable here — headers only. "
                  "This is expected on a dev host; build on the board.")
endif()

# ── examples / python ────────────────────────────────────────────────────────
if(BLLM_BUILD_EXAMPLES AND BLLM_HAVE_NATIVE)
  add_subdirectory(examples)
endif()

if(BLLM_BUILD_PYTHON)
  if(NOT BLLM_HAVE_NATIVE)
    message(FATAL_ERROR "BLLM_BUILD_PYTHON needs the native backend (board only).")
  endif()
  add_subdirectory(python)
endif()

# ── install / export (downstream: find_package(bllm)) ────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(BLLM_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/bllm)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(BLLM_HAVE_NATIVE)
  install(TARGETS bllm_native EXPORT bllmTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(EXPORT bllmTargets NAMESPACE bllm:: DESTINATION ${BLLM_CMAKE_DIR})
endif()

# The exported target names hobot::dnn / tokenizers::tokenizers, so a consumer needs
# the Find modules that create them.
install(FILES
  cmake/FindHobot.cmake
  cmake/FindTokenizers.cmake
  DESTINATION ${BLLM_CMAKE_DIR})

configure_package_config_file(
  cmake/bllmConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/bllmConfig.cmake"
  INSTALL_DESTINATION ${BLLM_CMAKE_DIR})
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/bllmConfigVersion.cmake"
  COMPATIBILITY SameMinorVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/bllmConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/bllmConfigVersion.cmake"
  DESTINATION ${BLLM_CMAKE_DIR})
