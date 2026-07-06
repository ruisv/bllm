cmake_minimum_required(VERSION 3.22)
project(bllm LANGUAGES CXX VERSION 0.1.0)

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
option(BLLM_BUILD_PYTHON   "Build the nanobind Python bindings (M3)" OFF)

# ── OE-LLM runtime (board-only) ──────────────────────────────────────────────
# When the SDK is present we build the real library (src/ linked against
# libxlm.so). On a host without the SDK (e.g. a Mac checkout) we degrade to a
# header-only INTERFACE target so `cmake` still configures for tooling.
find_package(OELLM QUIET)

set(BLLM_PUBLIC_HEADERS
  include/bllm/bllm.h
  include/bllm/common.h
  include/bllm/types.h
  include/bllm/llm_session.h
  include/bllm/omni_session.h)

if(OELLM_FOUND)
  message(STATUS "BLLM: OE-LLM SDK ${OELLM_VERSION} at ${OELLM_LIB_DIR} — building full library")
  add_library(bllm
    src/common.cc
    src/types.cc
    src/llm_session.cc
    src/omni_session.cc)
  target_include_directories(bllm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  # xlm.h is an implementation detail — link it PRIVATE so downstream users of
  # bllm's headers never need the OE-LLM SDK.
  target_link_libraries(bllm PRIVATE oellm::xlm)
  set(BLLM_HAVE_RUNTIME ON)
else()
  message(WARNING
    "BLLM: OE-LLM SDK not found — configuring headers-only INTERFACE target.\n"
    "      Set -DBLLM_OELLM_SDK=<...>/oellm_runtime (or env OELLM_SDK) on the board\n"
    "      to build the real libxlm-backed library. See docs/LLM_ONBOARD.md.")
  add_library(bllm INTERFACE)
  target_include_directories(bllm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  set(BLLM_HAVE_RUNTIME OFF)
endif()

add_library(bllm::bllm ALIAS bllm)

if(BLLM_BUILD_EXAMPLES AND BLLM_HAVE_RUNTIME)
  add_subdirectory(examples)
endif()

if(BLLM_BUILD_PYTHON)
  if(NOT BLLM_HAVE_RUNTIME)
    message(FATAL_ERROR "BLLM_BUILD_PYTHON needs the OE-LLM SDK (board only).")
  endif()
  add_subdirectory(python)
endif()

# ── install / export (downstream: find_package(bllm)) ────────────────────────
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS bllm EXPORT bllmTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT bllmTargets
  NAMESPACE bllm::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bllm)
