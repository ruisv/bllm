# FindOELLM.cmake — locate the D-Robotics OE-LLM runtime (libxlm.so + xlm.h).
#
# The runtime exists ONLY on the board / inside the LLM S100 SDK. This module
# searches, in order:
#   1. -DBLLM_OELLM_SDK=<oellm_runtime dir>   (cache var, highest priority)
#   2. $ENV{OELLM_SDK}
#   3. common install roots under $HOME/llm_s100/*/oellm_runtime
#
# On success it defines the imported target `oellm::xlm` (include dirs + link
# libs + rpath to the bundled runtime), plus:
#   OELLM_FOUND, OELLM_INCLUDE_DIR, OELLM_LIB_DIR, OELLM_VERSION
#
# libxlm.so pulls its own libdnn/libhbucp/... from the same lib/ dir, so we link
# only `xlm`, add that dir to the rpath, and ignore-unresolved-in-shared-libs
# (matching the SDK's own example CMakeLists).

set(_oellm_hints "")
if(BLLM_OELLM_SDK)
  list(APPEND _oellm_hints "${BLLM_OELLM_SDK}")
endif()
if(DEFINED ENV{OELLM_SDK})
  list(APPEND _oellm_hints "$ENV{OELLM_SDK}")
endif()
# Auto-discover unpacked SDKs (newest-sorted last so it wins).
file(GLOB _oellm_globs
     "$ENV{HOME}/llm_s100/*/oellm_runtime"
     "$ENV{HOME}/llm_s100/*/*/oellm_runtime")
if(_oellm_globs)
  list(SORT _oellm_globs)
  list(APPEND _oellm_hints ${_oellm_globs})
endif()

find_path(OELLM_INCLUDE_DIR
  NAMES xlm.h
  HINTS ${_oellm_hints}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)

find_library(OELLM_XLM_LIBRARY
  NAMES xlm
  HINTS ${_oellm_hints}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH)

if(OELLM_XLM_LIBRARY)
  get_filename_component(OELLM_LIB_DIR "${OELLM_XLM_LIBRARY}" DIRECTORY)
endif()

# Derive a version string from the SDK path (…/D-Robotics_LLM_S100_1.0.0_SDK/…).
set(OELLM_VERSION "unknown")
if(OELLM_INCLUDE_DIR)
  string(REGEX MATCH "LLM_S100_([0-9]+\\.[0-9]+\\.[0-9]+)" _m "${OELLM_INCLUDE_DIR}")
  if(CMAKE_MATCH_1)
    set(OELLM_VERSION "${CMAKE_MATCH_1}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OELLM
  REQUIRED_VARS OELLM_XLM_LIBRARY OELLM_INCLUDE_DIR
  VERSION_VAR OELLM_VERSION)

if(OELLM_FOUND AND NOT TARGET oellm::xlm)
  add_library(oellm::xlm SHARED IMPORTED)
  set_target_properties(oellm::xlm PROPERTIES
    IMPORTED_LOCATION "${OELLM_XLM_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OELLM_INCLUDE_DIR}"
    # libxlm resolves its own bundled deps (libdnn/libhbucp/...) at runtime from
    # OELLM_LIB_DIR — rpath so the loader finds them.
    INTERFACE_LINK_OPTIONS "-Wl,-rpath,${OELLM_LIB_DIR}")
  # NOTE: libxlm.so references symbols from its sibling .so's without declaring
  # them NEEDED, so anything linking it must relax unresolved-symbol checking.
  # That is left to each consumer (executables: ignore-in-shared-libs; the
  # Python module: ignore-all, which also covers its undefined CPython symbols)
  # so the flag never leaks into a static-archive/module link where it would
  # wrongly reject legitimately-deferred symbols.
endif()

mark_as_advanced(OELLM_INCLUDE_DIR OELLM_XLM_LIBRARY)
