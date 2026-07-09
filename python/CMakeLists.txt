# nanobind extension modules, placed next to the pure-Python package so `python/`
# is importable in-tree (PYTHONPATH=.../python) and installable later.
#
# The two backends are separate modules on purpose: `import bllm` works with only
# one of them present (see python/bllm/__init__.py, which imports each in a try).

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate nanobind's CMake config via the active interpreter (pip/conda install).
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_CMAKE_DIR
  RESULT_VARIABLE NB_RESULT)
if(NOT NB_RESULT EQUAL 0)
  message(FATAL_ERROR
    "nanobind not found in ${Python_EXECUTABLE}. `pip install nanobind` in the "
    "build env, or configure with -DBLLM_BUILD_PYTHON=OFF.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Drop built modules into the in-tree package dir so `import bllm` works with
# PYTHONPATH=<repo>/python without an install step.
set(BLLM_PY_PKG "${CMAKE_CURRENT_SOURCE_DIR}/bllm")

# ── libxlm backend module ────────────────────────────────────────────────────
if(BLLM_HAVE_RUNTIME)
  nanobind_add_module(_bllm STABLE_ABI NB_STATIC _bllm.cc)
  target_link_libraries(_bllm PRIVATE bllm::bllm oellm::xlm)
  # The module has legitimately-undefined symbols from two sources resolved at
  # load time: CPython (provided by the interpreter) and libxlm's sibling libs.
  target_link_options(_bllm PRIVATE -Wl,--unresolved-symbols=ignore-all)
  set_target_properties(_bllm PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${BLLM_PY_PKG}")
  message(STATUS "_bllm: libxlm sessions")
endif()

# ── native backend module ────────────────────────────────────────────────────
if(BLLM_HAVE_NATIVE)
  nanobind_add_module(_bllm_native STABLE_ABI NB_STATIC _bllm_native.cc)
  target_link_libraries(_bllm_native PRIVATE bllm::native)
  if(BLLM_HAVE_TOKENIZERS)
    # NativeVlm's convenience overloads decode media files with the vendored headers.
    target_include_directories(_bllm_native PRIVATE
      "${CMAKE_SOURCE_DIR}/vendor/stb" "${CMAKE_SOURCE_DIR}/vendor/dr_libs")
    message(STATUS "_bllm_native: native engine + NativeLlm + NativeVlm (no libxlm)")
  else()
    message(STATUS "_bllm_native: native engine only (no tokenizer -> no NativeLlm/NativeVlm)")
  endif()
  target_link_options(_bllm_native PRIVATE -Wl,--unresolved-symbols=ignore-all)
  set_target_properties(_bllm_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${BLLM_PY_PKG}")
endif()
