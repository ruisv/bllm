# nanobind extension module `_bllm`, placed next to the pure-Python package so
# `python/` is importable in-tree (PYTHONPATH=.../python) and installable later.

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

nanobind_add_module(_bllm STABLE_ABI NB_STATIC _bllm.cc)
target_link_libraries(_bllm PRIVATE bllm::bllm oellm::xlm)
# The module has legitimately-undefined symbols from two sources resolved at
# load time: CPython (provided by the interpreter) and libxlm's sibling libs.
# ignore-all covers both (the -shared default would too, but we must override
# any stricter flag inherited transitively).
target_link_options(_bllm PRIVATE -Wl,--unresolved-symbols=ignore-all)

# Drop the built module into the in-tree package dir so `import bllm` works with
# PYTHONPATH=<repo>/python without an install step.
set_target_properties(_bllm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bllm")
