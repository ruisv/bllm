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

# The NATIVE engine binding — its own module linking only the generic hobot
# runtime (no libxlm), so `bllm._bllm_native.NativeEngine` runs a .hbm on hbDNN.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(Hobot QUIET)
find_package(Tokenizers QUIET)
if(Hobot_FOUND)
  # With the C++ tokenizer, the module also exposes the unified string-in NativeLlm.
  if(Tokenizers_FOUND)
    nanobind_add_module(_bllm_native STABLE_ABI NB_STATIC
      _bllm_native.cc "${CMAKE_SOURCE_DIR}/src/model_config.cc")
    target_link_libraries(_bllm_native PRIVATE hobot::dnn tokenizers::tokenizers)
    target_compile_definitions(_bllm_native PRIVATE BLLM_HAVE_TOKENIZERS)
    target_compile_options(_bllm_native PRIVATE -march=armv8.2-a+fp16)
    message(STATUS "_bllm_native: native engine + unified NativeLlm (no libxlm)")
  else()
    nanobind_add_module(_bllm_native STABLE_ABI NB_STATIC _bllm_native.cc)
    target_link_libraries(_bllm_native PRIVATE hobot::dnn)
    message(STATUS "_bllm_native: native engine binding (no NativeLlm — tokenizers not found)")
  endif()
  target_include_directories(_bllm_native PRIVATE "${CMAKE_SOURCE_DIR}/include")
  target_link_options(_bllm_native PRIVATE -Wl,--unresolved-symbols=ignore-all)
  set_target_properties(_bllm_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bllm")
else()
  message(STATUS "_bllm_native: skipped (hobot runtime not found)")
endif()
