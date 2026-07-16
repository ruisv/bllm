# nanobind extension module, placed next to the pure-Python package so `python/`
# is importable in-tree (PYTHONPATH=.../python) and installable later.
# `import bllm` still tolerates the module's absence (see python/bllm/__init__.py).

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

  # Install the extension + the pure-Python package into site-packages, so a conda/pip
  # build lands an importable `bllm` (the in-tree layout above is for PYTHONPATH dev use).
  # Gated so a plain `cmake --install` on a dev host doesn't scribble into a site-packages.
  if(BLLM_INSTALL_PYTHON)
    set(_bllm_sp "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
    # The extension goes INSIDE the package dir — bllm/__init__.py imports it relatively
    # (`from ._bllm_native import ...`), so it must land at site-packages/bllm/, not the root.
    install(TARGETS _bllm_native LIBRARY DESTINATION "${_bllm_sp}/bllm")
    install(DIRECTORY bllm/ DESTINATION "${_bllm_sp}/bllm"
            FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi"
            PATTERN "__pycache__" EXCLUDE)
    # The model-dir tool on PATH: `bllm.make_model_dir` ships inside the package above,
    # so an installed-from-conda user can assemble a model.json without cloning the repo.
    install(PROGRAMS bllm-make-model-dir DESTINATION bin)
  endif()
endif()
