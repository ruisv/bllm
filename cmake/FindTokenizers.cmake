# FindTokenizers — locate the mlc-ai/tokenizers-cpp build (HuggingFace `tokenizers`
# via the official Rust crate) so BLLM's native engine can tokenize in pure C++.
# Point it at the tokenizers-cpp checkout with -DTOKENIZERS_ROOT=<path> (defaults to
# a sibling ~/projects/tokenizers-cpp). Defines the imported target
# `tokenizers::tokenizers` (include dir + the static libs + pthread/dl/m).
#
# Build tokenizers-cpp once on the board:
#   conda create -n tokbuild -c conda-forge rust cmake ninja
#   # prefixed-gcc symlinks so cc-rs finds a compiler for onig_sys:
#   for t in gcc g++ ar; do ln -sf $(which $t) ~/.local/xbin/aarch64-unknown-linux-gnu-$t; done
#   PATH=~/.local/xbin:$PATH cmake -S tokenizers-cpp -B tokenizers-cpp/build -G Ninja
#   PATH=~/.local/xbin:$PATH cmake --build tokenizers-cpp/build -j

set(TOKENIZERS_ROOT "$ENV{HOME}/projects/tokenizers-cpp" CACHE PATH "tokenizers-cpp checkout")

find_path(TOKENIZERS_INCLUDE_DIR tokenizers_cpp.h HINTS "${TOKENIZERS_ROOT}/include")
find_library(TOKENIZERS_CPP_LIB  tokenizers_cpp    HINTS "${TOKENIZERS_ROOT}/build")
find_library(TOKENIZERS_C_LIB    tokenizers_c      HINTS "${TOKENIZERS_ROOT}/build" "${TOKENIZERS_ROOT}/build/aarch64-unknown-linux-gnu/release")
find_library(SENTENCEPIECE_LIB   sentencepiece     HINTS "${TOKENIZERS_ROOT}/build/sentencepiece/src")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Tokenizers DEFAULT_MSG
  TOKENIZERS_INCLUDE_DIR TOKENIZERS_CPP_LIB TOKENIZERS_C_LIB)

if(Tokenizers_FOUND AND NOT TARGET tokenizers::tokenizers)
  add_library(tokenizers::tokenizers INTERFACE IMPORTED)
  set_target_properties(tokenizers::tokenizers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TOKENIZERS_INCLUDE_DIR}")
  # order matters: cpp wrapper -> c/rust lib -> sentencepiece -> system
  target_link_libraries(tokenizers::tokenizers INTERFACE
    "${TOKENIZERS_CPP_LIB}" "${TOKENIZERS_C_LIB}"
    $<$<BOOL:${SENTENCEPIECE_LIB}>:${SENTENCEPIECE_LIB}>
    pthread dl m)
endif()
