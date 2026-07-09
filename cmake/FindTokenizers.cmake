# FindTokenizers — locate the HuggingFace `tokenizers` C++ binding (mlc-ai/
# tokenizers-cpp + the official Rust `tokenizers` crate), linked as a single SHARED
# library so BLLM targets relink fast (static-linking the 37 MB Rust archive into
# every binary was the slow path). Point at the checkout with -DTOKENIZERS_ROOT=<p>
# (default ~/projects/tokenizers-cpp). Defines the imported target
# `tokenizers::tokenizers` (include dir + libtokenizers_hf.so + rpath).
#
# One-time build on the board (drops sentencepiece — we only use tokenizer.json):
#   for t in gcc g++ ar; do ln -sf $(which $t) ~/.local/xbin/aarch64-unknown-linux-gnu-$t; done
#   PATH=~/.local/xbin:$PATH cmake -S tokenizers-cpp -B tokenizers-cpp/build \
#       -DMLC_ENABLE_SENTENCEPIECE_TOKENIZER=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
#   PATH=~/.local/xbin:$PATH cmake --build tokenizers-cpp/build --target tokenizers_cpp -j
#   RUST=$(find tokenizers-cpp/build -name libtokenizers_c.a); \
#   g++ -shared -fPIC -o tokenizers-cpp/build/libtokenizers_hf.so \
#       -Wl,--whole-archive tokenizers-cpp/build/libtokenizers_cpp.a $RUST \
#       -Wl,--no-whole-archive -lpthread -ldl -lm

set(TOKENIZERS_ROOT "$ENV{HOME}/projects/tokenizers-cpp" CACHE PATH "tokenizers-cpp checkout")

find_path(TOKENIZERS_INCLUDE_DIR tokenizers_cpp.h HINTS "${TOKENIZERS_ROOT}/include")
find_library(TOKENIZERS_SO tokenizers_hf HINTS "${TOKENIZERS_ROOT}/build")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Tokenizers DEFAULT_MSG
  TOKENIZERS_INCLUDE_DIR TOKENIZERS_SO)

if(Tokenizers_FOUND AND NOT TARGET tokenizers::tokenizers)
  get_filename_component(_tok_libdir "${TOKENIZERS_SO}" DIRECTORY)
  add_library(tokenizers::tokenizers INTERFACE IMPORTED)
  set_target_properties(tokenizers::tokenizers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TOKENIZERS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${TOKENIZERS_SO}"
    INTERFACE_LINK_OPTIONS "-Wl,-rpath,${_tok_libdir}")   # find the .so at runtime
endif()
