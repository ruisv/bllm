# vendor/

Third-party single-header libraries, copied in so BLLM's native runtime has no
external dependencies beyond the board's hobot runtime.

- `stb/stb_image.h` — [nothings/stb](https://github.com/nothings/stb) v2.30,
  public domain (or MIT, dual-licensed). Used only by `bllm/image_io.h` to decode
  JPEG/PNG for the CLI and the `NativeVlm` image-path convenience overload; the
  library API itself takes raw RGB8, which is what a camera pipeline holds.
