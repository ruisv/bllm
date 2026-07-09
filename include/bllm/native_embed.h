// bllm::EmbedTable — the host-side input-embedding table.
//
// Models compiled for the BPU keep `embed_tokens` OUT of the graph (it is a
// 1.2 GB gather, not compute), so the runtime looks rows up on the CPU and feeds
// the resulting hidden vector to the decoder. That same seam is what makes VLM
// possible: a vision/audio tower's output rows are spliced into this stream in
// place of the image/audio placeholder tokens.
//
// The file is a flat [vocab][hidden] array of f32 or fp16 — dtype is inferred
// from the file size. It is mmap'd, so only the touched rows are paged in.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace bllm {

class EmbedTable {
 public:
  EmbedTable(const std::string& path, int vocab, int hidden)
      : vocab_(vocab), hidden_(hidden) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("[bllm] cannot open embed table: " + path);
    struct stat st {};
    if (::fstat(fd_, &st) != 0) { ::close(fd_); throw std::runtime_error("[bllm] fstat failed: " + path); }
    bytes_ = (size_t)st.st_size;

    const size_t rows = (size_t)vocab_ * hidden_;
    if (bytes_ >= rows * 4)      elem_ = 4;
    else if (bytes_ >= rows * 2) elem_ = 2;
    else { ::close(fd_); throw std::runtime_error("[bllm] embed table too small: " + path); }

    base_ = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (base_ == MAP_FAILED) { ::close(fd_); throw std::runtime_error("[bllm] mmap failed: " + path); }
  }
  ~EmbedTable() {
    if (base_ && base_ != MAP_FAILED) ::munmap(base_, bytes_);
    if (fd_ >= 0) ::close(fd_);
  }
  EmbedTable(const EmbedTable&) = delete;
  EmbedTable& operator=(const EmbedTable&) = delete;

  int vocab() const { return vocab_; }
  int hidden() const { return hidden_; }
  bool is_fp16() const { return elem_ == 2; }

  // Write embedding row `id` into `dst` (hidden floats).
  void lookup(int id, float* dst) const {
    if (id < 0 || id >= vocab_) throw std::runtime_error("[bllm] embed id out of range");
    const size_t off = (size_t)id * hidden_;
    if (elem_ == 4) {
      const float* src = reinterpret_cast<const float*>(base_) + off;
      for (int i = 0; i < hidden_; ++i) dst[i] = src[i];
    } else {
      const __fp16* src = reinterpret_cast<const __fp16*>(base_) + off;
      for (int i = 0; i < hidden_; ++i) dst[i] = (float)src[i];
    }
  }

 private:
  int fd_ = -1;
  void* base_ = nullptr;
  size_t bytes_ = 0;
  int elem_ = 4;
  int vocab_ = 0, hidden_ = 0;
};

}  // namespace bllm
