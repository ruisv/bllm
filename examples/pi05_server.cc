// π0.5 as a resident service: load the three graphs once, answer observations
// over TCP.
//
// Scoring a policy means thousands of inferences, and the three .hbm take ~9 s
// to load and 4.57 GB to hold, so the one-shot probe is not a harness. The wire
// format is deliberately fixed-size and dumb — every shape is known at startup,
// so a request is just four blobs back to back and a reply is one:
//
//   -> flags   [1]               i32   bit 0: a new prompt (and rope) follows
//      patches [cameras*256, 588] f16   (host-side patchify, already verified)
//      prompt  [prompt_len, 2048] f16   (only when it changed; one per task)
//      cos     [horizon, 256]     f32   (with the prompt: RoPE for the action rows)
//      sin     [horizon, 256]     f32
//      bias    [prefix]           f32   (0 for a real token, very negative for padding)
//      noise   [horizon, 32]      f32
//   <- actions [horizon, 32]      f32
//
// cos/sin ride along with the prompt because both depend only on the
// instruction: the action rows sit at `cameras*256 + n + [0, horizon)` where
// `n` is the real token count, so the table changes exactly when the prompt
// does. Sending it rather than baking it is what makes a short instruction
// correct -- see the comment on `Pi05Stack::sampleActions`.
//
// fp16 on the wire and a sticky prompt because the board's socket read measured
// **4757 ms for 1.47 MB** against 1018 ms of inference — 0.31 MB/s, where plain
// `nc` to the same board gets 2.9 MB/s and `scp` 11 MB/s. That gap is not
// understood yet; halving the payload and dropping the per-call prompt is worth
// doing regardless of what explains it.
//
// One connection is served at a time and each request is independent, which is
// what a simulator driving one arm actually needs.
#include "bllm/native_pi05.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

const char* argOf(int argc, char** argv, const char* flag, const char* dflt = nullptr) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return dflt;
}

// Read exactly n bytes or report the connection gone; a short read on a socket
// is normal and not an error, which is the whole reason this loop exists.
bool readAll(int fd, void* buf, size_t n) {
  auto* p = (uint8_t*)buf;
  while (n) {
    const ssize_t got = ::read(fd, p, n);
    if (got <= 0) return false;
    p += got;
    n -= (size_t)got;
  }
  return true;
}

bool writeAll(int fd, const void* buf, size_t n) {
  const auto* p = (const uint8_t*)buf;
  while (n) {
    const ssize_t put = ::write(fd, p, n);
    if (put <= 0) return false;
    p += put;
    n -= (size_t)put;
  }
  return true;
}

std::vector<float> readF32(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  std::fseek(f, 0, SEEK_END);
  const size_t n = (size_t)std::ftell(f) / sizeof(float);
  std::fseek(f, 0, SEEK_SET);
  std::vector<float> v(n);
  if (std::fread(v.data(), sizeof(float), n, f) != n) {
    std::fclose(f);
    throw std::runtime_error("short read on " + path);
  }
  std::fclose(f);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bllm::Pi05Shape shape;
    shape.cameras = std::atoi(argOf(argc, argv, "--cameras", "2"));
    shape.prompt_len = std::atoi(argOf(argc, argv, "--prompt-len", "32"));
    shape.horizon = std::atoi(argOf(argc, argv, "--horizon", "10"));
    shape.steps = std::atoi(argOf(argc, argv, "--steps", "10"));
    const int port = std::atoi(argOf(argc, argv, "--port", "9105"));

    bllm::Pi05Stack stack(argOf(argc, argv, "--vit"), argOf(argc, argv, "--prefill"),
                          argOf(argc, argv, "--expert"), shape);
    const auto cond = readF32(argOf(argc, argv, "--cond"));

    const size_t nPatch = (size_t)shape.cameras * shape.vision_tokens * shape.patch_row;
    const size_t nPrompt = (size_t)shape.prompt_len * shape.hidden;
    const size_t nBias = (size_t)shape.prefix();
    const size_t nAct = (size_t)shape.horizon * shape.action_dim;
    const size_t nRope = (size_t)shape.horizon * shape.head_dim;
    std::vector<float> patches(nPatch), prompt(nPrompt), bias(nBias), noise(nAct),
        actions(nAct), ropeCos(nRope), ropeSin(nRope);
    std::vector<uint16_t> half(std::max(nPatch, nPrompt));

    // fp16 -> f32 on arrival; the graphs take f32 and `writeRowsF32As` converts
    // again on the way into whatever the tensor declares.
    auto expand = [&](std::vector<float>& dst, size_t n) {
      const auto* h = (const bllm::native_detail::half_t*)half.data();
      for (size_t i = 0; i < n; ++i) dst[i] = (float)h[i];
    };

    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);
    if (::bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0)
      throw std::runtime_error(std::string("bind: ") + std::strerror(errno));
    ::listen(srv, 4);
    std::printf("[pi05] ready on :%d  prefix %d, horizon %d, %d steps  "
                "(request %.2f MB, reply %zu B)\n",
                port, shape.prefix(), shape.horizon, shape.steps,
                (nPatch * 2 + nBias * 4 + nAct * 4) / 1e6, nAct * 4);
    std::fflush(stdout);

    for (;;) {
      int fd = ::accept(srv, nullptr, nullptr);
      if (fd < 0) continue;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      std::printf("[pi05] client connected\n");
      std::fflush(stdout);
      size_t served = 0;
      double ms = 0;
      double msRead = 0;
      for (;;) {
        const auto tr = std::chrono::steady_clock::now();
        int32_t flags = 0;
        if (!readAll(fd, &flags, sizeof(flags))) break;
        if (!readAll(fd, half.data(), nPatch * 2)) break;
        expand(patches, nPatch);
        if (flags & 1) {
          if (!readAll(fd, half.data(), nPrompt * 2)) break;
          expand(prompt, nPrompt);
          if (!(readAll(fd, ropeCos.data(), nRope * 4) &&
                readAll(fd, ropeSin.data(), nRope * 4)))
            break;
        }
        if (!(readAll(fd, bias.data(), nBias * 4) &&
              readAll(fd, noise.data(), nAct * 4)))
          break;
        const auto t0 = std::chrono::steady_clock::now();
        msRead += std::chrono::duration<double, std::milli>(t0 - tr).count();
        stack.setObservation(patches.data(), prompt.data(), bias.data());
        stack.sampleActions(noise.data(), cond.data(), ropeCos.data(), ropeSin.data(),
                            actions.data());
        ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
        if (!writeAll(fd, actions.data(), nAct * 4)) break;
        if (++served % 5 == 0) {
          std::printf("[pi05] %zu inferences, %.1f ms mean (read %.1f ms mean)\n",
                      served, ms / served, msRead / served);
          std::fflush(stdout);
        }
      }
      ::close(fd);
      std::printf("[pi05] client gone after %zu inferences (%.1f ms infer, %.1f ms read)\n",
                  served, served ? ms / served : 0.0, served ? msRead / served : 0.0);
      std::fflush(stdout);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAILED: %s\n", e.what());
    return 1;
  }
}
