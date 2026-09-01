// SmolVLA as a resident service: load the package once, answer observations over TCP.
//
// Scoring a policy means thousands of inferences and the three .hbm take seconds
// to load, so the one-shot CLI is not a harness. What matters more: this serves
// `SmolVlaPolicy`, not `SmolVlaStack` — the board does the resize, the
// tokenization, the state projection and the unnormalisation, so a success rate
// measured through this socket is a success rate for **the code path users get**.
// π0.5 learned that distinction the expensive way: its packaged path and its
// development path differed by one episode in a hundred, and only a gate that
// drove the packaged path could see it.
//
//   -> flags  [1]        i32   bit 0: a prompt follows (send it when the task changes)
//      h0 w0 h1 w1       i32   image dimensions, so the caller need not resize
//      plen sdim         i32
//      image0 [h0*w0*3]  u8    interleaved RGB, any size
//      image1 [h1*w1*3]  u8
//      prompt [plen]     char  (only when flags & 1)
//      state  [sdim]     f32   the dataset's own units; the board normalises
//      noise  [H*32]     f32   pin it, or the comparison measures the policy's
//                              own stochasticity (position max|d| 0.408) rather
//                              than anything about the board
//   <- actions [H*action_dim_real] f32, already unnormalised
//
// Raw bytes rather than fp16 patches because the payload is 393 KB against
// π0.5's 1.47 MB, and because sending pixels is what makes this the packaged
// path. One connection at a time, each request independent — what a simulator
// driving one arm needs.
#include "bllm/native_smolvla_policy.h"

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

const char* argOf(int argc, char** argv, const char* k, const char* d = nullptr) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], k) == 0) return argv[i + 1];
  return d;
}

bool readAll(int fd, void* buf, size_t n) {
  auto* p = (uint8_t*)buf;
  while (n) {
    const ssize_t r = ::read(fd, p, n);
    if (r <= 0) return false;
    p += r;
    n -= (size_t)r;
  }
  return true;
}

bool writeAll(int fd, const void* buf, size_t n) {
  auto* p = (const uint8_t*)buf;
  while (n) {
    const ssize_t w = ::write(fd, p, n);
    if (w <= 0) return false;
    p += w;
    n -= (size_t)w;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const char* dir = argOf(argc, argv, "--model");
    const int port = std::atoi(argOf(argc, argv, "--port", "9601"));
    if (!dir) {
      std::fprintf(stderr, "usage: %s --model DIR [--port N]\n", argv[0]);
      return 2;
    }
    auto cfg = bllm::loadModelConfig(dir);
    if (!cfg.is_smolvla())
      throw std::runtime_error("[smolvla] " + std::string(dir) + " is arch=" + cfg.arch);
    bllm::SmolVlaPolicy policy(cfg);
    const int H = policy.horizon(), A = policy.actionDim();
    std::printf("[smolvla-server] %s ready on :%d  (%d x %d actions)\n",
                cfg.name.empty() ? dir : cfg.name.c_str(), port, H, A);
    std::fflush(stdout);

    const int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = INADDR_ANY;
    a.sin_port = htons((uint16_t)port);
    if (::bind(srv, (sockaddr*)&a, sizeof(a)) < 0) throw std::runtime_error("bind failed");
    ::listen(srv, 4);

    std::vector<uint8_t> img0, img1;
    std::vector<float> state, noise((size_t)H * 32), actions;
    std::string prompt;
    long calls = 0;
    for (;;) {
      const int c = ::accept(srv, nullptr, nullptr);
      if (c < 0) continue;
      ::setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      for (;;) {
        int32_t hdr[7];
        if (!readAll(c, hdr, sizeof(hdr))) break;
        const int flags = hdr[0], h0 = hdr[1], w0 = hdr[2], h1 = hdr[3], w1 = hdr[4];
        const int plen = hdr[5], sdim = hdr[6];
        img0.resize((size_t)h0 * w0 * 3);
        img1.resize((size_t)h1 * w1 * 3);
        state.resize((size_t)sdim);
        if (!readAll(c, img0.data(), img0.size())) break;
        if (h1 && !readAll(c, img1.data(), img1.size())) break;
        if (flags & 1) {
          prompt.assign((size_t)plen, '\0');
          if (plen && !readAll(c, &prompt[0], (size_t)plen)) break;
        }
        if (sdim && !readAll(c, state.data(), state.size() * 4)) break;
        if (!readAll(c, noise.data(), noise.size() * 4)) break;

        bllm::SmolVlaObservation obs;
        obs.image = img0.data();
        obs.image_h = h0;
        obs.image_w = w0;
        if (h1) {
          obs.wrist_image = img1.data();
          obs.wrist_h = h1;
          obs.wrist_w = w1;
        }
        obs.state = sdim ? state.data() : nullptr;
        obs.prompt = prompt;

        const auto t0 = std::chrono::steady_clock::now();
        actions = policy.act(obs, noise.data());
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (!writeAll(c, actions.data(), actions.size() * sizeof(float))) break;
        if ((++calls % 20) == 1)
          std::printf("[smolvla-server] call %ld: %.0f ms  %s\n", calls, ms,
                      prompt.substr(0, 48).c_str()),
              std::fflush(stdout);
      }
      ::close(c);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAILED: %s\n", e.what());
    return 1;
  }
}
