// bllm_beingh_probe — run one Being-H0.5 action-token pass on the board through
// the four-way shared-KV replay, and dump the gen hidden states.
//
// Deliberately fed pre-built stream embeddings rather than raw observations: it
// isolates the graph plumbing (shared window, mask, padding, rope) from the
// tokenizer and the host-side flow-matching modules, so a mismatch here has one
// possible cause. The host reference for the very same inputs comes from
// host_toolchain/convert/beingh/llm_eager_check.py.
//
//   ./bllm_beingh_probe --und und.hbm --gen1 gen1.hbm --gen15 gen15.hbm \
//        --dir /path/with/x_und.f32,x_gen.f32,segs.txt --out gen_out.f32
//
// segs.txt is one "start end und|gen" per line, in position order.
#include "bllm/native_beingh.h"
#include "bllm/native_beingh_host.h"
#include "bllm/native_vision.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char* argOf(int argc, char** argv, const char* flag, const char* dflt = nullptr) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return dflt;
}

std::vector<float> readF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const size_t n = (size_t)f.tellg() / sizeof(float);
  std::vector<float> v(n);
  f.seekg(0);
  f.read((char*)v.data(), n * sizeof(float));
  return v;
}

struct Seg { int start, end; bool gen; };

// The flat {"key": number} config export_flow_board.py writes next to the
// noise and direction dumps. Reading it is what keeps the loop's shape tied to
// the golden it is replaying: run a 10-step dump with the 4-step default and
// the integration is simply wrong, with no error anywhere — precisely the
// failure this probe exists to rule out.
std::map<std::string, double> readJson(const std::string& path) {
  std::map<std::string, double> m;
  std::ifstream f(path);
  if (!f) return m;
  std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t i = 0;
  while ((i = all.find('"', i)) != std::string::npos) {
    const size_t e = all.find('"', i + 1);
    if (e == std::string::npos) break;
    const std::string key = all.substr(i + 1, e - i - 1);
    size_t c = all.find(':', e);
    if (c == std::string::npos) break;
    ++c;
    while (c < all.size() && std::isspace((unsigned char)all[c])) ++c;
    m[key] = std::atof(all.c_str() + c);
    i = c;
  }
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  const char* dir = argOf(argc, argv, "--dir");
  const char* und = argOf(argc, argv, "--und");
  const char* gen1 = argOf(argc, argv, "--gen1");
  const char* gen15 = argOf(argc, argv, "--gen15");
  if (!dir || !und || !gen1 || !gen15) {
    std::fprintf(stderr,
                 "usage: %s --und u.hbm --gen1 g1.hbm --gen15 g15.hbm --dir D\n"
                 "          [--out f32] [--und-n 192] [--chunk 15] [--cache-len 512] [--repeat N]\n"
                 "          [--vit t.hbm [--vit-graph beingh_vit]] [--und-small N]\n"
                 "          [--host host.bin --flow DIR]  run the flow loop; DIR/flow.json,\n"
                 "            written by export_flow_board.py, supplies flow_steps / mpg_iters /\n"
                 "            num_timestep_buckets / mpg_lambda / mpg_gate_temperature /\n"
                 "            n_swd_dirs, and overrides the corresponding flags.\n",
                 argv[0]);
    return 2;
  }
  const int undN = std::atoi(argOf(argc, argv, "--und-n", "192"));
  const int chunk = std::atoi(argOf(argc, argv, "--chunk", "15"));
  const int repeat = std::atoi(argOf(argc, argv, "--repeat", "1"));

  try {
    bllm::BeingHShape shape;
    shape.cache_len = std::atoi(argOf(argc, argv, "--cache-len", "512"));

    // The tower is loaded first when asked for, so "do all three graphs fit at
    // once" is answered before anything else runs — 3.5 GB of weights against an
    // ION budget the plan put at 2.5 GB.
    std::unique_ptr<bllm::VisionTower> tower;
    if (const char* vitHbm = argOf(argc, argv, "--vit")) {
      bllm::VisionSpec vs;
      vs.patch = 14; vs.merge = 2; vs.temporal = 1;
      vs.order = bllm::PatchOrder::RowMajor;
      const float m[3] = {0.485f, 0.456f, 0.406f}, s[3] = {0.229f, 0.224f, 0.225f};
      std::memcpy(vs.mean, m, sizeof m);
      std::memcpy(vs.std, s, sizeof s);
      tower.reset(new bllm::VisionTower(vitHbm, argOf(argc, argv, "--vit-graph", "beingh_vit"), vs));
      std::printf("[beingh] vision tower: %dpx -> %d tokens x %d\n", tower->image_size(),
                  tower->n_token(), tower->hidden());
    }

    bllm::BeingHStack stack({und, gen1, gen15}, shape);
    stack.add("und", ("beingh_und" + std::to_string(undN)).c_str(), undN,
              shape.hidden_und, false, true);
    // Segment B is ~37 tokens; running it through the 192-slot graph wastes 81 %
    // of the call. A second, shorter und graph is optional — the two share their
    // 2.0 B weights when linked into one .hbm, so it costs 213 MB, not 2 GB.
    const int undSmall = std::atoi(argOf(argc, argv, "--und-small", "0"));
    if (undSmall)
      stack.add("und_small", ("beingh_und" + std::to_string(undSmall)).c_str(), undSmall,
                shape.hidden_und, false, true);
    stack.add("gen1", "beingh_gen1", 1, shape.hidden_gen, true, true);
    stack.add("gen", ("beingh_gen" + std::to_string(chunk) + "_nc").c_str(), chunk,
              shape.hidden_gen, true, false);

    const std::string d(dir);
    const auto xu = readF32(d + "/x_und.f32");
    const auto xg = readF32(d + "/x_gen.f32");
    std::vector<Seg> segs;
    {
      std::ifstream f(d + "/segs.txt");
      if (!f) throw std::runtime_error("cannot open " + d + "/segs.txt");
      std::string line;
      while (std::getline(f, line)) {
        std::istringstream is(line);
        Seg s{}; std::string kind;
        if (is >> s.start >> s.end >> kind) { s.gen = (kind == "gen"); segs.push_back(s); }
      }
    }
    const size_t P = xu.size() / shape.hidden_und;
    std::printf("[beingh] %zu positions, %zu segments, cache_len %d\n", P, segs.size(),
                shape.cache_len);

    // Optional: the whole flow-matching loop, host modules included.
    const char* hostBin = argOf(argc, argv, "--host");
    const char* flowDir = argOf(argc, argv, "--flow");

    std::unique_ptr<bllm::BeingHHost> host;
    std::vector<float> noise, dirs;
    // flow.json is the exporter's own record of the loop it dumped, so it wins
    // over any default here; a flag is only consulted when the key is absent.
    std::map<std::string, double> cfg;
    if (hostBin && flowDir) {
      host.reset(new bllm::BeingHHost(hostBin));
      cfg = readJson(std::string(flowDir) + "/flow.json");
      if (cfg.empty())
        std::fprintf(stderr, "[beingh] warning: no flow.json in %s, using flag defaults\n",
                     flowDir);
      noise = readF32(std::string(flowDir) + "/noise.f32");
      dirs = readF32(std::string(flowDir) + "/swd_dirs.f32");
      std::printf("[beingh] host modules loaded; %zu noise floats, %zu swd floats\n",
                  noise.size(), dirs.size());
    }
    auto cfgInt = [&](const char* key, const char* flag, const char* dflt) {
      auto it = cfg.find(key);
      return it != cfg.end() ? (int)it->second : std::atoi(argOf(argc, argv, flag, dflt));
    };
    auto cfgFloat = [&](const char* key, const char* flag, const char* dflt) {
      auto it = cfg.find(key);
      return it != cfg.end() ? (float)it->second : (float)std::atof(argOf(argc, argv, flag, dflt));
    };
    const int flowSteps = cfgInt("flow_steps", "--flow-steps", "4");
    const int mpgIters = cfgInt("mpg_iters", "--mpg-iters", "2");
    const int buckets = cfgInt("num_timestep_buckets", "--buckets", "1000");
    const float mpgLambda = cfgFloat("mpg_lambda", "--mpg-lambda", "0.1");
    const float mpgTemp = cfgFloat("mpg_gate_temperature", "--mpg-temp", "2.0");
    // The action width is the decoder's, not a flag's; flow.json's unified_dim
    // has to agree or the noise dump does not belong to this checkpoint.
    const int aDim = host ? host->actionDim() : cfgInt("unified_dim", "--action-dim", "200");
    if (host && cfg.count("action_chunk") && (int)cfg["action_chunk"] != chunk)
      throw std::runtime_error("[beingh] flow.json action_chunk " +
                               std::to_string((int)cfg["action_chunk"]) + " != --chunk " +
                               std::to_string(chunk));
    if (host && cfg.count("unified_dim") && (int)cfg["unified_dim"] != aDim)
      throw std::runtime_error("[beingh] flow.json unified_dim " +
                               std::to_string((int)cfg["unified_dim"]) +
                               " != decoder action dim " + std::to_string(aDim));

    const float* out = nullptr;
    int outRows = 0;
    double msPrefix = 0, msTotal = 0;
    std::vector<float> actions;
    for (int r = 0; r < repeat; ++r) {
      stack.reset();
      auto t0 = std::chrono::steady_clock::now();
      // ---- prefix: A -> s -> B, everything invariant across the flow steps ----
      const Seg* actionSeg = nullptr;
      for (const auto& s : segs) {
        const int n = s.end - s.start;
        if (s.gen && n > 1) { actionSeg = &s; continue; }
        auto& e = stack.graph(s.gen ? "gen1"
                                    : (undSmall && n <= undSmall ? "und_small" : "und"));
        const float* x = s.gen ? xg.data() + (size_t)s.start * shape.hidden_gen
                               : xu.data() + (size_t)s.start * shape.hidden_und;
        stack.run(e, x, n, s.start);
        if (r == 0)
          std::printf("  %-3s [%d,%d) n=%d N=%d\n", s.gen ? "gen" : "und", s.start,
                      s.end, n, e.n);
      }
      if (!actionSeg) { std::fprintf(stderr, "no action segment in segs.txt\n"); return 1; }
      const int T = actionSeg->end - actionSeg->start;
      outRows = T;
      auto& ga = stack.graph("gen");
      msPrefix += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();

      if (!hostBin || !flowDir) {
        // graph plumbing only: one action pass from the dumped features
        out = stack.run(ga, xg.data() + (size_t)actionSeg->start * shape.hidden_gen, T,
                        actionSeg->start);
        msTotal += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
        if (r == 0) std::printf("  gen [%d,%d) n=%d N=%d  (no cache out)\n",
                                actionSeg->start, actionSeg->end, T, ga.n);
        continue;
      }

      // ---- the flow-matching loop ------------------------------------------
      // Both dumps are cut by dimensions the checkpoint owns, not by flags: a
      // wrong stride here reads past the vector silently rather than failing.
      const int D = host->mpgEmbedDim();
      const int nDirs = cfgInt("n_swd_dirs", "--swd-dirs", "32");
      const size_t noiseDraw = (size_t)T * aDim, dirSet = (size_t)nDirs * D;
      if (noise.size() < noiseDraw * mpgIters)
        throw std::runtime_error("[beingh] noise.f32 holds " +
                                 std::to_string(noise.size() / noiseDraw) + " draws, the loop " +
                                 "needs " + std::to_string(mpgIters));
      std::vector<float> feats, vel, cleanEmb;
      int swdCall = 0;
      for (int it = 0; it < mpgIters; ++it) {
        actions.assign(noise.begin() + (size_t)it * noiseDraw,
                       noise.begin() + (size_t)(it + 1) * noiseDraw);
        for (int st = 0; st < flowSteps; ++st) {
          const int bucket = (int)((double)st / flowSteps * buckets);
          host->encodeActions(actions.data(), T, bucket, feats);
          if (it > 0 && host->hasMpg()) {
            if (dirs.size() < ((size_t)swdCall + 1) * dirSet)
              throw std::runtime_error("[beingh] swd_dirs.f32 holds " +
                                       std::to_string(dirs.size() / dirSet) +
                                       " direction sets, the loop needs more");
            host->mpgEnhance(feats, T, cleanEmb.data(),
                            dirs.data() + (size_t)swdCall++ * dirSet, nDirs,
                            mpgLambda, mpgTemp);
          }
          const float* h = stack.run(ga, feats.data(), T, actionSeg->start);
          host->decode(h, T, vel);
          if (vel.size() < (size_t)T * aDim)
            throw std::runtime_error("[beingh] decoder emitted " +
                                     std::to_string(vel.size() / T) + " values per step, expected " +
                                     std::to_string(aDim));
          for (int i = 0; i < T * aDim; ++i) actions[i] += vel[i] / (float)flowSteps;
          out = h;
        }
        if (it + 1 < mpgIters) host->encodeActions(actions.data(), T, 0, cleanEmb);
      }
      msTotal += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0).count();
      if (r == 0)
        std::printf("  flow: %d iterations x %d steps = %d action passes\n", mpgIters,
                    flowSteps, mpgIters * flowSteps);
    }
    std::printf("[beingh] prefix %.1f ms   full chunk %.1f ms\n", msPrefix / repeat,
                msTotal / repeat);

    if (!actions.empty()) {
      if (const char* af = argOf(argc, argv, "--actions-out")) {
        std::ofstream o(af, std::ios::binary);
        o.write((const char*)actions.data(), actions.size() * sizeof(float));
        std::printf("[beingh] wrote %s (%zu floats)\n", af, actions.size());
      }
      double s = 0;
      for (float a : actions) s += (double)a * a;
      std::printf("[beingh] action chunk rms=%.6f  head: %.4f %.4f %.4f\n",
                  std::sqrt(s / actions.size()), actions[0], actions[1], actions[2]);
    }
    if (!out) return 0;
    // The graph has `chunk` slots but only the segment's `outRows` are real;
    // the padded tail is the graph's answer to zero rows, and scoring or
    // dumping it both dilutes the rms and makes the file the wrong size to
    // diff against the reference's out_gen[:n].
    const size_t nOut = (size_t)outRows * shape.hidden_gen;
    double ss = 0;
    for (size_t i = 0; i < nOut; ++i) ss += (double)out[i] * out[i];
    std::printf("[beingh] gen out rms=%.6f (%d rows)  head: %.4f %.4f %.4f %.4f\n",
                std::sqrt(ss / nOut), outRows, out[0], out[1], out[2], out[3]);
    if (const char* of = argOf(argc, argv, "--out")) {
      std::ofstream o(of, std::ios::binary);
      o.write((const char*)out, nOut * sizeof(float));
      std::printf("[beingh] wrote %s (%zu floats)\n", of, nOut);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
