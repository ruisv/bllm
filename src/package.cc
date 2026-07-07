#include "bllm/package.h"

#include <cstdlib>
#include <filesystem>
#include <system_error>

#include "bllm/common.h"

namespace bllm {
namespace {
namespace fs = std::filesystem;

bool EndsWith(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// The single top-level *.hbm in `dir` (package convention: the .hbm sits at the
// package root next to config/). "" if none, or more than one.
std::string FindSingleHbm(const fs::path& dir) {
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return "";
  std::string found;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (ec) break;
    // Skip dotfiles, incl. macOS AppleDouble "._name" sidecars that a Mac-made
    // tar leaves next to real files (._model.hbm would falsely match .hbm).
    if (e.path().filename().string().rfind(".", 0) == 0) continue;
    if (e.is_regular_file(ec) && e.path().extension() == ".hbm") {
      if (!found.empty()) return "";  // ambiguous
      found = e.path().string();
    }
  }
  return found;
}

// The tokenizer/config dir inside a package: a "config" subdir, else the dir
// holding tokenizer.json, else the package dir itself.
std::string FindConfigDir(const fs::path& dir) {
  std::error_code ec;
  fs::path cfg = dir / "config";
  if (fs::is_directory(cfg, ec)) return cfg.string();
  for (const auto& e : fs::recursive_directory_iterator(dir, ec)) {
    if (ec) break;
    if (e.is_regular_file(ec) && e.path().filename() == "tokenizer.json")
      return e.path().parent_path().string();
  }
  return dir.string();
}

// Paths are quoted for the shell; reject embedded single quotes rather than
// risk an injection.
std::string ShellQuote(const std::string& s) {
  BLLM_CHECK(s.find('\'') == std::string::npos,
             "path contains a single quote, unsupported: " + s);
  return "'" + s + "'";
}

}  // namespace

ModelPackage ResolvePackage(const std::string& path0) {
  std::string path = path0;
  std::error_code ec;

  // 1. archive -> extract once (skip if already extracted)
  const bool is_targz = EndsWith(path, ".tar.gz");
  if (is_targz || EndsWith(path, ".tgz")) {
    fs::path arch(path);
    BLLM_CHECK(fs::exists(arch, ec), "package archive not found: " + path);
    fs::path parent = arch.has_parent_path() ? arch.parent_path() : fs::path(".");
    std::string stem = arch.filename().string();
    stem = stem.substr(0, stem.size() - (is_targz ? 7 : 4));  // strip .tar.gz/.tgz
    fs::path outdir = parent / stem;

    if (FindSingleHbm(outdir).empty()) {  // not extracted yet (first launch)
      std::string cmd = "tar -xzf " + ShellQuote(arch.string()) + " -C " +
                        ShellQuote(parent.string());
      int rc = std::system(cmd.c_str());
      BLLM_CHECK(rc == 0, "failed to extract package (tar -xzf): " + path);
    }
    BLLM_CHECK(fs::is_directory(outdir, ec),
               "package '" + path + "' did not extract to '" + outdir.string() +
                   "' — the archive should contain a top-level '" + stem + "/'");
    path = outdir.string();
  }

  // 2. directory -> find the .hbm + config dir
  fs::path p(path);
  if (fs::is_directory(p, ec)) {
    ModelPackage r;
    r.hbm_path = FindSingleHbm(p);
    BLLM_CHECK(!r.hbm_path.empty(),
               "no single .hbm found in package directory: " + path);
    r.tokenizer_dir = FindConfigDir(p);
    return r;
  }

  // 3. plain .hbm
  return ModelPackage{path, ""};
}

}  // namespace bllm
