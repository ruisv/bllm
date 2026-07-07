// BLLM — model package resolution: accept a .tar.gz / directory / .hbm and
// resolve it to concrete (hbm, config-dir) paths, auto-extracting an archive on
// first use. Lets a caller point BLLM straight at a downloaded package.

#ifndef BLLM_PACKAGE_H_
#define BLLM_PACKAGE_H_

#include <string>

namespace bllm {

// A model source resolved to concrete paths.
struct ModelPackage {
  std::string hbm_path;       // the .hbm to load
  std::string tokenizer_dir;  // the tokenizer/config dir ("" if none found)
};

// Resolve a model source:
//   - "<name>.tar.gz" / ".tgz": extracted ONCE next to the archive (into
//     "<name>/"); on later runs the already-extracted dir is reused. The archive
//     is expected to contain a top-level "<name>/". Then resolved as a directory.
//   - a directory: finds the single *.hbm and the config dir (a "config" subdir,
//     else the dir holding tokenizer.json, else the dir itself).
//   - a ".hbm" file: returned as-is (tokenizer_dir empty).
// Throws bllm::Error on failure (missing archive, extract failure, no/ambiguous
// .hbm).
ModelPackage ResolvePackage(const std::string& path);

}  // namespace bllm

#endif  // BLLM_PACKAGE_H_
