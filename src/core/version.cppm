// The project's own version, as a module partition.
//
// A partition rather than a separate module because it is part of
// free_breakfast.core's surface and has no independent consumer: `export import
// :version;` in core.cppm is what republishes it, and a partition cannot be
// imported from outside the module, so nothing can come to depend on it
// directly by accident.
export module free_breakfast.core:version;

import std;
import free_breakfast.core.types;

export namespace free_breakfast::core {

// Kept in step with project(VERSION ...) in the root CMakeLists by hand. There
// is deliberately no configure_file: a generated header would have to be
// included from a global module fragment, and that is the one thing this tree
// is organised to avoid.
inline constexpr int32_t kVersionMajor = 0;
inline constexpr int32_t kVersionMinor = 1;
inline constexpr int32_t kVersionPatch = 0;

[[nodiscard]] inline std::string version_string() {
  return std::format("{}.{}.{}", kVersionMajor, kVersionMinor, kVersionPatch);
}

} // namespace free_breakfast::core
