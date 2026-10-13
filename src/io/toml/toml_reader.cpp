/// @file
/// @brief Implementation of the TOML reader.

module free_breakfast.io.toml;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;

#include "diagnostics.h"

namespace free_breakfast {

[[nodiscard]] Result<toml::table> parse_toml(std::string_view toml_text) {
  // toml++ is built with exceptions ENABLED in this image, so parse() returns
  // a bare table and throws toml::parse_error on a syntax error. Catch it here
  // so a malformed document is a Result, never a throw.
  try {
    return toml::parse(toml_text);
  } catch (const toml::parse_error &e) {
    const auto &region = e.source();
    return err_io("malformed TOML at line {}, column {}: {}", region.begin.line,
                  region.begin.column, e.description());
  }
}

[[nodiscard]] Result<toml::table> read_toml(const std::filesystem::path &path) {
  auto text = FBF_TRY(read_file(path));
  return parse_toml(text);
}

} // namespace free_breakfast
