/// @file
/// @brief Implementation of the JSON reader.

module free_breakfast.io.json;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;

#include "diagnostics.h"

namespace free_breakfast {

[[nodiscard]] Result<json> parse_json(std::string_view json_text) {
  // nlohmann::json::parse throws on malformed input by default; catch it here
  // so a broken document is a Result rather than an escaping exception.
  try {
    return json::parse(json_text);
  } catch (const std::exception &e) {
    return err_io("malformed JSON: {}", e.what());
  }
}

[[nodiscard]] Result<json> read_json(const std::filesystem::path &path) {
  auto text = FBF_TRY(read_file(path));
  return parse_json(text);
}

} // namespace free_breakfast
