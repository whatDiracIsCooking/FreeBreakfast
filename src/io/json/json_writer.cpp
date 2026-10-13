/// @file
/// @brief Implementation of the JSON writer.

module free_breakfast.io.json;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;

namespace free_breakfast {

std::string json_to_string(const json &document, int indent) {
  return document.dump(indent);
}

[[nodiscard]] Status write_json(const std::filesystem::path &path, const json &document, int indent) {
  return write_file(path, json_to_string(document, indent));
}

} // namespace free_breakfast
