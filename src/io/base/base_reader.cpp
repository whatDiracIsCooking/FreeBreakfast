/// @file
/// @brief Implementation of the whole-file read primitive.

module free_breakfast.io.base;

import std;
import free_breakfast.core.error_handling;

namespace free_breakfast {

[[nodiscard]] Result<std::string> read_file(const std::filesystem::path &path) {
  const std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return err_io(std::format("cannot open '{}'", path.string()));
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  if (stream.bad()) {
    return err_io(std::format("error reading '{}'", path.string()));
  }
  return buffer.str();
}

} // namespace free_breakfast
