/// @file
/// @brief Implementation of the whole-file write primitive.

module free_breakfast.io.base;

import std;
import free_breakfast.core.error_handling;

namespace free_breakfast {

[[nodiscard]] Status write_file(const std::filesystem::path &path, std::string_view contents) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    return err_io(std::format("cannot open '{}' for writing", path.string()));
  }
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!stream) {
    return err_io(std::format("error writing '{}'", path.string()));
  }
  return {};
}

} // namespace free_breakfast
