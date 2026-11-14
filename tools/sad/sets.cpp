/// @file
/// @brief Implementation of @ref free_breakfast::sad_stamp_sets.

module free_breakfast.tools.sad;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.json;

#include "diagnostics.h"

namespace free_breakfast {

Result<std::vector<std::string>> sad_stamp_sets(const std::filesystem::path &basis_dir) {
  std::error_code ec;
  std::vector<std::filesystem::path> files;
  for (const auto &entry : std::filesystem::directory_iterator(basis_dir, ec)) {
    if (entry.path().extension() == ".json") {
      files.push_back(entry.path());
    }
  }
  if (ec) {
    return err_io("cannot list {}: {}", basis_dir.string(), ec.message());
  }
  std::ranges::sort(files);

  std::vector<std::string> names;
  for (const std::filesystem::path &file : files) {
    const json document = FBF_TRY(read_json(file));
    const auto name = document.find("name");
    if (name == document.end() || !name->is_string()) {
      return err_config("sad_stamp_sets: {} has no BSE 'name' string", file.string());
    }
    names.push_back(name->get<std::string>());
  }
  return names;
}

} // namespace free_breakfast
