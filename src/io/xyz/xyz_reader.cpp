/// @file
/// @brief Implementation of the XYZ geometry reader.

module free_breakfast.io.xyz;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants;
import free_breakfast.atom;
import free_breakfast.io.base;

#include "diagnostics.h"

namespace free_breakfast {

[[nodiscard]] Result<std::vector<Atom>> parse_xyz(std::string_view xyz_text) {
  // Split into non-empty lines; a trailing newline or blank lines at the end
  // of the file are tolerated rather than counted as atoms.
  std::vector<std::string_view> lines;
  size_t pos = 0;
  while (pos <= xyz_text.size()) {
    const size_t next = xyz_text.find('\n', pos);
    const size_t end = (next == std::string_view::npos) ? xyz_text.size() : next;
    std::string_view line = xyz_text.substr(pos, end - pos);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    lines.push_back(line);
    if (next == std::string_view::npos) {
      break;
    }
    pos = next + 1;
  }

  if (lines.size() < 2) {
    return err_io("XYZ input is too short: expected a count line and a comment line");
  }

  // Line 1: the declared atom count. A geometry has at least one atom.
  int32_t declared_count = 0;
  {
    std::string first(lines[0]);
    std::istringstream stream(first);
    if (!(stream >> declared_count) || declared_count < 1) {
      return err_io("XYZ line 1 is not a positive atom count: '{}'", first);
    }
  }

  // Lines 3.. : one atom each. Line 2 is a free-form comment and is skipped.
  std::vector<Atom> atoms;
  atoms.reserve(static_cast<size_t>(declared_count));
  for (size_t i = 2; i < lines.size(); ++i) {
    const std::string_view line = lines[i];
    // A blank tail line after the atoms is not an atom.
    const bool all_space = std::ranges::all_of(
        line, [](char c) { return std::isspace(static_cast<unsigned char>(c)); });
    if (all_space) {
      continue;
    }
    std::string record(line);
    std::istringstream stream(record);
    std::string symbol;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(stream >> symbol >> x >> y >> z)) {
      return err_io("XYZ atom line {} is malformed: '{}'", i + 1, record);
    }
    const Element element = element_from_symbol(symbol);
    if (element == Element::Sentinel) {
      return err_io("XYZ atom line {} has unknown element '{}'", i + 1, symbol);
    }
    // XYZ coordinates are in Angstrom; an Atom holds bohr. Convert here.
    atoms.push_back(Atom{{x * free_breakfast::kBohrPerAngstrom<double>,
                          y * free_breakfast::kBohrPerAngstrom<double>,
                          z * free_breakfast::kBohrPerAngstrom<double>},
                         element});
  }

  if (static_cast<int32_t>(atoms.size()) != declared_count) {
    return err_io("XYZ declares {} atoms but {} were found", declared_count, atoms.size());
  }

  return atoms;
}

[[nodiscard]] Result<std::vector<Atom>> read_xyz(const std::filesystem::path &path) {
  auto text = FBF_TRY(read_file(path));
  return parse_xyz(text);
}

} // namespace free_breakfast
