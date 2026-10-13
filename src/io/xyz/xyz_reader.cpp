/// @file
/// @brief Implementation of the XYZ geometry reader.

module free_breakfast.io.xyz;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants;
import free_breakfast.molecule;
import free_breakfast.io.base;

#include "diagnostics.h"

namespace free_breakfast {

/// @brief Map an element symbol (e.g. "H", "He", "c") to its atomic number.
///
/// Case-insensitive on the first letter's neighbours: XYZ files vary between
/// "CL" and "Cl", so the lookup normalises to a leading capital and a trailing
/// lowercase before matching. Returns 0 for an unrecognised symbol, which the
/// caller turns into an IO error.
[[nodiscard]] int atomic_number(std::string_view symbol) {
  // Index 0 is a placeholder so element E sits at index Z.
  static constexpr std::array<std::string_view, 119> kSymbols{
      "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
      "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
      "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
      "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
      "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
      "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
      "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
      "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

  if (symbol.empty() || symbol.size() > 2) {
    return 0;
  }
  // Normalise to "Xx" capitalisation so "cl", "CL" and "Cl" all match "Cl".
  std::string normalised(symbol);
  normalised[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(normalised[0])));
  if (normalised.size() == 2) {
    normalised[1] = static_cast<char>(std::tolower(static_cast<unsigned char>(normalised[1])));
  }
  for (int z = 1; z < static_cast<int>(kSymbols.size()); ++z) {
    if (kSymbols[static_cast<std::size_t>(z)] == normalised) {
      return z;
    }
  }
  return 0;
}

[[nodiscard]] Result<Molecule> parse_xyz(std::string_view xyz_text, int charge, int multiplicity) {
  // Split into non-empty lines; a trailing newline or blank lines at the end
  // of the file are tolerated rather than counted as atoms.
  std::vector<std::string_view> lines;
  std::size_t pos = 0;
  while (pos <= xyz_text.size()) {
    const std::size_t next = xyz_text.find('\n', pos);
    const std::size_t end = (next == std::string_view::npos) ? xyz_text.size() : next;
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

  // Line 1: the declared atom count.
  int declared_count = 0;
  {
    std::string first(lines[0]);
    std::istringstream stream(first);
    if (!(stream >> declared_count) || declared_count < 0) {
      return err_io("XYZ line 1 is not a non-negative atom count: '{}'", first);
    }
  }

  // Lines 3.. : one atom each. Line 2 is a free-form comment and is skipped.
  std::vector<Atom<double>> atoms;
  atoms.reserve(static_cast<std::size_t>(declared_count));
  for (std::size_t i = 2; i < lines.size(); ++i) {
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
    const int number = atomic_number(symbol);
    if (number == 0) {
      return err_io("XYZ atom line {} has unknown element '{}'", i + 1, symbol);
    }
    // XYZ coordinates are in Angstrom; Molecule stores bohr. Convert here.
    atoms.push_back(Atom<double>{
        x * free_breakfast::kBohrPerAngstrom<double>, y * free_breakfast::kBohrPerAngstrom<double>,
        z * free_breakfast::kBohrPerAngstrom<double>, static_cast<Element>(number)});
  }

  if (static_cast<int>(atoms.size()) != declared_count) {
    return err_io(
        std::format("XYZ declares {} atoms but {} were found", declared_count, atoms.size()));
  }

  return Molecule{std::move(atoms), charge, multiplicity};
}

[[nodiscard]] Result<Molecule> read_xyz(const std::filesystem::path &path, int charge,
                                        int multiplicity) {
  auto text = FBF_TRY(read_file(path));
  return parse_xyz(text, charge, multiplicity);
}

} // namespace free_breakfast
