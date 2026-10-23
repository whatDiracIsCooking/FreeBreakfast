/// @file
/// @brief The `import` front end for the atom types: re-exports Element and
///        Atom from the headers, and adds the host-only symbol table.
///
/// element.h / atom.h hold the single definitions (for CUDA `#include`). This
/// module includes them in its global module fragment and re-exports the names.
///
/// The re-export must be written *inside* `namespace free_breakfast`
/// (not as `export using free_breakfast::Element;` at global scope):
/// a name brought in from the global module fragment is only usable by importers
/// when the re-exporting using-declaration sits in the namespace that owns the
/// name. At global scope importers see the name but cannot form a declaration
/// with it ("declaration is not visible"). Because we re-export the header's own
/// entity, `import`ers and CUDA `#include`rs share one type, not two.
module;

#include "atom.h"

export module free_breakfast.atom;

import std;

export namespace free_breakfast {

using free_breakfast::Atom;
using free_breakfast::Element;

} // namespace free_breakfast

// Not exported: the table both lookups below read.
namespace free_breakfast {

// Index 0 is the Sentinel's empty symbol, so element Z sits at index Z.
inline constexpr std::array<std::string_view, 119> kElementSymbols{
    "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief The symbol of @p element ("H", "Og"); empty for Element::Sentinel.
[[nodiscard]] constexpr std::string_view element_symbol(Element element) {
  return kElementSymbols[static_cast<size_t>(element)];
}

/// @brief The element a symbol names, or Element::Sentinel if none does.
///
/// Case-insensitive: files vary between "CL" and "Cl", so the lookup
/// normalises to a leading capital and a trailing lowercase before matching.
[[nodiscard]] constexpr Element element_from_symbol(std::string_view symbol) {
  if (symbol.empty() || symbol.size() > 2) {
    return Element::Sentinel;
  }
  const auto upper = [](char c) {
    return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
  };
  const auto lower = [](char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
  };
  const std::array<char, 2> normalised{upper(symbol[0]),
                                       symbol.size() == 2 ? lower(symbol[1]) : '\0'};
  const std::string_view key(normalised.data(), symbol.size());
  for (size_t z = 1; z < kElementSymbols.size(); ++z) {
    if (kElementSymbols[z] == key) {
      return static_cast<Element>(z);
    }
  }
  return Element::Sentinel;
}

} // namespace free_breakfast
