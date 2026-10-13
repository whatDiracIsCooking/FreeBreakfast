/// @file
/// @brief Implementation of the BSE JSON basis parser and molecule assignment.

module free_breakfast.io.basis;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.io.base;
import free_breakfast.io.json;

#include "diagnostics.h"

namespace free_breakfast {

/// @brief Convert one BSE decimal string to a double.
///
/// BSE writes exponents and coefficients as strings ("0.1873113696E+02") to
/// keep full precision independent of a JSON number parser. std::from_chars is
/// used so a trailing non-numeric character is caught rather than silently
/// dropped, and the whole token must be consumed.
[[nodiscard]] Result<double> parse_decimal(const json &node, std::string_view field) {
  if (!node.is_string()) {
    return err_config(std::format("BSE '{}' entry is not a string", field));
  }
  const auto text = node.get<std::string>();
  double value = 0.0;
  const char *first = text.data();
  const char *last = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(first, last, value);
  if (ec != std::errc{} || ptr != last) {
    return err_config(std::format("BSE '{}' entry '{}' is not a number", field, text));
  }
  return value;
}

/// @brief Resolve a BSE shell's harmonic type from its optional
/// @c function_type string and its angular momentum @p l.
///
/// BSE's own enum (@c electron_shell_function_type) is
/// @c ["gto", "gto_spherical", "gto_cartesian", "sto"]; there is no separate
/// @c harmonic_type field -- @c function_type carries it. @c gto_spherical and
/// @c gto_cartesian are unambiguous. Bare @c "gto" (or the field being
/// absent) is exact as Cartesian for @c l <= 1 (the two conventions coincide
/// there), but genuinely ambiguous at @c l >= 2 -- real BSE data tags that
/// case explicitly (e.g. cc-pVDZ's own d shell is @c gto_spherical while its
/// s/p are bare @c gto), so rejecting the untagged case here only rejects
/// input that was already underspecified, rather than silently guessing a
/// harmonic type that would corrupt every downstream integral if wrong.
/// @c sto and any other value are rejected outright: this reader only builds
/// Gaussian shells.
[[nodiscard]] Result<Harmonic> resolve_harmonic(const json &block, int l) {
  const auto ft_it = block.find("function_type");
  if (ft_it == block.end()) {
    if (l >= 2) {
      return err_config(std::format(
          "BSE shell has no function_type and angular_momentum={} >= 2 is ambiguous between "
          "Cartesian and spherical; tag it \"gto_cartesian\" or \"gto_spherical\"",
          l));
    }
    return Harmonic::Cartesian;
  }
  if (!ft_it->is_string()) {
    return err_config("BSE shell function_type is not a string");
  }
  const std::string function_type = ft_it->get<std::string>();
  if (function_type == "gto_spherical") {
    return Harmonic::Spherical;
  }
  if (function_type == "gto_cartesian") {
    return Harmonic::Cartesian;
  }
  if (function_type == "gto") {
    if (l >= 2) {
      return err_config(std::format(
          "BSE shell function_type \"gto\" with angular_momentum={} >= 2 is ambiguous between "
          "Cartesian and spherical; tag it \"gto_cartesian\" or \"gto_spherical\"",
          l));
    }
    return Harmonic::Cartesian;
  }
  return err_config(std::format("BSE shell function_type \"{}\" is not a supported Gaussian type",
                                function_type));
}

/// @brief Turn one BSE electron_shells block into one or more ShellStamp.
///
/// The block pairs a list of angular momenta with a list of coefficient
/// columns. With equal lengths (>1) it is an sp/spd-style shell: momentum i
/// uses column i. With a single momentum and several columns it is a general
/// contraction: every column becomes a shell of that one momentum. The stamps
/// are geometry-free -- place() supplies the centers. @c function_type (see
/// @ref resolve_harmonic) is a property of the whole block, so it is resolved
/// once per emitted shell against that shell's own @c l.
[[nodiscard]] Result<std::vector<ShellStamp>> shells_from_block(const json &block) {
  if (!block.is_object()) {
    return err_config("BSE electron_shells entry is not an object");
  }

  const auto am_it = block.find("angular_momentum");
  const auto exp_it = block.find("exponents");
  const auto coef_it = block.find("coefficients");
  if (am_it == block.end() || exp_it == block.end() || coef_it == block.end()) {
    return err_config("BSE shell is missing angular_momentum, exponents or coefficients");
  }
  if (!am_it->is_array() || !exp_it->is_array() || !coef_it->is_array()) {
    return err_config("BSE shell angular_momentum, exponents and coefficients must be arrays");
  }

  // Angular momenta.
  std::vector<int> momenta;
  momenta.reserve(am_it->size());
  for (const json &l_node : *am_it) {
    if (!l_node.is_number_integer() && !l_node.is_number_unsigned()) {
      return err_config("BSE angular_momentum entry is not an integer");
    }
    momenta.push_back(l_node.get<int>());
  }
  if (momenta.empty()) {
    return err_config("BSE shell has an empty angular_momentum list");
  }

  // Exponents (strings -> double).
  std::vector<double> exponents;
  exponents.reserve(exp_it->size());
  for (const json &e_node : *exp_it) {
    exponents.push_back(FBF_TRY(parse_decimal(e_node, "exponents")));
  }
  if (exponents.empty()) {
    return err_config("BSE shell has no exponents");
  }

  const std::uint32_t ncols = coef_it->size();
  if (ncols == 0) {
    return err_config("BSE shell has no coefficient columns");
  }

  // Two admissible shapes: one momentum per column (sp/spd), or one momentum
  // with several columns (a general contraction). Anything else is ambiguous.
  const bool per_column_momentum = momenta.size() == ncols;
  const bool general_contraction = momenta.size() == 1;
  if (!per_column_momentum && !general_contraction) {
    return err_config(std::format("BSE shell has {} angular momenta but {} coefficient columns",
                                  momenta.size(), ncols));
  }

  std::vector<ShellStamp> shells;
  shells.reserve(ncols);
  for (std::uint32_t col = 0; col < ncols; ++col) {
    const json &column = (*coef_it)[col];
    if (!column.is_array()) {
      return err_config("BSE coefficients column is not an array");
    }
    if (column.size() != exponents.size()) {
      return err_config(
          std::format("BSE coefficients column has {} entries but there are {} exponents",
                      column.size(), exponents.size()));
    }
    std::vector<double> coefficients;
    coefficients.reserve(column.size());
    for (const json &c_node : column) {
      coefficients.push_back(FBF_TRY(parse_decimal(c_node, "coefficients")));
    }

    const int l = per_column_momentum ? momenta[col] : momenta.front();
    auto harmonic = FBF_TRY(resolve_harmonic(block, l));
    // ShellStamp's constructor aborts on a bad shape (negative L, empty or
    // mismatched primitive lists), rather than returning a value this parser
    // could catch -- every condition it checks is already screened above
    // (non-empty angular_momentum entries, non-empty exponents, and the
    // column-length check just above), so by the time this constructs, the
    // shape is provably valid and the abort path is unreachable.
    shells.emplace_back(l, exponents, coefficients, harmonic);
  }
  return shells;
}

[[nodiscard]] Result<BasisStamp> parse_basis(std::string_view json_text) {
  auto parsed = FBF_TRY(parse_json(json_text));
  const json &document = parsed;

  if (!document.is_object()) {
    return err_config("BSE document is not a JSON object");
  }
  const auto elements_it = document.find("elements");
  if (elements_it == document.end()) {
    return err_config("BSE document has no 'elements'");
  }
  if (!elements_it->is_object()) {
    return err_config("BSE 'elements' is not an object");
  }

  std::map<int, std::vector<ShellStamp>> table;
  for (auto it = elements_it->begin(); it != elements_it->end(); ++it) {
    // The element key is the atomic number as a decimal string.
    const std::string &z_key = it.key();
    const json &element = it.value();
    int z = 0;
    const char *first = z_key.data();
    const char *last = z_key.data() + z_key.size();
    const auto [ptr, ec] = std::from_chars(first, last, z);
    if (ec != std::errc{} || ptr != last || z <= 0) {
      return err_config(std::format("BSE element key '{}' is not an atomic number", z_key));
    }

    if (!element.is_object()) {
      return err_config(std::format("BSE element '{}' is not an object", z_key));
    }
    const auto shells_it = element.find("electron_shells");
    if (shells_it == element.end()) {
      return err_config(std::format("BSE element '{}' has no electron_shells", z_key));
    }
    if (!shells_it->is_array()) {
      return err_config(std::format("BSE element '{}' electron_shells is not an array", z_key));
    }

    std::vector<ShellStamp> shells;
    for (const json &block : *shells_it) {
      auto block_shells = FBF_TRY(shells_from_block(block));
      for (auto &shell : block_shells) {
        shells.push_back(std::move(shell));
      }
    }
    table.emplace(z, std::move(shells));
  }

  return BasisStamp{std::move(table)};
}

[[nodiscard]] Result<BasisStamp> read_basis(const std::filesystem::path &path) {
  auto text = FBF_TRY(read_file(path));
  return parse_basis(text);
}

[[nodiscard]] Result<BasisSet> parse_and_assign_basis(std::string_view json_text,
                                                      const Molecule &molecule) {
  auto basis_stamp = FBF_TRY(parse_basis(json_text));
  return place(basis_stamp, molecule);
}

[[nodiscard]] Result<BasisSet> parse_and_assign_aux_basis(std::string_view json_text,
                                                          const Molecule &molecule) {
  // A fitting set is an ordinary BSE record placed onto the same molecule, so
  // this is parse_and_assign_basis with one difference: a coverage failure is
  // re-tagged as an *auxiliary* pairing error, so the message names the DF role
  // rather than reading as if the orbital basis were short an element. The
  // classification (ErrorKind::InvalidConfig) is preserved -- only the text is
  // enriched. Parse errors pass through untouched; they are not about pairing.
  auto basis_stamp = FBF_TRY(parse_basis(json_text));
  auto placed = place(basis_stamp, molecule);
  if (!placed) {
    return err_config(
        std::format("auxiliary (density-fitting) basis cannot be paired with the molecule: {}",
                    placed.error().message));
  }
  return placed;
}

} // namespace free_breakfast
