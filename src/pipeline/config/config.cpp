/// @file
/// @brief Implementation of the TOML front end.

module free_breakfast.pipeline.config;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;
import free_breakfast.io.toml;

#include "diagnostics.h"

namespace free_breakfast {

/// @brief The `[step]` table of @p doc, refusing a key not in @p keys: a
/// misspelt option is an error, not a default.
[[nodiscard]] Result<const toml::table *> step_table(const toml::table &doc, std::string_view step,
                                                     std::initializer_list<std::string_view> keys) {
  const toml::table *table = doc[step].as_table();
  if (table == nullptr) {
    return err_config("the config needs a [{}] table", step);
  }
  for (const auto &[key, value] : *table) {
    if (std::ranges::find(keys, key.str()) == keys.end()) {
      return err_config("[{}] has no option '{}'", step, key.str());
    }
  }
  return table;
}

/// @brief Read @p key of the `[step]` table into @p out if present, leaving
/// the step's default otherwise. A path is resolved against @p base_dir.
template <class T>
[[nodiscard]] Status read_key(const toml::table &table, std::string_view step, std::string_view key,
                              const std::filesystem::path &base_dir, T &out) {
  const auto node = table[key];
  if (!node) {
    return {};
  }
  if constexpr (std::same_as<T, int32_t>) {
    const auto value = node.value_exact<int64_t>();
    if (!value || !std::in_range<int32_t>(*value)) {
      return err_config("[{}] {} is not a 32-bit integer", step, key);
    }
    out = static_cast<int32_t>(*value);
  } else if constexpr (std::same_as<T, double>) {
    // value, not value_exact: an integer node is accepted, so 1 reads as 1.0.
    const auto value = node.value<double>();
    if (!value) {
      return err_config("[{}] {} is not a number", step, key);
    }
    out = *value;
  } else {
    static_assert(std::same_as<T, std::filesystem::path>);
    const auto value = node.value_exact<std::string>();
    if (!value) {
      return err_config("[{}] {} is not a string", step, key);
    }
    out = base_dir / *value; // an absolute *value replaces base_dir
  }
  return {};
}

/// @brief As read_key, for a key the step has no default for.
template <class T>
[[nodiscard]] Status read_required_key(const toml::table &table, std::string_view step,
                                       std::string_view key, const std::filesystem::path &base_dir,
                                       T &out) {
  if (!table.contains(key)) {
    return err_config("[{}] needs '{}'", step, key);
  }
  return read_key(table, step, key, base_dir, out);
}

[[nodiscard]] Result<ConfigSteps> parse_config(std::string_view toml_text,
                                               const std::filesystem::path &base_dir) {
  const toml::table doc = FBF_TRY(parse_toml(toml_text));
  for (const auto &[key, value] : doc) {
    if (key.str() != MoleculeStep::kName && key.str() != BasisSetStep::kName &&
        key.str() != OrthogonalizerStep::kName) {
      return err_config("unknown step [{}]", key.str());
    }
  }

  MoleculeStep molecule;
  constexpr std::string_view kMolecule = MoleculeStep::kName;
  const toml::table &m = *FBF_TRY(step_table(doc, kMolecule, {"xyz", "charge", "multiplicity"}));
  FBF_TRY(read_required_key(m, kMolecule, "xyz", base_dir, molecule.xyz));
  FBF_TRY(read_key(m, kMolecule, "charge", base_dir, molecule.charge));
  FBF_TRY(read_key(m, kMolecule, "multiplicity", base_dir, molecule.multiplicity));

  BasisSetStep basis_set;
  constexpr std::string_view kBasisSet = BasisSetStep::kName;
  const toml::table &b = *FBF_TRY(step_table(doc, kBasisSet, {"basis"}));
  FBF_TRY(read_required_key(b, kBasisSet, "basis", base_dir, basis_set.basis));

  // Optional: its one option has a default.
  OrthogonalizerStep orthogonalizer;
  constexpr std::string_view kOrthogonalizer = OrthogonalizerStep::kName;
  if (doc.contains(kOrthogonalizer)) {
    const toml::table &o = *FBF_TRY(step_table(doc, kOrthogonalizer, {"eigenvalue_floor"}));
    FBF_TRY(read_key(o, kOrthogonalizer, "eigenvalue_floor", base_dir,
                     orthogonalizer.eigenvalue_floor));
  }

  return ConfigSteps{std::move(molecule), std::move(basis_set), std::move(orthogonalizer)};
}

[[nodiscard]] Result<ConfigSteps> read_config(const std::filesystem::path &path) {
  const std::string text = FBF_TRY(read_file(path));
  return parse_config(text, path.parent_path());
}

} // namespace free_breakfast
