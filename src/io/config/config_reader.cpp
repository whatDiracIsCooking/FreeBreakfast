/// @file
/// @brief Implementation of the TOML config reader.

module free_breakfast.io.config;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;
import free_breakfast.io.toml;

#include "diagnostics.h"

namespace free_breakfast {

namespace {

/// @brief Pull an integer keyed by @p key out of a TOML table, honouring
/// @p fallback when the key is absent and reporting a type mismatch as an
/// InvalidConfig error.
[[nodiscard]] Result<int> table_int(const free_breakfast::toml::table &table, std::string_view key,
                                    int fallback) {
  const auto node = table[key];
  if (!node) {
    return fallback;
  }
  const auto value = node.value<std::int64_t>();
  if (!value) {
    return err_config(std::format("config key '{}' is not an integer", key));
  }
  return static_cast<int>(*value);
}

/// @brief As table_int, for a floating-point value.
[[nodiscard]] Result<double> table_double(const free_breakfast::toml::table &table,
                                          std::string_view key, double fallback) {
  const auto node = table[key];
  if (!node) {
    return fallback;
  }
  const auto value = node.value<double>();
  if (!value) {
    return err_config(std::format("config key '{}' is not a number", key));
  }
  return *value;
}

/// @brief As table_int, for a string value.
[[nodiscard]] Result<std::string> table_string(const free_breakfast::toml::table &table,
                                               std::string_view key, std::string fallback) {
  const auto node = table[key];
  if (!node) {
    return fallback;
  }
  const auto value = node.value<std::string>();
  if (!value) {
    return err_config(std::format("config key '{}' is not a string", key));
  }
  return *value;
}

} // namespace

[[nodiscard]] Result<Config> parse_config(std::string_view toml_text) {
  auto parsed = FBF_TRY(parse_toml(toml_text));
  const toml::table &table = parsed;

  Config config{};

  config.method = FBF_TRY(table_string(table, "method", config.method));
  config.basis = FBF_TRY(table_string(table, "basis", config.basis));
  config.aux_basis = FBF_TRY(table_string(table, "aux_basis", config.aux_basis));
  config.charge = FBF_TRY(table_int(table, "charge", config.charge));
  config.multiplicity = FBF_TRY(table_int(table, "multiplicity", config.multiplicity));
  config.scf_energy_tolerance =
      FBF_TRY(table_double(table, "scf_energy_tolerance", config.scf_energy_tolerance));
  config.scf_density_tolerance =
      FBF_TRY(table_double(table, "scf_density_tolerance", config.scf_density_tolerance));
  config.max_iterations = FBF_TRY(table_int(table, "max_iterations", config.max_iterations));

  return config;
}

[[nodiscard]] Result<Config> read_config(const std::filesystem::path &path) {
  auto text = FBF_TRY(read_file(path));
  return parse_config(text);
}

} // namespace free_breakfast
