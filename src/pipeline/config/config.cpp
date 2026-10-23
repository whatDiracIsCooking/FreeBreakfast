/// @file
/// @brief Implementation of the TOML front end.

module free_breakfast.pipeline.config;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.io.toml;

#include "diagnostics.h"

namespace free_breakfast {

/// @brief Refuse a key of the `[step]` table not in @p keys: a misspelt option
/// is an error, not a default.
[[nodiscard]] Status check_keys(const toml::table &table, std::string_view step,
                                std::initializer_list<std::string_view> keys) {
  for (const auto &[key, value] : table) {
    if (std::ranges::find(keys, key.str()) == keys.end()) {
      return err_config("[{}] has no option '{}'", step, key.str());
    }
  }
  return {};
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
  } else if constexpr (std::same_as<T, std::string>) {
    // Verbatim: a string option is not necessarily a path, so anchoring it to
    // base_dir is the caller's decision rather than this function's.
    const auto value = node.value_exact<std::string>();
    if (!value) {
      return err_config("[{}] {} is not a string", step, key);
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

/// @brief Step `S` from its table. A filled-in step is read from an empty
/// one, so a required option it lacks is reported as for a written table.
template <class S>
[[nodiscard]] Result<S> read_step(const toml::table &table, const std::filesystem::path &base_dir) {
  constexpr std::string_view kStep = S::kName;
  S step;
  if constexpr (std::same_as<S, MoleculeStep>) {
    FBF_TRY(check_keys(table, kStep, {"xyz", "charge", "multiplicity"}));
    FBF_TRY(read_required_key(table, kStep, "xyz", base_dir, step.xyz));
    FBF_TRY(read_key(table, kStep, "charge", base_dir, step.charge));
    FBF_TRY(read_key(table, kStep, "multiplicity", base_dir, step.multiplicity));
  } else if constexpr (std::same_as<S, BasisSetStep>) {
    FBF_TRY(check_keys(table, kStep, {"basis"}));
    FBF_TRY(read_required_key(table, kStep, "basis", base_dir, step.basis));
    // 'basis' is a compiled-in set's name or a path to a BSE JSON file, and
    // only a path means anything relative to the config file. Joining a name
    // to base_dir would turn 'sto-3g' into a file nobody wrote.
    if (!is_builtin_basis(step.basis)) {
      step.basis = (base_dir / step.basis).string();
    }
  } else if constexpr (std::same_as<S, OrthogonalizerStep>) {
    FBF_TRY(check_keys(table, kStep, {"eigenvalue_floor"}));
    FBF_TRY(read_key(table, kStep, "eigenvalue_floor", base_dir, step.eigenvalue_floor));
  } else if constexpr (std::same_as<S, GuessStep>) {
    FBF_TRY(check_keys(table, kStep, {"type"}));
    std::string type = "sad";
    FBF_TRY(read_key(table, kStep, "type", base_dir, type));
    if (type != "sad") {
      return err_config("[{}] has no type '{}' (known: 'sad')", kStep, type);
    }
    step.type = GuessType::Sad;
  } else if constexpr (std::same_as<S, ScfStep>) {
    FBF_TRY(check_keys(table, kStep,
                       {"max_iterations", "energy_tolerance", "commutator_tolerance",
                        "diis_history", "fock_rebuild_interval"}));
    FBF_TRY(read_key(table, kStep, "max_iterations", base_dir, step.max_iterations));
    FBF_TRY(read_key(table, kStep, "energy_tolerance", base_dir, step.energy_tolerance));
    FBF_TRY(read_key(table, kStep, "commutator_tolerance", base_dir, step.commutator_tolerance));
    FBF_TRY(read_key(table, kStep, "diis_history", base_dir, step.diis_history));
    FBF_TRY(read_key(table, kStep, "fock_rebuild_interval", base_dir, step.fock_rebuild_interval));
  } else {
    // A step with options needs its own branch above, or every key in its
    // table would be refused.
    static_assert(std::is_empty_v<S>, "read_step has no branch for this step's options");
    FBF_TRY(check_keys(table, kStep, {}));
  }
  return step;
}

/// @brief The step whose `kName` is @p name, read from @p node.
[[nodiscard]] Result<Calculation::Step> read_named_step(std::string_view name,
                                                        const toml::node &node,
                                                        const std::filesystem::path &base_dir) {
  return [&]<size_t... Is>(std::index_sequence<Is...>) -> Result<Calculation::Step> {
    std::optional<Result<Calculation::Step>> out;
    const auto read = [&]<class S>() -> Result<Calculation::Step> {
      const toml::table *table = node.as_table();
      if (table == nullptr) {
        return err_config("{0} is not a table: write it as [{0}]", name);
      }
      return Calculation::Step{std::in_place_type<S>, FBF_TRY(read_step<S>(*table, base_dir))};
    };
    (void)((name == std::variant_alternative_t<Is, Calculation::Step>::kName
                ? (out = read.template
                         operator()<std::variant_alternative_t<Is, Calculation::Step>>(),
                   false)
                : true) &&
           ...);
    if (!out) {
      return err_config("unknown step [{}]", name);
    }
    return std::move(*out);
  }(std::make_index_sequence<std::variant_size_v<Calculation::Step>>{});
}

[[nodiscard]] Result<Calculation> parse_config(std::string_view toml_text,
                                               const std::filesystem::path &base_dir) {
  const toml::table doc = FBF_TRY(parse_toml(toml_text));
  std::vector<Calculation::Step> steps;
  for (const auto &[key, value] : doc) {
    steps.push_back(FBF_TRY(read_named_step(key.str(), value, base_dir)));
  }
  if (steps.empty()) {
    return err_config("the config names no step");
  }
  const toml::table empty;
  return Calculation::create(std::move(steps),
                             [&]<class S>() { return read_step<S>(empty, base_dir); });
}

[[nodiscard]] Result<Calculation> read_config(const std::filesystem::path &path) {
  const std::string text = FBF_TRY(read_file(path));
  return parse_config(text, path.parent_path());
}

} // namespace free_breakfast
