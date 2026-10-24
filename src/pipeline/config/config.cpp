/// @file
/// @brief Implementation of the TOML front end.

module free_breakfast.pipeline.config;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.io.toml;
import free_breakfast.pipeline.step;

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
  } else if constexpr (std::same_as<T, std::vector<int32_t>>) {
    const toml::array *array = node.as_array();
    if (array == nullptr) {
      return err_config("[{}] {} is not an array of integers", step, key);
    }
    std::vector<int32_t> values;
    for (const toml::node &element : *array) {
      const auto value = element.value_exact<int64_t>();
      if (!value || !std::in_range<int32_t>(*value)) {
        return err_config("[{}] {} is not an array of 32-bit integers", step, key);
      }
      values.push_back(static_cast<int32_t>(*value));
    }
    out = std::move(values);
  } else if constexpr (std::same_as<T, std::vector<double>>) {
    const toml::array *array = node.as_array();
    if (array == nullptr) {
      return err_config("[{}] {} is not an array of numbers", step, key);
    }
    std::vector<double> values;
    for (const toml::node &element : *array) {
      const auto value = element.value<double>();
      if (!value) {
        return err_config("[{}] {} is not an array of numbers", step, key);
      }
      values.push_back(*value);
    }
    out = std::move(values);
  } else if constexpr (std::same_as<T, std::vector<std::string>>) {
    const toml::array *array = node.as_array();
    if (array == nullptr) {
      return err_config("[{}] {} is not an array of strings", step, key);
    }
    std::vector<std::string> values;
    for (const toml::node &element : *array) {
      const auto value = element.value_exact<std::string>();
      if (!value) {
        return err_config("[{}] {} is not an array of strings", step, key);
      }
      values.push_back(*value);
    }
    out = std::move(values);
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

/// @brief The active-space selection's keys of the `[step]` table, the same
/// in every step that takes one: `ncas` and `nelecas` required, with `active`;
/// or `avas` with `avas_threshold` in their place. The ranges need the
/// molecule and the basis, so ActiveSpaceSelection::orbital_order checks them
/// when the step runs.
[[nodiscard]] Status read_active_space(const toml::table &table, std::string_view step,
                                       const std::filesystem::path &base_dir,
                                       ActiveSpaceSelection &out) {
  if (table.contains("avas")) {
    // AVAS picks the active space, so nothing may name one beside it.
    for (std::string_view key : {"ncas", "nelecas", "active"}) {
      if (table.contains(key)) {
        return err_config("[{}] avas picks the active space, so takes no '{}'", step, key);
      }
    }
    FBF_TRY(read_key(table, step, "avas", base_dir, out.avas));
    if (out.avas.empty()) {
      return err_config("[{}] avas names no AO label", step);
    }
    FBF_TRY(read_key(table, step, "avas_threshold", base_dir, out.avas_threshold));
  } else if (table.contains("avas_threshold")) {
    return err_config("[{}] avas_threshold needs 'avas'", step);
  } else {
    FBF_TRY(read_required_key(table, step, "ncas", base_dir, out.ncas));
    FBF_TRY(read_required_key(table, step, "nelecas", base_dir, out.nelecas));
  }
  return read_key(table, step, "active", base_dir, out.active);
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
                        "diis_history", "fock_rebuild_interval", "molden"}));
    FBF_TRY(read_key(table, kStep, "max_iterations", base_dir, step.max_iterations));
    FBF_TRY(read_key(table, kStep, "energy_tolerance", base_dir, step.energy_tolerance));
    FBF_TRY(read_key(table, kStep, "commutator_tolerance", base_dir, step.commutator_tolerance));
    FBF_TRY(read_key(table, kStep, "diis_history", base_dir, step.diis_history));
    FBF_TRY(read_key(table, kStep, "fock_rebuild_interval", base_dir, step.fock_rebuild_interval));
    FBF_TRY(read_key(table, kStep, "molden", base_dir, step.molden));
  } else if constexpr (std::same_as<S, LocalizationStep>) {
    FBF_TRY(check_keys(table, kStep,
                       {"mode", "max_sweeps", "functional_tolerance", "rotation_gain_threshold",
                        "exponent", "molden"}));
    std::string mode = "pipek_mezey";
    FBF_TRY(read_key(table, kStep, "mode", base_dir, mode));
    if (mode != "pipek_mezey" && mode != "ibo") {
      return err_config("[{}] has no mode '{}' (known: 'pipek_mezey', 'ibo')", kStep, mode);
    }
    step.mode = mode == "ibo" ? LocalizationMode::Ibo : LocalizationMode::PipekMezey;
    // The sweep's options sit in the table itself; IBO is the same sweep over
    // other populations, at Knizia's exponent four unless the table says.
    PipekMezeyOptions &options = step.pipek_mezey;
    if (step.mode == LocalizationMode::Ibo) {
      options.exponent = PipekMezeyExponent::Four;
    }
    FBF_TRY(read_key(table, kStep, "max_sweeps", base_dir, options.max_sweeps));
    FBF_TRY(read_key(table, kStep, "functional_tolerance", base_dir, options.functional_tolerance));
    FBF_TRY(read_key(table, kStep, "rotation_gain_threshold", base_dir,
                     options.rotation_gain_threshold));
    auto exponent = static_cast<int32_t>(options.exponent);
    FBF_TRY(read_key(table, kStep, "exponent", base_dir, exponent));
    if (exponent != 2 && exponent != 4) {
      return err_config("[{}] exponent is {}, not 2 or 4", kStep, exponent);
    }
    options.exponent = static_cast<PipekMezeyExponent>(exponent);
    FBF_TRY(read_key(table, kStep, "molden", base_dir, step.molden));
  } else if constexpr (std::same_as<S, CasciStep>) {
    FBF_TRY(check_keys(
        table, kStep, {"ncas", "nelecas", "nroots", "active", "avas", "avas_threshold", "molden"}));
    FBF_TRY(read_active_space(table, kStep, base_dir, step));
    FBF_TRY(read_key(table, kStep, "nroots", base_dir, step.nroots));
    FBF_TRY(read_key(table, kStep, "molden", base_dir, step.molden));
  } else if constexpr (std::same_as<S, CasscfStep>) {
    FBF_TRY(check_keys(table, kStep,
                       {"ncas", "nelecas", "nroots", "active", "avas", "avas_threshold", "weights",
                        "conv_tol", "conv_tol_grad", "max_cycle", "molden"}));
    FBF_TRY(read_active_space(table, kStep, base_dir, step));
    FBF_TRY(read_key(table, kStep, "nroots", base_dir, step.nroots));
    FBF_TRY(read_key(table, kStep, "weights", base_dir, step.weights));
    FBF_TRY(read_key(table, kStep, "conv_tol", base_dir, step.conv_tol));
    FBF_TRY(read_key(table, kStep, "conv_tol_grad", base_dir, step.conv_tol_grad));
    FBF_TRY(read_key(table, kStep, "max_cycle", base_dir, step.max_cycle));
    FBF_TRY(read_key(table, kStep, "molden", base_dir, step.molden));
    // Unlike the active space's ranges these need nothing of the molecule, so
    // they are refused here rather than when casscf_solve runs.
    if (!step.weights.empty()) {
      if (std::cmp_not_equal(step.weights.size(), step.nroots)) {
        return err_config("[{}] weights has {} entries, not nroots {}", kStep, step.weights.size(),
                          step.nroots);
      }
      if (std::ranges::any_of(step.weights, [](double w) { return !(w >= 0.0); })) {
        return err_config("[{}] weights has a negative entry", kStep);
      }
      // casscf_solve's own tolerance.
      if (const double sum = std::ranges::fold_left(step.weights, 0.0, std::plus<>{});
          std::abs(sum - 1.0) > 1e-12) {
        return err_config("[{}] weights sum to {}, not 1", kStep, sum);
      }
    }
    if (!(step.conv_tol > 0.0) || !(step.conv_tol_grad > 0.0)) {
      return err_config("[{}] conv_tol and conv_tol_grad must be positive", kStep);
    }
    if (step.max_cycle < 1) {
      return err_config("[{}] max_cycle is {}, below 1", kStep, step.max_cycle);
    }
  } else if constexpr (std::same_as<S, FciStep>) {
    // Whether the space fits is the device's to say, so FciStep::run checks
    // it when the step runs.
    FBF_TRY(check_keys(table, kStep, {"nroots"}));
    FBF_TRY(read_key(table, kStep, "nroots", base_dir, step.nroots));
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
      if (!table->contains("from")) {
        return Calculation::Step{std::in_place_type<S>, FBF_TRY(read_step<S>(*table, base_dir))};
      }
      // 'from' loads the product instead, and the step's options configure a
      // run that never happens, so none may stand beside it.
      if constexpr (LoadableStep<S>) {
        for (const auto &[key, value] : *table) {
          if (key.str() != "from") {
            return err_config("[{}] loads 'from' a file, so takes no option '{}'", name, key.str());
          }
        }
        Load<S> load;
        FBF_TRY(read_key(*table, S::kName, "from", base_dir, load.from));
        return Calculation::Step{std::move(load)};
      } else {
        return err_config("[{}] cannot be loaded 'from' a file", name);
      }
    };
    // The Load<S> alternatives have no kName: only the steps are named.
    const auto match = [&]<class X>() {
      if constexpr (requires { X::kName; }) {
        if (name == X::kName) {
          out = read.template operator()<X>();
        }
      }
      return !out;
    };
    (void)(match.template operator()<std::variant_alternative_t<Is, Calculation::Step>>() && ...);
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
