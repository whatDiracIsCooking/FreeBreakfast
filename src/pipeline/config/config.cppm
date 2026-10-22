/// @file
/// @brief The TOML front end: a config file read into the steps it names.
///
/// One table per step, named by the step's `kName`; its keys are the step's
/// options, spelled as its members. Until `Pipeline<Steps...>` and its
/// `StepVariant` land, the steps come back as a fixed tuple; then this
/// returns the step vector instead. Re-exports the steps it returns.

export module free_breakfast.pipeline.config;

import std;
import free_breakfast.core.error_handling;
export import free_breakfast.pipeline.molecule;
export import free_breakfast.pipeline.basis_set;
export import free_breakfast.pipeline.orthogonalizer;

export namespace free_breakfast {

/// @brief The steps a config file names, in pipeline order.
using ConfigSteps = std::tuple<MoleculeStep, BasisSetStep, OrthogonalizerStep>;

/// @brief Parse a config from an in-memory TOML document.
///
/// A relative path in it is resolved against @p base_dir (empty: left
/// relative to the working directory). A step whose options all have defaults
/// may leave its table out. A syntactically invalid document is ErrorKind::IO.
/// A missing step table or required key, an unknown table or key, or a value
/// of the wrong type is ErrorKind::InvalidConfig naming it.
[[nodiscard]] Result<ConfigSteps> parse_config(std::string_view toml_text,
                                               const std::filesystem::path &base_dir);

/// @brief Parse a config file, resolving its relative paths against the
/// file's own directory. A file that cannot be opened is ErrorKind::IO.
[[nodiscard]] Result<ConfigSteps> read_config(const std::filesystem::path &path);

} // namespace free_breakfast
