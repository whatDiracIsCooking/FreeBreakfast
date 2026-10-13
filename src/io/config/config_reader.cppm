/// @file
/// @brief Reading of a calculation config: a TOML document -> @ref Config.
///
/// A parser returning through the project's Result<T> so a malformed file
/// becomes a classified Error rather than a crash or a thrown exception
/// escaping to the caller. The TOML document describes the method, basis set,
/// charge, multiplicity and SCF controls.

export module free_breakfast.io.config:reader;

import std;
import free_breakfast.core.error_handling;
import :shared;

export namespace free_breakfast {

/// @brief Parse a calculation config from an in-memory TOML document.
///
/// A syntactically invalid document yields ErrorKind::IO; a document that
/// parses but has a key of the wrong type (a string where a number is wanted,
/// say) yields ErrorKind::InvalidConfig. Missing keys are not an error -- the
/// corresponding Config default stands.
[[nodiscard]] Result<Config> parse_config(std::string_view toml_text);

/// @brief Parse a config from a TOML file on disk.
///
/// A file that cannot be opened yields ErrorKind::IO; otherwise the contents
/// flow through parse_config and carry its classification.
[[nodiscard]] Result<Config> read_config(const std::filesystem::path &path);

} // namespace free_breakfast
