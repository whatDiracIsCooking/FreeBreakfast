/// @file
/// @brief The read half of the TOML format module: text -> toml::table.
///
/// The one place toml++'s exception surface is caught and turned into the
/// project's Result, so no domain parser needs its own try/catch.

export module free_breakfast.io.toml:reader;

import std;
import free_breakfast.core.error_handling;
import :shared;

export namespace free_breakfast {

/// @brief Parse a TOML document from an in-memory string.
///
/// A syntactically invalid document yields ErrorKind::IO naming the line and
/// column. What the document *means* is the caller's job -- a well-formed
/// document of the wrong shape parses fine here.
[[nodiscard]] Result<toml::table> parse_toml(std::string_view toml_text);

/// @brief Parse a TOML document from a file on disk.
///
/// A file that cannot be opened yields ErrorKind::IO; otherwise the contents
/// flow through parse_toml.
[[nodiscard]] Result<toml::table> read_toml(const std::filesystem::path &path);

} // namespace free_breakfast
