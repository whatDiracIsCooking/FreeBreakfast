/// @file
/// @brief The read half of the JSON format module: text -> @ref json.
///
/// The one place `json::parse`'s exception surface is caught and turned into
/// the project's Result, so no domain parser needs its own try/catch.

export module free_breakfast.io.json:reader;

import std;
import free_breakfast.core.error_handling;
import :shared;

export namespace free_breakfast {

/// @brief Parse a JSON document from an in-memory string.
///
/// A syntactically invalid document yields ErrorKind::IO carrying the
/// backing parser's diagnostic. What the document *means* is the caller's
/// job -- a well-formed document of the wrong shape parses fine here.
[[nodiscard]] Result<json> parse_json(std::string_view json_text);

/// @brief Parse a JSON document from a file on disk.
///
/// A file that cannot be opened yields ErrorKind::IO; otherwise the contents
/// flow through parse_json.
[[nodiscard]] Result<json> read_json(const std::filesystem::path &path);

} // namespace free_breakfast
