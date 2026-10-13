/// @file
/// @brief The write half of the JSON format module: @ref json -> text.
///
/// Serialization itself cannot fail (nlohmann's dump round-trips doubles at
/// full precision), so the string half returns the string; only the on-disk
/// half carries a Status.

export module free_breakfast.io.json:writer;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import :shared;

export namespace free_breakfast {

/// @brief Serialize @p document to text, pretty-printed with @p indent spaces
/// (the project convention; pass a negative indent for the compact form).
[[nodiscard]] std::string json_to_string(const json &document, int32_t indent = 2);

/// @brief Write @p document to @p path as pretty-printed JSON.
[[nodiscard]] Status write_json(const std::filesystem::path &path, const json &document,
                                int32_t indent = 2);

} // namespace free_breakfast
