/// @file
/// @brief The JSON vocabulary type, shared by the reader and writer.
///
/// Re-exports the third-party module (so `free_breakfast::nlohmann::json`
/// stays reachable through one import) and blesses @ref free_breakfast::json
/// as the canonical spelling for domain code.

export module free_breakfast.io.json:shared;

export import free_breakfast.third_party.nlohmann;

export namespace free_breakfast {

/// @brief The project's JSON document type.
using nlohmann::json;

} // namespace free_breakfast
