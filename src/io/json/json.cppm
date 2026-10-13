/// @file
/// @brief Aggregated interface for the JSON format module.
///
/// `import free_breakfast.io.json;` gives the project's JSON
/// vocabulary type (@ref free_breakfast::json, plus the underlying
/// `free_breakfast::nlohmann` names) and the classified parse/serialize
/// entry points. Domain modules import this instead of the third-party
/// module, so the backing library is named in exactly one place.

export module free_breakfast.io.json;

export import :shared;
export import :reader;
export import :writer;
