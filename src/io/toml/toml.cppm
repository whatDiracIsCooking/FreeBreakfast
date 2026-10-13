/// @file
/// @brief Aggregated interface for the TOML format module.
///
/// `import free_breakfast.io.toml;` gives the TOML vocabulary types
/// (`free_breakfast::toml::table` and friends) and the classified parse entry
/// points. Domain modules import this instead of the third-party module, so
/// the backing library is named in exactly one place. Read-only for now -- a
/// `:writer` partition joins when something needs to emit TOML.

export module free_breakfast.io.toml;

export import :shared;
export import :reader;
