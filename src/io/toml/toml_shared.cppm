/// @file
/// @brief The TOML vocabulary types, shared by the reader (and any future
/// writer).
///
/// Re-exports the third-party module, so one import of `io.toml` makes
/// `free_breakfast::toml::table`, `node_view` and the rest reachable for
/// domain code walking a parsed document.

export module free_breakfast.io.toml:shared;

export import free_breakfast.third_party.tomlplusplus;
