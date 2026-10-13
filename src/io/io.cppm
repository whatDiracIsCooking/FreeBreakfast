/// @file
/// @brief Convenience module that re-exports the file-level I/O layer.
///
/// `import free_breakfast.io;` gives the whole-file primitives, the format
/// modules (JSON, TOML, NumPy .npy) and the TOML config reader. The
/// sub-modules remain importable individually.

export module free_breakfast.io;

export import free_breakfast.io.base;
export import free_breakfast.io.config;
export import free_breakfast.io.json;
export import free_breakfast.io.npy;
export import free_breakfast.io.toml;
