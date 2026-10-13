/// @file
/// @brief Aggregated interface for the file-level I/O primitives.
///
/// `import free_breakfast.io.base;` gives the whole-file read and
/// write primitives every format module builds on: @ref read_file slurps a
/// file into a string, @ref write_file puts a string back on disk, and both
/// report failure through the project's Result/Status rather than a throw.

export module free_breakfast.io.base;

export import :reader;
export import :writer;
