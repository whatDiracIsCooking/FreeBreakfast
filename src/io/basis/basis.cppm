/// @file
/// @brief Aggregated interface for the Basis Set Exchange (BSE) JSON reader.
///
/// `import free_breakfast.io.basis;` gives @ref free_breakfast::parse_basis /
/// @ref free_breakfast::read_basis (JSON -> geometry-free BasisStamp) and the
/// parse-then-place conveniences. A domain module: it says what a BSE document
/// means, over the format module @c free_breakfast.io.json.

export module free_breakfast.io.basis;

export import :reader;
