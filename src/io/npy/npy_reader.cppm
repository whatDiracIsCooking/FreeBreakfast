/// @file
/// @brief The read half of the .npy format module: bytes -> @ref NpyArray.
///
/// Accepts format versions 1.0 through 3.0 (they differ only in the header
/// length field), but only the dtype this project writes: `'<f8'`
/// (little-endian float64). Either layout reads: a Fortran-order payload is
/// taken as is, a C-order one is reordered into the column-major
/// @ref NpyArray. A differently-typed file is a classified error, not a
/// silent misinterpretation.

export module free_breakfast.io.npy:reader;

import std;
import free_breakfast.core.error_handling;
import :shared;

export namespace free_breakfast {

/// @brief Parse a .npy document from in-memory bytes.
///
/// A bad magic, a truncated header or payload, an unsupported dtype, or a
/// payload whose length disagrees with the header's shape
/// each yield ErrorKind::IO.
[[nodiscard]] Result<NpyArray> parse_npy(std::string_view npy_bytes);

/// @brief Read a .npy file from disk.
///
/// A file that cannot be opened yields ErrorKind::IO; otherwise the contents
/// flow through parse_npy.
[[nodiscard]] Result<NpyArray> read_npy(const std::filesystem::path &path);

} // namespace free_breakfast
