/// @file
/// @brief The write half of the .npy format module: @ref NpyArray -> bytes.
///
/// Always emits format version 1.0 (the header this project writes never
/// outgrows its 16-bit length field), dtype `'<f8'`, C order -- exactly what
/// NumPy's own `np.save` emits for a float64 C-order array, so the output
/// loads with a plain `np.load`.

export module free_breakfast.io.npy:writer;

import std;
import free_breakfast.core.error_handling;
import :shared;

export namespace free_breakfast {

/// @brief Serialize @p array to .npy bytes.
///
/// A shape whose element count disagrees with `data.size()` (or overflows)
/// yields ErrorKind::InvalidConfig -- that is a malformed array, not a file
/// problem.
[[nodiscard]] Result<std::string> npy_to_bytes(const NpyArray &array);

/// @brief Write @p array to @p path as a .npy file.
[[nodiscard]] Status write_npy(const std::filesystem::path &path, const NpyArray &array);

} // namespace free_breakfast
