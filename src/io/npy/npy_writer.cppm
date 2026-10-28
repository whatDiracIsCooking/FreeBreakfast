/// @file
/// @brief The write half of the .npy format module: @ref NpyArray -> bytes.
///
/// Always emits format version 1.0 (the header this project writes never
/// outgrows its 16-bit length field), dtype `'<f8'`, the column-major payload
/// as is -- exactly what NumPy's own `np.save` emits for a float64
/// Fortran-order array, so the output loads with a plain `np.load`.

export module free_breakfast.io.npy:writer;

import std;
import free_breakfast.core.error_handling;
import :shared;

export namespace free_breakfast {

/// @brief The .npy header for a float64 array of @p shape: magic, version,
/// length field and the padded dict, ending where the payload begins (a
/// multiple of 64 bytes). npy_to_bytes is this followed by the payload, so a
/// writer that streams the payload in pieces emits the same file.
///
/// A shape whose element count overflows yields ErrorKind::InvalidConfig.
[[nodiscard]] Result<std::string> npy_header(std::span<const size_t> shape);

/// @brief Serialize @p array to .npy bytes.
///
/// A shape whose element count disagrees with `data.size()` (or overflows)
/// yields ErrorKind::InvalidConfig -- that is a malformed array, not a file
/// problem.
[[nodiscard]] Result<std::string> npy_to_bytes(const NpyArray &array);

/// @brief Write @p array to @p path as a .npy file.
[[nodiscard]] Status write_npy(const std::filesystem::path &path, const NpyArray &array);

} // namespace free_breakfast
