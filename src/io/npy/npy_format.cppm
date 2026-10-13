/// @file
/// @brief Format constants shared by the .npy reader and writer.
///
/// An internal partition (`module`, not `export module`): only this module's
/// implementation units import it, so nothing here reaches an importer of
/// free_breakfast.io.npy. See docs/style.md, "Module-private code".

module free_breakfast.io.npy:format;

import std;

namespace free_breakfast {

/// @brief The 6-byte magic that opens every .npy file.
inline constexpr std::string_view kMagic{"\x93NUMPY", 6};

} // namespace free_breakfast
