/// @file
/// @brief The write half of the I/O primitives: put a string on disk.
///
/// The mirror of @ref read_file: writers serialize to a string in memory
/// (`x_to_json(...)` and friends) and then hand the finished document to
/// @ref write_file. This partition is the one copy of that last step.

export module free_breakfast.io.base:writer;

import std;
import free_breakfast.core.error_handling;

export namespace free_breakfast {

/// @brief Write @p contents to @p path, replacing whatever was there.
///
/// The file is opened in binary mode, so the bytes land exactly as given. A
/// file that cannot be opened for writing, or a stream that goes bad while
/// writing, yields ErrorKind::IO naming the path.
[[nodiscard]] Status write_file(const std::filesystem::path &path, std::string_view contents);

} // namespace free_breakfast
