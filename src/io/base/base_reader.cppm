/// @file
/// @brief The read half of the I/O primitives: slurp a whole file.
///
/// Every reader in the I/O layer follows the same two-step shape -- a
/// `parse_x(std::string_view)` that works on an in-memory document and a
/// `read_x(path)` that is @ref read_file piped into it. This partition is the
/// one copy of that first step.

export module free_breakfast.io.base:reader;

import std;
import free_breakfast.core.error_handling;

export namespace free_breakfast {

/// @brief Slurp a file into a string, or an IO error if it will not open.
///
/// The file is opened in binary mode, so the bytes come back exactly as they
/// are on disk -- no newline translation. A file that cannot be opened, or a
/// stream that goes bad mid-read, yields ErrorKind::IO naming the path.
[[nodiscard]] Result<std::string> read_file(const std::filesystem::path &path);

} // namespace free_breakfast
