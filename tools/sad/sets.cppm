/// @file
/// @brief free_breakfast.tools.sad: which basis sets `free-breakfast-sad-stamps
///        --all` regenerates.
///
/// A module library rather than a function in main.cpp so that a host test
/// (test/io/sad_host.cpp) can pin it: a file name is not always a spelling
/// @c resolve_basis folds to (`6-31gstar.json` is `6-31G*`), and an `--all`
/// that passed file names refused the two Pople star sets (#1446).

export module free_breakfast.tools.sad;

import std;
import free_breakfast.core.error_handling;

export namespace free_breakfast {

/// @brief The BSE `name` of every `*.json` document in @p basis_dir, in file
/// name order: the sets `--all` regenerates, each spelt as the document calls
/// itself, which is the name @c resolve_basis carries it under.
///
/// A directory that cannot be listed or a document that cannot be read is
/// ErrorKind::IO; a document with no `name` string is ErrorKind::InvalidConfig.
[[nodiscard]] Result<std::vector<std::string>>
sad_stamp_sets(const std::filesystem::path &basis_dir);

} // namespace free_breakfast
