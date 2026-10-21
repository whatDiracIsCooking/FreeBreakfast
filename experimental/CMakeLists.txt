# ---------------------------------------------------------------------------
# experimental/ -- opt-in scratch space
# ---------------------------------------------------------------------------
# Configured only under -DFREE_BREAKFAST_BUILD_EXPERIMENTAL=ON (the
# `experimental` preset), so nothing in here can slow or break a normal build.
#
# Add targets with the same three macros the rest of the tree uses, and
# add_subdirectory() them below. Registering a ctest entry works here too: the
# root CMakeLists calls enable_testing() before any add_subdirectory, so an
# add_test() in this directory is recorded rather than silently dropped.
#
# Keep this file even if every subdirectory below is promoted or deleted.
# `add_subdirectory(experimental)` in the root CMakeLists is unconditional
# within its option guard, so a missing directory fails the configure outright
# with CMake's "add_subdirectory given source ... which is not an existing
# directory" -- which is what the `experimental` preset did before this file
# existed.
#
# A GPU test of something here goes in test/experimental, not in this tree: the
# root CMakeLists adds that directory under this same option and also behind
# FREE_BREAKFAST_COMPILE_TIME_ONLY, which skips everything that needs
# GoogleTest.

# Stable LSD radix sort of double keys over core/kernel's block_reduce. Its
# GPU test is test/experimental, which the root CMakeLists adds under this
# same option.
add_subdirectory(sort/radix)

# Stable merge sort of double keys, radix_sort's contract and order. Its
# benchmark times it against radix_sort, so it comes after sort/radix.
add_subdirectory(sort/merge)

# The spherically averaged atomic SCF a production SAD guess is made of, on the
# host. Its test is test/experimental's host binary.
add_subdirectory(atomic_scf)
