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
# Deliberately empty rather than absent. `add_subdirectory(experimental)` in the
# root CMakeLists is unconditional within its option guard, so a missing
# directory fails the configure outright with CMake's
# "add_subdirectory given source ... which is not an existing directory" --
# which is what the `experimental` preset did before this file existed.
