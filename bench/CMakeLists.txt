# ---------------------------------------------------------------------------
# bench/ -- wall-clock benchmarks over the real modules
# ---------------------------------------------------------------------------
# Configured only under -DFREE_BREAKFAST_BUILD_BENCH=ON, so a benchmark's build
# cost never lands on a normal configure. There is no `bench` preset: add the
# option on top of one that already matches the machine, e.g.
#
#   cmake --preset default -B build-bench -DFREE_BREAKFAST_BUILD_BENCH=ON
#
# A benchmark is an ordinary executable linking free_breakfast::* -- it is not a
# ctest entry, because a timing that fails a threshold on a busy machine is
# noise rather than a regression. Bound the container first (CPUSET/CPUS in
# devtools/config.sh) or the number means nothing; see .devcontainer/README.md.
#
# Deliberately empty rather than absent -- see experimental/CMakeLists.txt for
# why a missing directory is a configure error rather than a no-op.
