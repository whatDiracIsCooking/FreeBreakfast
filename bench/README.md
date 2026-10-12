# `bench/`

Wall-clock benchmarks that drive the real modules.

```sh
cmake --preset default -B build-bench -DFREE_BREAKFAST_BUILD_BENCH=ON
cmake --build build-bench
```

Opt-in, and with no preset of its own: the option rides on whichever preset
already matches the machine, into a build directory of its own so it never
reconfigures `build/`.

**A benchmark is not a test.** These are plain executables, not ctest entries —
a timing threshold that fails on a busy machine is noise, and a benchmark that
cries wolf gets ignored. Read the numbers; do not gate on them.

**Bound the container or the number is meaningless.** Two containers sharing a
box contend, and nothing in the output says so. Pin disjoint cores with
`CPUSET` (see `devtools/config.sh` and `.devcontainer/README.md`) before
comparing one run to another.
