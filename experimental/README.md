# `experimental/`

Opt-in scratch space: spikes, throwaway benchmarks of an idea, a module you are
not ready to put under `src/` yet.

```sh
devtools/cpp-tier.sh --preset experimental     # own build dir, build-experimental/
```

Nothing here is built by a normal configure — the directory is guarded by
`FREE_BREAKFAST_BUILD_EXPERIMENTAL`, which only the `experimental` preset sets.
That is the point: code in here is free to be broken, slow, or half-finished
without costing anyone else a build.

Two things still apply:

- **It is compiled, so it must compile.** A preset nobody runs rots; this one is
  cheap to run, so run it before you leave something in here.
- **ctest entries work.** `enable_testing()` is called before any
  `add_subdirectory` in the root `CMakeLists.txt`, so an `add_test()` here is
  recorded normally. GoogleTest cases go in `test/experimental/`. The root
  `CMakeLists.txt` adds that directory under this same option, and skips it
  under `FREE_BREAKFAST_COMPILE_TIME_ONLY`, which has no GoogleTest. It builds
  its own two binaries, `free_breakfast_tests_experimental_gpu` and
  `free_breakfast_tests_experimental_host`.

## What is here

| Directory | What |
|-----------|------|
| [`sort/radix/`](sort/radix/README.md) | Stable LSD radix sort of `double` keys (with optional `uint32_t` values), in IEEE-754 total order, over `core/kernel`'s `block_reduce` |
| [`sort/merge/`](sort/merge/README.md) | Stable merge sort with `radix_sort`'s contract and order; faster than it and than CUB's `DeviceMergeSort` at every measured size, and than CUB's radix sort below 2^20 keys |
| [`atomic_scf/`](atomic_scf/README.md) | The spherically averaged atomic SCF behind a production SAD guess, on the host, over closed-form one-center integrals; matches PySCF's `AtomSphAverageRHF` |

Promote something by moving it under `src/` (its tests into `test/gtest/`),
wiring it into `src/CMakeLists.txt`, and deleting it from here.
