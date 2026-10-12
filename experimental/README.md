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
  recorded normally.

Promote something by moving it under `src/`, wiring it into
`src/CMakeLists.txt`, and deleting it from here.
