# `experimental/`

Opt-in scratch space: spikes, throwaway benchmarks of an idea, a module you are
not ready to put under `src/` yet.

```sh
devtools/test.py pr --only cpp --preset experimental     # own build dir, build-experimental/
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
  its own binary, `free_breakfast_tests_experimental_gpu`.

## What is here

| Directory | What |
|-----------|------|
| [`sort/radix/`](sort/radix/README.md) | Stable LSD radix sort of `double` keys (with optional `uint32_t` values), in IEEE-754 total order, over `core/kernel`'s `block_reduce` |
| [`sort/merge/`](sort/merge/README.md) | Stable merge sort with `radix_sort`'s contract and order; faster than it and than CUB's `DeviceMergeSort` at every measured size, and than CUB's radix sort below 2^20 keys |
| [`fp16_gemm/`](fp16_gemm/README.md) | `__half` GEMM through cuBLAS / hipBLAS (`Hgemm`, and `GemmEx` with fp32 accumulation): does each backend support it, how accurately, and how fast against `Sgemm` |
| [`output_ring/`](output_ring/README.md) | Device arrays streamed into `.npy` files through a fixed ring of pinned staging slots: copies on the device stream, writes on a taskflow executor, several files at once |
| [`blas_lt/`](blas_lt/README.md) | fp64 GEMM through the cuBLASLt / hipBLASLt heuristic, on the shapes `nevpt2::rdms`' digest runs: does the Lt surface beat plain `Dgemm` there, and does hipBLASLt offer an fp64 algorithm at all |
| [`compress_rdm3/`](compress_rdm3/README.md) | A complete SC-NEVPT2 case with no symmetry, C1 glycolaldehyde DF-CASSCF(10,10), dm3/f3ca/f3ac included: data and its generator for 3-RDM compression experiments; nothing built |
| [`split_gemm/`](split_gemm/README.md) | fp64 GEMM composed from vendor `Sgemm` calls on a hi/lo fp32 split, on the same digest shapes: each card's own fp64 ceiling, and the speed-against-accuracy curve of spending fp32 throughput to buy fp64 accuracy |
| [`gemm_host_offload/`](gemm_host_offload/README.md) | The DF exchange build K = X^T X cut into batches, over m slabs or aux blocks, with B resident or streamed from pinned memory through a ring of device slots: the per-batch cost delta on top of the one-call build, and whether it is launch overhead, event waits or the GEMMs' tail |
| [`hip_comp/`](hip_comp/README.md) | GPU lossless compression on HIP: hipCOMP's batched LZ4 and Cascaded over device arrays, round-tripped bit-exact on the gfx1200. Cascaded packs sorted int64 indices 8.9× at 71 GB/s compress; LZ4 gets 5.5× on 90%-zero fp64 but compresses at only 1.4–7.5 GB/s; random fp64 stays at 1.00×. HIP only, and only where hipCOMP is installed (not in the image; README has the build) |
| [`routine/`](routine/README.md) | A check-before-invoke CRTP base (`Routine<Self, Alloc>`, one designated-initializer `In`, a shape-only `Shape` for `footprint`) and a compile-time allocation key (`AllocKey`, minted only by a root in the key's module), spiked on a port of `mcscf::root_rotation`. The port matches the original bit for bit on the gfx1200. `footprint(shape)`, built from sizes alone, is the pool's peak exactly. A death test covers `check`'s aborts, and five must-not-compile ctest entries pin the diagnostics. Verdict (#1324): promote the base, `In`, `Shape` and `check`; drop the key, because the one unforgeable friend placement is a module cycle in `src/`'s layering |
| [`rust/gemm/`](rust/gemm/README.md) | fp64 GEMM driven from Rust over WarpWraps' C ABI (the `wwr-sys` crate), with no C++ of this tree's own: a stream, device buffers and `dgemm` in safe, lifetime-checked types, with a demo checked against a host loop. Cargo is run by hand, not by CMake |
| [`rust/s_overlap/`](rust/s_overlap/README.md) | The spherical AO overlap S of a molecule in a basis, from Rust: a C ABI over the pipeline's setup and overlap steps, copying S into a `rust/gemm` buffer, checked against PySCF to ~1e-15. Also documents that free_breakfast never renormalizes a contraction, so diag(S) is 1 only to the published coefficients' digits (up to 1.3e-6 off in cc-pVTZ) |
| [`rust/rebuild/`](rust/rebuild/README.md) | free_breakfast's `src/` rebuilt in Rust one layer at a time over `wwr-sys`, standing alone from the other Rust crates. So far wwr's `extension/common` (error codes, policies, `gpu_check`, `ScopedDeviceIndex`), `extension/handle` (owning handles bound to a device or a stream, views that borrow them, the device-handle ladder) and `extension/runtime` (streams, events, memory pools, graphs, the kit device handle), with two `contracts` preconditions. Cargo by hand |
| [`rust/rayon_npy/`](rust/rayon_npy/README.md) | One column-major float64 `.npy` written by many rayon threads at once, each `pwrite`-ing its own columns at their fixed offsets; byte-identical to `np.save`. Pure Rust, no GPU, cargo by hand. Threads do not speed up the write itself (it stays ~1.7 GB/s on ext4); they speed up computing the columns |
| [`rust/rayon_npy_mmap/`](rust/rayon_npy_mmap/README.md) | The same file filled in place through a `MAP_SHARED` mapping, each rayon task computing into its own disjoint `&mut [f64]` block of the file's pages. Scales where `pwrite` does not: 3.6 GB/s on 8 threads, 3× its own single thread and 1.9× `rayon_npy`'s best; the per-file page-cache insert is the ceiling |
| [`rust/rayon_npy_direct/`](rust/rayon_npy_direct/README.md) | The same file with `O_DIRECT`: tasks own 4 KiB-aligned stripes of the *file* (flat element ranges, header in stripe 0), filled into page-aligned buffers; preallocated and pre-extended so no write allocates or extends. Device-bound on this box's shared SSD (~0.08 GB/s, `dd`'s rate); through `fdatasync` all three writers tie there |
| [`cope_rearrangement/`](cope_rearrangement/README.md) | A study of the chair Cope rearrangement of 1,5-hexadiene (CASSCF(6,6) vs NEVPT2). Steps 1 and 2 are PySCF/block2 references: done, and partial. Step 4 runs on free-breakfast's own DF-PC-NEVPT2: 4a, its energies against block2's at all 23 scan points, is done (4e-10 on the same orbitals and CI); 4b is done: PC-NEVPT2's one C2h stationary point (R = 1.922 Å) has a single a_u imaginary frequency and descends both ways to 1,5-hexadiene (barrier 31.2 kcal/mol), while CASSCF has a diyl minimum and an a_u saddle. Nothing built |

Promote something by moving it under `src/` (its tests into `test/gtest/`),
wiring it into `src/CMakeLists.txt`, and deleting it from here.
