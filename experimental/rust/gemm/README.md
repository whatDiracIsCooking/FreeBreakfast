# rust/gemm — fp64 GEMM from Rust

Can free_breakfast's GPU objects be driven from Rust? Yes, through a C layer:
`DeviceResources`, `Buffers::device<double>` and `wwr::gemm` sit behind a small C
ABI, and a cargo crate wraps that in safe types.

## Why a C layer

wwr and `free_breakfast.core.wwr` are C++23 named modules, mostly templates.
Neither `bindgen` nor `cxx` reads a module or instantiates a template, so the
only thing Rust can link against is a C ABI:

| Layer | File | What |
|---|---|---|
| C ABI | `ffi/fb_gemm.h`, `ffi/fb_gemm.cpp` | Opaque `fb_resources` / `fb_buffer` handles, upload / download, `fb_dgemm`; `int32_t` status codes. Every function is `noexcept`: an exception unwinding into Rust is undefined behaviour, so one that escapes (only `bad_alloc` in practice) terminates, as a failed GPU create or alloc aborts (`CreatePolicy`) |
| Raw Rust | `src/ffi.rs` | The same functions declared by hand. **Nothing checks it against the header**; keep them in step |
| Safe Rust | `src/lib.rs` | `Gpu` (owns a `DeviceResources`; `!Send`, `!Sync`, its one-driving-thread rule), `DeviceBuffer<'g>` (borrows its `Gpu`, freed on drop), column-major `Matrix<'g>`, and `gemm(gpu, alpha, op_a, a, op_b, b, beta, c)` with shapes checked in Rust and again in C |
| Demo | `src/main.rs` | `gemm-demo [M [N [K [REPS]]]]`: C = A B^T, checked against a host loop (bar 1e-12 relative to max \|C\|), then timed |

## How it links: a recorded link line, not a shared library

The C ABI would most simply be a `.so`, but `/opt/spdlog/lib/libspdlog.a` is not
built `-fPIC` (its thread-local is local-exec, `R_X86_64_TPOFF32`), so no shared
object can contain it. So the C ABI is a **static** archive, and the Rust binary
links it together with everything behind it.

Only CMake knows that set: the core and wwr module archives, spdlog, taskflow, the
vendor runtime and BLAS, and libc++. `free_breakfast_rust_gemm_linkprobe`, a
do-nothing C++ executable, links the archive through `ffi/record_link.sh` as its
`CXX_LINKER_LAUNCHER`. That script writes the command to `fb_gemm.link` in the
build directory and then runs it. `build.rs` reads that file back and passes every
argument except the probe's own to rustc, plus `-lc++` (rustc links with
`-nodefaultlibs`). The recorded rpath comes along with it, so nothing needs
`LD_LIBRARY_PATH`.

Rebuilding spdlog with `-fPIC` in `docker/Dockerfile.base` would allow a plain
`.so`, but that changes the image for every build.

## Build and run

The Rust toolchain is not in the image: install it with
[rustup](https://rustup.rs) (`~/.cargo`, gone at a container rebuild).

```bash
cmake --preset experimental
cmake --build build-experimental --target free_breakfast_rust_gemm_linkprobe
cd experimental/rust/gemm
cargo test --release          # needs a GPU: 2x2 every op, beta, shape errors, round trip
cargo run --release           # gemm-demo 512 512 512 20
```

`build.rs` looks for `build-experimental/experimental/rust/gemm/fb_gemm.link`.
Set `FB_RUST_GEMM_FFI_DIR` to use a different build directory, such as
`build-hip-experimental` (untried).

Nothing in ctest or either gate runs cargo. This directory is checked by hand only.

## Measured (RTX 3080, CUDA, 2026-11-10)

| M = N = K | max rel. error vs host | GPU |
|---|---|---|
| 512 | 6.3e-16 | 0.72 ms, 371 GFLOP/s |
| 2048 | 7.8e-16 | 82 ms, 210 GFLOP/s |

Each GPU figure is the mean over 20 calls, timed host-side between two syncs, on a
card other worktrees share. They show that the call path adds no visible overhead.
They are not a benchmark.
