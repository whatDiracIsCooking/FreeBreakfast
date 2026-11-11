# rust/gemm — fp64 GEMM from Rust

Can the GPU be driven from Rust without writing any C++ of our own? Yes. This
crate depends on WarpWraps' `wwr-sys` crate, the Rust bindings over WarpWraps'
C ABI (`deps/WarpWraps/src/capi`). It wraps a stream, device buffers and
`wwr_blas_dgemm` in safe types. Nothing here names CUDA or HIP: the backend is
whichever one the linked `libwwr_capi_*.so` was built for.

Until 2026-11-11 this crate carried its own C shim over free_breakfast's
`DeviceResources`. It was a static archive whose link line CMake recorded for
`build.rs` to replay, and WarpWraps' C ABI has replaced it. `../s_overlap` still
has to go through a shim of that kind, because what it calls is
free_breakfast's own pipeline, not wwr.

## Layers

| Layer | File | What |
|---|---|---|
| C ABI | `deps/WarpWraps/src/capi` | WarpWraps' generated `extern "C"` thunks, one shared library per surface. This crate links two of them: `libwwr_capi_runtime_api.so` and `libwwr_capi_blas.so` |
| Raw Rust | `wwr-sys` (in that directory) | Generated `extern "C"` declarations, one Cargo feature per surface; here `default-features = false, features = ["runtime_api", "blas"]` |
| Safe Rust | `src/lib.rs` | `Gpu` (a stream on one device and a BLAS handle bound to it; `!Send`, `!Sync`), `DeviceBuffer<'g>` (borrows its `Gpu`; allocated and freed in stream order), column-major `Matrix<'g>`, and `gemm(gpu, alpha, op_a, a, op_b, b, beta, c)` with shapes checked in Rust. A failure is an `Error` that names the call and its `wwr_status` / `wwr_blas_status`; nothing aborts. For a C ABI built beside it (`../s_overlap`): `Gpu::device`, `Gpu::stream`, `DeviceBuffer::as_ptr` / `as_mut_ptr` and `Matrix::from_buffer` |
| Link | `build.rs`, `capi_dir.rs` | Find the capi libraries in the CMake build directory; add the search path and an rpath |
| Demo | `src/main.rs` | `gemm-demo [M [N [K [REPS]]]]`: C = A B^T, checked against a host loop (bar 1e-12 relative to max \|C\|), then timed |

## Copies go through pinned memory and `dcopy`

The C ABI does not wrap `wwrMemcpy` or `wwrMemcpyAsync` yet. Their
`wwrMemcpyKind` argument is a by-value enum that wwr's generator has not
registered, and it logs both functions as skipped (`src/capi/README.md`
"Surface completeness", bucket C). BLAS `SetVector` / `GetVector` are not in
its BLAS surface either.

So `copy_from` and `copy_to` stage through pinned host memory instead
(`wwr_host_alloc`, and `wwr_host_get_device_pointer` for the device's view of
it). A `wwr_blas_dcopy_64` on the stream moves the data between that view and
the device buffer, and the copy is then waited for. That is a kernel reading or
writing host memory over PCIe rather than a DMA copy. For the demo's sizes it
does not matter.

Once WarpWraps registers `wwrMemcpyKind`, the two copies become
`wwr_memcpy_async` calls and `Pinned` goes away.

## Build and run

The Rust toolchain is in the image at `/opt/rust` (`docker/Dockerfile.base`),
or in `~/.cargo` from rustup in a container built before that.

```bash
cmake --preset experimental
cmake --build build-experimental --target wwr.capi.runtime_api wwr.capi.blas
cd experimental/rust/gemm
cargo test --release          # needs a GPU: 2x2 every op, beta, empty, shape and copy errors, round trip
cargo run --release           # gemm-demo 512 512 512 20
```

`build.rs` looks for the libraries in
`build-experimental/deps/WarpWraps/src/capi`. To use another CMake build
directory, set `FB_RUST_BUILD_DIR` to it, for example `build-hip-experimental`
(not tried). Any preset builds the capi, since WarpWraps adds it unconditionally.

`gemm-demo` NEEDs `libwwr_capi_runtime_api.so.1`, `libwwr_capi_blas.so.1` and
libc, nothing else (`readelf -d`). cuBLAS / hipBLAS comes in behind the blas
unit.

Nothing in ctest or either gate runs cargo. This directory is checked by hand only.

## Measured (RTX 3080, CUDA)

| M = N = K | max rel. error vs host | GPU, C shim (2026-11-10) | GPU, wwr-sys (2026-11-11) |
|---|---|---|---|
| 512 | 6.3e-16 | 0.72 ms, 371 GFLOP/s | 0.82 ms, 327 GFLOP/s |
| 2048 | 7.8e-16 | 82 ms, 210 GFLOP/s | 44 ms, 388 GFLOP/s |

The errors are identical, digit for digit, across the two paths. Each GPU figure
is the mean over 20 calls, timed host-side between two syncs. Both runs were on a
card that other worktrees share: it read 100% busy during the 2026-11-11 run.
So the two GPU columns do not compare with each other. They show that neither
call path adds visible overhead, and they are not a benchmark.
