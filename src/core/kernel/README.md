# Kernel

Building blocks for writing kernels: the launch-shape constants (`kWarpSize`,
`kFourWarps`, `kMaxThreads`, `NumWarps`), the `DeviceFunctor` concept, and warp- and block-wide reductions.
The integer rounding a launch is sized with (`math::idivup`,
`math::align_up`) is in [`primitives/`](../primitives/README.md).

## Module

`free_breakfast.core.kernel` — target `free_breakfast::core::kernel`, also
re-exported by `free_breakfast.core`. It carries only what host code needs to
size a launch: `kWarpSize`, `kFourWarps`, `kMaxThreads` and `NumWarps`, in
`free_breakfast`.

Everything else here is device code with no module face, `#include`d into a
`.cu` that links `free_breakfast::core::kernel_headers`. Its names are in
`free_breakfast::device`.

## Launch shape (`block_params.h`)

| Name | What it is |
|------|------------|
| `kWarpSize` | `uint32_t`, the build's warp/wavefront size: `WWR_WARP_SIZE`, which wwr sets at configure time (32; 64 for a CDNA build) |
| `kFourWarps` | `uint32_t`, `4 * kWarpSize`: the block size of a one-thread-per-item launch (128; 256 for a CDNA build) |
| `kMaxThreads` | `uint32_t`, 1024: the most threads a block may hold, on every target. The block size of a one-block launch that strides a whole array |
| `NumWarps<N>` | Concept over a *value*: `N` is integral, >= 1, a power of two, and `N * kWarpSize <= kMaxThreads`. Use it as `template <auto W> requires NumWarps<W>` |

## Functors (`device_functor.h`)

`DeviceFunctor<F>`: a trivially-copyable class type, which is what can be
passed to a kernel by value. The concept checks only that. Whether `F` can be
called with the right signature is checked by the device function that takes
it, because a host-side check would reject a `__device__`-only `operator()`.
This is not wwr's `device_functor` from `parallel_for`. That one also
requires the functor to be non-copy-assignable, because it is passed as
`__grid_constant__`; these functors never are.

`IdentityFunctor<T>` (device pass only) passes a value through unchanged. It is
the pre-transform for a plain reduce.

## Reductions (`warp_reduce.cuh`, `block_reduce.cuh`)

Both fold one arithmetic value per thread under an associative op and return
the result in **every** thread. Only the first `nactive` threads' values enter
the fold, and they are combined in thread order. So the op needs neither an
identity nor commutativity. Every thread must still make the call: the shuffles
and barriers are collectives.

| Call | Mechanism |
|------|-----------|
| `warp_reduce(tile, v, op, nactive)` | `shfl_down` ladder + broadcast, one warp |
| `block_reduce(block, shared_arr, v, op, nactive)` | `warp_reduce` per warp, then over the per-warp partials; one barrier |

Values must be arithmetic, because HIP's tile shuffles resolve only for the
vendor's scalar types. There is no reduction here for a class type (a complex
number, an argmax pair). `block_reduce` keeps the per-warp partials in the
caller's `shared_arr`, so the caller must `block.sync()` (or switch buffers)
between two calls that share it.

There are no stock ops. The caller writes the fold as a small functor with a
`__device__` `operator()`, as in the example below.

```cpp
#include "block_reduce.cuh"

namespace cg = cooperative_groups;
namespace fbd = free_breakfast::device;

struct Add {
  __device__ double operator()(double a, double b) const { return a + b; }
};

template <uint32_t kNumWarps>
__global__ void sum(const double *x, double *out, uint32_t n) {
  __shared__ double partials[kNumWarps];
  const double v = threadIdx.x < n ? x[threadIdx.x] : 0.0;
  const double s = fbd::block_reduce(cg::this_thread_block(), partials, v, Add{}, n);
  if (threadIdx.x == 0) {
    *out = s;
  }
}
```

## Include order under HIP

Under HIP, a libc++ header must not come after the vendor runtime header (see
`primitives.h`). Each header here includes its standard headers before any wwr
or vendor header. A `.cu` should include `block_reduce.cuh` (or
`warp_reduce.cuh`) before anything that brings in the vendor runtime.

## Tests

- Host: `test/gtest/core_vocabulary_host.cpp` pins `kWarpSize`, `kFourWarps`, `kMaxThreads` and the bounds
  `NumWarps` accepts and rejects.
- Device: `test/gtest/reduce_gpu.cpp` runs `block_reduce` (and so
  `warp_reduce`) through `test/gpu_kernels/reduce_kernels.cu`, over `double`
  and `int32_t`, checking every thread's result for `Add`, `MaxNan`, and two
  non-commutative folds (`First`, `Last`). It covers 1, 2, 4 and the maximum
  number of warps, a range of `nactive` values, and a NaN past `nactive` being
  ignored.

## Files

| File | Role |
|------|------|
| `block_params.h` | `kWarpSize`, `kFourWarps`, `kMaxThreads`, `NumWarps`, for host and device |
| `kernel.cppm` | Host module: GMF-includes `block_params.h` and re-exports its names. Must list everything the header defines |
| `device_functor.h` | `DeviceFunctor`; `IdentityFunctor` in a device pass |
| `warp_reduce.cuh` | `warp_reduce`, device only |
| `block_reduce.cuh` | `block_reduce`; device only |
| `CMakeLists.txt` | The module (links `wwr::device` privately for `WWR_WARP_SIZE`), and `free_breakfast::core::kernel_headers` for device code (`wwr::device`, `primitives_header`, `types_headers`) |
