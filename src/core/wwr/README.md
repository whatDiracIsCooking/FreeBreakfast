# `free_breakfast.core.wwr`

The interface between free_breakfast and WarpWraps. **Code in the module tree
that needs something from wwr imports this module, not `wwr.*` directly** —
with the exceptions listed below. It holds four things:

- every wwr module the tree uses, re-exported;
- the bridge from wwr's error codes into `free_breakfast::Error`
  (`gpu_status`), and the tree's error policies for the wwr RAII wrappers;
- those wrappers, bound once to the policies, so the rest of the tree names a
  type rather than a five-argument template;
- `ForkStream`, a `Stream` plus the two `Event`s that fork it from and join it
  back into a parent stream, for fanning work out inside one graph capture;
  `Device`, this tree's pool-tier device handle, whose work stream is a
  `ForkStream`; and `DeviceStream`, one more work stream on a `Device`,
  allocating from its pool.

It re-exports, binds and composes. It does not wrap wwr functions under
free_breakfast names: this tree does not duplicate WarpWraps. `ForkStream`'s
members are the fork/join protocol, not renamed runtime calls; the raw
operations stay wwr's free functions (`record`, `wait_event`, `sync`, ...).
The BLAS conveniences over wwr's dispatch (`gemm` for square operands, the
sandwiches `gemmwich` and `gmmwich`) are numerics, not binding, and live in
[`free_breakfast.recipes.blas`](../../recipes/blas/README.md).

## Module

`free_breakfast.core.wwr`, target `free_breakfast::core::wwr`, also re-exported
by `free_breakfast.core`. Five partitions under the primary interface:

| Partition | Exports |
|-----------|---------|
| `:status` | `gpu_status` |
| `:policy` | `AbortPolicy` / `LogPolicy`, `CreatePolicy` / `DestroyPolicy` / `PolicyMap` |
| `:fork_stream` | `Stream`, `Event`, `ForkStreamConfig`, `ForkStream` |
| `:device` | `MemPool`, `PoolUsage`, `Device` |
| `:device_stream` | `DeviceStream` |
| *(primary)* | the other aliases below, and the re-exports |

`Stream` and `Event` live in `:fork_stream` rather than the primary interface
because `ForkStream` is built from them and a partition cannot import the
primary interface; `Device` is in `:device` for the same reason, being built
from `ForkStream`, and `DeviceStream` in `:device_stream`, being built from
`Device`. Importers see no difference: every partition is re-exported.

Re-exported: `wwr.runtime_api`, `wwr.vector_types`,
`wwr.wrappers.{common,blas,solver}` (the typed BLAS and dense-solver dispatch:
`gemm<T>`, `dgmm<T>`, `syevd<T>`, …), and
`wwr.extension.{common,handle,runtime,memory_buffer,blas,solver,tx}`, so their
free functions (copy, memset, views, pointer-mode guards, …) come with it.

| Alias | wwr type | Notes |
| --- | --- | --- |
| `Device` | *(composed here)* | One device: properties, an owned `ForkStream` as its work stream, and a memory pool (1 GiB release threshold by default; an optional cap). Hold as `std::shared_ptr`. See below. |
| `DeviceStream` | *(composed here)* | Another work stream on a `Device`: its own `ForkStream`, the `Device`'s pool. Hold as `std::shared_ptr`. See below. |
| `MemPool` | `MemPoolWrapper` | A memory pool on one device; `Device` owns one. |
| `Stream` | `StreamWrapper` | A non-blocking stream. |
| `Event` | `EventWrapper` | Default-constructed it is sync-only (timing disabled); pass `(dev_idx, 0)` for a timing-capable one. |
| `ForkStream` | *(composed here)* | A `Stream` and a fork / join `Event` pair; see below. |
| `Buffers` | `kit::device_buffer_suite<PolicyMap, Device>` | `::device<T>`, `::host<T>`, `::pinned<T>`, `::unified<T>` and their `_view`s. |
| `StreamBuffers` | `kit::device_buffer_suite<PolicyMap, DeviceStream>` | The same kinds, bound to a `DeviceStream`: the `Device`'s pool, that stream's order. |
| `BlasHandle` | `BlasHandleWrapper` | Bound to a `Device`'s stream. |
| `SolverHandle` | `SolverDnHandleWrapper` | Bound to a `Device`'s stream. |
| `ScopedRange` | `ScopedRange` | NVTX / rocTX range. |
| `ScopedDeviceIndex` | `ScopedDeviceIndex` | Makes a device current on this thread, restores the previous one on exit. See below. |

To reach for another wwr module (`fft`, `sparse`, …), link
`wwr::extension::<name>` in this directory's `CMakeLists.txt`, `export import`
it in `wwr.cppm`, and add an alias bound to `CreatePolicy` / `DestroyPolicy` the
same way. Do that here rather than linking it where it is used.

```cpp
import free_breakfast.core.wwr;
namespace fb = free_breakfast;

auto dev = std::make_shared<fb::Device>(0);
fb::Buffers::device<float> a(n * n, dev);   // aborts here if the allocation fails
fb::BlasHandle blas(dev);
FBF_GPU_MUST(wwr::wwrDeviceSynchronize()); // diagnostics.h
```

## ForkStream

A `Stream` that can be forked from, and joined back into, a parent stream.
That is the protocol for running work on several streams inside one graph
capture: a stream joins a capture by waiting on an event recorded in the
capturing stream, and leaves it when the capturing stream waits on an event
recorded in it. Outside a capture the same calls simply order the work.

```cpp
fb::ForkStream side{dev_idx};             // or ForkStream{ForkStreamConfig{...}}
FBF_GPU_MUST(side.fork_from(parent));     // record fork on parent; side waits on it
ext::copy(d, in, side.stream());          // work on the fork
FBF_GPU_MUST(side.record_join());         // record join on side
FBF_GPU_MUST(side.join_into(parent));     // parent waits on join
```

`parent` is anything convertible to `wwrStream_t`: a `Device`'s `stream()`, a
`Stream`, a `StreamView` or a raw handle. `ForkStreamConfig` chooses the device,
stream flags (non-blocking by default, like `Stream`), an optional priority and
the event flags (timing disabled by default: the events only order work).

The operations return `wwrError_t`, like wwr's own stream and event free
functions, so the caller picks the tier: `FBF_GPU_MUST` where nothing can
recover, `gpu_status` where a `Status` is wanted. Creation aborts on failure
through `CreatePolicy`, as every wrapper here does.

`ForkStream` exposes `dev_idx()` and `stream()`, so it satisfies wwr's
`device_handle_stream` and can back a `kit::device_buffer_suite` directly, the
way a bare `Stream` can; it has no pool, so a buffer bound to it would draw
from the device's default pool. `Buffers` stays bound to `Device`, whose own
work stream is a `ForkStream` (next section), and a further stream that should
allocate from the same pool is a `DeviceStream`.
A `ForkStream` must outlive any capture or launch it is part of: `Stream`
synchronises before destroying, but a graph that still references a destroyed
stream's events is not something the runtime recovers from.

Inside a capture the two events are dependency edges in the graph, not
records, and after the capture ends the runtime rejects a synchronize or query
on them until they are recorded again. `Event` synchronises before destroying,
so a `ForkStream` re-records both events on its own stream, outside capture,
in its destructor and move-assignment: otherwise the destroy path would log
two errors and leave one as the runtime's *last error*, which the next
unrelated `wwrGetLastError()` would report as its own. Do not wait on
`fork_event()` / `join_event()` yourself after a captured fork/join.

## Device

`Device` is this tree's own pool-tier device handle, in place of wwr's
`kit::DeviceHandle`: the device's properties, a memory pool, and a
`ForkStream` whose stream is the work stream. It exposes `dev_idx()`,
`stream()` and `pool()`, so it satisfies wwr's `device_handle_pool`, and
`Buffers`, `BlasHandle` and `SolverHandle` bind to it as they did to
`kit::DeviceHandle`. What the `ForkStream` adds is `fork_stream()`: everything
queued on the device's stream — allocations, frees, BLAS and solver calls — can
be forked from a parent stream and joined back into it, inside a graph capture
or out of one.

```cpp
auto dev = std::make_shared<fb::Device>(0);
FBF_GPU_MUST(dev->fork_stream().fork_from(parent));
// ... allocate from dev, BLAS / solver calls on handles bound to dev ...
FBF_GPU_MUST(dev->fork_stream().record_join());
FBF_GPU_MUST(dev->fork_stream().join_into(parent));
```

The pool's release threshold is the second constructor argument, 1 GiB
(`Device::kDefaultPoolReleaseThreshold`) by default rather than wwr's 1 MiB:
that is how much freed memory the pool keeps cached across stream syncs
instead of returning to the driver, not a cap on allocation. The pool is
declared before the `ForkStream`, so the stream — which synchronizes before
destroying — is torn down first and no queued free outlives its pool.

### Pool cap and usage

The third constructor argument is the pool's cap in bytes, 0 (none) by
default. A cap sets two pool attributes: `maxSize`, the hard limit, past which
an allocation fails (through `Buffers`, that failure is `CreatePolicy`'s
abort); and the release threshold, to the cap in place of the second argument,
so memory one step frees stays in the pool for the next instead of going back
to the driver, where another process could take it mid-run.

```cpp
auto dev = std::make_shared<fb::Device>(0, fb::Device::kDefaultPoolReleaseThreshold, cap);
dev->reset_pool_high_water();
// ... the call to measure ...
const fb::PoolUsage u = dev->pool_usage(); // used / reserved, current / high
```

`pool_usage()` reads the pool's `UsedMemCurrent/High` and
`ReservedMemCurrent/High`; `reset_pool_high_water()` sets both high-water marks
back to the current values. `test/support/pool_footprint.h` builds the
footprint tests' two helpers on them: `measure_pool_peak` (one call's peak
above where it started) and `expect_footprint` (the sound and tight bars).

What the counters count, measured on the 3080 (driver 580.126.20, CUDA 13.0):
`UsedMem*` is the bytes requested, unrounded — a one-byte allocation adds
1 — so the pool's granularity as the footprint helpers read it is 1 B there,
though each allocation's address is 512 B-aligned; `ReservedMem*` grows in
32 MiB chunks. `pool_granularity()` in the helper header measures it rather
than assuming it.

**HIP:** ROCm 7.2.4 on the gfx1200 enforces `maxSize` — the cap test's
allocation past the cap fails with `wwrErrorMemoryAllocation` there, as on CUDA —
and its `UsedMem*` also counts requested bytes (granularity 1 B). One
difference: setting a high-water mark to 0 resets it to 0 there, not to the
current value as CUDA does, so `pool_usage()` floors each high-water mark at
its current value; after `reset_pool_high_water()` both backends then read
the same.

**Reuse rounding (HIP).** ROCm's pool serves a request whole from a block an
earlier allocation freed, when the block is at most 9/8 of the request, and
`UsedMem*` then counts the block, not the request: one allocation is counted
at most request / 8 past what it asked. CUDA's pool counts the request and
serves nothing whole. Measured on the gfx1200 (ROCm 7.2.4) by
`test/core/device_pool_gpu.cpp`'s `DevicePool.ReuseRounding`: for each block
X from 2 B to 64 MiB, freed on a fresh `Device`, the smallest request served
from it whole, by bisection, is the smallest Y with 8 X ≤ 9 Y, so
X − Y ≤ Y / 8, the bound reached from about 12 kB up (smaller blocks round
it down). It is pool rounding, as the granularity is, only larger and only
on reuse: a footprint keeps counting requested bytes, and
`test/support/pool_footprint.h`'s sound bar allows each allocation live at a
peak its 1/8 on a pool that has it (`kPoolReuseDivisor`, `pool_rounds_reuse`).
#1143 and #1049 are where it showed: steps that free one phase's buffers
before the next phase allocates.

### The fixed reserve outside the pool

What a process holds on the device that no pool counts, measured on the 3080
(driver 580.126.20, CUDA 13.0) by a scratch gtest that stepped through
creating each piece, holding each step 4 s while `nvidia-smi
--query-compute-apps=used_memory` (the process's own) and
`--query-gpu=memory.used` (the whole card, idle at 646 MiB) were sampled every
0.5 s, the pool empty:

| Step | Process | Card delta |
| --- | --- | --- |
| CUDA context (`wwrFree(nullptr)`), then the `Device` | 216 MiB | 221 MiB |
| + one `BlasHandle` | 228 MiB | 233 MiB |
| + one `SolverHandle` | 240 MiB | 245 MiB |
| + one 512³ `dgemm` (pool trimmed back to 0 after) | 240 MiB | 249 MiB |

**The fixed reserve is 249 MiB per process on the 3080** (the card's view,
the larger of the two): a `max_device_mem_per_gpu` leaves that much outside the
pool cap. It is a floor, not a constant: the context grows as lazily loaded
kernel modules are first launched, so a run that touches many of the tree's
kernels holds more, and another driver or card will differ. Most of that
growth is the kernels' stack, which is counted apart (next section).

Under HIP the pool is created exportable (a POSIX file descriptor handle type)
though nothing exports it. That is a workaround for a ROCm 7.2.4 defect (#300):
a non-exportable pool is backed by ROCm's virtual-memory heap, whose
`hipMemPoolDestroy` intermittently dereferences a null after a run of
differently sized allocations, and an exportable pool never gets the heap.
`Device::pool_props()` has the details and the upstream fix that retires it;
it is one of the tree's three backend `#if`s in host code; the others are
`Device::stack_growth()` (below) and `recipes/expm`'s `heevdx` under HIP
(#999).

### Kernel stacks outside the pool

The reserve above is measured before the tree's big kernels run. The larger
part of what a process holds outside the pool comes later: **CUDA holds a
kernel's per-thread stack for every thread the device can keep resident**,
SMs × `maxThreadsPerMultiProcessor` (68 × 1536 = 104,448 on the 3080), from
that kernel's first launch on. The driver raises the stack limit
(`wwrLimitStackSize`, 1024 B in a fresh context) to the kernel's stack and
keeps it, so the reservation is the largest stack launched so far times the
thread count. The generated integral kernels' stacks reach 16,912 B
(`potential_hessian<3,3>`; `cuobjdump --dump-resource-usage`), so a run
that launches it holds 1.6 GiB more than a fresh context.

Measured on the 3080 (driver 580.126.20, CUDA 13.0) by a scratch program
reading `wwrMemGetInfo` and the limit around one-block launches of kernels
with 1 KiB, 10,000 B and 23,192 B of stack:

| Step | Device used | Stack limit |
| --- | --- | --- |
| context | 1096.4 MiB | 1024 |
| 1 KiB-stack kernel | 1096.4 MiB | 1024 |
| 10,000 B-stack kernel | +894 MiB | 10,000 |
| 23,192 B-stack kernel | +2210 MiB in all | 23,200 |
| 1 KiB-stack kernel again | unchanged | 23,200 |
| `wwrDeviceSetLimit(stack, 1024)` | back to the context's | 1024 |

894 MiB is (10,000 − 1024) × 104,448 B exactly, and 2210 MiB is
(23,200 − 1024) × 104,448 B. The grid's size does not matter (a 4096 × 256
launch reads the same), nor does `CUDA_MODULE_LOADING=EAGER`. Setting the
limit takes or returns the memory at once. On a card full but for 256 MiB,
the 23,192 B kernel's launch fails with `wwrErrorMemoryAllocation`; when the
pool has already handed that memory out, the next allocation through the
pool fails instead and aborts (`CreatePolicy`). That was #1259:
formaldehyde's PC-NEVPT2 Hessian went +1000 MiB at its first DF kernels
(`eri3<2,2,3>`, 10,696 B) and +706 MiB at the Hessian's
(`potential_hessian<1,2>`, 17,784 B), both after the steps had read their
budget.

**A pool's cache does not protect the memory it holds.** The driver takes
the stack from its free memory first and then trims pools' cached memory
for it: a pool holding 7712 MiB cached (release threshold past it), with
1178 MiB free on the card, held 0 B once the limit was raised to 23,192 B,
and an allocation of the old size from it then failed. So a pre-warmed,
capped pool can lose its cache to the stack mid-run.

**The tree's kernels are emitted to keep this down** (#1263, #1265). Every
class whose FelizJueves default form needed more than 8 KiB takes the
lowest-stack form FelizJueves offers that was not measured slower
(`tools/emit/lowstack.py`): an existing low-stack form the default picks
only from a higher `L` floor, the low-stack `d2s` / `d2t`, compact tables
(`d2eri3`, the low-stack `geri3`, `xeri3`, and `d2v` at (pf) and (df)), or
`d2eri3`'s warp-shared tower, which forms the bra tables and the Hermite
tower once per record pair in shared memory instead of in every lane. Read
off the built binary (3586 kernels, sm_86):

| | Before #1263 | After #1263 | After #1265 |
| --- | --- | --- | --- |
| largest stack | 23,192 B, `kinetic_hessian<3,3>` | 18,912 B, `d2eri3<3,3,6>` | 16,912 B, `potential_hessian<3,3>` |
| reservation at that stack | 2310 MiB | 1884 MiB | 1685 MiB |
| kernels over 8 KiB | 89 | 63 | 37 |
| a DF cc-pVDZ/cc-pVDZ-JKFIT Hessian's largest (d orbitals, f aux) | 17,784 B, `potential_hessian<1,2>` (1771 MiB) | 9,360 B, `d2eri3<2,2,3>` (932 MiB) | 8,976 B, `potential_hessian<2,2>` (894 MiB) |

The warp-shared tower took `d2eri3<3,3,6>` from 18,912 to 8,216 B, and it
is bitwise the per-item form. Its speed depends on the class, though
(timed on the RX 9060 XT; the RTX 3080 is not timed): faster or level from
aux `l` = 3 up, but 1.4-2.1x slower at aux `l` <= 2 of (pf), (dd), (df)
with contracted aux sets. So those classes, and `<3,3,0>` and `<3,3,3>`,
keep the per-item form (`d2eri3<3,3,3>` 15,664 B). The compact `d2v` was
1.05x slower at (dd) and 1.08x at (ff), so `potential_hessian<3,3>` keeps
its dense tables (13,024 B compact, opt-in in FelizJueves) and sets the
maximum. The target, 8 KiB (816 MiB), is not reached. Next below are
`d2eri3<3,3,3>` and `<3,3,0>` (per-item, 15,664 and 13,584 B),
`kb<2,2,2,2>` (12,512 B), and the compact low-stack `geri3` / `xeri3` and
`gk` at f. `e_hermite<7,6>`, the gradient's raised `[p|q]` row, was
15,808 B until FelizJueves's `fused` lever (2,944 B, #1337), which every
`e_hermite` class it is not slower at now takes (#1345).

`Device::stack_growth(per_thread)` is what is still to come: the limit's
shortfall from `per_thread` times the thread count.
`DeviceResources::memory_budget()` leaves `stack_growth(kKernelStackBytes)`
out of the driver's supply, capped or not, so the pool is never handed what
the stack will take; once the largest kernel has run it is 0, and the free
memory already shows the stack. `kKernelStackBytes` is the binary's largest stack,
and ctest's `core_kernel_stack_bound` reads the built binary with
`cuobjdump` and fails naming the kernel when a regenerated one passes it.

**HIP:** ROCm 7.2.4 on the gfx1200 neither raises the limit nor reserves
scratch device-wide. The same program read no growth past the first
launch's 180 MiB, at 1 KiB, 10,000 B or 23,200 B of scratch a lane, one block
or 4096 × 256; and a 23,200 B launch on a card full but for 256 MiB ran.
ROCm sizes scratch per dispatch. So `stack_growth()` is 0 under HIP.

## DeviceStream

A `Device`'s pool paired with a `ForkStream` of its own: another work stream
on the same device. It exposes `dev_idx()`, `stream()` and `pool()`, so it is a
`device_handle_pool` like `Device`, and `StreamBuffers` bind to it: their
blocks come from the `Device`'s pool, and their allocation, zero-fill and free
are ordered on the `DeviceStream`'s stream.

```cpp
auto dev = std::make_shared<fb::Device>(0);
auto a = std::make_shared<fb::DeviceStream>(dev);
auto b = std::make_shared<fb::DeviceStream>(dev);   // independent of a
fb::StreamBuffers::device<double> x(n, a);          // dev's pool, a's stream
```

Ownership runs one way. Each `DeviceStream` retains its `Device`, and the
`Device` keeps no list of them, so any number can be made from one, and the
pool outlives every stream that allocates from it. The `Device` pointer is
declared before the `ForkStream`, so the stream, which synchronizes before
destroying, drains its queued frees into the pool before the `DeviceStream`
can drop the last reference to the `Device`.

Why not bind buffers to a bare `ForkStream`: that satisfies only
`device_handle_stream`, and wwr's device buffer then allocates with
`wwrMallocAsync` from the device's *default* pool. That loses `Device`'s
release threshold, and does it silently.

A `DeviceStream` orders nothing against the `Device`'s own stream, or another
`DeviceStream`'s. Work that crosses between them goes through
`fork_stream()`'s fork/join, as above.

## ScopedDeviceIndex

The current device is per-thread runtime state, and wwr sets it in only two
places. Creating a device-bound handle (`Stream`, `Event`, `MemPool`,
`BlasHandle`, …) selects its device and **leaves it current**. A buffer
allocates and frees under its own `ScopedDeviceIndex`, which restores the
previous device. Nothing else selects one: BLAS and solver calls, kernel
launches and the raw runtime calls all use whatever the calling thread has
current. Code that drives more than one device, or runs on a pool thread that
may have touched another device last, opens a guard first:

```cpp
{
  fb::ScopedDeviceIndex on{res.dev_idx()};  // res's device is current
  // ... BLAS / solver calls, kernel launches on res.stream() ...
}                                           // previous device restored
```

Both the switch and the restore go through `CreatePolicy` (abort). The restore
runs in a `noexcept` destructor, so a policy that threw would terminate there.
A restore that only logged would leave later work running on the wrong device
without any error.

## Where it sits, and the exceptions

```
types, log ─► error_handling (Error, Result, check_or_abort; no wwr)
                     │
                 core.wwr  ◄── everything else that needs wwr
```

It has to sit above `error_handling`, because the policies report through
`Error` and `check_or_abort`. That puts three kinds of code outside it, and
they link wwr directly:

1. **`core/types`.** `error_handling` is built on `types`, so `types` cannot
   import a module above `error_handling`. It takes `wwr.wrappers.common` (the
   scalar concepts) and `wwr::vector_types::host` (`vec.h`) itself. Keep it a
   thin vocabulary layer.
2. **Header and device code**: `primitives.h`, `core/kernel`'s `.h`/`.cuh`, and
   every `.cu` library. nvcc cannot import a module, so these link
   `wwr::device` for wwr's `runtime.h` and `WWR_WARP_SIZE`. They include that
   header, never `<cuda_runtime.h>` or `<hip/hip_runtime.h>` themselves:
   selecting the vendor runtime is wwr's job too. The one carve-out is the
   kernels FelizJueves generates (`src/integrals/one_electron/*/kernels/`): they
   include FelizJueves's own `felizjueves/runtime.h`, which selects the vendor
   runtime from the compiler's macros the same way, because a generator that
   is not built against wwr cannot spell wwr's header. Their launchers still
   include wwr's first, through the bridge.
   `core/constants` is header code of this kind with a host-side twist:
   `constants.h` includes wwr's `complex.h` for the complex numeric literals,
   and its module compiles that header on the host, so it links
   `wwr::vector_types::host` (the backend define, wwr's `src/` root and the
   vendor include path) rather than `wwr::device`, as `core/types` does for
   `vec.h`.
3. **`diagnostics_macros` does not link this module**, although `FBF_GPU_MUST`
   expands to `gpu_status`. `io/` and `basis/` link it for `FBF_TRY` and
   `FBF_LOG_*` and have no GPU code. Code that calls `FBF_GPU_MUST` already
   holds a wwr error code, so it imports this module anyway.

A vendor-only entry point (`import wwr.cuda.cublas_v2;`) is a fourth,
deliberate exception: import it where it is used, and that file is then
CUDA-only.

## gpu_status

`gpu_status(code)` converts any WarpWraps `error_type` (`wwrError_t`, which is
`cudaError_t` or `hipError_t` depending on the build, or a library status code)
to a `Status`. A failure becomes `ErrorKind::GpuRuntime` carrying the code's
name and description, located at the caller. It is not named `gpu_check`,
which is `wwr::extension::gpu_check` (policy-driven, returns `bool`).
`FBF_GPU_MUST(call)` is `check_or_abort(gpu_status(call))`.

## GPU error policy

What a failed GPU call inside a wwr RAII wrapper does. WarpWraps' extension
layer forces no error policy: every wrapper takes its policies as template
arguments (`wwr.extension.common`'s `error_policy` concepts). `:policy`
supplies ours:

| Slot | Alias | Policy | On failure |
| --- | --- | --- | --- |
| Create / alloc / device-select | `CreatePolicy<E>` | `AbortPolicy` | Log at critical with the failing call's location, `std::abort()` |
| Destroy / free | `DestroyPolicy<E>` | `LogPolicy` | Log at error, carry on |

`PolicyMap` binds the two to wwr's `alloc` / `free` slots for the buffer
suites. Name the aliases rather than a concrete policy, or in practice neither:
use the bound wrappers (`Device`, `Buffers`, `BlasHandle`, …).

Because a failed creation aborts at the call that failed, a wrapper that exists
is a valid one: there is no empty-wrapper state to check for, and no error
value to propagate.

### Why it is shaped like this

- **Abort, not recover.** For this code, the answer to a failed device
  allocation or handle creation is "rerun with a smaller problem or on another
  device", not an in-process fallback. Nothing in the tree recovers, so a
  recoverable path would be machinery with no caller. If one is ever needed,
  add it for that site — wrap that one construction with a policy that
  reports instead of aborting — rather than as a tree-wide mode.
- **Destroy/free logs instead.** It runs in a destructor, often during
  teardown after an earlier failure, where aborting would bury the root cause
  under the knock-on one.
- **Both policies are empty**, so they fit every wwr slot and collapse to zero
  size (`policy_slot`).

## Files

| File | Role |
|------|------|
| `wwr.cppm` | Primary interface: the re-exports and the bound wrapper aliases |
| `status.cppm` | `:status`: `gpu_status` |
| `policy.cppm` | `:policy`: `AbortPolicy`, `LogPolicy` and the aliases over them |
| `fork_stream.cppm` | `:fork_stream`: `Stream`, `Event`, `ForkStreamConfig`, `ForkStream` |
| `device.cppm` | `:device`: `MemPool`, `PoolUsage`, `Device` |
| `device_stream.cppm` | `:device_stream`: `DeviceStream` |
| `CMakeLists.txt` | The module library, and the one list of wwr module targets it links |

## Dependencies

All PUBLIC: `gpu_status`, the policies and the aliases are templates
instantiated in the consumer's TU.

- `free_breakfast::core::types`: the fixed-width integers in `ForkStreamConfig`
- `free_breakfast::core::nontransferrable`: `NonCopyable`, the base of `ForkStream`, `Device` and `DeviceStream`
- `free_breakfast::core::error_handling`: `Error`, `Status`, `check_or_abort`
- `free_breakfast::core::log`: `log::critical()` / `log::error()` from the
  policies
- the `wwr::*` module targets for everything re-exported above
