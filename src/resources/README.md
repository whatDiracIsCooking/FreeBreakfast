# `free_breakfast.resources`

The machine's compute resources, and what this process gets of them.

## Module

`free_breakfast.resources`, target `free_breakfast::resources`. It sits above
`free_breakfast.core` and is not re-exported by it.

| Partition | Exports |
|-----------|---------|
| `:device_resources` | `DeviceResources` |
| `:sys_env` | `SysEnv`, `GpuInfo`, `query_sys_env()`, `available_gpus()`, `logical_cpu_cores()`, `log_sys_env()` |

## Files

| File | Role |
|------|------|
| `resources.cppm` | Primary interface: re-exports both partitions |
| `device_resources.cppm` | `:device_resources`: `DeviceResources`, defined inline |
| `sys_env.cppm` | `:sys_env`: the startup hardware inventory's declarations |
| `sys_env.cpp` | Its implementation: `/proc/cpuinfo`, `/proc/meminfo`, and the GPU roster through wwr's backend-neutral `wwrGetDeviceCount` / `wwrGetDeviceProperties` |
| `CMakeLists.txt` | The module library |

## DeviceResources

One device's working set: a shared `Device` (its `ForkStream` work stream and
memory pool) plus a `BlasHandle` and a `SolverHandle` bound to that stream.

```cpp
import free_breakfast.core;
import free_breakfast.resources;
namespace fb = free_breakfast;

fb::DeviceResources res{0};                      // Device + BLAS + solver on device 0
fb::Buffers::device<double> a(n, res.device());  // allocated from the Device's pool
// ... BLAS / solver calls on res.blas() / res.solver(), all on res.stream()

// Order all of it after a parent stream's work -- or pull it into that
// stream's graph capture -- and join back:
FBF_GPU_MUST(res.fork_stream().fork_from(parent));
// ... allocations, BLAS / solver calls ...
FBF_GPU_MUST(res.fork_stream().record_join());
FBF_GPU_MUST(res.fork_stream().join_into(parent));
```

**The handles anchor to the `Device`, not to `DeviceResources`.** wwr's
stream-bound handles keep a `std::shared_ptr` to the object whose stream they
bind. If that object were `DeviceResources` itself, its own members would keep
it alive — a reference cycle that never frees (wwr `docs/architecture.md` §18,
"Handle lifetime / ownership direction"). So ownership runs one way:

```
DeviceResources ─owns─▶ BlasHandle / SolverHandle ─shared_ptr─▶ Device
                ─shared_ptr───────────────────────────────────▶ Device
Buffers::device<T> ─shared_ptr────────────────────────────────▶ Device
```

A buffer taken from `device()` therefore stays valid after the
`DeviceResources` that handed it out is destroyed.

**One stream for allocation and work.** Buffers allocate and free
asynchronously on the `Device`'s stream, and the BLAS and solver handles queue
their work on that same stream, so an allocation, the kernels that use it and
its free run in order. A separate allocation stream would need an event wait
before first use and before every free; a free that races a kernel on another
stream is a silent use-after-free. If concurrent work is needed later, add
compute streams with their own handles and explicit event ordering rather than
splitting allocation off.

**The one stream is a `ForkStream`.** `Device`'s work stream is owned by a
`ForkStream` (`free_breakfast.core.wwr`), and `fork_stream()` hands it out, so
the whole working set can be forked from and joined into a parent stream —
including inside a graph capture — without a second stream or handle.

**The pool keeps 1 GiB cached by default.** The constructor passes its second
argument, `kDefaultPoolReleaseThreshold` (1 GiB, `Device`'s own default) unless
given, to the `Device` as its pool's release threshold. That is how much freed
memory the pool holds across stream syncs instead of returning to the driver —
not a cap on allocation. wwr's default is 1 MiB.

**Constructing one leaves its device current** on the calling thread: wwr's
handle creation selects the device and does not restore it. BLAS and solver
calls use the calling thread's current device, so code that drives more than
one device opens a `ScopedDeviceIndex` (`free_breakfast.core.wwr`) first.

Movable, not copyable (the handles are move-only). A moved-from
`DeviceResources` holds no `Device`.

## sys_env

Queries system hardware at startup: CPU cores, GPU devices, host RAM, and disk
space.

```cpp
if (auto env = free_breakfast::query_sys_env()) {
  free_breakfast::log_sys_env(*env);
}
```

The GPU roster is `vector<GpuInfo>`: plain data (index, name, memory, compute
capability), deliberately not `Device`, which owns a stream and a memory pool.
A startup inventory must not create one per device just to report what is on
the box. A runtime with no usable device yields an empty roster, not an error.

Linux-only: the CPU and RAM figures come from `/proc`. Where `/proc/cpuinfo`
carries no `physical id` / `core id` lines (as on most aarch64 kernels),
`cpu_physical_cores` falls back to the logical count rather than failing the
query, and a debug-level line says so.
