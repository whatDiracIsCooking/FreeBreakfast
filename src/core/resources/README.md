# `free_breakfast.core.resources`

The node's compute resources, and what this process gets of them.

## Module

`free_breakfast.core.resources`, target `free_breakfast::core::resources`.
It is one of `core`'s libraries, **not re-exported by `free_breakfast.core`**:
it pulls in taskflow, and importing `free_breakfast.core` should not pull
taskflow into every unit that does. Import and link it explicitly.

| Partition | Exports |
|-----------|---------|
| `:device_resources` | `DeviceResources` |
| `:node_resources` | `NodeResources`, `NodeResourcesConfig`; re-exports `free_breakfast.third_party.taskflow` (`tf::Executor`, `tf::Taskflow`) |
| `:sys_env` | nothing: an internal partition, the host queries `NodeResources` is built from |

## Files

| File | Role |
|------|------|
| `resources.cppm` | Primary interface: re-exports the two interface partitions |
| `device_resources.cppm` | `:device_resources`: `DeviceResources`, defined inline |
| `node_resources.cppm` | `:node_resources`: `NodeResources` and its config; accessors and `on_device()` inline |
| `node_resources.cpp` | Its constructor and `log()`, the only code that reaches `:sys_env` |
| `sys_env.cppm` | `:sys_env`: declarations of the host queries |
| `sys_env.cpp` | Their implementation: `/proc/cpuinfo`, `/proc/meminfo`, and wwr's backend-neutral `wwrGetDeviceCount` |
| `CMakeLists.txt` | The module library |

## DeviceResources

One device's working set: a shared `Device` (its `ForkStream` work stream and
memory pool) plus a `BlasHandle` and a `SolverHandle` bound to that stream.

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
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

## NodeResources

The whole node: a `DeviceResources` per device and a taskflow `tf::Executor`
(`free_breakfast.third_party.taskflow`) whose workers drive them.

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
namespace fb = free_breakfast;

fb::NodeResources node;           // every visible device; physical cores - 2 workers
node.log();                       // cores, RAM, workers, each device

fb::tf::Taskflow flow;
for (std::size_t i = 0; i < node.device_count(); ++i) {
  flow.emplace([&node, i] {
    node.on_device(i, [](fb::DeviceResources &res) {
      // res's device is current on this worker: allocate from res.device(),
      // BLAS / solver calls on res.blas() / res.solver(), all on res.stream()
    });
  });
}
node.executor().run(flow).wait();
```

`NodeResourcesConfig` narrows it. Every field defaults to "all of it":

| Field | Default | |
|---|---|---|
| `devices` | every device the runtime sees, in index order | An empty vector holds none, for host-only use. `device(i)` is position `i` in this list, not runtime index `i`. |
| `num_threads` | physical cores − `kReservedCores` (2), at least 1 | Leaves a core for the thread that submits work and one for the driver's own threads. A configured count below 1 aborts. |
| `pool_release_threshold` | 1 GiB | Passed to every `DeviceResources`. |

**Tasks reach a device through `on_device()`.** BLAS and solver calls and
kernel launches run on the calling thread's current device, and an executor
worker may last have run another device's work. `on_device(i, f)` calls
`f(device(i))` under a `ScopedDeviceIndex`, so the device is current for the
call and the worker's previous one is restored after. Buffers select their own
device, but the work done on them does not.

**Construction leaves the caller's current device alone.** Creating a
device's handles leaves that device current, so the constructor puts back the
device that was current before it built the first one. With no devices it
makes no device call at all, since a device-less runtime fails even the query.

**The executor is destroyed first.** It is declared after the devices, so its
destructor, which waits for running tasks, runs while the devices they use
still exist.

**Neither copyable nor movable.** The executor's workers hold its address.
Hold one per process and pass it by reference.

**Construction aborts** if a device fails to create (`CreatePolicy`, as for
any wrapper), if `/proc` cannot be read, or on a configured worker count below 1.
No `NodeResources` is ever left half-built.

### What it reads from the host

The internal `:sys_env` partition supplies the CPU and RAM figures, read once
at construction and exposed as `logical_cores()`, `physical_cores()` and
`host_ram_bytes()`. It also supplies the device count used when no device list
is given. The per-device figures that `log()` prints (name, memory, compute
capability) come from each `Device`'s own `props()`.

Linux-only: the CPU and RAM figures come from `/proc`. Where `/proc/cpuinfo`
carries no `physical id` / `core id` lines (as on most aarch64 kernels), the
physical count falls back to the logical one rather than failing, and a
debug-level line says so.
