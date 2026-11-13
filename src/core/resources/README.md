# `free_breakfast.core.resources`

The node's compute resources, and what this process gets of them.

## Module

`free_breakfast.core.resources`, target `free_breakfast::core::resources`.
It is one of `core`'s libraries, **not re-exported by `free_breakfast.core`**:
it pulls in taskflow, and importing `free_breakfast.core` should not pull
taskflow into every unit that does. Import and link it explicitly.

| Partition | Exports |
|-----------|---------|
| `:device_resources` | `DeviceResources`, `CountedPinned<T>` |
| `:node_resources` | `NodeResources`, `NodeResourcesConfig`; re-exports `free_breakfast.third_party.taskflow` (`tf::Executor`, `tf::Taskflow`, `tf::Task`, `tf::FlowBuilder`, `tf::Subflow`) |
| `:sys_env` | nothing: an internal partition, the host queries `NodeResources` is built from |

## Files

| File | Role |
|------|------|
| `resources.cppm` | Primary interface: re-exports the two interface partitions |
| `device_resources.cppm` | `:device_resources`: `DeviceResources`, defined inline |
| `node_resources.cppm` | `:node_resources`: `NodeResources` and its config; accessors inline |
| `node_resources.cpp` | Its constructor and `log()`, the only code that reaches `:sys_env` |
| `sys_env.cppm` | `:sys_env`: declarations of the host queries |
| `sys_env.cpp` | Their implementation: `/proc/cpuinfo`, `/proc/meminfo`, and wwr's backend-neutral `wwrGetDeviceCount` |
| `CMakeLists.txt` | The module library |

## DeviceResources

One device's working set: a shared `Device` (its `ForkStream` work stream and
memory pool) plus a `BlasHandle` and a `SolverHandle` bound to that stream, the
workspaces they share, sized at construction (see "The workspaces" below),
and the device's pinned host memory (see "Pinned memory").

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
namespace fb = free_breakfast;

// Device + BLAS + solver on device 0, with a 64 MiB device workspace and a
// 1 MiB pinned one, held until res is destroyed.
fb::DeviceResources res{0, {.device = {.workspace = std::size_t{64} << 20},
                            .pinned = {.workspace = std::size_t{1} << 20}}};
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

**The pool keeps 1 GiB cached by default.** The constructor passes its third
argument, `kDefaultPoolReleaseThreshold` (1 GiB, `Device`'s own default) unless
given, to the `Device` as its pool's release threshold. That is how much freed
memory the pool holds across stream syncs instead of returning to the driver —
not a cap on allocation. wwr's default is 1 MiB.

**One driving thread at a time.** The BLAS and solver handles, the stream and
the `ForkStream`'s events must not be used from two threads at once: cuBLAS and
hipBLAS forbid concurrent use of one handle, and nothing here locks. That is a
contract, not a limitation worth working around: a `DeviceResources` given
enough work to be worth having is kept busy by one submitting thread, and the
stream serialises its GPU work anyway. A second thread that wants to overlap
work on the same GPU builds its own `DeviceResources` (and so its own stream
and handles), not a share of this one.

**The driving thread selects the device.** BLAS and solver calls and kernel
launches run on the calling thread's current device; only buffer allocation
and free select their own. So the driver either

| Driver | Selects the device |
|---|---|
| a thread that only ever drives this device | once, at thread start |
| a thread that drives several devices in turn | with a `ScopedDeviceIndex{res.dev_idx()}` per switch |
| a taskflow task (any worker, any device last) | with a `ScopedDeviceIndex{res.dev_idx()}` in the task |

`ScopedDeviceIndex` is `free_breakfast.core.wwr`'s. Constructing a
`DeviceResources` leaves its device current on the calling thread: wwr's handle
creation selects the device and does not restore it.

### The workspaces

A dense-solver call wants a workspace and reports through an `info` flag, and
most routines want device scratch of their own. `DeviceResources` holds a
device workspace, a pinned one and the flag, so a routine takes them from
`res` instead of from its caller:

| Member | What |
|---|---|
| `DeviceResources(dev_idx, workspaces, ...)` | the workspaces' sizes, `workspaces.device.workspace` and `workspaces.pinned.workspace` (a `MemoryFootprint`'s; its other fields are unread), each rounded up to `kWorkspaceAlignment` and allocated here |
| `device_workspace(caller, bytes)` | `Result<std::span<std::byte>>`: the first `bytes` of the device workspace, a span of device memory |
| `device_regions<Ts...>(caller, counts...)` | `Result<std::tuple<Ts *...>>`: the device workspace carved into one region per count |
| `pinned_workspace(caller, bytes)`, `pinned_regions<Ts...>(caller, counts...)` | the same over the pinned workspace (see "Pinned memory") |
| `workspaces()` | the sizes held, as a `MemoryFootprint` with only its two workspace fields set |
| `workspace_high_water()` | the largest borrow of each since construction, the same way: what a routine's footprint states as its workspace, so a test holds the footprint tight against it |
| `info()` | `Buffers::device<int32_t>&`, one element, for `devInfo`, from the pool |
| `fetch_info()` | `Result<int32_t>`: copies `info()` to a pinned slot on `stream()`, syncs, returns it |

```cpp
int32_t lwork = 0;
FBF_TRY(gpu_status(wwr::syevd_bufferSize<double>(res.solver().get(), ..., &lwork)));
const auto [work] = FBF_TRY(res.device_regions<double>("diagonalize", lwork));
FBF_TRY(gpu_status(wwr::syevd<double>(res.solver().get(), ..., work, lwork, res.info().data())));
const int32_t info = FBF_TRY(res.fetch_info());  // < 0: bad argument; > 0: numerical
```

**Sized at construction, held for the run.** The constructor allocates both
workspaces (the device one from the pool, so it counts against a pool cap;
the pinned one in the pinned total, so against the pinned cap), and nothing
resizes them: every member above borrows or carves, none grows. Inside a
pipeline the sizes are pre-flight's plan maximum (`src/pipeline/shape/`),
`main` building the node with them before the first step; a test sizes them
for the routine under test (`test/support/test_workspaces.h`).

**A borrow past the size is refused.** `Unsupported`, naming the caller, the
request rounded up and the workspace's size, and nothing is allocated, so a
borrow at the size or past it leaves the pool's `UsedMemCurrent` as it was.
Inside a pipeline that is a footprint that undercounted: pre-flight sized the
workspace from the steps' footprints, and the borrow asked for more. A site
need not check the memory budget before a borrow, since a borrow allocates
nothing.

**Fixed-size uses only.** What goes in a workspace may depend on the
dimensions and the step's options, never on a memory budget: the size is set
before the first step, from footprints evaluated at the plan's dimensions. An
adaptive step's adaptive part, a tile or a batch sized from what the budget
holds, is the step's own scratch (`core/footprint`'s README), which it
allocates and frees.

**What goes in it.** The workspace holds its maximum through every step, so it
is for what many steps use at similar sizes: a library call's `lwork`, a
solver's work, the regions a routine carves for its own scratch. An
intermediate only one step needs, at a size no other step comes near, is that
step's named scratch, freed when it returns, not the workspace's: in the
workspace it would be held for the whole run.

**Bytes, not a solver type.** The legacy API sizes workspace in elements of
`T`, the generic API in bytes, and one routine's `T` is not the next one's, so
the workspaces hold `std::byte` and a caller carves them as its own type.

**Rounded to `kWorkspaceAlignment` (256 B).** The sizes and every region are
rounded up to it: a multiple of every solver element type and the pool's own
alignment, so each region starts aligned for its type. `kWorkspaceAlignmentIn<T>`
is the same step in elements of `T`, for a routine carving regions by hand; it
is constrained to a `T` whose size divides 256, so the step can neither
truncate nor be 0.

**Carve it with `device_regions<Ts...>(caller, counts...)`.** One `T` per
count, or one `T` for all of them; each region is rounded up to 256 B, the
total is borrowed at once, and the regions' starts come back as a
`std::tuple<Ts *...>` for a structured binding or a `std::tie`:

```cpp
const auto [work, block] = FBF_TRY(res.device_regions<double>("df_gradient", n_cart * nmo, nmo * nmo));
```

Two `Ts` put a comma in the template argument list, which would split
`FBF_TRY`'s one argument, so that form takes a second pair of parentheses:
`FBF_TRY((res.device_regions<double, int32_t>(who, n, m)))`.

**Pass a routine its own `lwork`**, not the workspace's size, which may be
larger.

**One routine owns the device workspace at a time.** It is not only for the
solver's own work: a routine can carve all of its device scratch from it,
asking once for the total and splitting the result, so its caller lends
nothing. Nothing nests: a routine holds no region of the workspace across a
call to another that uses it, and no argument to one may point into it, since
both carve from the workspace's start. Using it again after that call returns
is fine. `orthogonalize` is the example: U and the eigenvalues must outlive
`diagonalize`, which takes the workspace for syevd's work, so they are its
own pool buffers; M is written only after `diagonalize` returns, so it is
the workspace's.

**A callback is a nested user, whether or not it runs concurrently.** Nothing
here runs two routines at once, so the hazard is not a race. A routine that
calls back into caller-supplied code, such as an iterative solver's operator
or preconditioner, cannot know whether that code takes the workspace, so it
must assume the callback does, and that it writes its scratch over the
routine's. No error is reported, and the routine reads garbage on its next
step. So state an iterative routine keeps across its callbacks is its own
pool buffer. Scratch that is dead before the next callback may still come from
the workspace. `davidson_solve` (`src/recipes/davidson/`) is the example: its
subspace and residual blocks are its own buffers, while syevd's work, taken by
`diagonalize` between two operator applies, is the workspace's.
`test/recipes/davidson_gpu.cpp` pins this with an operator that overwrites the
whole workspace on every apply.

**`info` is on the work stream.** The solver handle writes `info()` on
`stream()`, and `fetch_info()` copies on that same stream, so the read is
ordered after the write. A buffer on any other stream would race the
solver's write. The pinned slot is four bytes, fixed, and read at any time, so
it is a member of its own, outside the pinned workspace and the cap.

### Memory budget

`memory_budget()` is the device memory a refusal check counts on: the bytes
the pool can still give before an allocation through it would abort
(`CreatePolicy`). A site that can refuse its need as `Unsupported` compares it
against this, never against `wwrMemGetInfo`'s free memory
(`devtools/alloc-sites.py check` flags a new call outside `src/core/wwr/`).

```cpp
if (const size_t budget = FBF_TRY(res.memory_budget()); need > budget) {
  return err_unsupported("...: {} bytes, and device {}'s budget is {}", need, res.dev_idx(), budget);
}
```

| Member | What |
|---|---|
| `memory_budget()` | `Result<size_t>`: the limit less the pool's `UsedMemCurrent`, at least 0 |
| `pool_cap()` | the `Device` pool's cap, in bytes, the constructor's fifth argument; 0 for none |
| `kDeviceReserve` | 256 MiB: what the process holds on the device outside any pool, the kernels' stack aside |
| `kKernelStackBytes` | 16,912 B: the largest per-thread stack of any kernel the tree launches; the budget leaves room for it (below) |
| `kOwnPoolBytes` | 4 B: what a `DeviceResources` holds in its own pool from construction besides the device workspace, the `info()` flag; pre-flight's device room takes it off the cap |

**The limit** is `pool_cap()` under a cap. With none it is the device's total
less `kDeviceReserve`: the context and the BLAS and solver handles, which no
pool counts. That is the 249 MiB `src/core/wwr/README.md` ("The fixed reserve
outside the pool") measured on the 3080, rounded up, since it is a floor that
grows as kernel modules are first loaded. A constant rather than a measurement
at construction: by then the context's lazy growth has not happened, and
another `DeviceResources` on the device would read as part of it. A cap is
set that much below a device's share, so it is not taken out of the cap again.

**Never more than the driver can supply**: the limit is bounded by the
driver's free memory plus the pool's `ReservedMemCurrent`, since another
process may hold part of the device, or of the cap, before the pool has taken
it.

**Nor what the kernels' stack will take.** CUDA holds a kernel's stack for
every resident thread, outside the pool, from that kernel's first launch on:
up to `kKernelStackBytes` × 104,448 threads, 1.6 GiB, on the 3080, most of
it arriving mid-run, after a step has read its budget (#1259). The supply
leaves out `Device::stack_growth(kKernelStackBytes)`, what the
stack can still grow by, so a step that fills the budget leaves the stack
its room; once the largest kernel has run that is 0, and the free memory
shows the stack instead. A cap does not change it: the driver takes the
stack from its free memory first and then from what the pool holds cached,
so a pre-warmed pool loses its cache to it on a card with too little free
(measured: all 7.7 GiB of a cached pool, for 2.2 GiB of stack). Pre-flight's
pre-warm check therefore refuses a cap the card cannot supply beside the
stack. It is 0 under HIP, whose scratch is sized per dispatch. `src/core/wwr/README.md` ("Kernel stacks
outside the pool") has the measurements.

**Reserved but unused counts as available.** That is the difference from the
driver's free memory: a block the pool has freed and keeps cached (under a cap
it keeps all of them, the cap being its release threshold) is the next
allocation's, so freeing a buffer gives its bytes back to the budget at once.
It is a read, not a sync, as `Device::pool_usage()` is.

### Pinned memory

Pinned host memory is a per-device resource too, for loading and offloading:
`DeviceResources` owns all of it outside `src/core/resources/`, so one total
can be held under one cap.

| Member | What |
|---|---|
| `pinned_workspace(caller, bytes)` | `Result<std::span<std::byte>>`: the first `bytes` of the pinned workspace |
| `pinned_regions<Ts...>(caller, counts...)` | `Result<std::tuple<Ts *...>>`: it carved as `device_regions` carves the device one |
| `counted_pinned<T>(caller, count)` | `Result<CountedPinned<T>>`: a pinned buffer counted while it lives |
| `pinned_bytes()` | the pinned total: the workspace, from construction, plus every live `CountedPinned` |
| `pinned_cap()` | the cap on it, in bytes, the constructor's fourth argument; 0 for none |

**The workspace is for fixed-size uses**: an upload's staging, a small
readback, the bounce buffer of a load from disk, the slots of an offload ring.
It is sized at construction as the device workspace is, and borrowed and
carved the same way, a request past it refused. Carve it with
`pinned_regions`, which lays the regions out exactly as `device_regions`
does:

```cpp
const auto [z, c] = FBF_TRY(res.pinned_regions<double>("Molecule", n, 3 * n));
// ... write z and c on the host, copy them on res.stream() ...
FBF_TRY(res.sync()); // drained: the carve can be released
```

**`CountedPinned<T>` is for a product that lives in pinned memory**, past the
call that made it: an offloaded block streamed through a ring. Its bytes join
`pinned_bytes()` when it is allocated and leave when it is freed or assigned
to. It shares the count with `DeviceResources`, so it may outlive it. It hands
out `data()` and `num_elements()` only. Its free is immediate: drain the copies
that use it first.

**The cap.** The pinned workspace counts against it from construction, and a
pinned workspace past the cap aborts there, as a failed creation does:
pre-flight sizes it from a plan it has held to the cap. A counted allocation
that would take `pinned_bytes()` past `pinned_cap()` returns `Unsupported`,
naming the caller, the request, the total and the cap, and allocates nothing. With no cap (0, the
default) nothing is refused. The caller names itself because the refusal is
the run-time end of a configured limit, and the message has to say which step
asked.

**The one rule the device workspace does not need.** The CPU writes the pinned
workspace, and a copy that reads it runs later, on the stream. So a routine
must not rewrite a region until the stream has drained the copy that reads it
(a sync, or an event the ring waits on), and it drains its copies before it
returns, since the next routine to take the workspace writes it at once. "One
routine at a time" applies as it does to the device workspace: ask once for
the total, hold no region across a call to another routine that uses it.

Movable, not copyable (the handles are move-only). A moved-from
`DeviceResources` holds no `Device`.

## NodeResources

The whole node: a `DeviceResources` per device and a taskflow `tf::Executor`
(`free_breakfast.third_party.taskflow`) for host-side work, including handing
each device to its one driving task.

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
namespace fb = free_breakfast;

fb::NodeResources node;           // every visible device; physical cores - 2 workers
node.log();                       // cores, RAM, workers, each device

// One task per device: each is the only thread driving its DeviceResources.
fb::tf::Taskflow flow;
for (auto &res : node.devices()) {
  flow.emplace([&res] {
    const fb::ScopedDeviceIndex on{res.dev_idx()};
    // allocate from res.device(), BLAS / solver calls on res.blas() /
    // res.solver(), all on res.stream()
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

**Tasks follow `DeviceResources`' single-driver rule.** At most one task
drives a given device at a time, and it opens a `ScopedDeviceIndex` before it
works, since a worker may last have run another device's work. Nothing in
`NodeResources` enforces either; schedule the task graph so it holds.

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
