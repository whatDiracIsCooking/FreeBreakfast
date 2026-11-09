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
solver scratch they share (see "Solver scratch" below), and the device's pinned
host memory (see "Pinned memory").

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

### Solver scratch

A dense-solver call wants a workspace and reports through an `info` flag.
`DeviceResources` holds one of each, so a routine that runs a solver takes
them from `res` instead of from its caller:

| Member | What | Kind |
|---|---|---|
| `device_workspace(bytes)` | `Buffers::device<std::byte>&`, grown to hold `bytes` | from the `Device`'s pool, on `stream()` |
| `info()` | `Buffers::device<int32_t>&`, one element, for `devInfo` | from the pool |
| `fetch_info()` | `Result<int32_t>`: copies `info()` to a pinned slot on `stream()`, syncs, returns it | |

```cpp
int32_t lwork = 0;
FBF_TRY(gpu_status(wwr::syevd_bufferSize<double>(res.solver().get(), ..., &lwork)));
auto work = wwr::extension::reinterpret_buffer_view<double>(
    res.device_workspace(static_cast<size_t>(lwork) * sizeof(double)));
FBF_TRY(gpu_status(wwr::syevd<double>(res.solver().get(), ..., work.data(), lwork,
                                      res.info().data())));
const int32_t info = FBF_TRY(res.fetch_info());  // < 0: bad argument; > 0: numerical
```

**Bytes, not a solver type.** The legacy API sizes workspace in elements of
`T`, the generic API in bytes, and one routine's `T` is not the next one's, so
the workspaces hold `std::byte` and a caller views them as its own type with
wwr's `reinterpret_buffer_view<T>`.

**Grow-only, in `kWorkspaceAlignment` (256 B) steps.** A request larger than
the current block replaces it with `align_up(bytes, 256)`; a smaller one hands
back the block as it is. The rounding is load-bearing:
`reinterpret_buffer_view` refuses a source whose size is not a whole multiple
of `sizeof(T)`, and refuses through the buffer's allocation policy, which here
aborts. Without it, a 1004-byte request followed by a `complex<double>` view
would abort. 256 is a multiple of every solver element type and the pool's own
alignment. `kWorkspaceAlignmentIn<T>` is the same step in elements of `T`, for
a routine carving the workspace into typed regions; it is constrained to a `T`
whose size divides 256, so the step can neither truncate nor be 0.

**Carve it with `device_regions<Ts...>(counts...)`.** One `T` per count, or one
`T` for all of them; each region is rounded up to 256 B, the total is asked for
in one `device_workspace` request, and the regions' starts come back as a
`std::tuple<Ts *...>` for a structured binding or a `std::tie`:

```cpp
const auto [work, block] = res.device_regions<double>(n_cart * nmo, nmo * nmo);
```

The regions are the workspace's, so the rule below holds for them too.

**Fetch the buffer just before the call; never keep its pointer.** A growth
frees the old block — on `stream()` for the device workspace, so work already
queued against it is safe, but a pointer or view held across another request
dangles. Pass the routine its own `lwork`, not the buffer's size, which may be
larger.

**One routine owns the device workspace at a time.** It is not only for the
solver's own work: a routine can carve all of its device scratch from it,
asking once for the total and splitting the result, so its caller lends
nothing. Ask for the total in one request, since a second could grow it and
free the first's region. And nothing nests: a routine holds no region of the
workspace across a call to another that uses it, and no argument to one may
point into it. Using it again after that call returns is fine.
`orthogonalize` is the example: U and the eigenvalues must outlive
`diagonalize`, which takes the workspace for syevd's work, so they are its
own pool buffers; M is written only after `diagonalize` returns, so it is
the workspace's.

**A callback is a nested user, whether or not it runs concurrently.** Nothing
here runs two routines at once, so the hazard is not a race. It is
*lifetime*. A routine that calls back into caller-supplied code, such as an
iterative solver's operator or preconditioner, cannot know whether that code
takes the workspace, so it must assume the callback does. Any of the routine's
own scratch still live when the callback returns is then lost, in one of two
ways:

- the callback asks for more bytes than the routine took, the workspace grows,
  and the old block is freed under the routine's pointers;
- the callback asks for fewer bytes, gets the same block, and writes its
  scratch over the routine's.

Either way, no error is reported, and the routine reads garbage on its next
step. So state an iterative routine keeps across its callbacks is its own
pool buffer. Scratch that is dead before the next callback may still come from
the workspace. `davidson_solve` (`src/recipes/davidson/`) is the example: its
subspace and residual blocks are its own buffers, while syevd's work, taken by
`diagonalize` between two operator applies, is the workspace's.
`test/recipes/davidson_gpu.cpp` pins this with an operator that grows and
overwrites the workspace on every apply.

**`info` is on the work stream.** The solver handle writes `info()` on
`stream()`, and `fetch_info()` copies on that same stream, so the read is
ordered after the write. A buffer on any other stream would race the
solver's write. The pinned slot is four bytes, fixed, and read at any time, so
it is a member of its own, outside the pinned workspace and the cap.

### Pinned memory

Pinned host memory is a per-device resource too, for loading and offloading:
`DeviceResources` owns all of it outside `src/core/resources/`, so one total
can be held under one cap.

| Member | What |
|---|---|
| `pinned_workspace(caller, bytes)` | `Result<std::span<std::byte>>`: the pinned workspace, grown to hold `bytes`, the whole block |
| `pinned_regions<Ts...>(caller, counts...)` | `Result<std::tuple<Ts *...>>`: it carved as `device_regions` carves the device one |
| `counted_pinned<T>(caller, count)` | `Result<CountedPinned<T>>`: a pinned buffer counted while it lives |
| `pinned_bytes()` | the pinned total: the workspace plus every live `CountedPinned` |
| `pinned_cap()` | the cap on it, in bytes, the constructor's third argument; 0 for none |

**The workspace is for fixed-size uses**: an upload's staging, a small
readback, the bounce buffer of a load from disk, the slots of an offload ring.
It grows as the device workspace does, in 256 B steps, and never shrinks. A
growth synchronizes `stream()` and frees the old block before it allocates the
new one, since a pinned free is not stream-ordered; the two are never held at
once. Carve it with `pinned_regions`, which lays the regions out exactly as
`device_regions` does:

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

**The cap.** A workspace growth or a counted allocation that would take
`pinned_bytes()` past `pinned_cap()` returns `Unsupported`, naming the caller,
the request, the total and the cap, and allocates nothing. With no cap (0, the
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
