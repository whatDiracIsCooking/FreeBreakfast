# `free_breakfast.core.device_resources`

`DeviceResources`: one device's working set — a shared `Device` (its stream and
memory pool) plus a `BlasHandle` and a `SolverHandle` bound to that stream.

## Module

`free_breakfast.core.device_resources`, target
`free_breakfast::core::device_resources`, also re-exported by
`free_breakfast.core`. Built on the aliases in `free_breakfast.core.wwr`.

## Usage

```cpp
import free_breakfast.core;
namespace fb = free_breakfast;

fb::DeviceResources res{0};                      // Device + BLAS + solver on device 0
fb::Buffers::device<double> a(n, res.device());  // allocated from the Device's pool
// ... BLAS / solver calls on res.blas() / res.solver(), all on res.stream()
```

## Design

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

**The pool keeps 1 GiB cached by default.** The constructor sets the
`Device` pool's release threshold to `kDefaultPoolReleaseThreshold` (1 GiB) or
its second argument. That is how much freed memory the pool holds across stream
syncs instead of returning to the driver — not a cap on allocation. wwr's
default is 1 MiB, and `kit::DeviceHandle` takes no override, so it is set on the
live pool after construction.

Movable, not copyable (the handles are move-only). A moved-from
`DeviceResources` holds no `Device`.
