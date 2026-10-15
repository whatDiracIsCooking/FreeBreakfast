# `free_breakfast.core.wwr`

The interface between free_breakfast and WarpWraps. **Code in the module tree
that needs something from wwr imports this module, not `wwr.*` directly** —
with the exceptions listed below. It holds three things:

- every wwr module the tree uses, re-exported;
- the bridge from wwr's error codes into `free_breakfast::Error`
  (`gpu_status`), and the tree's error policies for the wwr RAII wrappers;
- those wrappers, bound once to the policies, so the rest of the tree names a
  type rather than a five-argument template.

It only re-exports and binds. It does not wrap wwr functions under
free_breakfast names: this tree does not duplicate WarpWraps.

## Module

`free_breakfast.core.wwr`, target `free_breakfast::core::wwr`, also re-exported
by `free_breakfast.core`. Two partitions under the primary interface:

| Partition | Exports |
|-----------|---------|
| `:status` | `gpu_status` |
| `:policy` | `AbortPolicy` / `LogPolicy`, `CreatePolicy` / `DestroyPolicy` / `PolicyMap` |
| *(primary)* | the aliases below, and the re-exports |

Re-exported: `wwr.runtime_api`, `wwr.vector_types`, `wwr.wrappers.common`, and
`wwr.extension.{common,handle,runtime,memory_buffer,blas,solver,tx}`, so their
free functions (copy, memset, views, pointer-mode guards, …) come with it.

| Alias | wwr type | Notes |
| --- | --- | --- |
| `Device` | `kit::DeviceHandle` | One device: properties, an owned stream and memory pool. Hold as `std::shared_ptr`. |
| `Stream` | `StreamWrapper` | A non-blocking stream. |
| `Event` | `EventWrapper` | Default-constructed it is sync-only (timing disabled); pass `(dev_idx, 0)` for a timing-capable one. |
| `Buffers` | `kit::device_buffer_suite<PolicyMap, Device>` | `::device<T>`, `::host<T>`, `::pinned<T>`, `::unified<T>` and their `_view`s. |
| `BlasHandle` | `BlasHandleWrapper` | Bound to a `Device`'s stream. |
| `SolverHandle` | `SolverDnHandleWrapper` | Bound to a `Device`'s stream. |
| `ScopedRange` | `ScopedRange` | NVTX / rocTX range. |

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
   selecting the vendor runtime is wwr's job too.
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
  size (`policy_slot`), and `kit::DeviceHandle` can default-construct its
  device policy.

## Files

| File | Role |
|------|------|
| `wwr.cppm` | Primary interface: the re-exports and the bound wrapper aliases |
| `status.cppm` | `:status`: `gpu_status` |
| `policy.cppm` | `:policy`: `AbortPolicy`, `LogPolicy` and the aliases over them |
| `CMakeLists.txt` | The module library, and the one list of wwr module targets it links |

## Dependencies

All PUBLIC: `gpu_status`, the policies and the aliases are templates
instantiated in the consumer's TU.

- `free_breakfast::core::error_handling`: `Error`, `Status`, `check_or_abort`
- `free_breakfast::core::log`: `log::critical()` / `log::error()` from the
  policies
- the `wwr::*` module targets for everything re-exported above
