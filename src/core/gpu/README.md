# `free_breakfast.core.gpu`

The wwr RAII wrappers, bound once to this build's error policies
(`free_breakfast.core.error_handling`), so the rest of the tree names a type rather
than a five-argument template.

| Alias | wwr type | Notes |
| --- | --- | --- |
| `Device` | `kit::DeviceHandle` | One device: properties, an owned stream and memory pool. Hold as `std::shared_ptr`. |
| `Stream` | `StreamWrapper` | A non-blocking stream. |
| `Event` | `EventWrapper` | Default-constructed it is sync-only (timing disabled); pass `(dev_idx, 0)` for a timing-capable one. |
| `Buffers` | `kit::device_buffer_suite<PolicyMap, Device>` | `::device<T>`, `::host<T>`, `::pinned<T>`, `::unified<T>` and their `_view`s. |
| `BlasHandle` | `BlasHandleWrapper` | Bound to a `Device`'s stream. |
| `SolverHandle` | `SolverDnHandleWrapper` | Bound to a `Device`'s stream. |
| `ScopedRange` | `ScopedRange` | NVTX / rocTX range. |

It also re-exports the wwr modules it binds — `runtime_api`, and
`wwr.extension.{common,handle,runtime,memory_buffer,blas,solver,tx}` — so their
free functions (copy, memset, views, pointer-mode guards, …) come with it.

These are the wwr extension modules **used in place, from the submodule**, not
copies: this tree does not duplicate WarpWraps. To reach for another one
(`fft`, `sparse`, …), link `wwr::extension::<name>` here and add an alias bound
to `CreatePolicy` / `DestroyPolicy` the same way.

```cpp
namespace fb = free_breakfast;

auto r = fb::capture([] {
  auto dev = std::make_shared<fb::Device>(0);
  fb::Buffers::device<float> a(n * n, dev);
  fb::BlasHandle blas(dev);
  // ...
});
```

Link it as `free_breakfast::core::gpu`.
