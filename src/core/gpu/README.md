# `free_breakfast.core.gpu`

The wwr RAII wrappers, bound once to this build's error policies
(`free_breakfast.core.gpu_policy`), so the rest of the tree names a type rather
than a five-argument template.

| Alias | wwr type | Notes |
| --- | --- | --- |
| `gpu::Device` | `kit::DeviceHandle` | One device: properties, an owned stream and memory pool. Hold as `std::shared_ptr`. |
| `gpu::Stream` | `StreamWrapper` | A non-blocking stream. |
| `gpu::Buffers` | `kit::device_buffer_suite<gpu::PolicyMap, gpu::Device>` | `::device<T>`, `::host<T>`, `::pinned<T>`, `::unified<T>` and their `_view`s. |
| `gpu::BlasHandle` | `BlasHandleWrapper` | Bound to a `Device`'s stream. |
| `gpu::SolverHandle` | `SolverDnHandleWrapper` | Bound to a `Device`'s stream. |
| `gpu::ScopedRange` | `ScopedRange` | NVTX / rocTX range. |

It also re-exports the wwr modules it binds — `runtime_api`, and
`wwr.extension.{common,handle,runtime,memory_buffer,blas,solver,tx}` — so their
free functions (copy, memset, views, pointer-mode guards, …) come with it.

These are the wwr extension modules **used in place, from the submodule**, not
copies: this tree does not duplicate WarpWraps. To reach for another one
(`fft`, `sparse`, …), link `wwr::extension::<name>` here and add an alias bound
to `gpu::CreatePolicy` / `gpu::DestroyPolicy` the same way.

```cpp
auto r = gpu::capture([] {
  auto dev = std::make_shared<gpu::Device>(0);
  gpu::Buffers::device<float> a(n * n, dev);
  gpu::BlasHandle blas(dev);
  // ...
});
```

Link it as `free_breakfast::core::gpu`.
