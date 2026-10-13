# `free_breakfast.core.gpu_policy`

What a failed GPU call inside a wwr RAII wrapper does — decided once, at
configure time, for the whole tree.

WarpWraps' extension layer forces no error policy: every wrapper takes its
policies as template arguments (`wwr.extension.common`'s `error_policy`
concepts). This module supplies ours, and a switch between two behaviours:

| `FREE_BREAKFAST_GPU_ERROR_POLICY` | Create / alloc / device-select slots | Destroy / free slots |
| --- | --- | --- |
| `ABORT` | `AbortPolicy` — log critical, `std::abort()` at the failing call | `AbortPolicy` |
| `RECORD` | `RecordPolicy` — record for the enclosing `gpu::capture` | `LogPolicy` — log at error |
| *(empty, default)* | `ABORT` for `CMAKE_BUILD_TYPE=Debug`, `RECORD` otherwise | |

The resolved mode is printed at configure (`-- GPU error policy: …`) and is
readable in code as `gpu::aborts_on_error`.

## Using it

Name the switched aliases — `gpu::CreatePolicy<E>`, `gpu::DestroyPolicy<E>`,
`gpu::PolicyMap` — rather than a concrete policy. In practice you rarely do even
that: `free_breakfast.core.gpu` has the wrappers already bound (`gpu::Device`,
`gpu::Buffers`, `gpu::BlasHandle`, …).

Construct inside `gpu::capture`, which turns a recorded failure into a `Result`:

```cpp
import free_breakfast.core;
namespace gpu = free_breakfast::gpu;

free_breakfast::Result<gpu::Buffers::device<float>> make_buffer(std::size_t n) {
  auto dev = FBF_TRY(gpu::capture([] { return std::make_shared<gpu::Device>(0); }));
  return gpu::capture([&] { return gpu::Buffers::device<float>(n, dev); });
}
```

Under `ABORT` the same code compiles and never sees an error value — the abort
happens first, at the call that failed — so callers are written once.

## Behaviour worth knowing

- **Write the construction inside the lambda.** A wrapper's defaulted
  `std::source_location` is then your line, and that is what the returned
  `Error::origin` points at.
- **First error wins** within a capture: later failures are usually knock-on
  effects of the first.
- **Captures nest.** An inner capture neither reports nor wipes an error its
  outer one has pending — also when the inner lambda throws.
- **A `RECORD` failure outside any capture is logged at error**, since nobody
  is there to return it to. It is also kept for `gpu::take_error()`.
- **Destroy/free failures are logged, not returned** in `RECORD`: they happen in
  a destructor, after any capture that could have returned them has finished.
- A failed create leaves the wrapper empty — `valid()` is false for a handle,
  `num_elements()` is 0 for a buffer — which is wwr's record-and-continue path.

## Why it is shaped like this

- **Per-thread recording, not a stateful policy.** The policies are empty so
  they fit every wwr slot and collapse to zero size (`policy_slot`), and because
  `kit::DeviceHandle` default-constructs its device policy — a policy holding a
  pointer to "its" error could not be threaded through there. The error belongs
  to whichever capture is running on the thread that made the call.
- **The switch is a compile definition on this target only**, read once into
  the `constexpr` `gpu::aborts_on_error`. Every importer gets that value through
  the BMI, so no TU can be built against the other mode.
- **Not `NDEBUG`.** That is `assert`'s switch, invisible in presets and the
  configure log; this one is a cache variable, so a Release build can opt into
  `ABORT` to chase a failure.

Link it as `free_breakfast::core::gpu_policy`.
