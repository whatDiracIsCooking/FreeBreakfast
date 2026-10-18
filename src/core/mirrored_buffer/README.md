# Mirrored buffer

`MirroredBuffer<T>`: a `Buffers::device<T>` and a `Buffers::pinned<T>` of the
same size, owned together, with the copies between them.

## Module

`free_breakfast.core.mirrored_buffer` — target
`free_breakfast::core::mirrored_buffer`, also re-exported by
`free_breakfast.core`.

A module of its own rather than a partition of `free_breakfast.core.wwr`: it is
built from `Buffers`, which that module's primary interface defines, and a
partition cannot import its primary interface.

## Usage

```cpp
import free_breakfast.core;

auto device = std::make_shared<free_breakfast::Device>(0);
free_breakfast::MirroredBuffer<double> m{n, device};

std::ranges::copy(values, m.pinned().data());
FBF_GPU_MUST(m.to_device());        // enqueue h2d on device->stream()
// ... launch on m.device() ...
FBF_GPU_MUST(m.to_pinned_sync());   // enqueue d2h, then wait for it
```

| Call | Direction | Returns after |
| --- | --- | --- |
| `to_device()` | pinned → device | enqueueing |
| `to_pinned()` | device → pinned | enqueueing |
| `to_device_sync()` | pinned → device | the work stream has drained |
| `to_pinned_sync()` | device → pinned | the work stream has drained |

All four queue on the `Device`'s work stream, the stream the device side was
allocated and zeroed on, and return the runtime's error code for the caller to
`FBF_GPU_MUST` or `gpu_status`.

## Things to know

- **Async means the pinned side is busy.** After `to_pinned()` it is not safe
  to read, and after `to_device()` it is not safe to overwrite, until the stream
  has been synchronized. Use the `_sync` variant unless you overlap the copy
  with other work.
- **Destruction synchronizes the stream.** The pinned free is not
  stream-ordered, so the destructor (and move assignment, for the side it
  overwrites) waits for any copy in flight first. It blocks, once.
- **Size is fixed.** Both sides are allocated once, at construction, with the
  same element count.
- **It costs twice the memory, and the host half is pinned.** Use it for data
  that really does move back and forth, not as a general container for large
  data that should stay on the device.
