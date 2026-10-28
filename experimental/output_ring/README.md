# Output ring (experimental)

A demo of writing device arrays to `.npy` through a **fixed ring of pinned
staging slots**, so several files are written at once and no write allocates
pinned memory. Built only under `FREE_BREAKFAST_BUILD_EXPERIMENTAL` (the
`experimental` preset).

## Why a ring

The download it replaces (`write_overlap_npy` in `src/pipeline/overlap/`) is
a pageable `std::vector`, `wwrMemcpyAsync`, `res.sync()`, `write_npy`. That
has two problems once there is more than one array, or a big one:

- **A copy into pageable memory does not overlap anything.** The runtime
  stages it through its own pinned buffer, and the call does not return
  until the copy is done, so file writes and the next array's copy run one
  after another.
- **A pinned buffer per write is worse.** Page-locking is a driver call whose
  cost grows with the size, and freeing pinned memory synchronizes the
  device. A staging buffer sized to each array would also leave a big dump's
  worth pinned if it were kept.

So the ring is allocated **once**, with a fixed size (4 × 32 MiB by default),
and an array of any size streams through it in slot-sized chunks.

## Module

`free_breakfast.experimental.output_ring`, target
`free_breakfast::experimental::output_ring`:

```cpp
import free_breakfast.experimental.output_ring;

fb::OutputRing ring{res.dev_idx(), node.executor()};    // OutputRingConfig{} = 4 x 32 MiB
FBF_TRY(ring.write_npy(res, dir / "h.npy", {n, n}, {h.data(), n * n}));
FBF_TRY(ring.write_npy(res, dir / "t.npy", {n, n}, {t.data(), n * n}));
FBF_TRY(ring.write_npy(res, dir / "v.npy", {n, n}, {v.data(), n * n}));
FBF_TRY(ring.wait());                                    // all three on disk
```

Per chunk, on the thread driving `res`: take a free slot, queue the
device-to-host copy into it on `res.stream()`, record the slot's `Event`, and
hand the slot to a task on the executor. That task waits on the event, writes
the chunk at its own offset in the file, and frees the slot.

- **Chunks carry their offsets**, so they land in any order: there is no
  per-file chain, and chunks of different files share the slots.
- **`write_npy` returns once its chunks are queued**, not written. `wait()`
  blocks until everything is on disk and returns the first error since the
  last `wait()`.
- **The header is written synchronously** first, truncating the file, so it
  exists before any chunk opens it. It is io.npy's own `npy_header` — the
  piece `npy_to_bytes` was refactored into — so a streamed file is
  byte-identical to `write_npy`'s.
- **A slot has one owner at a time.** The driving thread takes it, copies and
  records; the writer task waits on its event and frees it; only then can the
  driving thread take it again. No event is touched from two threads at once,
  so `DeviceResources`' one-driving-thread rule holds.
- **Blocking inside the executor co-runs.** Taking a slot, and `wait()`, block
  until a writer frees one. Called from one of the executor's own workers, as
  a pipeline step is, they `corun_until` instead, so the writers progress
  even with one worker.

## Where it would go

Not decided. It knows npy, so it is I/O, and `core/resources` cannot own it
(`core` cannot depend on `io`). The candidate discussed is the pipeline: one
ring per device for the run, steps publish outputs, and the pipeline writes
the ones the config asks for.

## Tests and bench

`test/experimental/output_ring_gpu.cpp` streams arrays through 37-double
slots, so each is many chunks, and compares every file byte for byte with
`npy_to_bytes` of the same values. It also drives the ring from inside a
one-worker executor (the `corun_until` path), writes an empty array, and
checks the shape-mismatch and unwritable-path errors:

```sh
devtools/cpp-tier.sh --preset experimental
ctest --test-dir build-experimental -L experimental -R OutputRing
```

`free_breakfast_output_ring_bench [max_n] [k] [reps]` times k arrays of n × n,
the serial download against the ring, wall clock to the last file closed.

## Findings

Measured 2026-10-28 on the RTX 3080 (CUDA 13.0), both of its GPU-lock slots
held: `free_breakfast_output_ring_bench 8192 3 5`, three n × n arrays per
run, default ring (4 × 32 MiB), files in the container's `/tmp`:

| n | MiB/array | serial ms | ring ms | speedup |
|---|---|---|---|---|
| 512  | 2   | 10.9   | 5.0    | 2.2× |
| 1024 | 8   | 62.7   | 15.2   | 4.1× |
| 2048 | 32  | 299.9  | 68.9   | 4.4× |
| 4096 | 128 | 1445.7 | 360.0  | 4.0× |
| 8192 | 512 | 7030.9 | 2124.5 | 3.3× |

Not all of that is the overlap. The serial path also pays for
`npy_to_bytes`, which copies the whole payload into a `std::string` before
writing it, and the ring never builds that copy. Neither path is near PCIe
bandwidth: at n = 8192 the ring moves 1.5 GiB in 2.1 s, so the file writes
set the pace, not the copies.
