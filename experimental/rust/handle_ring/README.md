# experimental/rust/handle_ring — the handle ring's two measurements

`docs/rust-redesign.html`, "A ring of library handles, and no library
stream", proposes that every library call run on the calling step's own
stream. A handle set (BLAS + solver) belongs to one step for the length of
its `run`, and steps share one only if the DAG orders them. Two measurements
pick between that ring and a handle per step, and check that the ring's move
is safe:

1. **What a second, third and fourth handle cost in one process**, BLAS and
   solver (`handles-blas 4`, `handles-solver 4`).
2. **One idle handle moved between two streams, a workspace per stream**
   (`ring <routine> <order>`): does the call use the workspace it was given
   after `SetStream` then `SetWorkspace`; is anything taken or grown by the
   move; what do the two calls cost.

It goes through `wwr-sys`, the path the Rust tree takes (on HIP, hipBLAS and
hipSOLVER over rocBLAS and rocSOLVER). It supersedes the C++ cases once
drafted for `experimental/library_handle_workspace`; that probe's README has
the single-handle results this one builds on.

```bash
# from rust/: the C ABI libraries, per backend
cmake --preset cuda && cmake --build build-cuda --target wwr.capi.runtime_api wwr.capi.blas wwr.capi.solver
cmake --preset hip  && cmake --build build-hip  --target wwr.capi.runtime_api wwr.capi.blas wwr.capi.solver
# from the checkout root: build, then every case under the card's lock
experimental/rust/handle_ring/run_all.sh cuda
experimental/rust/handle_ring/run_all.sh hip
```

`main.rs`'s header says what each case does and prints. The card's free memory
is read after a device sync around every step. One process per case. Nothing
in ctest or either gate runs it.

### Why the `_tiny` modes

cuBLAS writes none of a 32 MiB sentinel-filled workspace on either shape
(`touched_mib` is 0 everywhere), so the sentinel cannot say which workspace a
call saw. With a 256-byte workspace, cuBLAS picks another split-K algorithm,
1.72e-12 from the default. So on `dgemm_splitk`, a `_tiny` call that equals
the tiny reference saw the tiny workspace, and one that equals the default
did not. The square shape picks the same algorithm either way, so it tells
nothing.

## Results, 2026-11-24

RTX 3080 (CUDA 13) and RX 9060 XT (gfx1200, ROCm). Each card was held by
`with_gpu_card`. Figures are in MiB, and "+" means memory taken.

### 1. Handles in one process

| | cuBLAS | cuSOLVER | hipBLAS | hipSOLVER |
|---|---|---|---|---|
| First handle | +12 | +14 | +58 | +58 |
| Each further handle | **+8** | **+12** | **+58** | **+58** |
| `SetWorkspace(32 MiB)` | 0 | | −32 (rocBLAS's pool freed) | |
| First call, handle 0 | 0 | 0 or +14.4 (two runs) | +10 | +10 |
| First call, handles 1–3 | 0 | 0 | 0 | **−32** |
| Net per further handle | 8 | 12 | 26 | 26 |

- **A further handle is not nearly free on either backend.** On CUDA one
  handle set costs 20 MiB. On HIP it costs 116 MiB at creation and 52 MiB
  once rocBLAS's default pool is given back.
- **The pool is given back differently by the two libraries.** A hipBLAS
  handle gives it back at `SetWorkspace`. A hipSOLVER handle gives it back at
  its first call with a caller's work buffer. That is my reading of the −32,
  not checked.
- **The first-call growth is once per process, not per handle.** On HIP it is
  +10, probably code objects. On cuSOLVER it was +14.4 in one run and 0 in the
  other, likely CUDA's lazy module loading.

### 2. One handle moved between two streams

| | cuBLAS | hipBLAS |
|---|---|---|
| Result in every order and shape | equal to the default handle's, or to the tiny reference's | bitwise equal to the default |
| `stream_then_ws_tiny` (split-K) | **equals the tiny reference**: the call used the workspace it was given | rocBLAS `gemm` uses none (as before), so no signature |
| `ws_then_stream_tiny` (split-K) | **equals the default**: `SetStream` threw the workspace away | no signature |
| `stream_only_tiny` (split-K) | equals the default: the workspace was set before the first `SetStream` and lost to it | no signature |
| Memory taken by a move | 0 | 0 after the first `SetWorkspace`, which frees the 32 MiB pool |
| One `SetStream` + `SetWorkspace` before a call | 0.3–3.9 µs | 3–7 µs; the first ~150–200 µs (the pool's free) |
| Idle switch, mean of 1000 | 0.02 µs (stream alone 0.01) | 0.15 µs (stream alone 0.11) |
| Solver `SetStream`, idle | 0.02 µs; moved handle's `syevd` bitwise equal, 0 taken | 0.12 µs; the same |

- **The order is load-bearing on cuBLAS, as its documentation says.**
  `SetStream` resets the handle to cuBLAS's own default workspace. A
  `SetWorkspace` before it is silently lost, and the call then runs on memory
  outside the budget. The wrapper must call `SetStream` first, then
  `SetWorkspace`, every time.
- **HIP cannot show the order here.** rocBLAS's `gemm` takes no workspace, so
  nothing distinguishes the two orders. Nothing re-grows after a move in
  either order: once the pool is freed, a later `SetStream` does not bring it
  back.
- **The move costs nothing that matters.** No memory, and well under a
  microsecond while idle. The one-time ~150 µs on HIP is the pool's free, and
  it would happen once per handle at budget time.

An unexplained −6.3 MiB showed up on one cuBLAS square call
(`ws_then_stream` call 2): memory was freed, not taken. Every other reading
during a call was 0.

## What it means for the ring

1. **The ring wins over a handle per step**, by the pipeline's length over
   its width, at about 20 MiB per handle set on CUDA and 52 on HIP. For
   example, ten library-using steps on one card with a ring of two: on HIP,
   520 MiB with a handle per step against 104 with the ring. On CUDA, 200
   against 40.
2. **The wrapper's preparation is safe as proposed**, on the condition that
   it calls `SetStream` and then `SetWorkspace`. The reverse order is a
   silent budget leak on CUDA, not an error, and no getter reads the
   workspace back, so the two calls belong in one function that nothing
   else can reorder.
3. **The handles row of the budget table** should carry per-backend numbers:
   20 MiB per set on CUDA and 52 on HIP (after the pool is given back), plus a
   once-per-process first-call allowance (10 on HIP, up to ~15 on CUDA).
