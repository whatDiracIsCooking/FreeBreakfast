# experimental/rust/handle_ring — the handle ring's two measurements

The v0.0.4 prototype's page (removed; at `b0935805`), "A ring of library handles, and no library
stream", proposed that every library call run on the calling step's own
stream. A handle set (BLAS + solver) belongs to one step for the length of
its `run`, and steps share one only if the DAG orders them. Two measurements
pick between that ring and a handle per step, and check that the ring's move
is safe:

1. **What a second, third and fourth handle cost in one process**, BLAS and
   solver (`handles-blas 4`, `handles-solver 4`).
2. **One idle handle moved between two streams, a workspace per stream**
   (`ring <dgemm_square|dgemm_splitk|dtrsm> <order>`): does the call use the workspace it was given
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

Split-K `dgemm` reads the workspace on cuBLAS, through its algorithm choice.
`dtrsm` (L⁻¹B in place, L lower 4096 × 4096, B 4096 × 4096: the DF fit's
shape) reads it on hipBLAS. There the call writes its workspace, and the
default handle's pool grows by 12 MiB for it. rocBLAS's `gemm` uses no
workspace on either shape, so it reads nothing on HIP. cuBLAS's `trsm` takes
none either, so `dtrsm` reads nothing on CUDA.

| | cuBLAS, split-K `dgemm` | hipBLAS, `dtrsm` |
|---|---|---|
| `stream_then_ws` (32 MiB) | | **wrote 8 of the 32 MiB given**, took 0 |
| `ws_then_stream` (32 MiB) | | **the same**: `SetStream` kept the workspace |
| `stream_only` (32 MiB) | | **the same**: the workspace set before the first `SetStream` was still used on every call |
| `stream_then_ws_tiny` | **equals the tiny reference**: the call used the workspace it was given | **+8 MiB taken by the call**; the next `SetWorkspace` frees it |
| `ws_then_stream_tiny` | **equals the default**: `SetStream` threw the workspace away | +8 MiB taken by the call, likewise |
| `stream_only_tiny` | equals the default: the workspace was set before the first `SetStream` and lost to it | +8 MiB on the first call, kept by the handle after it |
| Results | equal to the default or the tiny reference, as above | bitwise equal to the default in every mode |
| Memory taken by a move | 0 | 0 after the first `SetWorkspace`, which frees the 32 MiB pool |
| One `SetStream` + `SetWorkspace` before a call | 0.3–3.9 µs | 3–8 µs; ~150–200 µs when it frees memory |
| Idle switch, mean of 1000 | 0.02 µs (stream alone 0.01) | 0.15–0.26 µs (stream alone 0.11–0.19) |
| Solver `SetStream`, idle | 0.02 µs; moved handle's `syevd` bitwise equal, 0 taken | 0.12 µs; the same |

- **The order matters on cuBLAS, as its documentation says.** `SetStream`
  resets the handle to cuBLAS's own default workspace. A `SetWorkspace` made
  before it is silently lost, and the call then runs on memory outside the
  budget.
- **On hipBLAS the order does not matter.** `SetStream` keeps the caller's
  workspace. So `SetStream` then `SetWorkspace` is correct on both backends.
- **On hipBLAS, a workspace that is too small is not an error: the call
  grows.** With 256 B, each `dtrsm` took 8 MiB of its own, outside the budget,
  and gave the same result. With 32 MiB it used 8 MiB inside the workspace.
  cuBLAS does the opposite: a small workspace only changes the algorithm. So
  on HIP, a step's BLAS workspace has to cover what its calls need. Not
  measured:
  - the threshold between 256 B and 8 MiB;
  - how the need scales with the shape;
  - whether the growth happens in hipBLAS or in rocBLAS.
- **The move costs nothing that matters.** No memory, and well under a
  microsecond while idle. The ~150 µs on HIP is a free (the pool's, or the
  8 MiB a tiny-workspace call grew), and it happens once per handle at budget
  time when the workspace is big enough.

An unexplained −6.3 MiB showed up on one cuBLAS square call
(`ws_then_stream` call 2): memory was freed, not taken. Every other reading
during a call was 0.

## What it means for the ring

**Outcome, 2026-11-24:** neither the ring nor a handle per step. The decision
is one handle set per card under one lock, held for a step's whole run, with a
fixed 128 MiB BLAS workspace (`rust/docs/gpu_layer.md`, "The v0.0.4 runtime", "One handle set per
GPU, under its lock"). The overlap the ring would buy, between library-using
steps the DAG leaves unordered on one card, is worth little when heavy steps
fill the card on their own. The points below still apply to that design: the
cost per set, the binding order, and HIP's growth past a small workspace.

1. **The ring wins over a handle per step**, by the pipeline's length over
   its width, at about 20 MiB per handle set on CUDA and 52 on HIP. For
   example, ten library-using steps on one card with a ring of two: on HIP,
   520 MiB with a handle per step against 104 with the ring. On CUDA, 200
   against 40.
2. **The wrapper's preparation is safe as proposed**, on the condition that
   it calls `SetStream` and then `SetWorkspace`. That order is right on both
   backends. The reverse order is a silent budget leak on CUDA, not an error,
   and no getter reads the workspace back, so the two calls belong in one
   function that nothing else can reorder.
3. **On HIP the BLAS workspace is a real declaration.** A call whose need
   exceeds it grows outside the budget, with no error. So a step's declared
   BLAS workspace must cover its largest call. Where that number comes from
   is open: rocBLAS has a size query (`rocblas_start_device_memory_size_query`),
   but it may not be reachable through hipBLAS and `wwr-sys` (to check). The
   alternative is a measured allowance per routine and shape. The allocator
   record (`rust/steps/tests/chain_vram.rs`) would catch a miss only
   after the fact.
4. **The handles row of the budget table** should carry per-backend numbers:
   20 MiB per set on CUDA and 52 on HIP (after the pool is given back), plus a
   once-per-process first-call allowance (10 on HIP, up to ~15 on CUDA).
