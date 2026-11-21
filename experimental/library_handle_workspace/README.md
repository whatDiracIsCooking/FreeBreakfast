# `experimental/library_handle_workspace/`

Does a vendor solver or BLAS handle take device memory behind a caller-sized
workspace, and what does it do with a workspace that is too small? This is
the question the Rust redesign's memory budget turns on
(`.temp/rust-redesign/rustflow_redesign.html`, "BLAS and solver
workspace"). The budget sizes each device's library workspace once, as the
max of what every step declares. If a library can still allocate during a
run, the budget's numbers are false. If it silently accepts a buffer that is
too small, a call `mem_budget` never declared corrupts memory.

```sh
cmake --build build-experimental --target free_breakfast_library_handle_workspace
cmake --build build-hip-experimental --target free_breakfast_library_handle_workspace
experimental/library_handle_workspace/run_all.sh cuda
experimental/library_handle_workspace/run_all.sh hip
```

The probe goes through wwr's raw `wwrsolverDn*` and `wwrblas*` names, the
path the tree takes. On HIP that is **hipSOLVER's cuSOLVER-style API and
hipBLAS** over rocSOLVER and rocBLAS, not bare rocSOLVER. Each case is its own
process, so the context and handle are fresh. The card's free memory is read,
after a device sync, around each step: creating the handle, the size query,
allocating the work buffer, and two calls. A *guarded* case passes the same
`lwork` inside a buffer twice the queried size, filled with a sentinel, and
reads back the highest element the call wrote.

Solvers at n = 2048 in fp64: `syevd` (vectors, lower), `potrf` (lower),
`gesvd` (`jobu = jobvt = 'N'`). `dgemm` at 4096³ and at 256 × 256 × 65536
(a split-K shape).

## Results, 2026-11-21

RTX 3080 (CUDA 13), RX 9060 XT (gfx1200, ROCm), one process per case, the
card held by `with_gpu_card`. "+" is memory taken; MiB.

### Solver: workspace too small (half the queried `lwork`)

| | cuSOLVER | hipSOLVER over rocSOLVER |
|---|---|---|
| `syevd`, tight buffer | **illegal memory access**, context lost | `ALLOC_FAILED`, nothing written |
| `syevd`, guarded | **`SUCCESS`, wrote 130.24 of the 65.12 passed** (the full query) | `ALLOC_FAILED`, nothing written |
| `potrf` | `SUCCESS` (its query is 0.02 MiB and the write fits inside the allocation's rounding) | `ALLOC_FAILED`, nothing written |
| `gesvd` | `SUCCESS`: queries 100.56, touches 2.05 | `SUCCESS`: queries 18.01, touches 1.03 |

**cuSOLVER does not check `lwork` against what it needs.** `syevd` writes its
whole requirement, whatever length it was given, and reports success. With a
tight buffer that is a crash. With anything mapped past the end it is silent
corruption of a neighbour. hipSOLVER does check, against what the call
actually needs (so `gesvd`'s half-buffer is enough), and refuses before
writing anything.

### Solver: `work = nullptr`, `lwork = 0`

| | cuSOLVER | hipSOLVER over rocSOLVER |
|---|---|---|
| `syevd` | `INVALID_VALUE` | **process abort**: "rocBLAS internal error: Cannot reallocate device memory while it is already in use" |
| `potrf` | illegal memory access | **`SUCCESS`, +34 MiB taken by the call** |
| `gesvd` | illegal memory access | **`SUCCESS`, +34 MiB taken by the call** |

**This is the only silent growth found, and only on HIP.** With no buffer,
hipSOLVER leaves rocBLAS to manage the handle's memory, and the call
allocates in the middle of the run, outside any budget.

### Solver: the exact workspace

Every call succeeds with results equal to the reference, and every guarded
exact call touches exactly what was queried (`syevd`), or less (`gesvd`,
`potrf`). A second call takes nothing on either backend. A handle takes 14 MiB
on CUDA and 60 MiB on HIP when it is created. On HIP the *first* call also
takes 2–10 MiB (`potrf`, `gesvd` 2; `syevd` 10). That happens even with an
exact buffer, so it is not workspace. It looks like code objects loaded at
first launch, but this probe does not separate the two.

### BLAS (`dgemm`)

| | cuBLAS | hipBLAS over rocBLAS |
|---|---|---|
| Handle creation | +10 to +12 | +58 |
| First call, any mode | 0 | +10 (the same in every mode, so not workspace; likely code objects) |
| Second call | 0 | 0 |
| `SetWorkspace(32 MiB)` | the buffer is the only change | net 0: **the handle frees its own 32 MiB pool** |
| `SetWorkspace(256 B)` | works; the split-K shape picks another algorithm (7.4e-12 from default) | works, the same result; −30 to −32 (the pool freed) |
| `SetWorkspace(nullptr, 0)` after a buffer | works, as the 256 B case | works, nothing taken |

Neither BLAS took memory on a call in any mode. A tiny workspace degrades the
algorithm rather than failing.

## What it means for the redesign

1. **On HIP, the guarantee is configuration.** Pass every hipSOLVER call a
   non-null buffer, never `lwork = 0` with `nullptr`. That includes a routine
   whose query returns 0: hand it a minimal non-null region. Then an
   undersized call is a clean `ALLOC_FAILED` and nothing grows. Setting a
   workspace on the hipBLAS handle also gives back rocBLAS's default 32 MiB
   pool.
2. **On CUDA, the library guarantees nothing.** cuSOLVER never grows, but it
   also never checks: a call `mem_budget` did not declare, or declared at a
   smaller order, writes past the workspace and reports success. So the
   declaration has to be right by construction (a call token minted by
   `Budget` that the call goes through), or checked by the wrapper before the
   call. This is the opposite of the expectation going in, that cuSOLVER
   would reject a short `lwork`.
3. The handles' own memory (10–14 MiB on CUDA, 58–60 MiB on HIP) and the HIP
   first-launch growth (2–10 MiB) are outside any workspace. The design's
   compliance statement already excludes handles and loaded modules. Its
   margin constant has to cover them.

Not covered: other solver routines (`getrf` takes no `lwork` at all, so it
cannot check one), rocBLAS-backed hipBLASLt, more than one handle per
device, or whether the first-call growth on HIP is code objects.
