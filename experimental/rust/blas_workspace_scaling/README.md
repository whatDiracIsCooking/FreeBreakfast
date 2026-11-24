# experimental/rust/blas_workspace_scaling — what a `dtrsm` needs from its handle

`docs/rust-redesign.html`, "One handle set per card, under one lock", gives
each card's BLAS handle a fixed 128 MiB workspace.
`experimental/rust/handle_ring` had one reading behind that number: a
4096 × 4096 `dtrsm` on hipBLAS writes 8 MiB of its workspace, and takes 8 MiB
of its own when the workspace is too small. This probe asks how that need
grows with the shape, on both backends: nine readings up to 128 MiB of B,
then nine more at 2, 4 and 8 GiB.

It goes through `wwr-sys` (on HIP, hipBLAS over rocBLAS).

```bash
# from rust/: the C ABI libraries, per backend
cmake --preset cuda && cmake --build build-cuda --target wwr.capi.runtime_api wwr.capi.blas
cmake --preset hip  && cmake --build build-hip  --target wwr.capi.runtime_api wwr.capi.blas
# from the checkout root: build, then every shape under the card's lock
experimental/rust/blas_workspace_scaling/run_all.sh cuda
experimental/rust/blas_workspace_scaling/run_all.sh hip
experimental/rust/blas_workspace_scaling/run_all.sh hip "right 4096 4100"
# a fourth word sets the large workspace in MiB, for a B three copies of
# which do not fit the card
experimental/rust/blas_workspace_scaling/run_all.sh hip "right 262144 4096 1024"
```

One process per shape. B is m × n, and L is lower, of order k = m (`left`,
B ← L⁻¹B) or k = n (`right`, B ← BL⁻¹). Each process makes two calls, each on
a fresh handle, on the same B. `main.rs`'s header says what each prints:

- **used**: the handle has a 0xFF-filled workspace larger than three copies
  of B. The highest byte the call changed is what it used.
- **own**: the handle has a 256-byte workspace. The card's free memory lost
  over the call is what the library took by itself.

Nothing in ctest or either gate runs it.

## Results, 2026-11-24

RTX 3080 (CUDA 13) and RX 9060 XT (gfx1200, ROCm 10.0.0: rocBLAS 5.6.0,
hipBLAS 3.6), each card held by `with_gpu_card`. The formulas are that
release's.

### hipBLAS

| Side | m × n | k | B (MiB) | Used (bytes) | Used (MiB) | Own (MiB) |
|---|---|---|---|---|---|---|
| left | 4096 × 4096 | 4096 | 128 | 8,388,608 | 8.0 | 8 |
| left | 4096 × 1024 | 4096 | 32 | 5,242,880 | 5.0 | 6 |
| left | 16384 × 1024 | 16384 | 128 | 20,971,520 | 20.0 | 20 |
| left | 4100 × 4096 | 4100 | 128.1 | 138,546,208 | **132.1** | 134 |
| left | 16400 × 1024 | 16400 | 128.1 | 151,141,504 | **144.1** | 146 |
| right | 4096 × 4100 | 4100 | 128.1 | 138,546,208 | **132.1** | 134 |

- **When k is not a multiple of 128, the call needs a copy of B.** The three
  such readings are 8·m·n + 1024·k bytes to within 1 kB: all of B, plus 128
  doubles per row of L. The side does not matter: left 4100 × 4096 and right
  4096 × 4100 use the same number of bytes.
- **When k is a multiple of 128, the need is small but still grows with B.**
  The three readings equal 1024·k + m·n/4 bytes exactly: 128 doubles per row
  of L, plus 1/32 of B. The large shapes below hold the same form.
- **"Own" agrees with "used".** With a 256-byte workspace the library took
  the same amount by itself, rounded up to 2 MiB, with no error. That is the
  growth outside the budget `handle_ring` found, now at every shape.
- **With the large workspace the call still lost 12 to 14 MiB of free
  memory.** That is the once-per-process first-call growth `handle_ring`
  measured at +10, not workspace.

### hipBLAS, B of 2, 4 and 8 GiB

Right side, so L stays small (k = n) while B grows with m.

| m × n | k | B (MiB) | Used (bytes) | Used (MiB) | Own (MiB) |
|---|---|---|---|---|---|
| 65536 × 4096 | 4096 | 2048 | 71,303,168 | 68.0 | 68 |
| 131072 × 4096 | 4096 | 4096 | 138,412,032 | 132.0 | 132 |
| 262144 × 4096 | 4096 | 8192 | 272,629,760 | 260.0 | 260 |
| 65536 × 4100 | 4100 | 2050 | 2,153,778,208 | **2054.0** | 2056 |
| 131072 × 4100 | 4100 | 4100 | 4,303,359,008 | **4104.0** | 4106 |
| 262144 × 4100 | 4100 | 8200 | | `ALLOC_FAILED` | |

- **Both forms hold, linear in B, over a factor of 64 in its size.** A
  multiple of 128: 1024·k + m·n/4 bytes, exact on all three, so B/32 plus
  4 MiB here. Not a multiple: 8·m·n + 1024·k bytes less 992, on both, so B
  plus 4 MiB.
- **A need the card cannot meet is an error, not a crash.** At 8.2 GiB of B
  with k = 4100 the call wants 8.2 GiB more, the 16 GiB card has 7.7 free,
  and the call returns `ALLOC_FAILED` (given a 64 MiB workspace, so the
  library tried to take its own). That is the one case where growth outside
  the caller's workspace is not silent.

### cuBLAS

| Side | m × n | B (MiB) | Used (bytes) | Free memory lost over a call (MiB) |
|---|---|---|---|---|
| left | 4096 × 4096 | 128 | 0 | 0, 0 |
| left | 4100 × 4096 | 128 | 0 | 0, 0 |
| left | 16400 × 1024 | 128 | 512 | 0, −0.6 |
| right | 65536 × 4100 | 2050 | 0 | 0, 0 |
| right | 131072 × 4100 | 4100 | 0 | +29.3, −4.8 |
| right | 262144 × 4100 | 8200 | 0 | 0, +26.6 |

cuBLAS's `dtrsm` uses none of the handle's workspace at any shape. Over the
two calls of each process (a 448 or 64 MiB workspace, then 256 bytes) the
card's free memory moved by at most 30 MiB, in either direction and with no
relation to the size of B: something cuBLAS keeps of its own, not explained,
and the 512 bytes neither.

## The other routines, hipBLAS and hipSOLVER

`routine-sweep <routine> <n>` (`src/sweep.rs`, run by `sweep.sh hip "routine
n" ...`) asks the same of the other calls the Rust tree makes, and of those
the ports will bring. One routine at one order per process, 40 readings: each
routine at an order that is a multiple of 128 and one that is not, and a
third only where the first two showed a need. HIP below; the same sweep on
CUDA follows it.

### BLAS: what the call uses of the handle's workspace

Matrices are n × n at n = 4096 and 4100 (128 MiB in double).

| Routine | n = 4096 | n = 4100 | Trend |
|---|---|---|---|
| `dtrsm`, right, transposed (the DF fit's form) | 8.0 MiB | 132.3 MiB | as `dtrsm` above: the side and the transpose change nothing |
| `strsm` | 4.0 MiB | 66.1 MiB | the same formulas at 4 bytes an element |
| `ztrsm` | 16.0 MiB | 264.5 MiB | the same at 16 bytes an element |
| `dtrmm` | 0 | 0 | none |
| `dtrsv` | 4 B | 4 B | none |
| `dgemm_strided_batched` (16 of order n/4) | 0 | 0 | none |
| `dsyrk` | 0 | 0 | none |
| `dgemv` | 0 | 0 | none |
| `dgeam` | 0 | 0 | none |
| `ddgmm` | 0 | 0 | none |
| `ddot`, `dnrm2` (vectors of n² elements) | 131,080 B | 131,344 B | 8 bytes per 1024 elements (rounded up), plus 8 |

- **`trsm` is the only BLAS call here whose need grows with its operands.**
  For an element of s bytes: s·(m·n + 128·k) when k is not a multiple of
  128, and s·(m·n/32 + 128·k) when it is.
- **The reductions need a little, and fail without it.** `ddot` and `dnrm2`
  keep one partial result per 1024 elements: 128 KiB for 16.8 million
  elements, so 8 MB per billion. Given a 256-byte workspace they return
  `ALLOC_FAILED` and take nothing, where `trsm` grows silently.
- **Everything else took nothing** at either order.

### Solvers: the size query, and what the call takes beyond it

Each call is handed a work buffer of exactly its size query and made twice.
"Beyond" is the free memory lost over the first call; the second lost 0 in
every case.

| Routine | Size query (doubles) | Beyond the work buffer | Trend |
|---|---|---|---|
| `dsyevd`, n = 2048, 2100, 4100 | 13.7 M, 14.4 M, 54.8 M (105, 110, 418 MiB) | +10 MiB | the query is 3.26 n²; no step at an odd order |
| `dpotrf`, n = 4096, 4100 | 128 | +2 MiB | the query is constant |
| `dgetrf`, n = 4096, 4100 | 17.35 M, 17.37 M (132 MiB) | −4 MiB | the query is 1.03 n²: a copy of A |
| `dgesvd`, n = 1024, 1030, 2048 | 0.654 M, 0.661 M, 2.36 M (5, 5, 18 MiB) | +22 MiB | the query is about n²/2 + 128 n |
| `dpotrs`, n = nrhs = 4096, 4100, 8192 | takes no buffer | **+170, +174, +554 MiB** | B + 42 MiB |
| `dgetrs`, n = nrhs = 4096, 4100, 8192 | takes no buffer | **+168, +170, +556 MiB** | B + 42 MiB |

- **A call with a size query stays inside it.** `syevd`, `potrf`, `getrf`
  and `gesvd` took nothing that grows with n beyond their work buffer; the
  +2 to +22 MiB is the same at every order, so it is first-call growth.
- **`potrs` and `getrs` take a copy of the right-hand sides by themselves.**
  They have no work argument, so hipSOLVER's internal rocBLAS handle manages
  its own memory for the triangular solves inside: B's size plus about
  42 MiB, at multiples of 128 as well as not, and the handle keeps it (the
  second call takes nothing more). There is no way to hand them a buffer.

### The rest of the C++ tree's calls

`src/` also calls `iamax` (`recipes::norm`, `recipes::expm`), `asum`
(`recipes::norm`), `dotc` (`recipes::cg_unitary`), `axpy`, `scal`, `copy`,
and `heevd`/`heevdx` (`recipes::expm`, `heevdx` over the whole range on HIP).
The same two readings, on the same card and release, at n = 4096, 4100 and
8192 (vectors of n² elements: 16.8, 16.8 and 67.1 million) for BLAS, and
n = 2048, 2100 and 4100 for the solvers. Each process was run once; the
three readings that looked odd were run twice more and held to the others
(below).

| Routine | Used, n = 4096 / 4100 / 8192 | 256-byte workspace | Trend |
|---|---|---|---|
| `idamax`, `izamax` | 262,148 / 262,676 / 1,048,580 B | `ALLOC_FAILED`, nothing taken | 16 bytes per 1024 elements, plus 4: a value and an index per block |
| `dasum` | 131,080 / 131,344 / 524,296 B | `ALLOC_FAILED`, nothing taken | 8 bytes per 1024 elements, plus 8, as `ddot` |
| `zdotc` | 262,160 / 262,688 / 1,048,592 B | `ALLOC_FAILED`, nothing taken | 16 bytes per 1024 elements, plus 16: `ddot`'s at a complex element |
| `daxpy`, `dscal`, `dcopy` | 0 / 0 / 0 | success, nothing taken | none |

| Routine | Size query (complex doubles) | Beyond the work buffer, call 1 / call 2 | Trend |
|---|---|---|---|
| `zheevd`, n = 2048, 2100, 4100 | 9.50 M, 9.98 M, 37.9 M (145, 152, 579 MiB) | +24 / 0 MiB at every order | the query is about 2.26 n²; nothing beyond it |
| `zheevdx`, range all, n = 2048, 2100, 4100 | 13.7 M, 14.4 M, 54.7 M (209, 219, 835 MiB) | +24 / 0 MiB at every order | the query is about 3.26 n²; nothing beyond it |

- **None of them grows silently.** The reductions behave as `ddot` and
  `dnrm2` do: a little workspace, linear in the vector, and `ALLOC_FAILED`
  with nothing taken when the handle's is too small. The fixed 128 MiB
  covers `iamax` and `zdotc` to about 8.6 billion elements and `asum` to
  about 17 billion. `axpy`, `scal` and `copy` use none.
- **`heevd` and `heevdx` stay inside their size query**, like `syevd`. The
  +24 MiB is the same at every order and gone at the second call: first-call
  growth.
- **The `heevdx` query is 1.44 times `heevd`'s.** Not a growth, but a cost
  of the #999 workaround that `recipes::expm` pays on HIP.
- **Three readings of the first run did not reproduce.** `zdotc` at 4100
  lost 356 MiB of free memory over its large-workspace call, while touching
  0.25 MiB of it; `zheevd` lost 56 MiB at 2100, and gained 122 and 162 MiB
  at 4100. Rerun twice each, all six gave the table's numbers. `rocm-smi`
  listed no other process afterwards. The card lock serializes the gates'
  GPU work, not every process on the host, so something else moved the
  card's free memory during those calls.

## The same sweep on cuBLAS and cuSOLVER

RTX 3080, CUDA 13, every routine above at the same orders, 67 processes,
then the disturbed readings (below) twice more.

### BLAS

| Routine | Used, n = 4096 / 4100 (/ 8192) | 256-byte workspace | Trend |
|---|---|---|---|
| `dtrsm`, right, transposed | 2,048 / 2,080 B | success, nothing taken | none: the same at both orders |
| `strsm`, `ztrsm` | 0 / 0, and 0 / 516 B | success, nothing taken | none |
| `dtrmm`, `dgemm_strided_batched`, `dsyrk`, `dgemv`, `dgeam`, `ddgmm` | 0 | success, nothing taken | none |
| `dtrsv` | 8 B | success, nothing taken | none |
| `ddot`, `dnrm2`, `dasum` | 6,528 to 6,536 B at every length | success, nothing taken | none: a fixed size, 16.8 to 67 million elements |
| `idamax`, `izamax`, `zdotc` | 2,460, 828 and 17,408 B at every length | success, nothing taken | none |
| `daxpy`, `dscal`, `dcopy` | 0 | success, nothing taken | none |

- **No cuBLAS call's need grows with its operands.** The reductions keep a
  fixed few kilobytes of partial results, where rocBLAS keeps one per 1024
  elements.
- **A 256-byte workspace is never an error and never a growth.** Every call
  succeeded on it and took nothing measurable, the reductions included,
  though they use more than 256 bytes of a large one. Where they put their
  partials then is cuBLAS's own business, inside memory it holds from
  handle creation.

### Solvers

| Routine | Size query | Beyond the work buffer |
|---|---|---|
| `dsyevd`, n = 2048, 2100, 4100 | 17.1 M, 17.9 M, 67.8 M doubles (130, 137, 517 MiB) | 0 to 2 MiB |
| `zheevd`, `zheevdx` (range all), same orders | 17.1 M, 17.9 M, 67.8 M complex doubles (260, 274, 1034 MiB) | 0 to 2 MiB |
| `dpotrf`, n = 4096, 4100 | 8208, 8340 doubles | 0 |
| `dgetrf`, n = 4096, 4100, 8192 | 13,344, 13,376, 40,992 doubles | 0 |
| `dgesvd`, n = 1024, 1030, 2048 | 3.48 M, 3.52 M, 13.2 M doubles (27, 27, 101 MiB) | 0 |
| `dpotrs`, `dgetrs`, n = nrhs = 4096, 4100, 8192 | take no buffer | **0** |

- **`potrs` and `getrs` take nothing on CUDA.** Up to 512 MiB of
  right-hand sides, where hipSOLVER takes B plus 42 MiB.
- **Every call with a size query stays inside it**, as on HIP. The queries
  are larger than hipSOLVER's for the eigensolvers (about 4.07 n² against
  3.26 n² for `syevd`) and far smaller for the factorizations (`getrf`'s is
  a panel, not a copy of A).
- **`zheevd` and `zheevdx` ask for the same count as `dsyevd`**, in complex
  elements, so twice its bytes. `heevdx` over the whole range costs nothing
  more than `heevd` here.

### Readings that did not reproduce

The card was shared with another session's GPU tests during the run, so the
free memory moved under some calls. Each of these was rerun twice and gave
0 at least once on the measured call:

- `dgetrs` at 8192: the factorization before the solves "took" 1194 and then
  1189 MiB on two of three runs, and 0 on the third; `dgetrf` at 8192 alone
  took 0. A second solve took 32 MiB once, 0 twice.
- `dpotrs` at 4096: 13.5 MiB on the first call once, 0 on two reruns; the
  factorization before it "gave back" 815 MiB once.
- `dgesvd` at 2048: 14 MiB on the second call once, 0 on two reruns.
- `dtrmm` and `dsyrk` at 4096 gained 1.8 MiB once each.

Nothing that reproduced is above 2 MiB.

## What it means for the fixed workspace

1. **On CUDA nothing measured grows.** No cuBLAS call's need depends on its
   operands, `trsm` included, and cuSOLVER's `potrs` and `getrs` take
   nothing; any constant workspace covers them all. What cuBLAS takes by
   itself stayed under 30 MiB up to 8 GiB of B, with no relation to its size.
2. **On HIP no constant covers `dtrsm`.** The need is a function of the
   call: about B's own size when L's order is not a multiple of 128, and 1/32
   of B when it is. n_aux is whatever the basis gives (3606 on gly10), so the
   density-fitting solve is the first case unless something pads it. With
   128 MiB fixed, a solve over more than about 128 MiB of B grows outside the
   budget.
3. **So on HIP `trsm`'s workspace is a per-call quantity, like a solver's
   work buffer.** The ways to treat it as one:
   - size it by the formula above, declare it in step scratch, and set it on
     the handle around the call, since the step holds the card's lock;
   - pad L to a multiple of 128, which cuts the need to 1/32 of B but still
     does not bound it;
   - use the tree's own triangular solve, as the C++ tree does for Cholesky
     on HIP.

4. **`potrs` and `getrs` cannot be budgeted through hipSOLVER at all.** A
   step that needs a solve against a factor has to make it from two `trsm`
   calls with a declared workspace, or by the tree's own recipe.
5. **The fixed 128 MiB covers everything else measured**: the reductions up
   to 17 billion elements (`iamax` and `zdotc` to 8.6), and the calls that
   take none. With the last of the C++ tree's calls measured, `trsm`,
   `potrs` and `getrs` are the only ones that take memory outside the
   caller's by themselves, and only on HIP.

Not measured: plain `gemm` beyond `handle_ring`'s 4096, upper solves,
rectangular `syrk` and `gemv`, the batched solver calls, hipBLASLt and
cuBLASLt.
