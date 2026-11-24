# experimental/rust/blas_workspace_scaling — what a `dtrsm` needs from its handle

`docs/rust-redesign.html`, "One handle set per card, under one lock", gives
each card's BLAS handle a fixed 128 MiB workspace.
`experimental/rust/handle_ring` had one reading behind that number: a
4096 × 4096 `dtrsm` on hipBLAS writes 8 MiB of its workspace, and takes 8 MiB
of its own when the workspace is too small. This probe asks how that need
grows with the shape, on both backends, in nine readings.

It goes through `wwr-sys` (on HIP, hipBLAS over rocBLAS).

```bash
# from rust/: the C ABI libraries, per backend
cmake --preset cuda && cmake --build build-cuda --target wwr.capi.runtime_api wwr.capi.blas
cmake --preset hip  && cmake --build build-hip  --target wwr.capi.runtime_api wwr.capi.blas
# from the checkout root: build, then every shape under the card's lock
experimental/rust/blas_workspace_scaling/run_all.sh cuda
experimental/rust/blas_workspace_scaling/run_all.sh hip
experimental/rust/blas_workspace_scaling/run_all.sh hip "right 4096 4100"
```

One process per shape. B is m × n, and L is lower, of order k = m (`left`,
B ← L⁻¹B) or k = n (`right`, B ← BL⁻¹). Each process makes two calls, each on
a fresh handle. `main.rs`'s header says what each prints:

- **used**: the handle has a 0xFF-filled workspace larger than three copies
  of B. The highest byte the call changed is what it used.
- **own**: the handle has a 256-byte workspace. The card's free memory lost
  over the call is what the library took by itself.

Nothing in ctest or either gate runs it.

## Results, 2026-11-24

RTX 3080 (CUDA 13) and RX 9060 XT (gfx1200, ROCm), each card held by
`with_gpu_card`.

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
  of L, plus 1/32 of B. Three points fit that form and do not prove it.
- **"Own" agrees with "used".** With a 256-byte workspace the library took
  the same amount by itself, rounded up to 2 MiB, with no error. That is the
  growth outside the budget `handle_ring` found, now at every shape.
- **With the large workspace the call still lost 12 to 14 MiB of free
  memory.** That is the once-per-process first-call growth `handle_ring`
  measured at +10, not workspace.

### cuBLAS

| Side | m × n | Used (bytes) | Own (MiB) |
|---|---|---|---|
| left | 4096 × 4096 | 0 | 0 |
| left | 4100 × 4096 | 0 | 0 |
| left | 16400 × 1024 | 512 | −0.6 |

cuBLAS's `dtrsm` uses none of the handle's workspace and takes nothing at
any of the three shapes. The 512 bytes and the −0.6 MiB (memory freed) are
not explained.

## What it means for the fixed workspace

1. **On CUDA the size does not matter for `dtrsm`.** Any constant covers it.
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

Not measured: `gemm`, `syrk` and `gemv` at production shapes (`handle_ring`
found `gemm` takes none at 4096), transposed or upper solves, other types
than double, and whether the formulas hold past these six shapes.
