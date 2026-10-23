# recipes/expm

`free_breakfast.recipes.expm`: the exponential of a **skew-Hermitian** matrix
(real skew-symmetric, for a real `T`), `exp(tH)` with `H^H = -H`. The result is
unitary (orthogonal), and `t ↦ exp(tH) W` is the geodesic of U(n) (O(n))
through `W` in direction `H`. It is the step of a Riemannian solver on the
unitary group, the Abrudan–Eriksson–Koivunen conjugate gradient that orbital
localization (Pipek–Mezey, Boys) runs on, and of orbital rotations `exp(κ)`
in general.

## The route

`iH` is Hermitian, so one Hermitian eigendecomposition gives every `t`:

```
iH      = U diag(w) U^H            skew_eig: one heevd
exp(tH) = U diag(e^{-i t w}) U^H   skew_expm: one phase-scale kernel, one gemm
```

The work is split in two on purpose. A line search evaluates `exp(tH)` at
many `t` for one `H`, so it pays for `heevd` once and each further `t` costs
an elementwise kernel and a gemm. The decomposition also hands the search
`|ω_max| = max |w|`, the frequency bound its almost-periodicity interval is
built on, with no separate power iteration.

The phases are formed directly, so the result is exact up to the
decomposition's accuracy at any `t`: there is no scaling and squaring, no
Padé degree to pick, and no host sync to read a norm. A degenerate `w` is
fine. `heevd` may return any basis of that eigenspace, and the exponential
depends only on the projector.

**A real `T` still decomposes in complex arithmetic.** A real skew-symmetric
`H` has imaginary eigenvalues in `±` pairs (and a `0` for odd n) and complex
eigenvectors, so `u` is always complex: `T` itself when `T` is complex, `T`'s
complex counterpart when it is real. `skew_expm` then takes the real part of a
product that is real in exact arithmetic. A complex `heevd` costs roughly four
times a real `syevd`. The real-arithmetic alternative, a real Schur form with
2×2 blocks, is not a dense-solver call wwr offers.

**Forming `iH` takes no kernel.** For a complex `T` it is a `copy` and a
`scal` by `kImaginaryUnit<T>`. For a real `T`, `iH` is `H` in the imaginary
plane, so `u` is zeroed and `h` copied into every second real component
(stride 2). Likewise, a real `T`'s result is the real plane copied out at
stride 2.

## Why not a general `expm`

calaman's `expm` is scaling and squaring with a diagonal Padé approximant,
plus a 1-norm reduction to pick the degree, about 2300 lines over five
partitions. Nothing here needs `exp(A)` for a general `A`. If something does
later, it belongs beside these two as its own routine; it is not a reason to
route the skew case through Padé.

## Workspace

Both calls take `res.device_workspace()`: `skew_eig` for `heevd`'s work,
`skew_expm` for `U diag(e^{-i t w})` (and, for a real `T`, the complex
product), `n²` or `2n²` complex elements. Neither keeps anything there between
calls, so `u` and `w` must live in the caller's own buffers.

## Tests

`test/gtest/expm_gpu.cpp`, against a closed-form ground truth rather than a
reference implementation: `H = Q B Q^H`, with `Q` a dense unitary (two
Householder reflectors) and `B` a canonical form whose exponential is cos and
sin alone. For a real `T`, `B` is 2×2 rotation generators; for a complex one,
`diag(iφ)`. One frequency repeats, to cover a degenerate eigenspace.

## Files

| File | Role |
|------|------|
| `expm.cppm` | Module interface: `skew_eig`, `skew_expm` |
| `expm.cpp` | Implementation, and the four instantiations |
| `expm.cu` | The phase-scale kernel |
| `expm_bridge.h` | The launcher's declaration, shared by `expm.cpp` and `expm.cu` |
| `CMakeLists.txt` | The device library and the module |
