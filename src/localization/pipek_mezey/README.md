# Pipek-Mezey

Pipek-Mezey orbital localization on the GPU: the occupied orbitals rotated,
within their own span, to sit on as few atoms as the Mulliken populations allow.
Custom kernels over `core/kernel`'s `block_reduce` and `core/primitives`' math,
and two wwr `gemm`s.

## Module

`free_breakfast.localization.pipek_mezey`, target
`free_breakfast::localization::pipek_mezey`.

| Name | Role |
|---|---|
| `pipek_mezey_localize(res, basis, n_occ, s, c, c_local, u, options)` | The localization; returns `Result<PipekMezeyInfo>` |
| `PipekMezeyOptions` | Sweep cap, tolerances, exponent |
| `PipekMezeyExponent` | `Two` (Pipek-Mezey) or `Four` (Knizia's IBO exponent) |
| `PipekMezeyInfo` | Sweeps taken, the functional before and after |

Double precision only. Every matrix is a column-major device `double *` with a
packed leading dimension, over the spherical AOs of `basis`: `s` is n_ao x
n_ao, `c` and `c_local` n_ao x n_occ, `u` (optional) n_occ x n_occ, with
C_local = C U. The first `n_occ` columns of a square C, such as `ScfOutput::c`, serve
as `c` directly.

```cpp
const PipekMezeyInfo info = FBF_TRY(pipek_mezey_localize(
    res, basis, n_occ, overlap.s.data(), scf.c.data(), c_local.data(), u.data(),
    {.exponent = PipekMezeyExponent::Four}));
```

`c` is expected S-orthonormal (C^T S C = I): that is what makes an orbital's
populations sum to one. It is the caller's invariant, not checked.

**Not converging is an error**, as for `cg_unitary` and the SCF:
`MaxIterationsExceeded` after `max_sweeps`, with `c_local` and `u` holding the
last sweep. An option out of range, a bad pointer, `c_local` aliasing `c`, or
a basis with no atoms is `InvalidConfig`; a failed launch, copy or BLAS call is
`GpuRuntime`.

## What it maximizes

```
L_p(U) = sum_i sum_A (Q^A_ii)^p
Q^A_ij = 1/2 sum_{mu in A} (C_mu,i W_mu,j + C_mu,j W_mu,i),   W = S C
```

Q^A is atom A's symmetrized Mulliken population matrix. Because
sum_A Q^A_ij = (C^T S C)_ij = delta_ij, an orbital's populations sum to one, so
raising them to a power and summing rewards concentrating each orbital on few
atoms. Only the relative value matters, and the algorithm only raises it.

**An atom owns its shells' AOs, as one range.** A basis placed over a molecule
records which shells each atom owns (`BasisSet::atom_shell_offset`), and the
spherical layout follows the shell order, so atom A's AOs are
[ao_offset_sphe(first shell), ao_offset_sphe(next atom's first shell)). No
geometric matching of shell centres to nuclei is needed, and a basis built
from a bare shell list, with no atoms, is refused.

**p = 2** is Pipek and Mezey's (J. Chem. Phys. 90, 4916 (1989)), with an exact
rotation angle per pair. **p = 4** is Knizia's IBO exponent (J. Chem. Theory
Comput. 9, 4834 (2013), eq. 4 and Appendix D): p = 2's rotation Hessian can
carry an exact zero mode for a symmetric pi system (benzene's six equivalent pi
orbitals), leaving a continuum of answers, and p = 4 picks one. The pair
change in L_4 has 4gamma and 8gamma harmonics, so it has no closed-form
maximum. The angle is Knizia's Appendix D one, 1/4 atan2(B, -A): B is half the
exact slope at gamma = 0 and -A an eighth of the negated curvature, so it is
zero only where the pair is stationary, and a converged sweep is a stationary
point of L_4. (Dropping the 8gamma harmonic and maximizing the rest lands
nearer each pair's own maximum, but its angle vanishes off the stationary
points, and a sweep driven by it stalls on a slope.) The angle is approximate, so the exact gain
there is evaluated from the same pass's four sums, and a pair that would lose
is left alone. Both exponents read the same population tensor and share every
kernel but the pair angle and the functional.

The functional is eq. 4's, but over Mulliken populations rather than the
paper's IAO ones: this is Pipek-Mezey with Knizia's exponent, not IBO.
`localization/ibo` is IBO: it writes the orbitals over the orthonormal IAOs,
where S is the identity and these populations are the IAO ones, and calls this
sweep.

## Design

The classical solution is a Jacobi sweep: rotate one orbital pair by the angle
that maximizes L, repeat. Two things make it a poor fit for a device, and both
are changed here.

**1. Sweep the populations, not the coefficients.** Q^A transforms as
U^T Q^A U, so the sweep runs on the n_atoms x n_occ x n_occ population tensor
while the rotations accumulate into an n_occ x n_occ U. The coefficients are
touched twice, W = S C at the start and C U at the end. A rotation costs
O(n_atoms n_occ) instead of O(n_ao n_occ), and nothing in between leaves the
device.

**2. Rotate a whole round of disjoint pairs at once.** A round-robin ("circle
method") tournament splits a sweep into n_occ - 1 rounds of n_occ/2 disjoint
pairs; an odd count is padded with a dummy index, the pivot, so one real
orbital sits each round out. Disjoint rotations commute into one
block-diagonal R, so a round is three launches (angles, Q <- R^T Q R,
U <- U R) however many pairs it holds.

That costs nothing in quality. A pair's rotation changes only its own Q_pp and
Q_qq, from its own Q_pp, Q_qq and Q_pq, and L reads only diagonals, so a round
gains the sum of its pairs' gains, each non-negative. L rises round by round as
under sequential Jacobi, and the two reach the same maximum: the step's test
matches PySCF's to 1e-8 in L.

### The kernels

The population tensor is atom-fastest, ((i n_occ) + j) n_atoms + A, and every
kernel puts the atom on `threadIdx.x`, so a warp walks contiguous memory. The
arithmetic is small `__device__` functions in the same `.cu`, each kernel only
picking its elements; nothing of it runs on the host. The round-robin schedule
is the one host part, built by the driver once and uploaded.

| Kernel | Grid | What it does |
|---|---|---|
| `populations_kernel` | (atom tiles, n_occ, n_occ) | Q^A_ij. The pair rides y and z, so the i <= j triangle test is a whole-block exit |
| `pair_angle_kernel<p>` | one `kFourWarps` block per pair | Reduces the pair's sums over the atoms with `block_reduce` and writes per-orbital cosine, sine and partner |
| `rotate_populations_kernel` | (atom tiles, n_occ, n_occ) | Q_out = R^T Q_in R: out-of-place (each output reads four inputs), so the driver ping-pongs |
| `rotate_columns_kernel` | (row tiles, n_occ) | U_out = U_in R |
| `functional_kernel<p>` | one `kMaxThreads` block | L_p. One block on purpose: `block_reduce` folds in thread order, so L, and so the convergence test, is bit-reproducible |
| `identity_kernel` | flat | U = I |

The population build and the rotation compute the i <= j triangle and mirror
it, which keeps the tensor exactly symmetric across a sweep rather than letting
(i, j) and (j, i) drift apart by a rounding difference.

### Cost

Time is O(n_ao n_occ^2) once for the populations, then O(n_atoms n_occ^2) a
round with n_occ - 1 rounds a sweep. The host synchronizes once a sweep, for
the convergence test. Space is the practical limit: every scratch buffer,
dominated by the two population tensors (2 n_atoms n_occ^2 elements, 64 MB in
double for 100 atoms and 200 orbitals), comes from `res.device_workspace()`.

The launch count, 3 (n_occ - 1) a sweep, makes a sweep launch-bound for a few
dozen orbitals. Capturing a sweep into a graph is the obvious next step, and is
not done here.

## Not done here

The closed-shell occupied space, and Mulliken populations only (IAO ones are
`localization/ibo`'s, over this sweep): Loewdin populations and the virtual
space are further work. Either exponent works unchanged on another population tensor built the
same way. `recipes/cg_unitary` is the other route to the same maximum, as a
cost function on O(n); the population build would be shared, and this sweep is
the reference to check it against.

## Tests

Every test needs a GPU.

- `test/gtest/pipeline_localization_gpu.cpp`, the oracle: water in STO-3G and
  cc-pVDZ and NH4+ in cc-pVDZ, both exponents, through the step
  (`pipeline/localization`), against PySCF's Pipek-Mezey
  (`tools/emit_localization_reference.py`): L to 1e-8, and every orbital's
  Mulliken population on every atom matched to a PySCF orbital to 1e-5. The
  populations stand in for the orbitals, which the two codes express over
  differently normalised and ordered AOs. Water has an odd occupied count, so
  this also runs the schedule's padding.
- `test/gtest/pipek_mezey_gpu.cpp`, over model bases: for both exponents, U
  orthogonal, C_local = C U and S-orthonormal, and the reported functional the
  host's from the definition, over atoms owning s+p and s shells; separated
  centres localizing onto single atoms; on a symmetric six-ring p = 2 reaching
  one value by many localizations where p = 4 picks one; one orbital and none;
  the sweep cap; every refusal.

## Files

| File | Role |
|---|---|
| `pipek_mezey.cppm` | Module interface: the options, the info and `pipek_mezey_localize` |
| `pipek_mezey.cpp` | The sweep driver and the round-robin schedule |
| `pipek_mezey_bridge.h` | The launchers, declared for the module and the `.cu` |
| `pipek_mezey_kernels.cu` | The arithmetic, the kernels and their launchers |
| `CMakeLists.txt` | The module and the device library |
