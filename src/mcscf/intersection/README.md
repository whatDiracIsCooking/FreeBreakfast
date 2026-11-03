# Intersection

What a minimum-energy conical intersection (MECI) search takes at each
geometry of a converged DF-SA-CASSCF, from one call: `mcscf::df_intersection`,
the nuclear gradients g_I and g_J of two roots of equal weight and their
coupling h_IJ, the numerator of `df_nac`.

## Module

`free_breakfast.mcscf.intersection`, target
`free_breakfast::mcscf::intersection`, re-exported by `free_breakfast.mcscf`.

```cpp
Result<mcscf::Intersection> mcscf::df_intersection(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux,
    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
    const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t i,
    int32_t j, const mcscf::IntersectionOptions &options = {});
// x->g_i, x->g_j, x->h_ij: each n_atom x 3, row-major, element 3 A + k;
// x->iterations, x->residual: the block Z-vector solve's
```

Every argument through `ncas` is `df_gradient`'s and `df_nac`'s
([`../gradient/`](../gradient/README.md), [`../nac/`](../nac/README.md)).
`options.tolerance` is each Z-vector column's, 1e-10 by default, the one
`df_gradient`'s own solve takes.

## What it is

- `g_i` is `df_gradient(…, i).total`, dE_I/dR.
- `g_j` is `df_gradient(…, j).total`.
- `h_ij` is `df_nac(…, i, j).numerator`, h_IJ = ⟨I|∂H/∂R|J⟩ with the orbitals
  and CI relaxed: symmetric in I and J, and changing sign with either root's
  CI phase as `solution` carries it. No CSF term and no division by the gap:
  an MECI search spans the branching plane with g_J − g_I and h_IJ, and the
  gap goes to zero there.

Each equals its own call to the Z-vector solves' convergence. The three are
solved here, not in those calls, so the agreement is the solver's, not
round-off's (Tests).

## What the three share

Three separate calls build the CI objects and the CI-relaxed Hessian H̃ at
the solution three times, and solve three Z-vectors one after another. Here:

1. The CI strings, sigma, RDMs, the fitted integrals, the averaged Fock, the
   orbital Hessian and H̃ are built once, with the weights `df_gradient`
   recovers (`sa_gradient_weights`).
2. The three right-hand sides — root I's and root J's orbital gradients
   (`response::fock`'s at each root's own RDMs, as `df_gradient`'s Z-vector
   takes them) and the transition orbital gradient at the symmetrized
   transition RDMs (`transition_orbital_gradient`, `df_nac`'s) — are solved
   together, one `zvectors` over that H̃, the Schur solver (its default).
3. Each column goes, as given multipliers, to its own derivative call:
   `df_gradient` for root I and root J, `df_transition_gradient` at the
   transition RDMs for h_IJ. Those calls solve nothing.

**The derivative contractions are not stacked.** The issue (#773) asked for
the three-center and metric contractions to be done once over the three
density sets. Measured on the RTX 3080 (the skewed H₂CO SA-3, cc-pVDZ /
cc-pVDZ-JKFIT), each derivative call is 57 ms, of which
`three_center_gradient` is 53 ms, and of that the geri3 kernels are 50 ms
(the staging 2.6 ms, the Cartesian transform 0.7 ms); `metric_gradient` and
the one-electron terms are under 1 ms each. So the only stacking that pays is
a geri3 that contracts several Γ against each integral it evaluates, a
generator change in `deps/FelizJueves` (`tools/emit/df_gradient_kernels.py`)
and a new `three_center_gradient` overload, outside this module. The
public route that does exist, the materialized derivative integrals
(`three_center_derivative` over every aux atom, then one contraction per set),
costs 167 ms for the integrals alone, more than three geri3 passes (150 ms),
so it is not taken.

## Refusals

- **Unequal weights on I and J** (`Unsupported`, naming the reason): two
  weights differing by more than `kRotationWeightTolerance` of the larger.
  The rotation between I and J is then a parameter of the Lagrangian, and its
  multiplier divides by E_I − E_J, which is singular at an intersection.
  Other roots may carry other weights.
- `InvalidConfig`: fewer than two roots, I = J, an index outside the average,
  a tolerance not positive, and `df_nac`'s checks of the solution, the
  orbitals and the fit.

## Timing

The wall clock against the three separate calls (`df_gradient(I)`,
`df_gradient(J)`, `df_nac(I, J)`), on each case's roots (0, 1): one warm-up of
each, then the median of three (`McscfIntersectionTiming.AgainstThreeCalls`,
slow tier only), on the **RTX 3080 (CUDA)**, the card held. The gfx1200 was
not measured.

| Case | One call | Three calls | |
|---|---|---|---|
| H₂CO SA-2, (0, 1) | 0.220 s | 0.242 s | 1.10× |
| skewed H₂CO SA-3, (0, 1) | 0.247 s | 0.311 s | 1.26× |

What it saves is two of the three H̃ builds and the serial solves: its setup
and block solve take about 70 ms on the SA-3, where the three calls' own take
about 140 ms. The three derivative calls, 57 ms each, are most of what is
left. It is faster, so it is not behind an
option.

## Memory and stream

`df_gradient`'s, with H̃, the CI objects and the three columns of multipliers
held beside them. Queued on `res.stream()`; every call it makes synchronizes,
so it returns with everything done.

## Tests

`test/gtest/mcscf_intersection_gpu.cpp`, on `test/gtest/data/casscf_nac/`'s
two cases at their equal weights, every pair I < J: g_I, g_J and h_IJ against
`df_gradient(I)`, `df_gradient(J)` and `df_nac(I, J)`'s numerator, each
**1e-9 relative to the largest element of the two vectors compared**. Every
one of the three compares independently converged Z-vector solves, so each is
solver-limited, and 1e-9 is the bar the ruling on #767 sets for such a check;
none is a round-off-only path agreement, so none is held at 1e-10. Measured
on the RTX 3080: g_I within 1.2e-11, g_J 5.0e-13, h_IJ 2.2e-10 (the skewed
SA-3's (0, 1)), over two runs. Then the refusals: I = J and a root past the average
(`InvalidConfig`), and unequal weights (`Unsupported`, the weighted γ re-mixed
so the recovered weights differ).

## Files

| File | Role |
|------|------|
| `intersection.cppm` | Module interface: `IntersectionOptions`, `Intersection`, `df_intersection` |
| `intersection.cpp` | Implementation: the shared H̃, the three right-hand sides, the block solve, the three derivative calls |
| `CMakeLists.txt` | The `free_breakfast.mcscf.intersection` module library |
