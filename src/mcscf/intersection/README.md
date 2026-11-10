# Intersection

What a minimum-energy conical intersection (MECI) search takes at each
geometry of a converged DF-SA-CASSCF, from one call: `mcscf::df_intersection`,
the nuclear gradients g_I and g_J of two roots of equal weight and their
coupling h_IJ, the numerator of `df_nac`, computed so that nothing divides by
the gap E_I − E_J and the call is as good at the seam as away from it.

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
`df_gradient`'s own solve takes, and it sets the CI response's tolerance
inside every H̃ product too, relative: `response_tolerance_for` it, 1e-2 of
it and at least 1e-10 ("The floor under the true residual"). `options.max_iterations` caps the
block solve's products, 200 by default, `zvectors`' own.

| `IntersectionOptions` | Default | Settles |
|---|---|---|
| `tolerance` | 1e-10 | each Z-vector column's true residual, and through `response_tolerance_for` the CI response's inside H̃ |
| `max_iterations` | 200 | the block solve's H̃ products |

There is no separate option for the CI response's tolerance: #861 listed one
(`kIntersectionResponseTolerance`), but #858 replaced that constant with
`response_tolerance_for(tolerance)`, so it already follows `tolerance`.

## What it is

The derivative of the pair's 2 × 2 Hamiltonian, turned to its eigenvectors:

- `g_i` and `g_j` are dE_I/dR and dE_J/dR, the adiabatic gradients.
- `h_ij` is h_IJ = ⟨I|∂H/∂R|J⟩ with the orbitals and CI relaxed, as
  `df_nac`'s `numerator`: symmetric in I and J, and changing sign with
  either root's CI phase as `solution` carries it. No CSF term and no
  division by the gap.

**Away from a crossing** each equals its own call — `df_gradient(…, i)`,
`df_gradient(…, j)`, `df_nac(…, i, j).numerator` — to the Z-vector solves'
convergence (Tests). **Near one**, the two roots' CI vectors are fixed only
to the CI solver's convergence within their 2-dimensional space, and so are
g_I, g_J and h_IJ: at a gap of 3e-13 Eh the pair (x/2, h_IJ), x = g_J − g_I,
is a rotation within the branching plane by an angle the noise sets. What is
fixed there, and what a search reads, is the **mean gradient**
ḡ = (g_I + g_J)/2 and the **branching plane** span{x, h_IJ}: both are
invariant to that rotation, and neither is computed through a 1/gap.

## The derivation

**The 2 × 2.** Let c_I and c_J be the solution's two roots, at equal weight
w in the state average, and H_KL(R) = ⟨c_K|Ĥ(R)|c_L⟩, K, L ∈ {I, J}, with
the orbitals and the CI relaxed with the geometry. The state-averaged energy
E_SA = Σ_r w_r ⟨c_r|Ĥ|c_r⟩ is stationary in the orbital rotations κ and in
the CI rotations P out of the averaged space, and *not* in the rotation
between I and J: at equal weights it leaves E_SA unchanged, so it is no
parameter of the state average at all (redundant). The relaxation of H_KL is
therefore the state average's alone. Its Lagrangian,

    L_KL = H_KL + z_KL · ∂E_SA/∂(κ, P),

is stationary in (κ, P) when A z_KL = −∂H_KL/∂(κ, P), A the SA Hessian, and
dH_KL/dR = ∂L_KL/∂R. The CI part of the right-hand side is zero: every root
is an eigenvector of the same active-space H₀, so P Ĥ c_L = 0 for K = L and
K ≠ L alike. Eliminating the CI leaves the CI-relaxed orbital Hessian H̃
([`../../response/relaxed_hessian/`](../../response/relaxed_hessian/README.md)) and its Z-vector
H̃ z_o = −g_KL, g_KL the orbital gradient of H_KL at the solution
([`../../response/zvector/`](../../response/zvector/README.md)). This is the SA-CASSCF coupling
numerator of Galván et al. (I. F. Galván, M. G. Delcey, T. B. Pedersen, F.
Aquilante and R. Lindh, JCTC 12, 3636 (2016)) for K ≠ L, and the SA-CASSCF
gradient's Lagrangian for K = L, at the same multiplier equations.

**Three columns, not four.** H_KL is linear in the pair's RDMs, so the
derivatives are taken in the combinations a search reads:

| Column | Combination | RDMs | Orbital right-hand side | Derivative call |
|---|---|---|---|---|
| 0 | (H_II + H_JJ)/2 | the mix (½, ½) | `response::fock`'s gradient, core included | `df_gradient` at the mix (½, ½) |
| 1 | (H_II − H_JJ)/2 | ½(γ_I − γ_J) | `transition_orbital_gradient` (the core cancels) | `df_gradient` at the sum-0 mix (½, −½) |
| 2 | H_IJ | the symmetrized transition RDMs | `transition_orbital_gradient` | `df_transition_gradient` |

The three are solved together, one `zvectors` (Schur) over the one H̃, and
each column's multipliers go to its derivative call, which solves nothing.
The mix calls are #741's (`df_gradient(…, mix, …)`): a sum-0 mix is the
transition gradient at the mix's RDMs, so column 1 is the half difference of
the two roots' derivatives with no core or nuclear term, as it must be.

**The turn to the eigenvectors.** At the solution the 2 × 2 is diagonal to
the CI's convergence. It is formed from the sigma (H_II, H_JJ and H_IJ =
½(c_I · H₀ c_J + c_J · H₀ c_I)) and diagonalized: its eigenvectors are c_I
and c_J turned by φ, tan 2φ = H_IJ / ((H_II − H_JJ)/2), |φ| ≤ π/4, the
smallest turn, so root I stays root I. The eigenvalues' derivatives are the
turned matrix's diagonal and the coupling its off-diagonal (Hellmann–Feynman
within the 2 × 2, the eigenvector held: J. W. Park, JCTC,
doi:10.1021/acs.jctc.9b00919 (2019), whose QD-NEVPT2 gradient is
E_P = Σ R_PPMN H_MN at a fixed eigenvector R, Eqs. 17–20, and whose
interstate coupling replaces R_PPMN by R_PQMN, Eqs. 46–47: the same
construction over a model space),
with ḡ = dH_mean, x̃ = dH_diff and h̃ = dH_IJ the three columns:

    g_I = ḡ + cos 2φ x̃ + sin 2φ h̃,
    g_J = ḡ − cos 2φ x̃ − sin 2φ h̃,
    h_IJ = −sin 2φ x̃ + cos 2φ h̃.

Away from a crossing φ is H_IJ over a gap of hundredths of a hartree, a
turn of roundoff. At the seam it is whatever the CI's noise makes it, and any
value is right: ḡ does not depend on φ, and (x/2, h_IJ) is (x̃, h̃) turned by
2φ, the same plane — the branching plane of Bearpark, Robb and Schlegel
(CPL 223, 269 (1994)), along which the gap rises at the two slopes
twice the singular values of [x/2; h_IJ] (`../../../test/gtest/data/casscf_meci/`).

**What the previous version lost, and where.** #773's version solved g_I's,
g_J's and h_IJ's own right-hand sides; in exact arithmetic those are the
same three derivatives (the I–J rotation is no parameter there either). It
diverged in the CI-relaxed Hessian's product: the CI response δ_r =
−(H₀ − E_r)⁻¹ (1 − c_r c_rᵀ) H′ c_r carries every other root's part exactly,
−(c_s · H′ c_r)/(E_s − E_r) c_s, and the I–J pair's adds (w_I − w_J) times
that, over the transition RDMs, to H̃ x. That is zero at equal weights, but
weights recovered from the solution's averaged RDMs by least squares
(`sa_gradient_weights`, then the only route) are equal only to about 1e-8
relative, and the
term is an O(1/gap) number cancelling through the transition RDMs (at #774
the difference of two RDMs of c ± δ, since #691 `ci::Rdm::trans_rdm12`
summed over the roots). Measured in #774 near ethylene's seam, the true
residual of every column grew as 5e-12 / gap from a gap of 1e-5 Eh down to
the RelaxedHessian's floor of 1e-8 (where the mixing is dropped outright),
reaching 1e-3, while the conjugate gradients' recurrence still met 1e-10 and
the multipliers came back with no error. Here H̃ is created with `RelaxedHessian::create`'s `mixing`
mask from `rotating_pairs(weights)`: a pair of equal weights, I and J among
them, carries no mixing, which is the state average's response exactly, and
a pair of unequal weights (another root's against I or J) keeps its own, as
`df_gradient` has it.

**An unconverged Z-vector is an error.** After the block solve each column's
true residual, ||H̃ z + g|| from one more product, is checked against the
tolerance; a column above it is `Numerical`, naming the column and the
residual. A solve that runs out of products is `zvectors`' own
`MaxIterationsExceeded`. Multipliers short of the tolerance never reach the
derivative calls.

## The floor under the true residual

Each H̃ product solves the CI response, (H₀ − E_r) δ_r = −P H′(x) c_r, by
conjugate gradients to a tolerance relative to its right-hand side, so the
product is exact only to that. The Z-vector's own conjugate gradients take
those products, and their errors go into z: the true residual from one
more product is then bounded below by the products' error, whatever the
recurrence reports. Until #858 and #864 the response stopped at
`RelaxedHessian::create`'s default 1e-8. Measured on the RTX 3080 (CUDA),
with each column's true residual taken from a product at the response's
1e-8 and also from one at 1e-13, effectively exact ("exact" below):

| Case | Products | ‖g‖ | ‖z‖ | Recurrence | True (1e-8) | Exact |
|---|---|---|---|---|---|---|
| ethylene C3v, cc-pVDZ, PC-QD state 0 | 13 | 0.13 | 0.057 | 2.07e-11 | 2.07e-11 | 2.07e-11 |
| PSB3 13–14 start, CASSCF MECI, eval 1, column 1 | 34 | 0.48 | 0.16 | 6.3e-11 | 7.4e-11 | 7.1e-11 |
| PSB3 twisted 90°, cc-pVDZ, SC-QD MECI, column 0 | 31 | 0.39 | 0.79 | 9.8e-11 | **2.55e-10** | 2.19e-10 |
| PSB3 twisted 90°, cc-pVTZ, SC-QD state 0 (one column) | 32 | 0.51 | 0.84 | 8.1e-11 | **3.84e-10** | 2.42e-10 |

- **It is the response's tolerance, not a cap.** The response's own silent
  cap of 200 sigmas was never reached in these runs. The block solve's cap
  of 200 products was not either: #858's failing column stopped at 30
  because its recurrence had met 1e-10, as every column here did. The
  failure was the floor, the same as #864's.
- **The multipliers really are off.** "Exact" is close to "True (1e-8)", so
  the error is in z itself and not only in the check.
- **How it scales.** (true − recurrence) / (1e-8 ‖z‖) is 0.02 to 0.04 at
  cc-pVDZ and cc-pVTZ. The products' error is about **0.04 ‖z‖** times the
  response tolerance. ‖z‖ grows with the system (0.06 at ethylene, 0.8 at
  PSB3), and so did the floor, until it crossed 1e-10 (#864). Near a seam
  the mean column's ‖z‖ is steady but the solve takes more products. #858's
  trail rose from 6.4e-11 to 1.19e-10 over evaluations 0 to 7 as the gap
  closed from 3.2e-2 to about 5e-4 Eh.
- **Converged further, it is gone.** At a response tolerance of 1e-10, every
  true and exact residual above equals its recurrence to under 0.2%, in the
  same products: 9.83e-11, 5.98e-11, 4.16e-11 for the cc-pVDZ MECI's columns,
  and 8.09e-11 at cc-pVTZ. At 1e-12 they are the same again. The cost is
  small: the response's conjugate gradients converge geometrically. The
  cc-pVTZ state gradient's whole run took 37 s at 1e-8 and 38 s at 1e-10, and
  the cc-pVDZ MECI's 46 s at 1e-10 and 48 s at 1e-12.

So the response is converged to `response_tolerance_for` the Z-vector's
tolerance t (`response.zvector`): 1e-2 t, relative, but never below 1e-10. At
the default t = 1e-10 that is 1e-10, the setting measured above: the true
residual is within 0.2% of the recurrence, and a column whose true residual
still lands just over the bar, in (t, 1.5 t], gets one restart
([`../../response/zvector/`](../../response/zvector/README.md) "The restart").
A looser t takes a cheaper response at 1e-2 t, well under the bar's own
error. `nevpt2`'s Lagrangian (`lagrangian_zvector`) takes the same rule for its
own tolerance.

**Why not tighter.** The response cannot be pushed below round-off, and
where the roots nearly span the space, round-off arrives early. Ethylene's
SA2-CASSCF(2,2) has 4 determinants and 3 singlet roots, so a right-hand side
projected off the roots is mostly round-off. Its conjugate gradients then
chase noise: p·Ap is round-off and the steps are garbage. At a response of
1e-12, `PipelineOptimize.EthyleneMeciReachesTheReferenceMeci` failed at
evaluation 12: column 2's true residual was 3.7e-3 after 100 products. At
1e-14, H₂CO's broke a product (p·H̃p = −3.5e2, a `Divergence`). At 1e-10
both pass, and so does every MECI, NAC and QD case in the tiers.

**The walks it stopped** (RTX 3080, CUDA, a `--preset default` build):

- #848's PSB3 13–14 CASSCF MECI (SA3-CASSCF(6,6)/cc-pVDZ, `max_gap` 1e-6),
  refused at evaluation 7 before, converges in 53 evaluations to a gap of
  6.3e-9 Eh. Every column's true residual was 4.6e-11 to 9.9e-11, in 31 to
  35 products. That run used `[casscf] conv_tol_grad = 1e-9`. At 1e-10,
  `mcscf::solve` itself stalled near ‖g‖ = 3.9e-10 in one run, which was #867
  and not this solve (the CI's floor, since tied to `conv_tol_grad`:
  `../solve/README.md` "The CI's floor under ‖g‖"). At 1e-9, s_o ‖g‖ is
  about 2e-10 Eh.
- PSB3 twisted 90° SC-QD-NEVPT2 MECI (`nevpt2`'s Lagrangian, the same
  floor), refused at evaluation 0 before, runs its 4 allowed steps at
  cc-pVDZ (gap 1.9e-2 to 8.4e-5) and its 1 allowed step at cc-pVTZ.

**Why the bar stays 1e-10, absolute.** What the multipliers feed is the
gradient (and, for the QD intersection, s_o and s_c in the gap bound).
With z off by δz = H̃⁻¹ r, the gradient is off by δz · ∂g/∂R, at most
‖r‖ ‖H̃⁻¹‖ ‖∂g/∂R‖. With ‖r‖ = 1e-10, that is about 1e-9 Eh/bohr, three orders
under any geometry criterion here. s_o and s_c are norms of the multipliers,
so they move by about ‖δz‖ / ‖z‖, about 1e-9 relative. A bar relative to ‖g‖
was considered and not taken: ‖g‖ is 0.07 to 0.5 in every case measured,
and with the floor gone nothing is left that grows with the system. Once the
products are exact enough, 1e-10 is reached in 31 to 36 products at PSB3,
against a cap of 200.

## What the three share

Three separate calls build the CI objects and the CI-relaxed Hessian H̃ at
the solution three times, and solve three Z-vectors one after another. Here:

1. The CI strings, sigma, RDMs, the fitted integrals, the averaged Fock, the
   orbital Hessian and H̃ are built once, with the weights `df_gradient`
   takes (`sa_gradient_weights`: the solution's own, recovered from its RDMs
   only when it carries none), the equal-weight pairs' mixing left out.
2. The three right-hand sides of "The derivation" are solved together, one
   `zvectors` over that H̃, the Schur solver (its default), and each
   column's true residual is checked.
3. Each column goes, as given multipliers, to its own derivative call:
   `df_gradient` at the mixes (½, ½) and (½, −½), `df_transition_gradient`
   at the transition RDMs. Those calls solve nothing.
4. The 2 × 2 from two sigmas, and the turn of the three to its
   eigenvectors, on the host.

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
  Other roots may carry other weights. The weights are the solution's own,
  so equal ones are equal exactly: before #856 they were recovered from the
  RDMs, and at ethylene's C3v-ethylidene MECI, where the roots' 1-RDMs are
  nearly collinear, equal weights came back unequal past the tolerance and
  every walk there was refused.
- `InvalidConfig`: fewer than two roots, I = J, an index outside the average,
  a tolerance not positive or `max_iterations` below 1, and `df_nac`'s checks
  of the solution, the orbitals and the fit.
- **An unconverged Z-vector**: `MaxIterationsExceeded` (`zvectors`', a
  column still above the tolerance after `max_iterations` products) or
  `Numerical` (a column's true residual above it, the recurrence having
  passed it). Never multipliers short of the tolerance.

## Timing

The wall clock against the three separate calls (`df_gradient(I)`,
`df_gradient(J)`, `df_nac(I, J)`), on each case's roots (0, 1): one warm-up of
each, then the median of three (`McscfIntersectionTiming.AgainstThreeCalls`,
slow tier only), on the **RTX 3080 (CUDA)**, the card held. The gfx1200 was
not measured.

| Case | One call | Three calls | |
|---|---|---|---|
| H₂CO SA-2, (0, 1) | 0.209 s | 0.260 s | 1.25× |
| skewed H₂CO SA-3, (0, 1) | 0.248 s | 0.317 s | 1.28× |

What it saves is two of the three H̃ builds and the serial solves: its setup
and block solve take about 70 ms on the SA-3, where the three calls' own take
about 140 ms. The three derivative calls, 57 ms each, are most of what is
left. It is faster, so it is not behind an
option.

## Memory and stream

`df_gradient`'s, with H̃, the CI objects and the three columns of multipliers
held beside them. Queued on `res.stream()`; every call it makes synchronizes,
so it returns with everything done.

```cpp
MemoryFootprint mcscf::df_intersection_footprint(
    const BasisSet &basis, const BasisSet &aux, int32_t ncore, int32_t ncas,
    int32_t nelecas, int32_t multiplicity, int32_t nroots, bool rotating);
```

is what the call holds, all of it freed on return (#1212), in its order:
`df_nac_footprint`'s setup (the strings, the integrals, the averaged Fock,
the sigma's tables, H and H̃, here over three block columns), kept; the
pair's RDMs; the three right-hand sides, with the mean's Fock and the two
transition orbital gradients each freed once placed; the sigma's apply
scratch and `zvectors_footprint` over the three columns, the ZVectors kept;
each column copied out, kept; the three derivative calls'
`df_gradient_footprint` (the mean's mix, the half difference's mix summing
to 0, the coupling's transition RDMs, each with its multipliers); the two
sigma products. `rotating` says two weights differ. `[optimize]`'s MECI
footprint reads it (`../../pipeline/optimize/`).
`CasscfReaderFootprint.DfIntersection`
(`test/gtest/pipeline_step_footprint_gpu.cpp`) holds it to the pool's peak
over water's SA-2 CAS(4,4).

## Tests

`test/gtest/mcscf_intersection_gpu.cpp`.

**Away from a crossing** (`McscfIntersection`), on
`test/gtest/data/casscf_nac/`'s two cases at their equal weights, every pair
I < J: g_I, g_J and h_IJ against `df_gradient(I)`, `df_gradient(J)` and
`df_nac(I, J)`'s numerator, each **1e-9 relative to the largest element of
the two vectors compared**. Every one of the three compares independently
converged Z-vector solves, so each is solver-limited, and 1e-9 is the bar the
ruling on #767 sets for such a check; none is a round-off-only path
agreement, so none is held at 1e-10. Then the refusals: I = J and a root past
the average (`InvalidConfig`), unequal weights (`Unsupported`, the weighted γ
re-mixed so that it is no longer the carried weights' sum and the recovered
weights differ), and the block solve capped at one product
(`MaxIterationsExceeded`).

**At the seam**, on `test/gtest/data/casscf_meci/`'s ethylene
SA2-CASSCF(2,2), cc-pVDZ / cc-pVDZ-JKFIT:

- `McscfIntersectionSeam.MeciAgainstReference`: a fresh evaluation at the
  reference's MECI. Every column's true residual at most 1e-10; −P ḡ (off the
  translations, the rotations about the centroid and our plane, as the
  reference's projector) within **1e-6 Eh/bohr** of the reference's
  projected force, element by element; and the gap's in-plane slopes, twice
  the singular values of [x/2; h_IJ], against those recomputed from the
  reference's vectors (0.2290 and 0.1413 to four places) within
  **2 √(6 n_atom) · 1e-8** of the largest element of its g₀, g₁ and h₀₁:
  the tree's 1e-8 bar against PySCF's gradients and couplings carried
  through Weyl's inequality (a singular value moves by at most the change's
  spectral norm, at most its Frobenius norm).
- `McscfIntersectionNearSeam.CarriedAgreesWithFresh/{Meci, AlongX,
  AlongH}`: at the MECI and at the MECI moved 1e-7 bohr along the
  reference's x and h₀₁ (gaps of about 2e-8 Eh), a fresh-orbital evaluation
  against a carried-orbital one (the CASSCF's orbitals at the reference's
  iterate 2.4e-3 bohr away, Löwdin-carried as `[optimize]` carries them).
  Every column's true residual at most 1e-10; ḡ within **1e-9 of the larger
  element** (#767's bar, two independent solves), and the two planes'
  largest principal angle below **1e-6 rad**. `AlongX` stands for the three
  in the fast tier: the MECI's own gap is under the RelaxedHessian's 1e-8
  floor, below which #773's version already dropped the mixing.

Measured on the RTX 3080 (CUDA):

| Check | Bar | Measured |
|---|---|---|
| True residual, every evaluation (15 products) | 1e-10 | 2.2e-11 to 7.7e-11 |
| MECI: −P ḡ against the reference's projected force | 1e-6 Eh/bohr | 8.8e-13 |
| MECI: in-plane slopes 0.2290407634, 0.1412706023 against the reference's | 7.98e-9 | 1.2e-11, 1.4e-11 |
| `Meci` (gap 1.2e-11): carried vs fresh ḡ, plane | 1e-9, 1e-6 rad | 2.7e-10, 7.0e-11 rad |
| `AlongX` (gap 1.97e-8): carried vs fresh ḡ, plane | 1e-9, 1e-6 rad | 3.4e-10, 1.1e-10 rad |
| `AlongH` (gap 2.17e-8): carried vs fresh ḡ, plane | 1e-9, 1e-6 rad | 4.4e-10, 1.2e-10 rad |
| Away: g_I, g_J, h_IJ against the separate calls (worst, skewed SA-3 (0, 1)) | 1e-9 | 1.8e-10, 1.0e-10, 2.2e-10 |

(the slow tier's run with the response at 1e-10, #858; the slopes are the
previous run's).

With the mixing kept for every pair (#773's H̃, everything else as here),
`AlongX` stops at `Numerical`: column 1's true residual is 1.6e-5 after 16
products, the recurrence having met 1e-10.

**The tolerance reaches the response** (`McscfIntersectionTight`, skewed
H₂CO SA-3, (0, 1)): asked for 1e-12, the call's true residual meets it,
and g_I, g_J and h_IJ agree with the default call's to 1e-9 relative (the
default solve's, as above). H₂CO's z is small enough that its floor under
the old 1e-8 response was itself below 1e-12. So this checks the option's
plumbing. The floor's regression is PSB3's
(`Nevpt2QdIntersection.Psb3TrueResidual`, slow tier,
[`../../nevpt2/intersection/`](../../nevpt2/intersection/README.md)), which
was refused at 2.55e-10 before the change.

## Files

| File | Role |
|------|------|
| `intersection.cppm` | Module interface: `IntersectionOptions`, `Intersection`, `df_intersection` |
| `intersection.cpp` | Implementation: the shared H̃ (equal-weight pairs unmixed), the three right-hand sides, the block solve and its residual check, the three derivative calls, the 2 × 2's turn |
| `CMakeLists.txt` | The `free_breakfast.mcscf.intersection` module library |
