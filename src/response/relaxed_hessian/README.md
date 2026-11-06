# Relaxed Hessian

The orbital Hessian with the CI relaxed, H̃ = H + R, over the packed pairs at
fixed orbitals and roots: `response::Hessian`'s H plus the CI's response R.
`mcscf::solve`'s augmented Hessian steps over it, and the state-averaged
Z-vector of a CASSCF gradient drives the same product.

## Module

`free_breakfast.response.relaxed_hessian`, target
`free_breakfast::response::relaxed_hessian`, over `ci` (strings, sigma, RDMs,
roots) and `response.hessian`. `free_breakfast.response` re-exports it;
`mcscf/solve` links it PRIVATE. Everything is in
`namespace free_breakfast::response`.

| Name | What it is |
|------|------------|
| `ResponseIntegrals` | F^I, ppaa and papa on the host, column-major as the device's: what H′(x) contracts |
| `response_integrals` | H′(x): h_eff′ and (tu\|vw)′ along κ(x), row-major as `ci::Sigma::set_integrals` takes them |
| `response_gradient` | Its adjoint: the antisymmetric nmo × nmo whose packed pairs are the gradient in x of h_eff′·γ + (tu\|vw)′·Γ |
| `RelaxedHessian` | H̃ at one set of orbitals and roots: `apply` (y = H̃ x, one column or a block), `respond` (δ and its transition RDMs γ_T, Γ_T for one column), `couple` (w = −(H₀ − E)⁻¹ P y for a CI right-hand side y, and H_oc w; one column or a block), `apply_coupled` (the unreduced operator over (x, x_c), and optionally the rotations among the roots, a block), `coupling` (H_oc x_c alone, a block), `ci_diagonal` (diag H₀), `delta`, `size`, `weights` |

`RelaxedHessian::create` takes the `response::Hessian`, the sigma whose
integrals are H₀ at the same orbitals, the `ci::Rdm` and the strings both are
over, the `ci::Roots` of H₀, the weights and the `ResponseIntegrals`. It keeps
pointers to the first four of those (and the roots), which must outlive it
and not change under it, and allocates its own second sigma (for H′(x)) and
the CI-sized scratch. A product synchronizes the stream: κ(x) and the
transition RDMs are read back, and H′ and its adjoint are host work.
| `create` argument | Default | Settles |
|---|---|---|
| `response_tolerance` | 1e-8 | the CI response's conjugate gradients' stop, relative to their right-hand side |
| `mixing` | empty (every pair) | which pairs of roots carry their other-roots part ("Leaving a pair's mixing out") |
| `response_max_iterations` | 200 | their sigmas per root, a silent stop |
| `pcg_ci_response` | off | those conjugate gradients through `recipes::pcg_solve` instead of the loops here ("The CI response through `recipes::pcg_solve`") |

`response_tolerance` (1e-8 by default) is where the CI response's conjugate
gradients stop, relative to their right-hand side, and
`response_max_iterations` (200 by default, the last argument, at least 1)
caps their sigmas per root: at the cap they stop **silently**, the product
then good only to where they got (#861). Every product is good to
about that: measured at PSB3, a product's error is about 0.04 ‖x‖ times it.
A Z-vector solved through H̃ to a tolerance t cannot have a true residual
below the products' error, so the callers that hold a true residual to t
pass `response::response_tolerance_for(t)`, 1e-2 t but at least 1e-10 (1e-10
at the default 1e-10): `mcscf::df_intersection` and `nevpt2`'s Lagrangian. At 1e-8 the
floor crossed 1e-10 at PSB3 (#858, #864;
[`../../mcscf/intersection/`](../../mcscf/intersection/README.md) "The floor under the true
residual" has the measurements and the argument). The conjugate gradients
converge geometrically, so the tighter tolerance cost about 3% of the
cc-pVTZ state gradient's run. A test that compares an exact solve with one
through H̃ tightens it too.

## What a product is

`apply` is `response::Hessian::apply`, then R x added:

1. κ(x) unpacked (`unpack_orbital_pairs`) and read back, and H′(x) from it
   on the host (`response_integrals`); no J or K, every integral it takes
   has two active indices.
2. b_r = H′ c_r, the second sigma applied to the roots.
3. δ_r = −(H₀ − E_r)⁻¹ (1 − Σ_s c_s c_sᵀ) b_r by conjugate gradients over
   H₀'s sigma, to `response_tolerance` of the right-hand side (at most
   `response_max_iterations`, 200, sigmas per root;
   none when the roots span b_r, below), and the other roots' part exactly,
   −(c_s · b_r) / (E_s − E_r) c_s (left out for a gap under 1e-8 hartree).
4. γ_T and Γ_T: Σ_r w_r 2 `ci::Rdm::trans_rdm12(c_r, δ_r)`, the
   symmetrized transition RDMs root by root (Γ_T's halved, for the ½ of the
   two-body energy).
5. R x = H_oc δ: `response_gradient` of γ_T and Γ_T, uploaded, packed
   (`pack_orbital_pairs`) and added to y.

Steps 1 to 4 are `respond`, so a caller that needs δ itself (the CI part of a
Z-vector's multipliers) or the transition RDMs takes them without H's product.
`couple` is steps 3 to 5 with b = y, a CI block the caller gives (projected
off every root first, so the other roots' part is nothing): w into `delta()`
and H_oc w over the packed pairs, the orbital side of a Z-vector's CI
right-hand side.
## Block calls

`apply`, `couple` and `apply_coupled` over a block of k columns are what
`response::zvectors` (`../zvector/README.md`, "Many right-hand sides") iterates
on. They share the per-column work above: H's product over the block in one
call (`response::Hessian::apply`, which still builds each column's J and K
on its own), then H′(x_j) c and H_oc δ_j per column (host work, a readback
each). What the block changes is the CI side:

- **The block `apply`** runs every column's CI response at once: the
  conjugate gradients of step 3 over all k · nroots vectors in lockstep, each
  vector at its own root's E_r, with one sigma of the whole block per step,
  every vector's dot by one strided-batched gemm and one readback, and the
  updates as dgmm and geam over the block. A vector stops at
  `response_tolerance` of its right-hand side as before (its step lengths
  zero after that). The other roots' part is one gemm of the overlaps.
  Each column agrees with the one-column `apply` to the CI response's
  tolerance, not to the bit (another summation order); `delta()` holds the k
  blocks afterwards.
- **The block `couple`** is the one-column `couple` over the block's lockstep
  solve.
- **`apply_coupled`** solves nothing: y = H x + H_oc x_c and
  y_c,r = P (H′(x) c_r + (H₀ − E_r) x_c,r), the CI rows unweighted (the
  Z-vector's convention), (H₀ − E_r) x_c one sigma of the block. With one
  string set for both spins (N_α = N_β), y_c is then put **exactly** in the
  roots' sector under the swap of the α and β strings, (y_c ± y_cᵀ)/2 per
  vector, the sign the roots' parity (`create` reads it off c₀ · c₀ᵀ: + for
  even S, as `ci::Layout::symmetric` has it). An iteration built from it by
  axpy and a swap-symmetric diagonal stays in that sector to the bit. Without
  it the projection's roundoff seeds the other one, which in the full
  determinant space holds the M_s = 0 triplets: below an excited singlet
  root, they make H₀ − E_r indefinite, and a conjugate gradients that
  reaches them diverges (water's SA-2 did, at p·Ap = −1.3e-13).

### The rotations among the roots

At unequal weights the Z-vector carries the rotations among the roots, Θ_sr
for r < s, as parameters (`../zvector/README.md`, "The rotations among the
roots"). Nothing in H̃ changes for them: step 3's other roots' part is their
elimination already (both halves of a pair weighted, its transition RDMs
carry (w_r − w_s) Θ_sr, zero at equal weights). Two calls take them
explicitly:

- **`coupling`**: H_oc x_c for a block of CI vectors and no solve, through
  their transition RDMs as step 5. A part Σ_s Θ_sr c_s of x_c,r along the
  roots is the rotations' coupling to the orbitals.
- **`apply_coupled` with `theta`** (host, nroots (nroots − 1) / 2 a column):
  Θ joins x_c as its part along the roots (one gemm of C by the
  antisymmetric matrices, into the lockstep scratch), so y gains its H_oc,
  and `y_theta` returns the rows c_s · H′(x) c_r + (E_s − E_r) Θ_sr, read off
  y_c's overlaps with the roots before the projection drops them (both
  orders averaged). With `theta` empty it is the call above, unchanged.

Their scratch is a member, grown to the widest block seen: r, p, A p and one
product (dim × k nroots each) and a few k nroots vectors, plus `delta()` and
b at k blocks, and k nroots² doubles for `apply_coupled`'s rotation
matrices. Six CI blocks per column, so a block of 3N columns over a
large active space is that much device memory.

### The CI response through `recipes::pcg_solve`

With `create`'s `pcg_ci_response`, step 3's conjugate gradients are
[`recipes::pcg_solve`](../../recipes/pcg/README.md) instead of the loops
above, unpreconditioned, over an operator local to the solve: one sigma of
the vectors it is handed, then each shifted by −E_r of its own root and
projected off every root (`project`'s gemv pair). The operator takes
`pcg_solve`'s column form, which names each gathered block's columns, so the
block calls make **one** `pcg_solve` over all k · nroots vectors, each at its
own root's E_r, straight into `delta()` (#1032; #831 made one per root, since
an operator told only a count could not shift per column). The stopping rule
is passed to it unchanged: `response_tolerance` · ‖(1 − Σ_s c_s c_sᵀ) b_r‖ as
each vector's absolute tolerance (infinite for a right-hand side the roots
span, below, so no sigma runs), `response_max_iterations` under
`stop_silently`, and `check_curvature` off, so a direction with p·Ap ≤ 0
(round-off's other spins) ends that vector's solve where the sweep before left
it, without an error. The one-column call hands `pcg_solve` the ‖P b_r‖ it
already took as `norms`, so no norm is taken twice.

**What it changes.** For one column nothing but the time: it is the
one-column loop's arithmetic step for step, so δ is the loop's to the bit
and every solve takes the loop's sigmas (#831 ran both on every solve of the
CASSCF, gradient, NAC, Z-vector, NEVPT2 and Hessian suites on the gfx1200:
201,199 solves, 1,889,546 sigmas, no count and no bit of δ different; #1032's
in-process A/B on the 3080 again found none). For a block it is the loop's
recurrence but not its arithmetic (`pcg_solve`'s dot and nrm2 per vector
against the strided-batched gemm, axpy against dgmm and geam): of 43,800
block vectors over the same suites, 105 took a sigma more or fewer (#831),
where a vector met its tolerance or its cap within round-off, and δ differed
by at most 1e-8 relative, the tolerance itself (~1e-13 elsewhere). The one
block call is #831's per-root calls to the bit, vector for vector.

**Why it is still off.** #1032 cut `pcg_solve`'s per-call overheads (the
duplicate norm, the closing sync, a readback per column per reduction, the
dead last direction update) and made the block one call. That leaves the
one-column path level with the loop, but the block path still slower on the
RTX 3080, which is the repo's rule for an option. What remains is
`pcg_solve`'s per-column BLAS-1 (a dot, two axpys, a nrm2, a scal and an
axpy per vector per sweep, and the operator's axpy and gemv pair), against
the loop's few block calls: on a small CI space each is a launch-bound
kernel, and an SA3 block is nine or more vectors. Batching them would change
the sums every caller's counts and `zvector`'s bitwise z rest on. Measured
on the 3080 with the card held, main's binary (loops) and #1032's with the
option on, alternated within one hold:

| Case (`test/gtest`, unskipped) | Loops (default) | `pcg_ci_response` |
|---|---|---|
| `McscfSolveCiFloor.Psb3MeciReachesTheGradientBar` (`mcscf::solve`, one column) | 28.97, 29.23, 29.33 s | 29.17, 29.09, 29.62 s |
| `McscfSolveSpan.EthyleneSa3OverEverySingletConverges` | 6.07, 6.08, 6.12 s | 6.03, 6.10, 6.14 s |
| `McscfSolve.FullDensityFitting…/h2co_ccpvdz_sa2` | 2.175, 2.174, 2.198 s | 2.155, 2.184, 2.187 s |
| `McscfIntersectionTiming`, H₂CO skewed SA3: `df_gradient` × 2 + `df_nac` (one column) | 0.355, 0.354, 0.354 s | 0.353, 0.353, 0.353 s |
| `McscfIntersectionTiming`, H₂CO SA2: `df_intersection`'s one call (block) | 0.241, 0.235, 0.241 s | 0.243, 0.249, 0.247 s |
| `McscfIntersectionTiming`, H₂CO skewed SA3: `df_intersection`'s one call (block) | 0.283, 0.281, 0.282 s | 0.305, 0.311, 0.311 s |

Before #1032 (#831's measurement) the one-column chain was 0.355 against
0.367 s and the two block rows +8% and +22%.

### A right-hand side the roots span

When the projected right-hand side (1 − Σ_s c_s c_sᵀ) b_r is at most 1e-12
of ‖b_r‖, no conjugate gradients run on it: δ_r off the roots is zero, and
δ_r is the other roots' part alone. The one-column solve and the block one
make the same test, per vector.

**Why that is exact.** b_r = H′ c_r lies in the roots' spin, and the
projection takes off every root's part of it. What is left is b_r's part in
the roots' spin outside their span, plus the projection's round-off, ~1e-16
‖b_r‖ in every direction, the other spins' among them. A genuine part of
1e-12 ‖b_r‖ or less would move δ_r, and so H̃ x, by less than the 1e-8
relative tolerance the gradients stop at already leaves; below that the
"part" is the round-off itself. So the skip changes a product only where the
conjugate gradients would have worked on round-off.

**When it fires: a state average over every root of the spin.** Ethylene's
SA3-CAS(2,2) averages all three singlets of a CAS(2,2), so the roots span
the whole singlet space and every projected b_r is round-off: ‖b_r‖ ≈ 0.1
to 0.9 against 1e-16 to 1.7e-15 after the projection, and that round-off sits
in the M_s = 0 triplet, the only direction left. The gradients then chase it
against a tolerance relative to itself, which they never meet: the steps blew
up, H̃ reported curvatures of −1e1 to −6.6e7 from the first macro iteration,
and `mcscf::solve` ran 200 macro iterations at ‖g‖ ≈ 0.2 at both of
`test/gtest/data/casscf_meci/`'s geometries (#820). With the skip each
converges in 6 macro iterations, every step accepted. An SA2 of the same
molecule keeps 1e-3 to 6e-2 of b_r off its roots, far above the test.

### Leaving a pair's mixing out

`create`'s last argument, `mixing` (empty by default: every pair mixes, as
above), is one flag per pair of roots r < s in (r, s) lexicographic order,
`ZVector::theta`'s. A pair flagged false gets no other-roots part in step 3,
in the one-column product and the block one alike. At equal weights that
part adds (w_r − w_s)(c_s · H′ c_r)/(E_s − E_r) times the pair's transition
RDMs to H̃ x, which is zero, so nothing H̃ means changes; what it removes is
an O(1/gap) term cancelling through the transition RDMs, against weights
recovered from RDMs only to about 1e-8 relative (a solution's own are exact,
but one built by hand still has them recovered). Near a degeneracy that term is
the product's error: `mcscf::df_intersection`, which passes
`rotating_pairs(weights)` (only the pairs of unequal weight mix), measured
true residuals of 1.6e-5 at a gap of 2e-8 Eh with every pair mixing
([`../../mcscf/intersection/README.md`](../../mcscf/intersection/README.md), "The
derivation"). `delta()` then has no part along the flagged pairs' roots,
which a Z-vector projects off in any case.

The derivation, the costs and what H̃ bought the loop are in
[`../../mcscf/solve/README.md`](../../mcscf/solve/README.md), "The CI response is computed, per
product".

## Tests

None of its own: `test/gtest/mcscf_solve_gpu.cpp` drives every product
through `mcscf::solve` (ethylene's SA3 there, the span above), `test/gtest/response_zvector_gpu.cpp` `couple`
through `response::zvector`'s CI right-hand side, and
`test/gtest/response_zvectors_gpu.cpp` the block calls through
`response::zvectors`, and `test/gtest/response_zvector_rotations_gpu.cpp`
`coupling` and `apply_coupled`'s rotations through its rotations at unequal
weights. `test/gtest/mcscf_root_rotation_gpu.cpp` solves its Z-vectors with
`pcg_ci_response` too, by `Schur` (the block calls) and by `PerColumn` (one
column), against the same fixture and bars as the default.

## Files

| File | Role |
|------|------|
| `relaxed_hessian.cppm` | Module interface: `ResponseIntegrals`, `response_integrals`, `response_gradient`, `RelaxedHessian` |
| `relaxed_hessian.cpp` | H′(x) and its adjoint, the CI response's conjugate gradients (one column, and a block's in lockstep, or one `pcg_solve`) and transition RDMs, and the products |
| `CMakeLists.txt` | The module `free_breakfast.response.relaxed_hessian` |
