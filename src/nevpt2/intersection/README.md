# `free_breakfast.nevpt2.intersection`

What a QD-NEVPT2 minimum-energy conical intersection search needs of one QD
pair (P, Q), SC or PC, on a DF-SA-CASSCF: the gradients g_P and g_Q and the
interstate coupling h_PQ at one set of H_eff eigenvectors R, with no refusal
of a degenerate pair, and two gap sensitivities that bound how far the
pair's gap moves when the SA-CASSCF is not fully converged. Link
`free_breakfast::nevpt2::intersection`; `free_breakfast.nevpt2` re-exports
it.

    qd_sc_intersection(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                       p, q, batch, options = {}) -> QdIntersection
        options.shared  one several-W gradient call (default) or three W-form calls
        options.nfrozen the frozen core, 0 by default: H_eff and every G(W) over it
    qd_pc_intersection(...the same...) -> QdIntersection
        g_p, g_q       dE_P/dR = G(u_P u_Pᵀ), dE_Q/dR = G(u_Q u_Qᵀ), n_atom × 3 (hartree / bohr)
        h_pq           h_PQ = u_Pᵀ (∇H_eff) u_Q = G(W^PQ), W^PQ = ½ (u_P u_Qᵀ + u_Q u_Pᵀ)
        eigenvalues    H_eff's, ascending: the QD energies
        eigenvectors   R, column K at [K n], each signed so its largest component is positive
        heff_gap       |E_Q − E_P|, reported, never gated
        root_gap       the smallest CASSCF |E_r − E_s|
        gap_sensitivity_orbital   s_o
        gap_sensitivity_ci        s_c

The arguments are `qd_sc_coupling`'s (`../gradient/`), without its fold.
`options.nfrozen` is the frozen core (#875): `qd_inputs` and the W forms
take it, so H_eff, R and the three G(W) are all the frozen-core
functional's (`../gradient/README.md`, "The frozen core"); s_o and s_c read
the multipliers that come back, and need nothing of their own. The
consumer is the QD MECI mode of the optimize step (#785), which hands g_P,
g_Q and h_PQ to `geometry::optimize_meci` and reads s_o and s_c for its gap
bar.

## How it is computed

1. H_eff = diag(E_r) + ½ (C + Cᵀ) over `nevpt2::qd_inputs` and
   `sc_effective_hamiltonian` (or `pc_effective_hamiltonian`), diagonalized
   once on the device. R is signed as `QdScLagrangian::eigenvectors`.
2. G(W) at u_P u_Pᵀ, u_Q u_Qᵀ and W^PQ, all built from that one R: by
   default one several-W call of `qd_sc_gradient` (or `qd_pc_gradient`),
   which shares the adjoint's forward arrays, the CI-relaxed Hessian and one
   `mcscf::zvectors` solve of the three columns (`../gradient/README.md`,
   "Several W's at once"); with `options.shared = false`, three W-form
   calls, each its own Lagrangian and Z-vector solve (#784's path).
3. Each solve's true residual (`ZVector::residual`, one more product at the
   solution, not the conjugate gradients' recurrence) must be at or below the
   solve's own tolerance, `options.zvector_tolerance` (1e-10 by default, the
   one parameter the W-form calls solve to), or the call returns `Numerical`
   naming the W (the gradient layer checks every column too). The CI
   response inside H̃ is converged to `mcscf::response_tolerance_for` it
   (`../gradient/README.md` "The Z-vector's floor"). Near a degeneracy the recurrence
   can report convergence while the true residual has not (#786).
4. s_o and s_c from the three solves' multipliers, below.

**No refusal of a degenerate pair.** `qd_*_coupling` and the state form of
`qd_*_gradient` refuse when E_P and E_Q are within `mcscf::kRootGapFloor`,
because each needs u_K's own derivative. The W forms do not: G(W) is linear
in W and nothing divides by E_Q − E_P. At exact degeneracy u_P and u_Q are
any orthonormal pair of the two-state space, and g_P, g_Q, h_PQ turn with
them, but the branching plane span{g_Q − g_P, h_PQ} and ḡ = (g_P + g_Q)/2
do not. That is all the optimizer uses. An MECI's own convergence takes the
gap below the floor, so this is the case that matters.

**Unequal SA weights are allowed.** The only division by a gap is
`mcscf::root_pair_multipliers`' by the CASSCF roots', which keeps its
refusal. (`mcscf::df_intersection` refuses unequal weights because there the
roots themselves are the crossing pair; here the QD states are.)

## Sharing the three calls

`QdIntersectionOptions::shared` (#787) picks the path. Both give the same
result to the Z-vector solves' convergence (the tests hold them to 1e-9 of
the largest summand at the same R; PC's s_c to 1e-7, #784's ruling). The
shared path builds the adjoint's forward arrays and RDMs once instead of
three times, and the Hessian and the CI objects once, and solves the three
right-hand sides in one block conjugate-gradient run. H_eff's own walk
(step 1) is still separate: R must exist before any W does.

The derivative passes stay three (four when a W's ρ needs the split pass):
stacking them would need a multi-density derivative kernel, since one
`geri3` kernel is ~50 ms of each ~57 ms `df_gradient` call (#773, PR #776).
That is out of scope here, and it is why the gain is modest.

Measured on the RTX 3080 (CUDA), pair (0, 1), `Nevpt2QdIntersection.Timing`
(one warm-up of each path, then three of each, alternating; min / mean):

| case | scheme | three calls | shared |
|---|---|---:|---:|
| `h2co_skew_ccpvdz_sa3` | SC | 0.636 / 0.637 s | 0.568 / 0.568 s |
| `h2co_skew_ccpvdz_sa3` | PC | 0.653 / 0.654 s | 0.585 / 0.585 s |
| ethylene SA2-CAS(2,2)/cc-pVDZ | SC | 0.566 / 0.567 s | 0.538 / 0.539 s |
| ethylene SA2-CAS(2,2)/cc-pVDZ | PC | 0.558 / 0.559 s | 0.530 / 0.531 s |

The shared path is faster on every case, so it is the default. The
ethylene case is `data/casscf_meci/`'s start geometry with its SA2 weights:
#787 asked for #785's SA3, which this tree's CASSCF did not converge from
RHF orbitals at the MECI, the start or planar ethylene (the test's comment
has the numbers). The timing case is skipped unless
`FBF_QD_INTERSECTION_TIMING` is set.

## The gap sensitivities

Write the SA-CASSCF's parameters p (orbital rotations x over the packed
pairs, the CI vectors in `gen_g_hop`'s c → c + δc coordinates, and the
rotations Θ among the roots) and its stationarity conditions G(p) = (g,
2 w_r P (H₀ − E_r) c_r, Θ's). A solution with residual G sits δp = A⁻¹G +
O(‖G‖²) from the exact one. An element a_W = tr(W H_eff) is not stationary in
p, so it moves at first order, δa_W = −z^W · G, where z^W is the Z-vector of
the W form G(W): exactly the `lagrangian.multipliers` (z_o, z_c, Θ) of the
W-form call (#785's derivation).

In H_eff's eigenbasis the pair's block is [[a, b], [b, d]], with gap
√((a − d)² + 4b²), a = a_{u_P u_Pᵀ}, d = a_{u_Q u_Qᵀ} and b = a_{W^PQ}. The gap
has no derivative at a degeneracy, but it moves by at most √(δ(a − d)² +
4 δb²), and the triangle inequality in that plane gives

    |δgap| ≤ s_o ‖g‖ + s_c √n_roots ρ
    s_o = √(‖z_o^Q − z_o^P‖² + 4 ‖z_o^PQ‖²)
    s_c = √(‖ẑ_c^Q − ẑ_c^P‖² + 4 ‖ẑ_c^PQ‖²),   ẑ_c = (2 w_r z_c,r)_r

**The pairing, pinned.** Each block of the multipliers pairs with one
residual the SA-CASSCF reports:

- **The orbital multipliers, factor 1, with ‖g‖** =
  `mcscf::Solution::gradient_norm`. That norm is `response::Fock`'s
  gradient alone, g_pq = 2 (F_pq − F_qp) = dE_SA/dx_pq over the packed
  pairs, the very vector the Z-vector system's orbital rows are the
  derivative of.
- **The CI multipliers, each root's block scaled by 2 w_r, with ρ** =
  `ci::Roots::davidson.max_residual_norm`. The Z-vector system's CI rows are
  the true ∂L/∂c_r divided by 2 w_r (`../../mcscf/zvector/README.md`), so
  z_c,r multiplies the condition 2 w_r P (H₀ − E_r) c_r. That is the
  Davidson residual, which is orthogonal to every root (Galerkin), so P
  drops nothing. The spin penalty adds nothing to z_c · r: z_c is in the
  roots' spin and λ (S² − S(S+1)) c is not. √n_roots turns the largest
  residual into a bound on the stacked one.
- **Θ needs no term.** Its condition c_s · H₀ c_r = c_s · r_r is zero by
  Rayleigh–Ritz (r_r is orthogonal to the subspace c_s lies in), at any
  weights.
- **z^c needs none either.** The core and virtual canonicalization over the
  SA Fock is a direct diagonalization, its off-diagonal residual round-off.

Both sensitivities are invariant to turning u_P and u_Q within their pair:
(z^Q − z^P)/2 and z^PQ turn as the pair (x/2, h) does, by twice the angle,
and each norm is Euclidean.

## Tests

`test/gtest/nevpt2_qd_intersection_gpu.cpp`, SC and PC, on this tree's own
DF-SA-CASSCF of `test/gtest/data/qd_nevpt2/`'s cases at #696's bars:

- `Nevpt2QdIntersectionWForms.MatchesWForms/<case>_<sc|pc>`: g_P, g_Q and
  h_PQ each against its own W-form call at the result's own u's, every pair
  of `h2co_skew_ccpvdz_sa3` and `h2co_skew_ccpvdz_sa3_weighted`, to 1e-9 of
  the largest summand of the derivative passes; s_o and s_c against their
  definitions on those calls' multipliers, to 1e-9 of the largest norm
  summed. Independently converged Z-vectors, hence 1e-9 (#784's bar
  amendment); PC's s_c alone at 1e-7, because PC's CI adjoint ȳ is
  reproducible between identical calls only to ~5e-8 (#729), and z_c is
  solved from it: two runs' s_c measured up to 4.8e-9 apart, relative. The
  check is for a wrong pairing factor (2 or 2 w_r), which is O(1)
  ([#784's ruling](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/784#issuecomment-6616656181)).
- In both of the above, the shared path (#787): one several-W call at the
  same u's against the three W-form calls, g_P, g_Q and h_PQ to 1e-9 of
  the larger side's largest summand, s_o and s_c to 1e-9 of the largest
  norm summed (PC's s_c 1e-7). `MatchesWForms` also runs the intersection's
  non-default path once per case, pair (0, 1), against the same calls.
- `Nevpt2QdIntersection.DegeneratePair`: `h2co_ccpvdz_sa2` (C2v, roots 1A1
  and 1A2) made exactly degenerate at a consistent solution: h + λ v vᵀ,
  v = S C e_k on the active orbital the roots' occupations differ most in,
  the DF-SA-CASSCF re-converged at #696's bars for each λ, λ by secant
  until H_eff's diagonal agrees to 1e-11. There `qd_*_coupling` refuses,
  and the intersection returns and matches the W forms; s_o and s_c are
  unchanged when u_P, u_Q are turned within the pair, against separate
  W-form calls (1e-9; PC's s_c 1e-7) and by linearity on one set of
  multipliers (1e-10). #784 named `Nevpt2QdNac.Refuses`' setup instead, one CASSCF root energy
  moved by hand; there the roots are no longer eigenvectors at their
  energies, and every W form's own gate refuses its Lagrangian (SC
  active-active asymmetry 2.4e-1), so it cannot carry a W-form comparison.
- `Nevpt2QdIntersectionBound.HoldsForAStoppedSolve/<sc|pc>`: a DF-SA-CASSCF
  of `h2co_skew_ccpvdz_sa3` from the converged orbitals turned by a fixed κ,
  stopped at `conv_tol_grad` = 1e-5 and canonicalized; every pair's gap
  moves by at most s_o ‖g‖ + s_c √n_roots ρ, the stopped solve's own ‖g‖
  and ρ, s_o and s_c the converged point's.
- `Nevpt2QdIntersection.ZvectorTolerance`: `h2co_ccpvdz_sa2`'s pair
  (0, 1), SC, at `options.zvector_tolerance` 1e-12. The intersection
  succeeds and g_P, g_Q and h_PQ agree with the default call's to 1e-9 of
  the W form's largest summand. The W form at u_P u_Pᵀ with the same
  trailing tolerance reports a true residual of at most 1e-12. A tolerance
  of 0 or below is `InvalidConfig`.
- `Nevpt2QdIntersection.Psb3TrueResidual` (slow tier): PSB3's
  DF-SA3-CASSCF(6,6)/cc-pVDZ with C13=C14 twisted 90° (root gap 2.9e-2 Eh).
  The SC intersection of QD states 0 and 1, and the SC state gradient of
  state 0 (one column, whose true residual is now checked too), both
  succeed: every true residual is at most 1e-10. Before the CI response
  inside H̃ was converged to `mcscf::response_tolerance_for` the Z-vector
  tolerance, the intersection was refused at 2.55e-10 after 31 products
  (`../../mcscf/intersection/` "The floor under the true residual"). Now
  the three columns end at their recurrence's 9.8e-11, 6.0e-11 and
  4.2e-11.
- `Nevpt2QdIntersection.Refuses`: P = Q, a state outside the average, and a
  class PC refuses (PC's error, as `qd_pc_coupling`'s).
- `Nevpt2QdIntersection.Timing`: the table above; a measurement, skipped
  unless `FBF_QD_INTERSECTION_TIMING` is set.

## Files

| File | Role |
|------|------|
| `intersection.cppm` | Module interface: `QdIntersectionOptions`, `QdIntersection`, `qd_sc_intersection`, `qd_pc_intersection` |
| `intersection.cpp` | H_eff and R, the shared or three W-form calls, the residual check, s_o and s_c |
| `CMakeLists.txt` | Build configuration |
