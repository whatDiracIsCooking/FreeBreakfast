# Gradient

The analytic nuclear gradient of a converged, state-specific DF-CASSCF on the
GPU: `mcscf::df_gradient`, every term returned on its own,

    dE/dR = Σ D dT + Σ D dV − Σ W dS + dE₂ + dV_nn

## Module

`free_breakfast.mcscf.gradient`, target `free_breakfast::mcscf::gradient`,
re-exported by `free_breakfast.mcscf`.

```cpp
Result<mcscf::GradientTerms> mcscf::df_gradient(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux,
    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
    const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t root);
// g->kinetic, g->potential, g->overlap, g->two_electron,
// g->nuclear_repulsion, g->total: each n_atom x 3, row-major, element 3 A + k
```

`solution` is `mcscf::solve`'s under `DensityFitting::Full` over the same B;
`aux`, B and L are `pipeline.eri`'s `DensityFitted` (`aux`, `b`, `l`), `h` the
core Hamiltonian and `classes` `primitive_pair_classes` of the basis, as
`gradient::df_rhf` takes them. The inputs are explicit, not pipeline types,
so this layer stays below `pipeline`.

**One root only.** A solution averaging more than one root is `Unsupported`:
its gradient needs the orbital and CI multipliers (a Z-vector over the
state-averaged Hessian), which this call does not solve for. `root` is in the
signature already, so a caller is written against it; today it must be 0.

**The formula is the DF-CASSCF energy's gradient, and only at convergence.**
W stands in for the orbitals' response (stationarity in the orbitals), and the
CI is variational. A solution from `Direct` or `ApproxHessian` converges to
the exact CASSCF, whose gradient this is not; nothing in a `Solution` says
which mode made it, so the call cannot refuse one.

## The terms

C's columns are core (i, j), active (t, u, v, w), virtual; γ and Γ the
solution's 1- and 2-RDM, E = Σ h_pq D_pq + ½ Σ Γ_pqrs (pq|rs) over MOs.

- **One-electron.** D = 2 C_c C_cᵀ + C_a γ C_aᵀ (a gemm and a `gemmwich`),
  against `kinetic_gradient` and `potential_gradient`; and −Σ W S′ with
  W = C sym(F) Cᵀ, F `response::fock`'s generalized Fock over the fitted
  integrals (F_pq = Σ_r h_pr D_rq + Σ (pr|st) Γ_qrst). The symmetric
  connection C(R) = C (1 − ½ S′_MO) puts −½ S′_pq into U_pq, and
  ∂E/∂U_pq = 2 F_pq, so the term is −Σ F_pq S′_pq; S′ is symmetric, so only
  F's symmetric part counts (F is symmetric only to the solve's convergence).
  These four go through `gradient::one_electron_terms`, exported from
  `free_breakfast.gradient` at any D and −W, which the RHF gradients there
  call too.
- **Two-electron.** Over the fit, (pq|rs) = Σ_PQ (pq|P) M⁻¹_PQ (Q|rs), M = L Lᵀ
  the metric, so any E₂ = ½ Σ Γᴬᴼ_mnls T_mnP M⁻¹ T_lsQ differentiates to
  Σ Γ_mnP T′_mnP + Σ W_PQ M′_PQ. All of E₂ goes into **one** Γ_mnP (in
  `three_center`'s layout, P fastest) and **one** W_PQ, then one
  `three_center_gradient` and one `metric_gradient`: the derivative
  integrals are evaluated once, whatever the number of pieces. E₂ splits as

      E₂ = ½ J[D]·D − ½ J[D_a]·D_a − Σ_i Σ_pq G_pq (ip|iq) + ½ Σ Γ_tuvw (tu|vw),

  D_a = C_a γ C_aᵀ, G = diag(1_core, γ) over the occupied orbitals:

  - **J** (core–core and core–active; the active–active J is in Γ's term).
    ½ J[D_c]·D_c + J[D_c]·D_a = ½ J[D]·D − ½ J[D_a]·D_a. Each density X fits
    to d_X = M⁻¹ T X = L⁻ᵀ (B vec X) (a gemv and a trsv), and
    ½ J[X]·X differentiates as `df_rhf`'s Coulomb, Γ = X ⊗ d_X and
    W = −½ d_X d_Xᵀ: so Γᴶ = D ⊗ d − D_a ⊗ d_a and
    Wᴶ = −½ d dᵀ + ½ d_a d_aᵀ, four rank-one gemms.
  - **K** of the core with itself and with the active space.
    −¼ K[D_c]·D_c − ½ K[D_c]·D_a = −Σ_ij (ij|ij) − Σ_i,tu γ_tu (it|iu)
    = −Σ_i Σ_pq G_pq (ip|iq). With the fitted pairs c_P,ip = (L⁻ᵀ B_ip)_P,
    i core and p occupied (`b_mo`, then a trsm), and V_P,iq = Σ_p c_P,ip G_pq
    (a gemm), Γᴷ_mnP = −2 Σ_i C_mi Σ_q V_P,iq C_nq (a gemm over q and a
    strided-batched gemm over i, as `df_rhf`'s exchange) and
    Wᴷ_PQ = Σ_iq V_P,iq c_Q,iq (a gemm). The core–core block of G is `df_rhf`'s
    exchange exactly; the core–active block is the same MO form with γ in it.
  - **Active–active**, ½ Σ Γ_tuvw (tu|vw). With c_P,tu the fitted active pairs
    and Z_P,vw = Σ_tu c_P,tu Γ_tuvw (one gemm against Γ as the ncas² square
    matrix it is in `response::Fock::gamma2`), Γᵃᵃ_mnP = Σ_vw C_mv C_nw Z_P,vw
    (a gemm and a strided-batched gemm) and Wᵃᵃ = −½ Z cᵀ (a gemm).

  `two_electron` is their sum; PySCF's `df.grad.casscf` reports the same
  sum (J, K, the 2-RDM term and the auxiliary basis's response) as one part.
- **Nuclear repulsion**: `nuclear_repulsion_gradient`, through the shared
  one-electron call.

## Memory

**Γ is held in full**: naux nao² doubles, the size of B, as in
`gradient::df_rhf`, plus the derivative call's Cartesian copy in
`res.device_workspace()`. Beside it the call holds W (naux²), the two fitted
densities, G (nocc²), the core pairs and V (naux ncore nocc each), one
back-transform (naux max(ncore, ncas) nao), and the active pairs and Z
(naux ncas² each), and refuses (`Unsupported`, naming the bytes) when those
would not fit the device's free memory, before anything is allocated.
`casscf_integrals` and `response::fock` hold their own nmo² ncas² integrals
for the call. naux nao² must fit `int32_t`.

## Tests

`test/gtest/mcscf_gradient_gpu.cpp`, against
`test/gtest/data/casscf_gradient/`'s state-specific cases (PySCF's
`df.grad.casscf` over a DF-RHF `_scf` sharing the CASSCF's `with_df`, so its W
is fitted as the energy is): H₂O and the skewed H₂O in cc-pVDZ /
cc-pVDZ-JKFIT, CAS(4,4). Each runs this tree's own `DensityFitting::Full`
solve from the exact RHF and the case's active orbitals to ‖g‖ < 1e-9, then
every term and the total to 1e-8 absolute, the total's sum over atoms below
1e-9, and the two refusals (a root past the solution's, and a solution of two
roots).
