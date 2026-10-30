# `free_breakfast.gradient`

Analytic nuclear gradients on the GPU, namespace `free_breakfast::gradient`.
For now the closed-shell DF-RHF one, every term returned on its own:

    dE/dR = Σ D dT + Σ D dV − Σ W dS + dE_J + dE_x + dV_nn

```cpp
import free_breakfast.gradient;

auto g = gradient::df_rhf(res, molecule, basis, classes, aux, b, l, scf.c, scf.mo_energies, nocc);
// g->total, g->kinetic, g->potential, g->overlap, g->coulomb, g->exchange,
// g->nuclear_repulsion: each n_atom x 3, row-major, element 3 A + k
```

Link `free_breakfast::gradient`. The inputs are explicit — the molecule, the
orbital basis and its MMD classes, the placed aux basis, B and L from
`density_fit`, C, ε and n_occ — not pipeline types, so this module stays
below `pipeline`.

## The terms

D = 2 C_occ C_occᵀ and W = 2 C_occ diag(ε_occ) C_occᵀ. The formula is the
gradient only at a converged SCF, where W carries the orbitals' response.

- **One-electron**: `kinetic_gradient` and `potential_gradient` at D,
  `overlap_gradient` at −W (one `gmmwich`), and `molecule`'s
  `nuclear_repulsion_gradient`. These are `one_electron_terms`, unexported in
  the interface and shared with every RHF gradient in the module, whatever its
  ERIs: it checks C, ε and n_occ and returns D for the two-electron part.
- **Coulomb (DF-J).** With M = L Lᵀ and T = (mn|P), E_J = ½ Dᵀ T M⁻¹ T D, so
  dE_J = Σ D_mn d_P T′ − ½ dᵀ M′ d, d = M⁻¹ T D = L⁻ᵀ (B vec D) (a gemv and a
  trsv). Γᴶ = D ⊗ d (a rank-one gemm) against `three_center_gradient`, and
  Wᴶ = −½ d dᵀ against `metric_gradient`.
- **Exchange (DF-K).** E_x = −¼ Σ D_ml D_ns (mn|ls) = −Σ_ij (ij|ij), and with
  the fitted occupied pairs c_P,ij = (M⁻¹ T_ij)_P, dE_x = −2 Σ Γᴷ T′ + Σ Wᴷ M′,
  Γᴷ_mnP = Σ_ij C_mi C_nj c_P,ij and Wᴷ_PQ = Σ_ij c_P,ij c_Q,ij. c is `b_mo`
  over the occupied columns, then a trsm with Lᵀ; Γᴷ is a gemm over j and a
  strided-batched gemm over i straight into `three_center`'s (m, n, P) layout,
  with the −2 in its α; Wᴷ is one gemm.

PySCF's `df.grad.rhf` reports the exchange term as −½ dE_K with
E_K = ½ Σ D_ml D_ns (mn|ls); that is the same number as `exchange` here.

## Memory

**Γ is held in full**: naux nao² doubles, the size of B, plus the derivative
call's Cartesian copy and its transform's intermediate in
`res.device_workspace()`. Beside it the call holds c (naux n_occ²), the
half back-transform (naux n_occ nao), one naux² metric weight and d. At
500 AOs and 1500 fitting functions Γ alone is 3 GB. The call refuses
(Unsupported, naming the bytes) when its own buffers would not fit the
device's free memory, and `three_center_gradient` refuses its scratch the
same way; batching Γ over P is not done. naux nao² must also fit `int32_t`.

## Tests

`test/gtest/df_rhf_gradient_gpu.cpp`, against `test/gtest/data/gradient/`'s
`df_rhf` section (PySCF's `df.grad.rhf`, term by term): this tree's own
DF-RHF, converged to 1e-12 Eh and a commutator of 1e-10, then every term and
the total to 1e-8 absolute, and the total's sum over atoms below 1e-9, for
H₂O, NH₄⁺ and the distorted H₂O in cc-pVDZ / cc-pVDZ-JKFIT and H₂O in
cc-pVDZ / cc-pVDZ-acCD, and H₂O in cc-pVTZ / cc-pVTZ-JKFIT (f orbitals; about
a second, so in the fast tier); a central difference (h = 1e-4 bohr) of the DF-RHF
energy on one coordinate of H₂O / cc-pVDZ to 1e-6; and the refusals.
