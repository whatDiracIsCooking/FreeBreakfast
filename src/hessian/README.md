# `free_breakfast.hessian`

Analytic nuclear Hessians on the GPU, namespace `free_breakfast::hessian`.
So far the **response part** of the closed-shell DF-RHF one: everything the
orbitals' first-order response adds to the explicit part (the
second-derivative integrals contracted with D, W and D ⊗ D, and V_nn's), in
the symmetric form of the design (`rhf_hessian.tex` eq. hess-sym, Algorithm 1
lines 2–9).

```cpp
import free_breakfast.hessian;

auto r = hessian::df_rhf_response(res, molecule, basis, classes, aux, b, l, c, mo_energies, nocc);
// r->hessian: 3 n_atom x 3 n_atom, element x + 3 n_atom y, symmetric
// r->cphf.solution: U^x, (n_vir n_occ) x 3 n_atom, U^x_ai at a + n_vir i
```

Link `free_breakfast::hessian`. The inputs are explicit — the molecule, the
orbital basis and its MMD classes, the placed aux basis, B (a device buffer,
as `response::cphf` takes it) and L from `density_fit`, C (nao × nmo), ε and
n_occ — not pipeline types, so this module stays below `pipeline`, as
`gradient` does.

## The terms

Per nuclear coordinate x = 3 A + k, with S^x and the skeleton Fock derivative
F^(x) = T^x + V^x + J^(x) − ½ K^(x) at fixed D, their MO blocks
X_pi = C_pᵀ X C_i (every MO p, occupied i), D_S^x = 2 C_o S^x_oo C_oᵀ and
G = J − ½ K over the fit:

    B^x_ai = −F^(x)_ai + ε_i S^x_ai + G(D_S^x)_ai,      M U^x = B^x
    H_xy   = −2 (B^x·U^y + B^y·U^x) − 2 (Σ_xy + Σ_yx) + 2 Ω_xy + Θ_xy
    Σ_xy   = Σ_ij S^x_ij F^(y)_ij
    Ω_xy   = Σ_ij S^x_ij S^y_ij (ε_i + ε_j)
    Θ_xy   = tr[D_S^x G(D_S^y)] = 2 Σ_ij S^x_ij G(D_S^y)_ij

and the full Hessian is H plus the explicit part. B^x·U^y is Bᵀ M⁻¹ B,
symmetric only for an exact solve; its symmetrized half is taken, which
discards exactly the antisymmetric part the CPHF residual leaves, so H is
symmetric to rounding. (The design's −4 Bᵀ U is the unsymmetrized term.)

## The build

1. **S^x and F^(x) to MO blocks.** `overlap_derivative`, then
   `kinetic_derivative` + `potential_derivative` + `df_jk_derivative`'s
   J^(x) − ½ K^(x), each a 3 n_atom stack of nao² AO matrices. Each stack is
   carried to its MO block C^T X^x C_o by one strided-batched gemm
   (X^x C_o, every x) and one gemm (Cᵀ over all of them side by side), and
   the AO stack freed.
2. **G(D_S^x)**, one DF J/K build per coordinate: with R = C_o S^x_oo,
   D_S^x / 2 = C_o Rᵀ, so J is `df_coulomb` of D_S^x and ½ K the two-factor
   `df_exchange` of (C_o, R), never factoring D_S^x (which is indefinite).
   Carried to its MO block like the others.
3. **B** in `response::cphf`'s layout: the vo blocks of a stack are one
   n_vir × (n_occ 3 n_atom) matrix of leading dimension nmo, so B is one geam
   (−F + G), one dgmm (S diag(ε_i)) and one axpy over every coordinate.
4. **The CPHF**, `response::cphf`, every column in lockstep. Its result is
   returned as is: U^x, the residuals and the sweep count. U^x is what the
   dipole derivative for IR intensities reuses (design §10, step 6).
5. **The terms**, over the oo blocks packed to n_occ² × 3 n_atom: P = Bᵀ U +
   S_ooᵀ F_oo (two gemms), H = −2 (P + Pᵀ) (one geam), then
   H += 2 S_ooᵀ (G_oo + S_oo ∘ E), E_ij = ε_i + ε_j (a dgmm, an axpy and a
   gemm). Only the 3 n_atom × 3 n_atom H comes back to the host.

## Memory

The call holds three MO stacks (nmo n_occ 3 n_atom doubles each), the
right-hand sides (n_vir n_occ 3 n_atom), and one coordinate's J/K build
(three nao² and nao n_occ). While F^(x) is built it holds three AO stacks of
3 n_atom nao² beside them (F^(x), J^(x), K^(x)), plus `df_jk_derivative`'s
own scratch: at 500 AOs and 30 atoms that is 540 MB. The call refuses
(Unsupported, naming the bytes) when its own peak would not fit the device's
free memory; `df_jk_derivative` refuses its own scratch the same way. The
`response::cphf` solve allocates its own. 3 n_atom nao² must fit `int32_t`.

The J/K builds are one density per call (3 n_atom for G(D_S^x), and 3 n_atom
per CPHF sweep): the batched multi-density build the design suggests does
not exist yet, in `jk` or here.

## Tests

`test/gtest/df_rhf_hessian_response_gpu.cpp` (suite `HessianDfRhfResponse`),
against `test/gtest/data/rhf_hessian/` (`tools/emit/rhf_hessian_reference.py`):
for the skew water and a skew formaldehyde in cc-pVDZ / cc-pVDZ-JKFIT,

- H against `pyscf.df.hessian.rhf`'s total less `partial_hess_elec` and
  `hess_nuc` over the same fitting set, to 1e-7 (both CPHFs to ~1e-10 in
  the residual; the tool solves PySCF's column by column, past its block
  Krylov's stall, and checks it against a dense solve of M);
- U^x against PySCF's CPHF solutions, to 1e-8;
- H's asymmetry, to 1e-14 of max |H|;
- the AO form (design eq. hess-ao), tr(D^y F^(x)) − tr(W^y S^x) with
  W^y = ½ (D^y F D + D F^y D + D F D^y), built in the test from the tree's
  S^x, F^(x), overlap and DF J/K and the returned U^x: against H, and its own
  asymmetry (the CPHF residual's), each to 1e-8;
- the refusals.

Measured on the gfx1200 (HIP): |H − PySCF| 7.7e-12 (water) and 9.1e-12
(formaldehyde), asymmetry 2.2e-16, |U − PySCF| 3e-11 and 5e-11, the AO form
within 7e-12 of H and asymmetric by 1.4e-11 and 7.4e-12, in 15 and 17 CPHF
sweeps, under 2 s a call.

## Files

| File | Role |
|------|------|
| `hessian.cppm` | Module interface: `hessian::RhfResponse`, `hessian::df_rhf_response` |
| `df_rhf_response.cpp` | The MO blocks, the G(D_S^x) builds, B, the CPHF and the terms |
| `CMakeLists.txt` | Build configuration |
