# `free_breakfast.hessian`

Analytic nuclear Hessians on the GPU, namespace `free_breakfast::hessian`:
the closed-shell DF-RHF one, **`df_rhf`**, every term of it returned beside
the symmetrized total. Its parts are entry points of their own: the
**response part**, everything the orbitals' first-order response adds to the
explicit part (the second-derivative integrals contracted with D, W and
D ⊗ D, and V_nn's), in the symmetric form of the design (`rhf_hessian.tex`
eq. hess-sym, Algorithm 1 lines 2–9); and the explicit part's
**two-electron terms**, the fitted Coulomb and exchange energies
differentiated twice at fixed orbitals, the fit's cross terms included (eqs.
dfj, dfk). Beside them, the **dipole derivative** the same CPHF solution
gives, `rhf_dipole_derivative`, and the **IR intensities** it projects to,
`infrared_intensities`. For DF-CASSCF: the **CASSCF two-electron
explicit part**, `df_casscf_two_electron` (`casscf_hessian.tex` eq. df2),
the **fixed-parameter second derivative** E_xy built on it,
`df_casscf_exy` (eq. exy) and, with a rotation, its derivative along
C → C (1 + ε K) (a state average's **L_xy z_o part**), the
**state-specific Hessian**, `df_casscf` (eq. ss), its CP-CASSCF step λ^x on
its own, `df_casscf_cp`, a state average's **z terms** at one root's
multipliers or a QD-NEVPT2 Lagrangian's, `df_sacasscf_z` (L_xy's z part
and L_xλ, eq. sa), its
CP-SA-CASSCF step on its own, `df_sacasscf_cp`, and for the
NEVPT2 Hessian (milestone 24) the **parameter response and total tangent**
t_x of (h_MO, B_MO, c), `df_nevpt2_tangents`, and the **DF second
derivative at an MO adjoint**, Σ h̄ h^xy + Σ Λ B^xy, `df_adjoint_xy`, and
the **SC- and PC-NEVPT2 Hessians** they assemble into, `df_sc_nevpt2` and
`df_pc_nevpt2`, and for the QD-NEVPT2 Hessian the same tangent over a state
average, `df_qd_nevpt2_tangents`, all below.

```cpp
import free_breakfast.hessian;

auto h = hessian::df_rhf(res, molecule, basis, classes, aux, b, l, c, mo_energies, nocc);
// h->hessian: 3 n_atom x 3 n_atom, element x + 3 n_atom y, exactly symmetric;
// h->one_electron, h->nuclear_repulsion, h->two_electron.{coulomb, exchange},
// h->response.hessian: the terms; h->response.cphf: U^x and the residuals;
// h->asymmetry: max |H - H^T| of the terms' sum, discarded by symmetrizing
```

```cpp
auto r = hessian::df_rhf_response(res, molecule, basis, classes, aux, b, l, c, mo_energies, nocc);
// r->hessian: 3 n_atom x 3 n_atom, element x + 3 n_atom y, symmetric
// r->cphf.solution: U^x, (n_vir n_occ) x 3 n_atom, U^x_ai at a + n_vir i
```

```cpp
auto e2 = hessian::df_rhf_two_electron(res, basis, classes, aux, b, l, c, nocc);
// e2->coulomb, e2->exchange: each 3 n_atom x 3 n_atom, symmetric
```

```cpp
auto d = hessian::rhf_dipole_derivative(res, molecule, basis, classes, c, nocc,
                                        h->response.cphf.solution);
// *d: dmu_c/dx, 3 n_atom x 3 row-major, (x, c) at 3 x + c
auto ir = hessian::infrared_intensities(vibrations.modes, *d);   // km/mol per mode
```

Link `free_breakfast::hessian`. The inputs are explicit — the molecule, the
orbital basis and its MMD classes, the placed aux basis, B (a device buffer,
as `response::cphf` takes it) and L from `density_fit`, C (nao × nmo), ε and
n_occ — not pipeline types, so this module stays below `pipeline`, as
`gradient` does.

## The whole Hessian

`df_rhf` is the design's Algorithm 1 end to end:

    H = H_1e + V_nn^xy + E_J^xy + E_x^xy + H_resp,
    H_1e = Σ_mn D_mn (T + V)^xy_mn − W_mn S^xy_mn,   D = 2 C_o C_oᵀ, W = 2 C_o ε_o C_oᵀ

`overlap_hessian` at −W, `kinetic_hessian` and `potential_hessian` at D
(`integrals/one_electron/hessian`, the diagonal blocks by translational
invariance), `nuclear_repulsion_hessian` (`molecule`), then
`df_rhf_two_electron` and `df_rhf_response` below. D and −W are made as
`gradient::df_rhf` makes them (a gemm and a gmmwich) and freed before the
two-electron calls, which each free their own before the next runs, so the
call's peak is the larger of theirs. The five terms are summed on the host
and the sum symmetrized; `asymmetry` is what that discards, the terms'
rounding (each is symmetric to rounding), while what the response part's
own symmetrization of Bᵀ U discards is bounded by the CPHF residuals it
reports. PySCF's `pyscf.df.hessian.rhf` total, over the same fitting set,
with auxbasis response 2.

Its inputs are checked as `df_rhf_response` checks them, before anything
runs. The memory and the `int32_t` bounds are the parts'.

### Tests

`test/gtest/df_rhf_hessian_gpu.cpp` (suite `HessianDfRhf`), against
`test/gtest/data/rhf_hessian/`'s `total`, `one_electron`,
`nuclear_repulsion`, `coulomb`, `exchange` and `response`
(`tools/emit/rhf_hessian_reference.py`), for the skew water and the skew
formaldehyde in cc-pVDZ / cc-pVDZ-JKFIT and the skew water in cc-pVTZ /
cc-pVTZ-JKFIT (f orbital shells, g fitting shells):

- the total and each term to 1e-7 absolute (the cc-pVTZ case under
  `FBF_SKIP_IF_FAST_TESTS()`); the sum's asymmetry to 1e-10
  of max |H|; the translational sum rule Σ_A H[3A + k][y] to 1e-8; every
  CPHF residual under 10 × its 1e-10 tolerance;
- the water against the tree's own central differences of `gradient::df_rhf`
  (the `[hessian]` step, h = 1e-3 bohr, each displacement a full DF-RHF SCF)
  to 1e-5;
- at PySCF's DF-RHF/cc-pVDZ minimum of water (largest gradient component
  1.5e-12), six mass-weighted eigenvalues under 1e-6 (hartree / bohr² / amu,
  against the bend's 0.1) and `geometry::harmonic_analysis`'s frequencies
  against PySCF's `thermo.harmonic_analysis` of its own Hessian to 1e-3 cm⁻¹;
- the refusals.

Measured on the RTX 3080 (CUDA): |H − PySCF| 4.4e-11 (water), 9.4e-11
(formaldehyde) and 1.3e-10 (cc-pVTZ water) of max |H| 0.72, 0.97 and 0.70,
the Coulomb term the largest part of it each time; asymmetry before
symmetrizing ≤ 1.4e-14, sum rule ≤ 8.7e-12; 1.4 s, 2.9 s and 10.9 s a call
(so the cc-pVTZ case is slow-gated). The central differences are 3.6e-7
from the analytic Hessian. At the minimum the largest rigid-body eigenvalue
is 1.1e-11 against the bend's 0.12, and the frequencies (1776.03, 4113.30,
4211.68 cm⁻¹) are 4.3e-6 cm⁻¹ from PySCF's.

## The dipole derivative and IR intensities

`rhf_dipole_derivative` is design §10 step 6: with μ_c = ⟨m| r_c |n⟩ about
the origin (`dipole`), electrons of charge −1 and D^x = D_U^x − D_S^x
(rhf_hessian.tex eq. dy),

    dμ/dx = −tr(D μ^x) − tr(D^x μ) + Z_A e_k
          = −tr(D μ^x) − 4 Σ_ai U^x_ai μ_ai + 2 Σ_ij S^x_ij μ_ij + Z_A e_k

with U^x the CPHF solution `df_rhf` already holds (`response.cphf.solution`),
so the derivative costs no solve of its own; the density fit enters only
through U^x.

1. **The explicit term**, −tr(D μ^x): `dipole_gradient`
   (`integrals/one_electron/dipole_gradient`, FelizJueves's `gmu_` kernels)
   at M = D = 2 C_o C_oᵀ, negated. The origin is fixed, so only the basis
   functions move.
2. **μ's MO blocks**: the three spherical dipole matrices (`dipole`, carried
   by `MatrixAOPair`), each to C^T μ_c C_o by two gemms, the three nmo × nocc
   slabs side by side.
3. **The U term**: the slabs' vo blocks are one nvir × 3 nocc matrix of
   leading dimension nmo, packed by one geam into three columns of U's
   (n_vir n_occ) rows, then −4 Uᵀ μ_vo, one gemm, 3 n_atom × 3.
4. **The S term**: P_c = C_o μ_c,oo C_oᵀ (two gemms each), so
   Σ_ij S^x_ij μ_c,ij = ⟨S^x, P_c⟩, and S^x (`overlap_derivative`, the
   n_atom × 3 nao² stack read as one nao² × 3 n_atom matrix) gives all of
   them by one gemm, 2 S^xᵀ P, added.
5. The 3 n_atom × 3 result comes back to the host, where the explicit term
   and the nuclear charges are added.

For a neutral molecule dμ/dx does not depend on the origin, and its sum over
the atoms vanishes to the CPHF's tolerance (a translation does not change the
dipole); for a charged one it is the origin-0 derivative. The dipole
integrals stop at d (`kDipoleMaxL`), so a basis with an f shell is refused
(`dipole`'s InvalidConfig). Memory: S^x (3 n_atom nao²) and
`overlap_derivative`'s own scratch, beside three dipole pairs and the MO
blocks; past the device's free memory that is Unsupported.

`infrared_intensities` projects it onto the normal modes,
I_q = K |Σ_x L_qx dμ/dx|², L as `geometry::harmonic_analysis` returns them
(each unit mass-weighted eigenvector over √(m_A / amu), PySCF's `norm_mode`),
so dμ/dQ is in e / amu^½ and K = N_A / (12 ε₀ c²) e² / u = 974.88 km/mol per
e² / amu, from `core/constants`'s N_A, ε₀ and e with the speed of light and
the dalton beside them in `infrared.cpp`.

### Tests

`test/gtest/infrared_gpu.cpp`, against `test/gtest/data/dipole_derivative/`
(`tools/emit/dipole_derivative_reference.py`: PySCF's dipole derivative from
its own CPHF, solved column by column to 1e-12, checked there against central
differences of its DF-RHF dipole to 1e-5 and measured 1e-7 to 6e-7):

- `HessianDipoleDerivativeGolden`: the skew water, the skew formaldehyde
  and the water at PySCF's DF-RHF minimum in cc-pVDZ / cc-pVDZ-JKFIT, the
  call fed PySCF's C and U (`data/rhf_hessian/`), to 1e-8 absolute, and the
  sum over the atoms to 1e-8;
- `HessianDipoleDerivative.Refusals`, including an f shell;
- `HessianInfraredIntensities`: a hand-made mode's intensity, a mode's sign
  not mattering, and the shape refusals;
- `PipelineInfrared`: the `[hessian]` step end to end (`pipeline/hessian`).

Measured on the RTX 3080 (CUDA): |dμ/dx − PySCF| 6.0e-12 (water), 3.4e-11
(formaldehyde) and 8.0e-12 (the minimum) of max |dμ/dx| 0.71, 1.08 and 0.72,
sums over the atoms ≤ 4.5e-14, under 10 ms a call. At the minimum the step's
own SCF and CPHF give intensities 80.815, 21.171 and 60.556 km/mol for the
bend and the two stretches, 5.9e-9 km/mol from PySCF's.

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
   returned as is: U^x, the residuals and the sweep count. U^x is what
   `rhf_dipole_derivative` reuses for the IR intensities (design §10, step 6).
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

## The two-electron explicit part

`df_rhf_two_electron`, with T_mn,P = (mn|P), M = (P|Q) = L Lᵀ,
u = Tᵀ vec D, d = M⁻¹ u, the fitted occupied pairs c_ij = M⁻¹ T_ij and
R^x_ij = T^x_ij − M^x c_ij:

    E_J^xy = u^xyᵀ d − ½ dᵀ M^xy d + g^xᵀ M⁻¹ g^y,           g^x = u^x − M^x d = 2 Σ_i R^x_ii
    E_x^xy = −Σ_ij (2 T^xyᵀ_ij c_ij − c_ijᵀ M^xy c_ij + 2 R^xᵀ_ij M⁻¹ R^y_ij)

with E_J = ½ uᵀ M⁻¹ u and E_x = −Σ_ij T_ijᵀ M⁻¹ T_ij, PySCF's
`_partial_hess_ejk` ej and −ek (`partial_hess_elec` is e1 + ej − ek).

1. **The fitted pairs**, c_P,ij, as `gradient::df_rhf` makes them (`b_mo`
   over the occupied columns, then a trsm with Lᵀ); d = 2 Σ_i c_ii is one
   gemv against a vector of twos over the diagonal pairs, leading dimension
   naux (n_occ + 1).
2. **The second-derivative pieces** are `gradient::df_rhf`'s weights handed
   to #593's contractions: `three_center_hessian` at Γ = D ⊗ d and
   `metric_hessian` at −½ d dᵀ for E_J; the same at Γᴷ_mnP = −2 Σ_ij C_mi
   C_nj c_P,ij and Wᴷ = Σ_ij c_ij c_ijᵀ for E_x.
3. **The cross terms** are Gram matrices of r^x = L⁻¹ R^x (each n_occ² naux)
   and of L⁻¹ g^x = 2 Σ_i r^x_ii. One **atom block** is R^x for the three
   coordinates x of atom A (3 naux n_occ² doubles), and one walk over the
   aux atoms builds a **batch** of them, atoms A₀ … A₀ + s − 1. With X = `three_center_derivative`'s
   block of aux atom B (the derivative on m's centre), d(mn|P)/dA_k =
   −[m on A] X_mn − [n on A] X_nm + [P on A] (X_mn + X_nm), so with
   Y_ij = Σ_{m on A, n} C_mi C_nj X_mn,P and Z the same over every m,
   T^x_ij = −(Y_ij + Y_ji) + [P on A] (Z_ij + Z_ji) for the rows P on B:
   one strided-batched gemm X C_o over every (k, m), made once per aux atom
   for the whole batch, then per atom of the batch and k two more (batched
   over j and over i, so both orders land without a transpose) accumulating
   straight into that block's rows. After the walk,
   (M^x c)_P = [P on A] Σ_Q M1[k][P][Q] c_Q + Σ_{Q on A} M1[k][Q][P] c_Q
   (`metric_derivative`'s M1, two gemms per atom and k), one trsm by L over
   the whole batch, and three gemvs per atom for the Coulomb vectors.
4. **The contraction**, one pair of batches (a, b ≤ a) at a time: the
   3s × 3s blocks −2 r_aᵀ r_b (exchange) and t_aᵀ t_b (Coulomb) by two
   gemms, downloaded and written to H_AB and H_BA for every A in a and B in
   b. The walk is where the time goes: `three_center_derivative` gives every
   m for one aux atom, of which a block uses only the rows m on A (and P on
   A), so the walk's cost is shared by every block it builds. When every
   atom's block fits the budget (below) the batch is the whole molecule and
   there is **one walk**. Otherwise two batches of s = ⌊slots / 2⌋ atoms are
   held, batch a across its inner loop and batch b rebuilt for each pair, so
   n_b (n_b + 1) / 2 walks for n_b = ⌈n_atom / s⌉ batches; at s = 1 that is
   the n_atom (n_atom + 1) / 2 walks every call took before #631.

### Memory

Held for the call: c (naux n_occ²). Then, in turn, the explicit part's Γ
(naux nao², the size of B), Y (naux n_occ nao), one naux² weight, d and D;
and the cross terms' M1 (3 naux²), one aux atom's X (3 nao² nblk, nblk the
largest atom's aux count), the two (3 n_atom)² products and the batches'
**slots**, each an atom block and its Coulomb vectors (3 naux (n_occ² + 1)).
The integral calls' scratch (the Cartesian Γ, X's Cartesian block) is the
workspace's. The budget is the device's free memory, or the call's
`max_bytes` when that is less. The call refuses (Unsupported, naming the
bytes) when c plus the larger of the two phases at two slots would not fit
it; naux nao², 3 naux n_occ² and naux n_occ nao must fit `int32_t`.

The slots are sized after the explicit part, from the budget less c, M1, X
and the products, less room for `three_center_derivative`'s own scratch past
what the workspace already holds (its Cartesian block, half-transforms and
staging) and 512 MiB of headroom for the BLAS libraries and the pool: every
atom's when they fit, else as many as fit, two at the least. So the call
takes **more memory than before when it has it**: all n_atom blocks,
3 n_atom naux n_occ² doubles, the whole exchange cross term, against two.
At 500 AOs, 1500 fitting functions and 60 occupied orbitals an atom block
is 130 MB and Γ is 3 GB; for 50 atoms the whole batch would be 6.5 GB, so
on a 10 GB card it is split.

### Tests

`test/gtest/df_rhf_hessian_two_electron_gpu.cpp` (suite
`HessianDfRhfTwoElectron`), against `test/gtest/data/rhf_hessian/`'s
`coulomb` and `exchange` (`tools/emit/rhf_hessian_reference.py`, PySCF's
`_partial_hess_ejk` ej and −ek, checked there against `partial_hess_elec`
and the sum rule), for the skew water and the skew formaldehyde in cc-pVDZ /
cc-pVDZ-JKFIT: each term to 1e-9 absolute, its asymmetry to 1e-12 and the
translational sum rule Σ_A H[A k][y] to 1e-9; the refusals; and
`BatchesAgreeAndMemoryRefusal`: a budget (`max_bytes`) below two slots is
Unsupported naming the bytes, that many bytes (one atom a batch) is accepted
on the water, and a water dimer at its need plus two slots (two atoms a
batch, three batches), each against the unbudgeted call (one walk) to 1e-10
of the term's max |H|, a path agreement.

Measured on the RTX 3080 (CUDA): |H − PySCF| 6.2e-11 (Coulomb) and 1.8e-11
(exchange) on the water, 9.1e-11 and 2.2e-11 on the formaldehyde, of max |H|
8.0 / 1.0 and 17.6 / 1.7; asymmetry ≤ 1.8e-15, sum rule ≤ 2.2e-12; 1.2 s and
2.6 s a call. The same pieces with the cross terms built in NumPy from
PySCF's integrals (Cholesky solves) reach ej and −ek to 5e-11 and 2e-10, so
the JKFIT sets sit well inside the 1e-9 bar. The batches agree with one walk
to 6.6e-13 / 2.8e-13 (water, one atom a batch) and 1.0e-12 / 1.4e-12 (the
dimer, two a batch).

`bench/hessian_df_rhf_two_electron.cpp` times the call over an alkane's
DF-RHF in cc-pVDZ / cc-pVDZ-JKFIT. One walk (#631) against the
n_atom (n_atom + 1) / 2 walks before it, best of the calls after a warm-up,
with the call's own buffers at their peak (the formula above). The
device's free memory polled during a first call moved by 358 MiB before and
after on the 3080 for propane (584 for pentane), and by 385 → 399 MiB on the
gfx1200: the workspace's and the pool's growth, which the extra blocks
barely show in.

| molecule | card | before | after | speed-up | own buffers before → after |
|---|---|---|---|---|---|
| propane, 11 atoms (nao 82, naux 394, n_occ 13) | RTX 3080 | 31.40 s | 1.39 s | 22.6× | 25.2 → 31.7 MiB |
| pentane, 17 atoms (nao 130, naux 626, n_occ 21) | RTX 3080 | 151.25 s | 2.39 s | 63× | 99.0 → 145.9 MiB |
| propane | gfx1200 (HIP) | 33.48 s | 1.31 s | 25.6× | 25.2 → 31.7 MiB |

## The CASSCF two-electron explicit part

`df_casscf_two_electron` is `casscf_hessian.tex` eq. (df2) at fixed orbitals
and RDMs, for the energy E₂ = ½ Σ Γ_ab (T M⁻¹ Tᵀ)_ab over AO pairs a, b:

    E₂^xy = Σ Γ̃ T^xy − ½ Σ Ŵ M^xy + Σ_ab Γ_ab [R^x M⁻¹ R^yᵀ]_ab,
    Γ̃ = Γ c, Ŵ = cᵀ Γ c, R^x = T^x − c M^x

```cpp
auto e2 = hessian::df_casscf_two_electron(res, basis, classes, aux, b, l, c, ncore, ncas,
                                          gamma, gamma2);
// e2->three_center, e2->metric, e2->cross_coulomb, e2->cross_exchange,
// e2->cross_active and their sum e2->total: each 3 n_atom x 3 n_atom
```

γ and Γ are the active RDMs on the host, row-major as `ci::Rdm::rdm12`
writes them: one root's own for its explicit part (the fixture's per-root
`df2_*`), or whatever densities the caller wants the same expression at.

1. **The second-derivative pieces** are `mcscf::df_gradient`'s Γ_mnP and W
   at these RDMs (that README's "Two-electron": J's D ⊗ d − D_a ⊗ d_a and
   −½ d dᵀ + ½ d_a d_aᵀ, K's −2 Σ_i C_mi Σ_q V_P,iq C_nq and V c_coreᵀ,
   the active Σ_tu C_mt C_nu Z_P,tu and −½ Z c_actᵀ), handed to #593's
   `three_center_hessian` and `metric_hessian`. Everything is read off one
   set of fitted occupied pairs c_P,ab = (L⁻ᵀ B_ab)_P (`b_mo` and a trsm):
   d = c · G_J, d_a = c · G_a by gemvs (G_J = diag(2_core, γ),
   G_a = diag(0_core, γ) over the occupied orbitals), and the core rows and
   active pairs packed out of it by geams.
2. **The cross term** is split as the design's §6, so no pair-space Γ is
   formed. With r^x = L⁻¹ R^x over the occupied pairs, one **atom block**
   (the three coordinates of one atom, 3 naux n_occ² doubles) built exactly
   as `df_rhf_two_electron` builds it (`occupied_atom_block`, the
   `:atom_block` partition both share):
   - **J**, Γ = D ⊗ D − D_a ⊗ D_a: t^x = r^x · G_J and t_a^x = r^x · G_a
     (one gemm per coordinate against the two weights), and the 3 × 3 block
     t_Aᵀ t_B − t_a,Aᵀ t_a,B from one 6 × 6 gemm;
   - **K**, −Σ_i Σ_pq G_pq (ip|iq), G = G_K = diag(1_core, γ): the core rows
     of r^y times G (3 gemms, s^y, 3 naux n_core n_occ, the issue's
     half-transform), then −2 r_A,coreᵀ s_B, one (3 n_occ)² gemm whose
     n_occ × n_occ blocks' traces are the 3 × 3;
   - **active**, Σ Γ_tuvw r^x_tuᵀ r^y_vw: the active pairs packed out of the
     block (one geam per coordinate), Z = act_B Γ (a strided-batched gemm),
     and act_Aᵀ Z, one gemm.

   The blocks are contracted one pair (A, B ≤ A) at a time, as
   `df_rhf_two_electron` did before it batched them (#631): n_atom
   (n_atom + 1) / 2 blocks built, each one walk. `occupied_atom_block`
   builds a batch of atoms from one walk (its `count`) for the RHF part; this
   call still asks for one atom at a time.

With n_core = n_occ (no active space) and γ = 0 the J and K pieces are
`df_rhf_two_electron`'s Coulomb and exchange cross terms.

With a `rotation` K (its last argument, C then over every orbital) the call
returns instead each piece's derivative along C → C (1 + ε K) at fixed
MO-basis RDMs, the product rule over every occupied C, for L_xy's z_o part
(below).

### Memory

Held for the call: the fitted occupied pairs (naux n_occ²), the three
weights and Γ (n_cas⁴). Then, in turn, the explicit part's Γ_mnP (naux nao²)
with W, Y (naux max(n_core, n_cas) nao), V and the packed core rows
(naux n_core n_occ each), the active pairs and Z, d, d_a, D and D_a; and the
cross term's M1 (3 naux²), one aux atom's X (3 nao² nblk), two atom blocks
(3 naux n_occ² each), K's s (3 naux n_core n_occ), three active blocks
(3 naux n_cas² each), the Coulomb vectors and the products. The call refuses
(Unsupported, naming the bytes and the atom block's size) when the held part
plus the larger phase would not fit the device's free memory; naux nao²,
3 naux n_occ², naux max(n_core, n_cas) nao and n_cas⁴ must fit `int32_t`.

### Tests

`test/gtest/df_casscf_hessian_two_electron_gpu.cpp` (suite
`HessianDfCasscfTwoElectron`), against `test/gtest/data/casscf_hessian/`'s
per-root `df2_three_center`, `df2_metric`, `df2_cross_j`, `df2_cross_k`,
`df2_cross_active` and `two_electron` (`tools/emit/casscf_hessian_reference.py`,
over the same cc-pVDZ-JKFIT fit), with the fixture's orbitals and the RDMs of
its CI vectors (`ci::Rdm`, never a re-converged CASSCF): the stretched water
CAS(4,4) in 6-31G and in cc-pVDZ (d shells), and root 1 of the two-root
average at (0.7, 0.3). Each piece and the total to 1e-9 absolute, each
piece's asymmetry to 1e-12, the total's translational sum rule to 1e-9; and
the refusals.

Measured on the RTX 3080 (CUDA): |H − reference| ≤ 3.2e-11 per piece
(the second-derivative contractions; the cross terms ≤ 1.4e-12) and
≤ 1.4e-12 for the total, of max |H| 1.3 (6-31G), 1.7 (cc-pVDZ) and 0.79 (the
average's root 1); asymmetry ≤ 4.4e-16, sum rule ≤ 1.6e-12; 0.21 s, 0.82 s
and 0.21 s a call. Peak device memory, the call's own buffers (the cross
term's phase, the larger) beside the workspace its integral calls grew to:
1.19 MB + 0.72 MB (6-31G) and 2.01 MB + 2.27 MB (cc-pVDZ). The same
refactor leaves `HessianDfRhfTwoElectron`'s figures above unchanged to the
digits quoted.

## The CASSCF fixed-parameter second derivative

`df_casscf_exy` is `casscf_hessian.tex` eq. (exy): the second derivative of
the energy at fixed orbital rotations and CI, the orbitals carried by the
symmetric (Löwdin) connection C T(x), T = (Cᵀ S(x) C)^−½, to second order:

    E_xy = Σ D^AO h^xy + V_nn^xy + E₂^xy                      (explicit)
         + 2 Σ F τ^xy + 2 Σ (F^[x] τ^y + F^[y] τ^x) + E^(2)[τ^x, τ^y],
    τ^x = −½ S^x,   τ^xy = −½ S^xy + ⅜ (S^x S^y + S^y S^x)

(S^x, S^xy in the MO basis), each piece returned beside the sum:

```cpp
auto fx = mcscf::df_derivatives(res, molecule, basis, classes, aux, h, b, l, c_matrix, ncore,
                                ncas, nalpha, nbeta, roots, one_hot_on_I);
auto e = hessian::df_casscf_exy(res, molecule, basis, classes, aux, h, b, l, c, ncore, ncas,
                                gamma, gamma2, fx->fock);
// e->one_electron, e->nuclear_repulsion, e->two_electron (df_casscf_two_electron's),
// e->connection_second, e->connection_fock, e->connection_twice and their sum e->exy:
// each 3 n_atom x 3 n_atom
```

**The ⅜ terms are not optional.** The linearized connection 1 − ½ S^x δx
is not orthonormal at second order; τ^xy's ⅜ (S^x S^y + S^y S^x) is what
makes it so, and without it E_xy is wrong by about its own size (the design's
§10: 0.93 against finite differences; `DroppingTheThreeEighthsTermsMisses`
below measures it here).

**Any densities.** γ and Γ are any active 1- and 2-RDMs over the closed-shell
core, row-major as `ci::Rdm::rdm12` writes them, and F^[x] is the caller's at
the same RDMs (`mcscf::Derivatives::fock`; `df_derivatives` at weights
one-hot on root I gives root I's). E_xy is linear in (γ, Γ, F^[x]) at a fixed
core, so a state average's Lagrangian takes it at root I's RDMs and at its
z_c's transition RDMs added to them, the way `mcscf::df_gradient` does. What
this form cannot hold is a density with a virtual index: the z_o-rotated
densities K_z D − D K_z, which are E_xy's derivative along the orbital
rotation C → C (1 + ε K_z) rather than E_xy at some (γ, Γ). The call's
`rotation` argument takes those (below, "L_xy's z_o part").

### How each piece is built

D and Γ are formed over the occupied orbitals on the host: D = diag(2_core, γ)
and Γ the closed-shell core's D ⊗ D − ½ exchange with itself and with γ,
plus the active Γ, 8-fold symmetrized (nocc⁴ doubles).

- **Explicit.** `kinetic_hessian` + `potential_hessian` at D^AO =
  C_o D C_oᵀ (S^xy is the connection's, below), `nuclear_repulsion_hessian`,
  and `df_casscf_two_electron` at the same RDMs.
- **F**, eq. (fock), from B in the MO basis (`b_mo`, every pair) and
  G_Q,ij = Σ_rs Γ_ijrs B_Q,rs (one gemm from the occupied pairs):
  F[:, occ] = h_MO D + Σ_Q,r B_Q,rp G_Q,qr, one gemm over B's (Q, r < nocc)
  rows.
- **2 Σ F τ^xy.** The S^xy part is −Σ F S^xy_MO = `overlap_hessian` at
  −C sym(F) Cᵀ. The ⅜ part, with τ^x = −½ S^x symmetric, is
  6 ⟨sym(F) τ^x, τ^y⟩: one gemm makes sym(F) τ^x for every x, one more the
  Gram matrix against the τ stack.
- **2 Σ (F^[x] τ^y + F^[y] τ^x)**: the Gram matrix ⟨F^[x], τ^y⟩, and twice it
  plus its transpose.
- **E^(2)[τ^x, τ^y]**, the energy at the integrals one-index transformed by
  τ^x on one index and τ^y on another, every ordered pair of positions. With
  Γ 8-fold symmetric and nonzero only over the occupied orbitals, those
  twelve placements are four on the same pair and eight across the pairs:

      E^(2) = 2 ⟨D, τ^xᵀ h τ^y⟩ + 2 X₁ + 4 X₂,
      X₁ = Σ Γ_pqrs τ^x_ap τ^y_bq (ab|rs) = ⟨τ^y_occ, W^x⟩,   W^x = Σ_Q B_Q τ^x_occ G_Q,
      X₂ = Σ Γ_pqrs τ^x_ap τ^y_cr (aq|cs) = ⟨R^x, R^y Γ⟩,   R^x_Q,ip = (B_Q τ^x_occ)_pi, p occupied.

  Per x, Y_Q = B_Q τ^x_occ is one strided-batched gemm over B's columns,
  stored (Q, i, p) so that R^x is its first naux nocc² doubles and
  W^x = Yᵀ G one gemm; V^x = 2 h τ^x_occ D + 2 W^x. Then
  E^(2) = ⟨V^x, τ^y_occ⟩ + 4 ⟨R^x, R^y Γ⟩, two Gram matrices.

**Why not `response::Hessian`.** E^(2)[A, B] is the machinery behind the
orbital Hessian, which for an antisymmetric κ is
xᵀ H y = E^(2)[κ_x, κ_y] + ⟨F, κ_x κ_y + κ_y κ_x⟩. But that operator is a
product over the packed non-redundant pairs that leans on κ's antisymmetry
throughout (κ's core rows stand for its core columns; the core–core,
active–active and virtual–virtual blocks are zero; the −½ (F + Fᵀ) κ term is
the exponential's κ²/2), and τ^x is symmetric and fills every block,
virtual–virtual included. Here every pair (x, y) of 3 n_atom generators is
wanted at once, which is a Gram matrix rather than 3 n_atom operator
products. So E^(2) is its own contraction over the fitted B, and
`response::Hessian` is untouched: its callers (`mcscf/solve`,
`RelaxedHessian`, the Z-vectors, `mcscf::df_gradient` and the NEVPT2
Lagrangian) are bitwise as before.

### Memory

The explicit parts run first and free what they hold
(`df_casscf_two_electron` checks its own). Then, held for the connection
terms: τ^x and sym(F) τ^x for every x (2 · 3 n_atom nmo²), B over every MO
pair (naux nmo²) and the occupied ones, G (naux nocc² each), h_MO and two
Fock matrices, R^x and R^x Γ for every x (2 · 3 n_atom naux nocc²), V^x
(3 n_atom nmo nocc) and Γ (nocc⁴); beside the AO S^x stack while τ is made,
or one coordinate's Y (naux nocc nmo) from the workspace. Past the device's
free memory that is Unsupported before it is allocated. Nothing batches the
coordinates yet; B over every MO pair is what `response::Hessian`'s
density-fitted operator reads too.

### Tests

`test/gtest/df_casscf_hessian_exy_gpu.cpp` (suite `HessianDfCasscfExy`),
against `test/gtest/data/casscf_hessian/`'s per-root `one_electron`,
`nuclear_repulsion`, `two_electron`, `connection_second`, `connection_fock`,
`connection_twice` and `exy` (`tools/emit/casscf_hessian_reference.py`, over
the same cc-pVDZ-JKFIT fit), at the fixture's orbitals and the RDMs of its CI
vectors (never a re-converged CASSCF), with F^[x] from `mcscf::df_derivatives`
at weights one-hot on the root: the stretched water CAS(4,4) in 6-31G and in
cc-pVDZ (d shells), root 1 of the two-root average at (0.7, 0.3) and root 2
of the three-root one at (0.5, 0.3, 0.2). Each piece and E_xy to 1e-8
absolute, each piece's asymmetry to 1e-11, E_xy's translational sum rule to
1e-8. `DroppingTheThreeEighthsTermsMisses` takes 2 Σ F ⅜ (S^x S^y + S^y S^x),
computed on the host from the fixture's F and this tree's S^x, out of E_xy
and requires the rest to miss the reference by more than a tenth of
max |E_xy|, by the size of the term; and the refusals.

Measured on the RTX 3080 (CUDA), of max |E_xy| 0.44 (6-31G), 1.56
(cc-pVDZ), 0.53 and 0.50 (the averages' roots): |H − reference| ≤ 5.6e-12
for the one-electron part, ≤ 4.7e-13 for the two-electron part, ≤ 5.7e-13,
7.4e-14 and 8.0e-14 for the three connection terms, and ≤ 6.1e-12 for E_xy;
asymmetry ≤ 3.6e-15, sum rule ≤ 5.2e-13; 0.22 s (6-31G) and 0.82 s
(cc-pVDZ) a call, `df_casscf_two_electron` included. Without the ⅜ terms
E_xy misses by 0.93 of max |E_xy| 0.44: the design's 0.93, twice E_xy's own
size.

## L_xy's z_o part

A state-averaged root's L_xy = E_I,xy + z · ∂_xy g_SA (`casscf_hessian.tex`
eq. sa) takes E_xy at z's densities: the transition RDMs of z_c, which fit
the form above (E_xy at root I's RDMs plus them, as `mcscf::df_gradient`
takes them), and the z_o-rotated densities K D − D K and K on each index of
Γ, K = K(z_o) at the averaged RDMs, which do not: they have a virtual
index. Those are E_xy's derivative along the orbital rotation
C → C (1 + ε K) at fixed MO-basis RDMs, and `df_casscf_exy`'s last argument,
`rotation`, returns exactly that, every piece of it, by the product rule
(no difference in parameter space anywhere):

```cpp
auto sa = mcscf::df_derivatives(res, molecule, basis, classes, aux, h, b, l, c_matrix, ncore,
                                ncas, nalpha, nbeta, roots, weights, k.data());   // K = K(z_o)
auto z_o = hessian::df_casscf_exy(res, molecule, basis, classes, aux, h, b, l, c, ncore, ncas,
                                  gamma_sa, gamma2_sa, *sa->fock_rotated, k.data());
// z_o->one_electron, z_o->two_electron, z_o->connection_second, z_o->connection_fock,
// z_o->connection_twice and their sum z_o->exy; z_o->nuclear_repulsion is zero
```

K is the antisymmetric nmo × nmo on the device (column-major, C's columns,
C → C e^K: the fixture's `z_orb`), γ and Γ the averaged RDMs, and F^[x] the
rotated one `mcscf::df_derivatives` returns at the same K and weights
(`../mcscf/derivative/README.md`, "The rotated densities"). In the rotating
basis the densities are the unrotated ones and every MO-basis quantity X
moves as its commutator, X′ = X K − K X; in the AO basis the occupied
coefficients move as C̃_o = C K[:, occ]. Piece by piece:

- **One-electron**: h^xy against D^AO's derivative, C̃_o D C_oᵀ + C_o D C̃_oᵀ.
- **Two-electron**: `df_casscf_two_electron`'s own `rotation`, the product
  rule on every C_o. Its Γ_mnP and W gain each intermediate's derivative,
  term for term as `mcscf::df_gradient`'s orbital response gains them (the
  fitted densities d′ = c′ · G, D′, V′ = c′_core G_K, Z′ = c′_act Γ, each
  back-transform once with C̃_o), with c′ = c_õo + c_oõ the fitted pairs'
  derivative. The cross term's atom blocks are built over [C_o | C̃_o],
  twice the occupied orbitals (`occupied_atom_block` unchanged, at four
  times the block), and cut into r and r′ = r_õo + r_oõ; R^x is bilinear in
  its two orbitals, so r′ is its derivative, and each pair of atoms is
  contracted as (r′, r) and (r, r′) by the same J, K and active pieces, all
  of them symmetric bilinear forms.
- **The two F terms.** Σ F τ^xy and Σ F^[x] τ^y are contractions over the
  whole MO basis, so their outer C's form C Cᵀ = S⁻¹, stationary to first
  order (K antisymmetric): only F's inner indices move, and the outer
  rotation of the last index leaves F Kᵀ. So both pieces are the unrotated
  code at F_L and at `fock_rotated`, with
  F_L = F Kᵀ + h K D + Σ_Q,a B_Q,ap G̃_Q,aq + Σ_Q,r B_Q,rp G′_Q,rq,
  G̃_Q,aq = Σ_r K_ar G_Q,rq (a strided-batched gemm over G's columns) and
  G′ = B′_oo Γ: F at the densities K D − D K and K·Γ, as
  `mcscf::df_gradient`'s W takes its z_o.
- **E^(2)[τ^x, τ^y]** is multilinear in (h, B, τ^x, τ^y) at the fixed
  occupied (D, Γ), so its derivative is the same formula with each in turn
  its commutator: B′ = B K − K B (one gemm over B's (Q, p) rows for B K,
  and K B_Q = slab_q Kᵀ per column q), h′ = h K + (h K)ᵀ,
  τ^x′ = τ^x K + (τ^x K)ᵀ;

      V′^x = 2 Y′ᵀ G + 2 Yᵀ G′ + 2 h τ′^x D + 2 h′ τ^x D,   Y′ = B′ τ^x_occ + B τ′^x_occ,
      E^(2)′ = ⟨V′^x, τ^y⟩ + ⟨V^x, τ′^y⟩ + 4 (⟨R′^x, R^y Γ⟩ + ⟨R^x, R′^y Γ⟩),

  R′^x the first naux nocc² of Y′, the last term the transpose of the one
  before (added on the host).

A numpy prototype of the F_L and E^(2)′ formulas against the densities'
direct contraction (random integrals and RDMs, a random K) agreed to 6e-13
of 600 and 5e-11 of 1.7e5 before the port. V_nn^xy does not see the
orbitals: that piece is zero.

### Memory

Beside the unrotated call's: in `df_casscf_two_electron`, [C_o | C̃_o], the
fitted pairs over it (4 naux nocc²) and c′; the dotted explicit intermediates
(a second Y, V′, c′_core, c′_act, Z′, d′, d′_a, D′, D′_a); and in the cross
phase the extended atom block (4 × 3 naux nocc²) and r′, t′ and act′ for
both atoms. In `df_casscf_exy`, B′ over every MO pair (naux nmo²) and the
occupied ones, G′ and G̃ (naux nmo nocc), h K, h′, F_L, and τ′^x, V′^x and
R′^x for every x; per coordinate Y′. Each is in its call's estimate, refused
(Unsupported) past the device's free memory before anything is allocated.

### Tests

`test/gtest/df_casscf_hessian_rotation_gpu.cpp` (suite
`HessianDfCasscfRotation`), against `test/gtest/data/casscf_hessian/`'s per
root L_xy z part (`z_one_electron`, `z_two_electron`, `z_connection_second`,
`z_connection_fock`, `z_connection_twice` and `exy_z`;
`tools/emit/casscf_hessian_reference.py`, eq. exy at z's densities), with
the orbitals, roots and z **loaded from the fixture** (PySCF's two-root
solutions are saddle points): every root of the stretched water's
SA-CAS(4,4)/6-31G at two roots (0.5, 0.5) and (0.7, 0.3) and three at equal
weights and (0.5, 0.3, 0.2). The fixture's z part is orbital and CI
together, so the test adds the CI part the merged entry points already give:
the roots' change dc_r = z_ci,r + Σ_s Θ_sr c_s, the transition RDMs as half
the difference of `rdm12` at c_r + dc_r and c_r − dc_r (and F^[x] the same
of two `df_derivatives` calls; the core parts cancel), and `df_casscf_exy`
at the averaged RDMs plus those less at the averaged RDMs alone. Each piece
and the sum to 1e-8 absolute, each z_o piece's asymmetry to 1e-11, the z_o
part's translational sum rule to 1e-8; and the refusal of a rotation with C
short of every orbital.

Measured on the RTX 3080 (CUDA), worst over the ten roots, of max |z| 0.20
(one-electron), 0.17 (two-electron) and 0.021 (`exy_z`), the z_o part about
as large as the whole: |z − reference| ≤ 5.9e-15 one-electron, 1.3e-12
two-electron, ≤ 8.7e-15 for the three connection terms and 1.3e-12 for the
sum; asymmetry ≤ 2.0e-16, sum rule ≤ 9.9e-15; 0.47 s for the z_o part
(`df_derivatives` with its rotation and `df_casscf_exy` with its own).
`HessianDfCasscfTwoElectron`, `HessianDfCasscfExy`, `HessianDfCasscf`,
`McscfDerivative` and the `HessianDfRhf*` suites are unchanged and measure
the same figures as above: without a rotation the calls run the same BLAS
sequence as before.

## The state-specific CASSCF Hessian

`df_casscf` is `casscf_hessian.tex` eq. (ss), Algorithm 1: at a converged
state-specific DF-CASSCF,

    d²E/dx dy = E_xy + E_xλ λ^y,   E_λλ λ^y = −E_λy,

every piece from a merged part, no integrals of its own:

```cpp
auto h = hessian::df_casscf(res, molecule, basis, classes, aux, h_core, b, l, solution, ncore, ncas);
// h->hessian: 3 n_atom x 3 n_atom, exactly symmetric; h->asymmetry: what symmetrizing discarded;
// h->exy: df_casscf_exy's E_xy and its pieces; h->response: E_xλ λ^y, element (x, y);
// h->cp: λ^x (zvectors' z_o and z_c) and the solver's own residuals;
// h->residuals: each column's residual on the unreduced equations
auto cp = hessian::df_casscf_cp(res, molecule, basis, classes, aux, h_core, b, l, solution, ncore, ncas);
// *cp: steps 1, 3 and 4 below alone, the same λ^x as h->cp
```

The inputs are `mcscf::df_gradient`'s, a one-root `mcscf::Solution`
included (a state average is refused: its Lagrangian is #621's). In order:

1. **E_λx**: `mcscf::df_derivatives` at weight 1, every coordinate at once:
   F^[x], `rhs_orbital` and `rhs_ci`.
2. **E_xy**: `df_casscf_exy` at the solution's γ, Γ and that F^[x].
3. **λ^x**: H̃ built at the solution as `mcscf::df_gradient` builds it for a
   Z-vector (`casscf_integrals`, `response::fock`, the sigma over H₀,
   `response::Hessian`, `RelaxedHessian`), then `response::zvectors` (Schur, its
   default) on all 3 n_atom columns: the orbital right-hand sides packed
   (`pack_orbital_pairs`; `rhs_orbital` holds G_pq at p > q, the order it
   reads), the CI ones **halved**, 2 Q H̃^x c → Q H̃^x c, into the Z-vector's
   unweighted, unscaled CI rows. With them the solution is λ^x itself:
   zvector's equations are E_λλ's with the CI rows divided by 2.
4. **The check**: one block product of the unreduced operator,
   `RelaxedHessian::apply_coupled`, at (z_o, z_c), plus the right-hand side:
   per column the norm of both blocks is `residuals`.
5. **The response**, Λᵀ E_λ· and not −Λᵀ E_λλ Λ: two gemms,
   G_packedᵀ Z_o + rhs_ciᵀ Z_c, the CI rows the energy's own (with their 2)
   against z_c. The CI rows' scaling enters the solve and its right-hand side
   together and cancels; −Λᵀ E_λλ Λ would need E_λλ in the energy's
   coordinates, which no product here applies.
6. E_xy + response, symmetrized; `asymmetry` is the antisymmetric part
   discarded, E_xy's rounding plus what the CP solve's residual leaves.

**A saddle is refused.** A state-specific excited root is a saddle in λ: E_λλ's
CI block 2 (H₀ − E) is negative along every lower root of the same spin. The
outer conjugate gradients of `zvectors` meet that as negative curvature of H̃
and return a Divergence, which is refused as H̃ not positive (as is its
MaxIterationsExceeded, "A singular H̃" below). They need not meet it: `RelaxedHessian`'s inner CI
conjugate gradients stop without an error on a direction H₀ − E is not
positive along (or after 200 sigmas), and an outer solve over the H̃ those
truncated responses make can still reach its tolerance. So the solution is
also checked on the unreduced equations, which nothing inner stands in for,
and a column above 1e-6 there is a Divergence naming it. A converged solve
sits near 1e-9 (the inner solves' 1e-8, relative), three orders below the
ceiling. Stationarity is assumed, not checked, as in
`mcscf::df_gradient`: at a geometry that is not a stationary point of the
orbitals and CI the formula is not the Hessian.

### A singular H̃

**A null direction of H̃ is refused too (#978), and the coordinates' columns
cannot see one.** HF in cc-pVDZ with CAS(10,7) over every electron
(`PipelineNevpt2.PcAcceptedThroughAContinuum`'s input) is the case. Its two
weakly occupied active orbitals (0.016 and 0.018) are σ* and one orbital of
the 2π pair, which correlates one 1π orbital more than the other (1π
occupations 1.9979 and 1.9902 in one run, 1.9992 and 1.9889 in another, at
the same energy). So the CASSCF is one point of a continuous family:
rotating the active 2π orbital about the bond into its virtual partner, with
the CI following, costs nothing. H̃'s smallest eigenvalue is −1.4e-11
(largest 121, next 6.4e-3); its eigenvector is the active orbitals 5 and 6
against the virtuals 9, 11 and 17. Every coordinate's right-hand side is
orthogonal to it by symmetry, up to round-off (g·v 5e-13 to 7e-12 of ‖g‖
11). The conjugate gradients divide that round-off by the null eigenvalue.
The transverse columns' |z_o| then reaches 58 to 7.4e3 against 0.9, and the
unreduced residual 1e-7 to 2e-5 (the CI response's 1e-8 of a right-hand side
that large). So whether a run passed the 1e-6 ceiling depended on its
round-off, or it met p·H̃p = −3e-13 instead: 9 of 27 runs refused on the RTX
3080, 1 of 10 on the gfx1200. No tolerance fixes that. The point the CASSCF
reached decides which way the active 2π orbital points, and bending in its
plane and out of it are different second derivatives, so the transverse
block of the Hessian would turn with it from run to run.

**The probe.** `zvectors` solves one more column beside the coordinates': H̃
u = −s for a fixed pseudo-random s (`kCasscfCpProbeSeed`, unit norm, no CI
part), generic along every eigenvector. Its Rayleigh quotient ρ = u·H̃u /
u·u = −u·s / u·u is at least λ_min(H̃). The bar is what the products
resolve. Each product's CI response leaves R x wrong by up to its tolerance
(1e-8) of R x, and at a minimum 0 ≤ −R ≤ H, so H̃'s eigenvalues are known to
1e-8 ‖H‖ (Weyl). ‖H‖ is at least max diag(H), so 1e-8 max diag(H) is at most
that resolution. ρ not above it is a Divergence that calls H̃ singular, and
so is the solve meeting a direction it is not positive along, or running out
of products: the probe's residual cannot fall below its part along a null
direction. Since ρ ≥ λ_min, the check refuses only an H̃ whose smallest
eigenvalue its products cannot resolve. The probe's column is dropped from
the result.

The refusal does not hang on round-off: s's part along the null direction
is O(1), not O(ε), so no run can solve the probe's column without ρ falling
to the null eigenvalue or the solve failing. At HF CAS(10,7) ρ is 7e-15 to
5e-13 against a bar of 1.2e-6, seven orders apart, or the probe's column
meets negative curvature or 200 products (measured on both cards). On every
other input of the Hessian tests ρ is at least 2.2e-2, five orders above its
bar, and the probe takes one fewer to 6 more products than the slowest
coordinate (34 solves, RTX 3080).

**`df_sacasscf` solves its λ^x the same way (#997).** The probe is one
column more of the same `zvectors` call (`probed_zvectors`, shared through
the `:state_specific` partition), its CI and Θ rows zero, and A's CI
response is solved to the same 1e-8, so the bar is the same derivation. A
state average does not lift the null direction. Rotating every orbital about
the bond, each root following, maps every root to one of the same energy, so
E_SA is unchanged at any weights. With one 2π orbital active it is not a
rotation inside the active space. HF CAS(10,7) at two roots and equal
weights is refused in one run on the gfx1200, with ρ = 1.7e-10 against its
bar of 1.21e-6. Its roots are −100.03821 and −99.66527; root 1 is one
component of the ¹Π pair, which a three-root average makes exactly
degenerate. In other runs of the same input, the probe's column met p·H̃p
< 0 or ran out of products (‖r‖ 8e-8 to 1.2e-5), and each was the same
refusal. Where the probe finished, the coordinates' columns took 23 and 26
products. z is solved first, over the same A, on root I's gradient. It met
nothing, because its right-hand side is axially symmetric, as the
coordinates' are.

On the existing state averages ρ is 4.9e-3 to 0.30 against a bar of 8.6e-7,
four orders and more above it (water SA-CAS(4,4)/6-31G, the four
`HessianDfSaCasscf` cases and the three analytic SA `PipelineCasscfHessian`
cases, gfx1200). Every root of one case has the same A and the same ρ. The
probe took one fewer to 3 more products than the slowest coordinate.

One caveat: for a diatomic, every transverse coordinate is a rigid
translation or rotation. So translational and rotational invariance fix the
transverse block, and the run-to-run turning #978 describes cannot show in
it. With the coordinates' columns alone, a CAS(4,3) SA-2 of HF (active 3σ,
one 1π, σ*), which has the same null direction, gave the same Hessian three
times: H_xx = H_yy = −0.0134034, off-diagonal under 4e-13. With the probe,
that input is refused too: its column meets p·H̃p = −6.4e-15 of p·p, a zero
eigenvalue's round-off, after 10 products. The refusal stands on H̃ being
singular, not on that input's numbers.

### Memory

The parts run one after another, each freeing what it held:
`df_derivatives`' stacks (about 20 of 3 n_atom nao² doubles), then
`df_casscf_exy`'s, then the CP solve's, H̃'s scratch over 3 n_atom columns
(six CI blocks per column) beside the right-hand sides (3 n_atom (pairs +
n_det)) and λ. Each part refuses its own memory, Unsupported, before it
allocates. `df_derivatives`' F^[x] and right-hand sides stay held across
the whole call (3 n_atom (2 nmo² + n_det)).

### Tests

`test/gtest/df_casscf_hessian_gpu.cpp` (suite `HessianDfCasscf`), against
`test/gtest/data/casscf_hessian/`'s state-specific cases (6-31G and cc-pVDZ,
d shells), over the same cc-pVDZ-JKFIT fit, the orbitals and CI vector
**loaded from the fixture** (PySCF's own cc-pVDZ solution is a saddle point),
the root's energy ⟨c|H₀|c⟩ and RDMs by `ci::Rdm`: E_xy against `exy` to
1e-8, the response against `response` and the total against `hessian` to
1e-7, the total against `relaxed_fd` (the analytic relaxed gradient
differenced at re-solved displaced geometries, fourth order, h = 1e-3 bohr) to
the design's 1e-6, λ^x's orbital part against `lambda_orb` to 1e-7 of its
largest element, every CP residual to 1e-8, the asymmetry before
symmetrizing to 1e-8, the translational sum rule to 1e-7. **No harmonic
analysis**: the fixture's stretched water is not a stationary point (its
gradient reaches 0.074 hartree/bohr), so the six near-zero mass-weighted
eigenvalues of a minimum do not apply to it, and no fixture here is at one.
`ExcitedRootIsRefused` builds a saddle: the second CASCI root at the 6-31G
fixture's orbitals, one root of a `Solution`, which must be a Divergence;
and the refusals.

Measured on the RTX 3080 (CUDA), of max |H| 0.065 (6-31G) and 0.064
(cc-pVDZ): E_xy ≤ 6.0e-12 from the reference, the response ≤ 2.2e-10 (of
max 0.48 and 1.61), the total ≤ 1.3e-10 from `hessian` and ≤ 2.1e-9 from
`relaxed_fd` (the fixture's own analytic-vs-difference gap, 2.1e-9 and
8e-11); λ^x's orbital part ≤ 3.4e-10 of 0.38; CP residuals ≤ 1.6e-9 after 19
and 25 Schur iterations; asymmetry ≤ 1.9e-10, sum rule ≤ 9.7e-11; 0.69 s
(6-31G) and 1.38 s (cc-pVDZ) a call. The excited root (−75.5923 over the
ground state's −75.9025) is refused by `zvectors` itself: p·H̃p = −0.23 at
its first product.

## The NEVPT2 parameter response and total tangent

`df_nevpt2_tangents` is the parameter response x^x of the single-state
NEVPT2 nuclear Hessian (`docs/nevpt2/nevpt2.tex` eqs. koo and tx, #813) and the total
first-order change t_x of the NEVPT2 functional's inputs, per coordinate:

    h_dot^x = h1^x + κ^xᵀ h_MO + h_MO κ^x
    B_dot^x = W^x  + κ^xᵀ B_Q  + B_Q κ^x          (each fitted index Q)
    c_dot^x = c^x
    κ^x     = κ_nr^x + κ_oo^x,   κ_oo^x_pq = −f^(x)_pq / (ε_p − ε_q)   (eq. koo)

```cpp
auto cp = hessian::df_casscf_cp(res, molecule, basis, classes, aux, h_core, b, l, solution, ncore, ncas);
auto pairs = hessian::df_nevpt2_tangents(
    res, molecule, basis, classes, aux, h_core, b, l, solution, ncore, ncas, *cp, /*pc=*/false,
    [&](int32_t x0, int32_t count, const double *h_dot, const double *b_dot,
        const double *c_dot, const double *kappa) -> Status { ...; return {}; });
// *pairs: PC's degenerate pairs (empty for SC, which refuses them)
```

- **(h1^x, W^x)**, the derivative integrals with the connection τ^x =
  −½ S^x_MO, are `mcscf::df_integral_derivatives`' chunks (#807), its
  consumer this call's: W^x (p, q, Q at p + nmo (q + nmo Q)) is transposed to
  NEVPT2's MO B layout (Q + naux (p + nmo q), as `b_mo` leaves it) by one
  geam per coordinate. W^x's metric gauge is not the Cholesky factor's at a
  moved geometry, which NEVPT2 never sees (`../mcscf/third_derivative/README.md`,
  "NEVPT2's B").
- **(κ_nr^x, c^x)** is λ^x, the CP-CASSCF solution: `cp` as
  `df_casscf_cp` or `df_casscf` (`CasscfTerms::cp`) returns it, the caller's,
  so a Hessian that has already solved it does not solve it twice. z_o is
  unpacked to the antisymmetric κ_nr (`unpack_orbital_pairs`, C → C exp κ);
  z_c is the CI response itself (zvector's CI rows are E_λλ's halved, its
  solution unscaled), handed over in place.
- **The one-index transform** by κ, C → C (1 + κ): h_MO κ + κᵀ h_MO as two
  gemms; on B, B_Q κ for every Q is one gemm over B as the (naux nmo) × nmo
  matrix of rows (Q, p), and κᵀ B_Q one strided-batched gemm over the
  naux × nmo slabs at each q (B_Q symmetric, so (κᵀ B_Q)_pq = Σ_r B_Q,rq κ_rp).
- **κ_oo^x**: f^(x) is `nevpt2::builders_tangent` (#806) along everything
  above but κ_oo, its f_core and f_virt; ε the diagonals of its value's f,
  the solution's core and virtuals canonical (an off-diagonal |f_pq| above
  1e-8 is refused: the division is f_pq = 0's stationarity only where f is
  diagonal). Then κ_oo's own one-index transform onto h_dot and B_dot, and
  `kappa`, the consumer's, is the total. Along κ_oo, f_pq moves by
  κ_pq (ε_p − ε_q) and nothing else f reads moves (a core-core or
  virtual-virtual rotation leaves D and γ), so that is the whole condition.

**The degenerate-pair rule** is the gradient's z^c rule
(`../nevpt2/gradient/zc.cpp`): below |ε_p − ε_q| = 1e-6, κ_oo^x_pq is 0
while |f^(x)_pq| ≤ 1e-8 (an abelian subgroup's split pair, or a
displacement that does not split it). Past that the division is not one the
state determines: **SC refuses**, Numerical, naming the pair, the gap,
f^(x)_pq and the coordinate (SC is not invariant to the rotation, and κ_oo
enters the Hessian quadratically, `docs/nevpt2/nevpt2.tex` eq. koo); **PC records**
it as a `DegeneratePair` (coordinate, p > q, gap, f), κ_oo^x_pq left 0, for
#817 to measure, and goes on.

**All-electron.** The call takes no frozen core: the NEVPT2 it feeds
correlates the whole core (milestone 24's update refuses `frozen_core ≠ 0`
in the Hessian paths), and `builders_tangent` builds the all-electron
inputs.

### Memory

Held throughout: h_MO and B_MO (naux nmo²). W^x's chunk takes a third of
the budget (the device's free memory at the call, or `max_bytes` when that
is less): `df_integral_derivatives` refuses one coordinate past it,
Unsupported, naming the bytes. Per chunk of k coordinates this call holds
B_dot (k naux nmo², as large as the chunk's W^x), h_dot and κ (k nmo² each),
and `builders_tangent`'s arrays, one set of `BuilderOutputs` per coordinate
(the MO blocks, three n_act⁶ RDM tangents) beside its own point: what the
other two thirds are for. naux nmo past `int32_t` is Unsupported.

### Tests

`test/gtest/df_nevpt2_tangents_gpu.cpp` (suite `HessianNevpt2Tangents`), on
this tree's DF-CASSCF through the registry (the Casscf step's canonical core
and virtuals, ‖g‖ < 1e-10):

- `MatchesTheLadder` (slow tier) and `MatchesTheLadderOnOneAtom` (the
  carbon's three coordinates, the fast set's): t_x on `nevpt2_hessian/`'s
  skew formaldehyde (#803, STO-3G / cc-pVDZ-JKFIT, CAS(4,4)) against a
  Richardson ladder of (h_MO, B_MO, c) of the CASSCF **re-solved** at each
  displaced geometry (`mcscf::solve` from the reference moved by the
  connection, C₀ (C₀ᵀ S C₀)^−½, its CI the guess, ‖g‖ < 1e-12) and
  **re-canonicalized** (f's core and virtual blocks diagonalized, each
  orbital's sign along its reference one), in the gauge t_x is in: the
  active orbitals turned by U = (MᵀM)^−½ Mᵀ, M the active block of
  C_refᵀ S C, so that no active-active rotation enters (the CI vector carried
  along by its strings' minors, c' = Dᵀ c D over the alpha-by-beta matrix c,
  D_JI the minor of U over strings J and I, its sign along the reference's), and B's fitted index in the symmetric metric gauge
  Z = L_M⁻ᵀ (L_M⁻¹ M L_M⁻ᵀ)^−½ at the reference's L_M (#807's ladder). h_MO
  and c at steps 1e-3 and 5e-4 bohr; B_MO at 5e-4 and 2.5e-4, its tight
  JKFIT functions on C and O giving it a Richardson remainder that falls
  16× a halving (7.2e-8 relative at 1e-3). Every coordinate within 1e-7 of
  the ladder's largest element of each, the chunks two coordinates each.
- `DegeneratePairRule`: a C3v ammonia in 6-31G, its full valence CAS(8,7),
  whose virtual e pairs (orbitals 9, 10 and 11, 12) sit 8e-14 apart: SC is
  Numerical, naming a pair and a coordinate; PC returns the records, each a
  pair below the floor with |f^(x)| above 1e-8, in coordinate order. (The
  `h2co_neardeg_sto3g` fixture's virtual pair is 6.0e-3 Eh apart, far above
  the rule's 1e-6, so it is not this case; a CAS(2,2) over ammonia's two a1
  orbitals breaks the symmetry in the CASSCF itself.)
- `Refusals`: no consumer, λ^x of one coordinate fewer, and two core
  orbitals turned into each other (not canonical).

Measured on the gfx1200 (`--preset hip`): worst 5.9e-8 (c along the
oxygen's y), h_MO ≤ 2.8e-8, B_MO ≤ 4.5e-9; the call 1.7 s over six chunks,
the full ladder 15 s (72 solves); the ammonia's SC refusal at
coordinate 0, PC's 22 records (both pairs at every coordinate but the
nitrogen's z, which keeps the C3 axis).

## The QD-NEVPT2 parameter response and total tangent

`df_qd_nevpt2_tangents` is the same over a state average, the parameter
response x^x of the QD-NEVPT2 Hessian and the total tangent t_x of every
input of H^eff, (h_MO, B_MO, {c_r}) (`docs/nevpt2/nevpt2.tex`, the QD
Hessian's parameter response: Eqs. qd-K, qd-theta, qd-gsa, and koo at
f^SA; #961):

    h_dot^x   = h1^x + κ^xᵀ h_MO + h_MO κ^x
    B_dot^x   = W^x  + κ^xᵀ B_Q  + B_Q κ^x          (each fitted index Q)
    c_dot^x_r = ξ^x_r + Σ_s θ^x_sr c_s              (θ_rs = −θ_sr)
    κ^x       = κ_nr^x + κ_oo^x

```cpp
auto cp = hessian::df_sacasscf_cp(res, molecule, basis, classes, aux, h_core, b, l, solution,
                                  weights, ncore, ncas);  // once, for every QD state
auto t = hessian::df_qd_nevpt2_tangents(
    res, molecule, basis, classes, aux, h_core, b, l, solution, weights, ncore, ncas, cp->cp,
    /*pc=*/false,
    [&](int32_t x0, int32_t count, const double *h_dot, const double *b_dot,
        const double *c_dot, const double *kappa) -> Status { ...; return {}; });
// t->theta: every rotation per coordinate; t->max_theta, t->max_kappa_oo;
// t->recorded: PC's degenerate pairs
```

Three steps, block triangular, each closed once the one before it is done:

1. **CP-SA-CASSCF**: `df_sacasscf_cp`, `df_sacasscf`'s λ^x on its own
   (below), the caller's: it does not depend on the state, so one solve
   serves every QD state. Its z_o is κ_nr (unpacked as in
   `df_nevpt2_tangents`), its z_c the ξ^x_r (off every root), and its Θ the
   rotations of the pairs of **unequal** weights, which E_SA sees.
2. **The equal-weight rotations**, closed form (Eq. qd-theta):
   θ^x_sr = 𝓗^x_sr / (E_r − E_s), 𝓗^x_sr the first-order change of
   ⟨c_s|H|c_r⟩ along step 1: Σ ḣ1eff_tu γ^(sr)_tu + ½ Σ (tu|vw)˙ Γ^(sr)_tuvw,
   the active Hamiltonian's tangent from `nevpt2::qd_builders_tangent`'s
   `h1e` and `eri` against the value's transition RDMs (both integrals are
   symmetric, so their column-major and the RDMs' row-major layouts dot
   flat). ξ adds nothing (⟨ξ_s|H|c_r⟩ = E_r ⟨ξ_s|c_r⟩ = 0), and neither does
   κ_oo (a core-core or virtual-virtual rotation leaves h1eff and (tu|vw)).
   An unequal pair's Θ meets the same condition: its CP row is
   (E_s − E_r) Θ_sr + c_s · H′ c_r + H̃^x_sr = 0.
3. **κ_oo at f^SA** (Eq. koo, f → f^SA): `qd_builders_tangent`'s f_core and
   f_virt along steps 1 and 2 (an equal-weight rotation does not move f^SA,
   Eq. qd-gsa, so the tangent is taken before step 2's θ is added and serves
   both), ε f^SA's diagonal, the solution canonical in f^SA (an off-diagonal
   above 1e-8 is refused). The degenerate-pair rule is `df_nevpt2_tangents`'
   (1e-6 Eh, 1e-8): SC refuses, naming the pair and the coordinate; PC
   records it.

Then c_dot = ξ + every θ, κ the total, both one-index transforms on h_dot and
B_dot, and the chunk is the consumer's. `theta` returns every rotation
(the CP solve's as they come, Eq. qd-theta's at the equal pairs) for the
Hessian's L_xx actions.

**Near a degeneracy it refuses.** Two roots under `mcscf::kRootGapFloor`
(1e-8 Eh) apart are Numerical before anything runs (`state_averaged_electrons`,
`df_sacasscf`'s check). A rotation |θ^x_sr| above 10 per bohr, the CP solve's
or Eq. qd-theta's, is Numerical, naming the pair and the coordinate:
`root_rotation`'s bound, where the quadratic model holds within about
1/20 bohr. The QD gap itself is not this call's: H^eff's eigenvectors are
the Hessian's.

### `df_sacasscf_cp`

`df_sacasscf`'s steps 1, 3 and 4 below, the same body stopped there (so
`df_sacasscf`'s existing tests cover it): A, λ^x by `zvectors` with the probe
column, the check on the unreduced equations and the bound on Θ. It takes
weights equal for some pairs only (0.4, 0.4, 0.2), which `df_sacasscf`
refuses before anything runs (eq. rot is for all-equal weights, λ^x's Θ for
pairwise different ones, and E_I's Hessian there would need both): each pair is rotating or not by
`response::rotating_pairs`, the rotating ones' Θ rows in the solve and in
the check, an equal pair's rotation row left out of both (its residual would
be 𝓗^x_sr itself, step 2's numerator). `SaCasscfCp` carries λ^x and the
residuals.

### Memory

`df_nevpt2_tangents`', with `qd_builders_tangent`'s arrays per coordinate
(the shared outputs and every ordered pair's RDM tangents, n² of them, three
n_act⁶ on the diagonal and dm3's off it) in place of `builders_tangent`'s,
and c_dot k n n_det for a chunk of k.

### Tests

`test/gtest/df_qd_nevpt2_tangents_gpu.cpp` (suite `HessianQdNevpt2Tangents`),
on this tree's DF-SA-CASSCF of `qd_nevpt2/`'s `h2co_skew_ccpvdz_sa3` (C1,
three roots at equal weights, slow tier) and `h2co_skew_ccpvdz_sa3_weighted`
(0.4, 0.4, 0.2, the representative) through the registry. Each defining
condition is checked on the tangent handed over, every coordinate:

- CP-SA-CASSCF: `df_sacasscf_cp`'s residuals within 1e-9.
- The rotations, every pair: ⟨ċ_s|H|c_r⟩ + ⟨c_s|H|ċ_r⟩ + ⟨c_s|Ḣ|c_r⟩ zero to
  1e-10 of its largest summand, Ḣ's h1eff and (tu|vw) rebuilt on the host from
  h_dot and B_dot and each product by `ci::Sigma` (H without its constant,
  the roots' orthonormality, ċ_s · c_r + c_s · ċ_r, checked to 1e-12 beside
  it).
- κ_oo: f^SA's off-diagonal change along t_x (`qd_builders_tangent` at the
  tangent handed over) zero to 1e-10 of the larger of f^(x) and κ_oo's part,
  κᵀ f + f κ, per block and coordinate.

The reference is converged to ‖g‖ < 1e-10, which takes the CI's residual to
~6e-14: Eq. qd-theta reads H c_r = E_r c_r (ξ adds nothing to ⟨c_s|H|c_r⟩
only at eigenvectors), and at 1e-8 the CI's ~1e-10 residual against ξ missed
the weighted case's bar by 1.06×.

`DegeneratePairRule` (slow tier): `df_nevpt2_tangents`' C3v ammonia in
6-31G, SA-2 CAS(8,7) over its full valence, both roots A1, whose virtual e
pairs stay degenerate: SC refused, naming a pair and a coordinate; PC's
records. (HF's SA-2 CAS(2,2) over the σ bond and antibond is a saddle: its
CP solve meets negative curvature.) `Refusals`: no consumer, λ^x
short, roots 0 and 1 of the weighted case moved within the gap floor, and to
1e-6 Eh apart (θ past the bound), and `df_sacasscf_cp` naming itself.

Measured on the gfx1200 (`--preset hip`):

| | equal weights | 0.4, 0.4, 0.2 |
|---|---|---|
| CP residual, worst column | 8.3e-11 | 7.4e-11 |
| Rotation condition, equal pairs, worst / largest summand | 5.6e-13 | 2.3e-13 |
| Unequal pairs' rows, absolute | (none) | 1.1e-14 |
| κ_oo condition, worst / larger summand | 1.0e-12 | 2.4e-12 |
| Orthonormality | 1.1e-15 | 1.7e-16 |
| max \|θ\|, max \|κ_oo\| | 0.69, 15.5 | 0.25, 32.6 |
| Largest equal-weight coupling 𝓗 | 0.21 | 0.037 |
| `df_sacasscf_cp`, the tangents | 1.0 s, 0.41 s | 0.94 s, 0.39 s |

The ammonia's SC refusal is at coordinate 0 (orbitals 9 and 8, 1e-11 Eh
apart); PC records 22 pairs.

## The DF second derivative at an MO adjoint

`df_adjoint_xy` is the first term of the NEVPT2 skeleton L_AB
(`docs/nevpt2/nevpt2.tex` eq. Nab's first term, #809): a fixed symmetric MO adjoint
(h̄, Λ), nevpt2::ScAdjoint's shapes, against the second derivatives of the
integrals it is the adjoint of,

    Φ_xy = Σ_pq h̄_pq h^xy_pq + Σ_L,pq Λ_L,pq B^xy_L,pq,

the E_corr + Φ share of L_AB without its integral curvature (the HVP's
⟨t_A, HVP t_B⟩ term is that).

```cpp
auto phi = hessian::df_adjoint_xy(res, molecule, basis, classes, aux, h_core, b, l, c,
                                  adjoint.h_bar.data(), adjoint.lambda.data());
// *phi: 3 n_atom x 3 n_atom, element x + 3 n_atom y, symmetric to rounding
```

**The path.** h_MO and B_MO are differentiated along

    C(R)    = C (Cᵀ S(R) C)^−½,                         τ^x = −½ S^x,  τ^xy = −½ S^xy + ⅜ {S^x, S^y}
    B_MO(R) = (L⁻¹ M(R) L⁻ᵀ)^−½ L⁻¹ (P|pq)(R)[C(R)],  μ^x = −½ m^x,  μ^xy = −½ m^xy + ⅜ {m^x, m^y}

(S^x in the MO basis, m^x = L⁻¹ M^x L⁻ᵀ, L the metric's Cholesky factor at
the reference, {a, b} = ab + ba): the Löwdin connection to second order, as
`df_casscf_exy`'s τ^xy, and the fitted index carried the same way.

**The gauge at second order.** B's fitted index is only defined up to an
orthogonal turn O(R) (any L(R) with L Lᵀ = M(R) fits), and a general Λ sees
it. The path above is the factor L(R) = L (1 + m(R))^½: its first order turns
the fitted index by μ^x = −½ m^x, symmetric, which is exactly
`mcscf::df_integral_derivatives`' W^x
(`../mcscf/third_derivative/README.md`, "The gauge"), and its second order is
the symmetric square root's, μ^xy above. So B^x and B^xy are the first two
derivatives of one curve, which a Hessian's two halves must share: L_AB's
HVP term pairs W^x (#807's gauge) and this term pairs B^xy, and their sum is
the derivative of one function only because the curve is one. Another gauge
with the same first order, O(R) = 1 + ½ A^xy δx δy + …, A^xy antisymmetric,
adds ⟨G, A^xy⟩, G_QP = Σ_pq Λ_Q,pq B_P,pq, which E_corr never sees: it
depends on B only through (pq|rs) = Σ_Q B_Q,pq B_Q,rs, so its Λ = 2 B V
(V = ∂E/∂(pq|rs)) makes G symmetric, and the exact turn O(R) leaves E_corr
unchanged at every order. The same holds for E_cas and every term bilinear
in B. #816 proves the pairing end to end.

**The pieces.** B = U_Q U_p U_q t, three connections (μ on Q, τ on the left
and on the right) acting on t = L⁻¹ (P|mn)[C], each U = 1 + u^x δx +
½ u^xy δx δy. With f^x = B^x the full first derivative, the second is

    f^xy = t^xy + Σ_k [u_k^x f^y + u_k^y f^x + (u_k^xy − {u_k^x, u_k^y}) B]
                − Σ_k<l (u_k^x u_l^y + u_k^y u_l^x) B,
    u^xy − {u^x, u^y} = −½ s^xy + ⅛ {s^x, s^y}   (s = S or m),

and h_MO the same over its two connections. Contracted with (h̄, Λ), with
X = 2 h_MO h̄ + 2 Σ_Q B_Q Λ_Q (ScAdjoint's X) and G as above:

| Term | How |
|---|---|
| ⟨C h̄ Cᵀ, h^xy⟩ | `kinetic_hessian` + `potential_hessian` |
| ⟨Λ, t^xy⟩ | `three_center_hessian` at Γ = L⁻ᵀ C Λ Cᵀ (`mcscf::df_gradient`'s extra Γ) |
| −½ ⟨sym G, m^xy⟩ | `metric_hessian` at −½ L⁻ᵀ sym(G) L⁻¹ |
| −½ ⟨sym X, S^xy⟩ | `overlap_hessian` at −½ C sym(X) Cᵀ |
| ⅛ ⟨sym G, {m^x, m^y}⟩ + ⅛ ⟨sym X, {S^x, S^y}⟩ | ⟨sym(G) μ^x, μ^y⟩ + ⟨sym(X) τ^x, τ^y⟩, Gram matrices |
| ⟨Λ, μ^x B^y⟩ − ⟨Λ, μ^x (τ^y B + B τ^y)⟩, + (x ↔ y) | ⟨μ^x, H^y⟩, H^y = Λ W^yᵀ − 2 Λ (B τ^y)ᵀ over the pairs (naux²) |
| ⟨Λ, τ^x B^y + B^y τ^x⟩ + ⟨h̄, τ^x h^y + h^y τ^x⟩, + (x ↔ y) | 2 ⟨τ^x, N^y⟩, N^y = Σ_Q Λ_Q W^y_Q + h̄ h1^y |
| −⟨Λ, τ^x B τ^y + τ^y B τ^x⟩ − ⟨h̄, τ^x h τ^y + τ^y h τ^x⟩ | −2 ⟨τ^x, V^y⟩, V^y = Σ_Q Λ_Q τ^y B_Q + h̄ τ^y h_MO |

(B^y, h^y the full first derivatives W^y and h1^y.) The first derivatives
(h1^x, τ^x, W^x) are `mcscf::df_integral_derivatives`' chunks, its consumer
this call's: per coordinate y, N^y is one gemm of Λ and W^y side by side as
nmo × (nmo naux) (Λ transposed once to W's layout, each slab symmetric), H^y
two gemms over the pairs and the B τ^y between them, and its column of
⟨μ^x, H^y⟩ is taken by atom blocks (below), so W^x is never held past its
chunk. V^y is one strided-batched gemm (τ^y B_Q over the naux × nmo slab at
each q) and one gemm, after the chunks. The τ Gram matrices are one gemm
over their stacks.

**The products by atom blocks.** T^x M^y and M^x M^y meet the adjoint as
⟨μ^x, H⟩, at H = H^y and at H = sym(G) μ^y, and μ^x = −½ L⁻¹ M^x L⁻ᵀ is
dense in both indices however local M^x is. So the trsms go on H's side
instead:

    ⟨μ^x, H⟩ = ⟨M^x, Z⟩ = ⟨Ŝ_x, Z + Zᵀ⟩,   Z = −½ L⁻ᵀ H L⁻¹,   M^x = Ŝ_x + Ŝ_xᵀ,

Ŝ_x being `metric_derivative`'s slab k at its columns on atom A (x = 3A + k,
zero elsewhere). Those columns are contiguous in M¹ and in Z + Zᵀ, so one
H's column over every x is two trsms, a geam and one gemv per atom (its
three k at once, over naux · n_A rows). H^y's column comes so as each
chunk's W^y arrives; for M^x M^y, μ^x is made one coordinate at a time
(M¹'s atom block, two trsms), multiplied by sym(G), and taken the same way.
Nothing per coordinate is naux², so no stack of 3 n_atom naux² is held:
the price is two more trsms per coordinate on each of the two products.

**No more kernels.** Every integral class B^xy holds (T^xy, M^xy, T^x M^y,
M^x M^y, the connection's S^xy, S^x S^y and S^x against T^y and h^y) is a
contraction above over integrals the tree already has: the products of first
derivatives are Gram matrices of first-derivative stacks, not new kernels.

**The ladder sees the connection's curvature.** The FD oracle below
differentiates `mcscf::df_gradient`'s extra-adjoint pass at displaced
geometries. There the pass differentiates along the connection and the
gauge re-centred at the displaced point, whose tangent differs from the
path's by ⅛ [S^x, S^y] δy (and ⅛ [m^x, m^y] δy on the fitted index) at first
order: the ladder's Jacobian is Φ_xy plus an antisymmetric K_xy, seen
through X's and G's antisymmetric parts, which a general adjoint has (a
stationary Lagrangian's total X is symmetric, and E_corr's G is). So Φ_xy is
the Jacobian's symmetric part, and the test compares against that.

### Memory

Held for the call (its own: b_mo, `metric_derivative`, the derivative calls
and the second-derivative integrals all take the workspace): B over every MO
pair, Λ in W's layout and one more naux nmo² scratch (3 naux nmo²), the τ^x
stack and one more (2 · 3 n_atom nmo²), M¹ (3 naux²), G and two naux²
scratches, four nmo² matrices and the AO matrix, and the five Gram matrices.
W^x is `df_integral_derivatives`' chunks, as many coordinates as the rest of
the budget (the device's free memory at the call, or `max_bytes` when that
is less) holds. When this call's buffers and `df_integral_derivatives`' own
beside one coordinate's W^x (its count, in `integral_derivative.cpp`) pass
the budget, the call is Unsupported, naming the bytes, before anything is
allocated.

The fitting set enters as naux² only through M¹ and the scratches, six
naux² in all at any n_atom: the products' atom blocks (above) replaced a
μ^x stack of 3 n_atom naux² (#922; 1500 fitting functions over 30 atoms was
1.6 GB of it). What scales with the system now is the naux nmo² factors:
three here, and `df_integral_derivatives`' factor and one W^x beside them.

Measured on the A100 (`--preset default`) for pHBI (21 atoms,
`phbi_tz1211_s0.xyz`) in cc-pVTZ over cc-pVTZ-JKFIT, 518 orbitals and 1316
fitting functions, at random (h̄, Λ), each run at the smallest budget it
takes (one coordinate a chunk): the refusal threshold went from 16.44 GB to
15.58 GB (the 62 naux² dropped, 0.86 GB), and the card's peak over the run
from 33 741 MiB to 32 941 MiB; the stack's version refuses at the blocked
one's budget. The two agree to 3.5e-11 of max |Φ| (9.3e5), and both took
17 min, the derivative walk's 63 chunks, not the products.

### Tests

`test/gtest/df_adjoint_xy_gpu.cpp` (suite `HessianDfAdjointXy`), a skew
water in cc-pVDZ (d on O) over cc-pVDZ-JKFIT, at random orthonormal orbitals
C₀ = S^−½ U and a random symmetric (h̄, Λ), elements uniform in [−1, 1]:

- `MatchesTheLadder`: every coordinate pair within 1e-7 of the ladder's
  largest element, against the symmetric part of a fourth-order central
  difference (±h, ±2h, h = 2.5e-4 bohr; the 5e-4 rung beside it for the
  report) of `mcscf::df_transition_gradient`'s extra-adjoint pass at zero
  RDMs over one active orbital and no core (so the pass alone), with the
  adjoint held: at each displaced geometry the orbitals carried along the
  connection, C₀ (C₀ᵀ S C₀)^−½, Λ turned into the Cholesky frame
  `density_fit` takes there (Λ' = Oᵀ Λ, O = (L₀⁻¹ M L₀⁻ᵀ)^−½ L₀⁻¹ L, so
  that O B_chol is B in the reference's symmetric gauge), and X rebuilt
  there. The analytic run is first refused below one coordinate's budget
  (Unsupported, naming the bytes), then run at the smallest budget that
  takes two coordinates a chunk, and reported against an unchunked run.
- `Refusals`: no h̄, no Λ, the fitting set on two of the three atoms, C
  over another basis.

Not gated, implied by the ladder: the match to `df_casscf_exy` at E_cas's
adjoint, x ↔ y symmetry, the translational sum rule.

Measured on the 3080 (`--preset default`): 24 orbitals, 116 fitting
functions, max |Φ_xy| 842; worst miss 3.1e-6 (3.7e-9 of it) at h = 2.5e-4,
4.6e-5 at 5e-4, falling 15× a halving, so the stencil's truncation, not the
analytic value; the Jacobian's antisymmetric part 796 (the curvature above,
as large as Φ itself at a random adjoint); the chunked run against the whole
4.6e-12; the call 0.9 s, the ladder 6.3 s (72 passes).

## The state-averaged z terms

A state average's root I has the Hessian of its Lagrangian (`casscf_hessian.tex`
eq. sa): E_I,xy plus the z terms, z = root I's multipliers,

    L_xy = E_I,xy + z · ∂_xy g_SA,      L_xλ = ∂_x ∂_λ E_I + z · ∂_x ∂_λ g_SA,

and `df_sacasscf_z` returns those two z terms at the gradient's own z (no z^x;
the remaining terms, E_xy, the response, the third derivative and the
root-rotation term, are #621's sum). A QD-NEVPT2 state's Lagrangian
(`docs/nevpt2/nevpt2.tex`, Eq. qd-Lxx) has the same terms at its own
multipliers (Z, {z_r}, Θ) (`nevpt2::qd_sc_lagrangian`'s and
`qd_pc_gradient`'s `lagrangian.multipliers`), its E_cas at ρ = diag W: Part
I's level-2 terms at the SA weights, the CI curvature the root set's,
½{K_u, K_v} c_r (Eq. qd-curv), which `ThirdDerivative`'s joint exponential
already is. So the call takes the right-hand side's weights rather than a
root: one-hot on I for `df_sacasscf`, ρ for a QD Lagrangian (linear in it),
or none (empty) for the multipliers' terms alone. It is assembly only:

```cpp
auto t = hessian::df_sacasscf_z(res, molecule, basis, classes, aux, h, b, l, c, ncore, ncas,
                                nalpha, nbeta, roots, weights, rhs_weights, z);
// t->lxy: L_xy's z part, CasscfExy's pieces (nuclear_repulsion zero), t->lxy.exy the sum
// t->lxlambda: size x 3 n_atom on the device, column x = L_xλ's covector in
//              ThirdDerivative's packed layout; L_xλ λ^y = column x . λ^y
```

`z` is one column in `mcscf::ThirdDerivative`'s layout and **the energy's
convention** (the packed pairs, d_r per root, Θ_sr for every r < s unless all
weights are equal: at weights equal for some pairs only, such as
(0.4, 0.4, 0.2), the equal pair's Θ is a slot too, zero in zvector's z),
which is `response::zvector`'s (z_o, z_c, Θ) as it comes: zvector
scales its equations' CI rows, not its unknowns (`df_sacasscf`'s step 2).
The d_r are taken off every root first, as `ThirdDerivative` takes them. In
order:

1. **L_xλ's right-hand side**, if `rhs_weights` is not empty:
   `mcscf::df_derivatives` at those weights (one-hot on root I, or ρ;
   projected off every root, Θ rows unless all weights are equal), packed:
   `pack_orbital_pairs` per coordinate, the CI rows copied as the energy's
   (2 ρ_r Q H̃^x c_r), `rhs_theta` for r < s.
2. **L_xy's z_o part**: K = K(z_o) by `unpack_orbital_pairs`, `df_derivatives`
   at the averaged weights with that rotation (`fock_rotated`) and
   `df_casscf_exy` with it, at the averaged RDMs.
3. **L_xy's z_c part**: dc_r = d_r + Σ_s Θ_sr c_s (one gemm C Θ), and E_xy at
   the symmetric transition RDMs Σ_r w_r (T(dc_r, c_r) + T(c_r, dc_r)) as half
   the difference of `df_casscf_exy` at the RDMs and F^[x] of c_r + dc_r and of
   c_r − dc_r. E_xy is affine in (γ, Γ, F^[x]) at a fixed core, so the core's
   constant part cancels, and the polarization is exact (bilinear), not a
   difference quotient. Added piece by piece to step 2's.
4. **I^x z**: `ThirdDerivative::create(z)`, then
   `mcscf::df_integral_derivatives` (h1^x and W^x, the connection included,
   W^x a chunk of coordinates at a time, each applied by `apply_integrals`),
   added to step 1's columns (written whole without step 1).

The test (`test/gtest/df_casscf_hessian_rotation_gpu.cpp`) that first
assembled steps 2–3, and `test/gtest/mcscf_integral_derivative_gpu.cpp`
steps 1 and 4, are the same formulas; this is their one entry point.

### Memory

The parts run one after another, each freeing what it held: four
`df_derivatives` calls (the right-hand side's, rotated, c + dc, c − dc), three
`df_casscf_exy` calls, then the operator, then h1^x and W^x (W^x the
largest, 3 GB per coordinate at 500 functions over 1500 fitting functions,
so built for as many coordinates at once as the device's free memory holds
and applied a chunk at a time). Held throughout: L_xλ (size × 3 n_atom)
and a copy of C. Each part refuses its own memory, Unsupported, before it
allocates.

### Tests

`test/gtest/df_sacasscf_hessian_z_gpu.cpp` (suite `HessianDfSaCasscfZ`),
every root of the four state-averaged cases of
`test/gtest/data/casscf_hessian/` (two roots at (0.5, 0.5) and (0.7, 0.3),
three at equal weights and at (0.5, 0.3, 0.2)), the orbitals, roots, z and
λ^x **loaded from the fixture**: L_xy's z part piece by piece against
`z_one_electron`, `z_two_electron`, `z_connection_second`,
`z_connection_fock`, `z_connection_twice` and `exy_z`; L_xλ's covectors part
by part against `root_rhs_*` + `lxl_z_*` (the CI rows' singlet part), and
L_xλ λ^y against `lxl_lambda`, all to 1e-10 absolute; each L_xy piece's
asymmetry to 1e-11 and both terms' translational sum rule to 1e-9. Every
case's z, λ^x and Θ come from the fixture, the three roots at unequal weights
included (`response::zvector` carries Θ there since #613), the right-hand
side at one-hot weights. `RefusesBadInputs`: right-hand side weights not one
per root, one root, and z of the wrong length (at unequal weights, and at
partly equal ones, it carries the roots' rotations).

Measured on the RTX 3080 (CUDA), worst over the ten roots: L_xy's z part
≤ 6.7e-13 from `exy_z` (of max 0.0088–0.021), ≤ 6.6e-13 two-electron,
≤ 7.6e-15 for the one-electron and connection pieces; L_xλ's covectors
≤ 3.1e-13 orbital (of 1.9), ≤ 1.3e-11 CI (of 0.19), ≤ 7.4e-14 Θ (of 0.20);
L_xλ λ^y ≤ 1.2e-13 from `lxl_lambda` (of 0.26–0.80); sum rules ≤ 6.6e-13;
1.2–1.4 s a call (the water in 6-31G, 9 coordinates).

`test/gtest/df_sacasscf_hessian_z_qd_gpu.cpp` (suite `HessianDfSaCasscfZQd`)
runs the call **at a QD-NEVPT2 Lagrangian's multipliers**:
`nevpt2::qd_sc_lagrangian`'s (Z, {z_r}, Θ) for state K = 1, on
`data/qd_nevpt2/`'s `h2co_skew_ccpvdz_sa3` (equal weights, slow tier) and
`h2co_skew_ccpvdz_sa3_weighted` (0.4, 0.4, 0.2, the representative). Each
case uses this tree's own DF-SA-CASSCF through the registry, with no
right-hand side. It checks the call and `ThirdDerivative` against a
Richardson ladder of F = Z · g_SA + Σ_r z_r · g_ci,r (+ Θ · g_Θ) = d/dε
E_SA(λ + ε z). F is evaluated on the host at each displaced geometry, along
the symmetric connection and the root set's joint exponential, with the
exponentials' derivatives exact. The ladder runs along two coordinates (O's
z, the first H's y) and along their sum e, with the parameters moving along
x_e: `df_sacasscf_cp`'s λ^x and, at partly equal weights,
`df_qd_nevpt2_tangents`' rotations (Eq. qd-theta's for the equal pair).
Checked: T(x_e, x_e, z) against d²/dt² F(R₀, t x_e), and L_ee + 2 L_eλ · x_e +
T(x_e, x_e, z) against d²/dt² F(R₀ + t e, t x_e), each to 1e-7 absolute. Two
shares must each exceed 10× the bar. The first is the root-set curvature's
share: the third term's ladder in the joint parametrization, less the same
ladder with each root on its own sphere. The second, at unequal weights, is
Θ's share: the ladders at Θ alone.

Measured on the gfx1200 (HIP), worst miss: weighted 2.1e-8 (third term) and
2.0e-8 (total) of max 0.15; equal 2.0e-8 and 2.0e-8 of max 0.040. The
polarized mixed element misses by ≤ 1.3e-8. The root-set curvature's share
is 6.6e-4 (weighted) and 1.6e-4 (equal); Θ's share is 3.6e-3. Each case
takes about 12 s, of which df_sacasscf_z and the operator take 9 s.

## The single-state Z terms

A single-state reference's Lagrangian (`docs/nevpt2/nevpt2.tex`, eq. hess; a
NEVPT2 gradient's E_cas + Z·g + …) has three places its multipliers
(Z, z) = Z meet the CASSCF gradient g = ∂E/∂λ: Z·∂²g/∂R∂R in L_AB,
Z·∂²g/∂R∂x in L_Ax and Z·∂²g/∂x∂x in L_xx. `df_casscf_z` returns the three
at caller-supplied Z, and solves nothing (λ^x is the caller's):

```cpp
auto t = hessian::df_casscf_z(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                              ncas, z, k, lambda);
// t->lxy:      Z·∂²g/∂R∂R, CasscfExy's pieces (nuclear_repulsion zero), t->lxy.exy the sum
// t->lxlambda: Z·∂²g/∂R∂x, size × 3 n_atom on the device, column x the covector I^x Z
// t->third:    Λᵀ (Z·∂²g/∂x∂x) Λ, k × k on the host, over lambda's k columns
```

It is `df_sacasscf_z` at one root of weight 1 (the two share one body), with
two differences: no step 1 (∂_x ∂_λ E is the energy's own term,
`df_casscf`'s E_λx, not Z's), and the operator step 4 builds is applied once
more, to Λ, for the third term (`ThirdDerivative::apply`, then Λᵀ Y). The
inputs are `df_casscf`'s (a one-root `Solution`, its checks and refusals) plus
z, pairs + n_det doubles in `response::zvector`'s coordinates, its CI part
taken off the root. Nothing restricted the parts to a state average but
`df_sacasscf_z`'s own two-root check: `ThirdDerivative`, `df_derivatives` and
`rdm12` take one weight of 1 as they are, and the operator has no rotations
among one root. `df_casscf_z` passes no right-hand side weights.

### Memory

`df_sacasscf_z`'s, less its one-hot `df_derivatives` call, plus Y
(size × k) while Λᵀ Y is formed.

### Tests

`test/gtest/df_casscf_hessian_z_gpu.cpp` (suite `HessianDfCasscfZ`), the
stretched water's CAS(4,4)/6-31G over cc-pVDZ-JKFIT at the fixture's
`h2o_631g` geometry and orbitals (made orthonormal over this tree's S, the
root re-solved by CASCI at them), with a random Z and a random Λ of three
columns (elements up to 0.1, CI parts off the root). Each term against a
Richardson ladder of **Z·g itself**, never of a Hessian-vector product: Z·g
is evaluated on the host at this tree's integrals at each displaced geometry,
along the symmetric connection C(R) = C (Cᵀ S(R) C)^−½ and the exponential
parametrization C(R) e^K, e^A c, the exponentials' derivatives by their
series (exact), the orbital part by ∂E/∂U_o and the CI part by the ±
polarization (the two vectors' densities differenced before they are
contracted: the energies themselves are ~80 hartree), only the RDMs from
`ci::Rdm`. The fitted factor at each geometry is M's own Cholesky factor
there, not the derivative's gauge: E sees it only as Σ_Q L_Q L_Q.

- Z·∂²g/∂R∂R against the mixed central difference of Z·g(R, 0) in
  (R_x, R_y), steps 4e-3 and 2e-3 bohr, every (x, y);
- Z·∂²g/∂R∂x (column x against λ_j) against the mixed difference of
  Z·g(R, t λ_j) in (R_x, t), steps (4e-3, 4e-3) and half of both;
- Λᵀ (Z·∂²g/∂x∂x) Λ against the mixed difference of
  Z·g(R₀, s λ_i + t λ_j), steps 4e-3 and 2e-3;

each to 1e-7 absolute. The host energy at the reference is checked against
the CASCI root's to 1e-10 first. `Refusals`: z one short, a negative k, no
Λ for k = 1, and ncas 0.

Measured on the RTX 3080 (CUDA): Z·∂²g/∂R∂R ≤ 3.6e-10 from its ladder (of
max 0.13), Z·∂²g/∂R∂x ≤ 1.2e-10 (of 0.030), Λᵀ (Z·∂²g/∂x∂x) Λ ≤ 4.1e-9 (of
0.19); 1.2 s for the call (the water in 6-31G, 9 coordinates), 2.3 s for the
ladders. The ladders are rounding-limited, not truncation-limited: the
third term's miss rises from 4.1e-9 at a 4e-3 step to 3.1e-8 at 2e-3 and
6.9e-8 at 1e-3, so the steps are the largest of those. `HessianDfSaCasscfZ` and `HessianDfSaCasscf` are unchanged (the
d_r's projection is a no-op on zvector's z, which is off the roots
already).

## The SC-NEVPT2 Hessian

`df_sc_nevpt2` is `docs/nevpt2/nevpt2.tex` eq. (hess) for SC-NEVPT2 on a
converged single-state DF-CASSCF (#816), assembled from the parts above and
the NEVPT2 gradient's, no integrals of its own:

    d²E/dR_A dR_B = L_AB + L_Ax x_B + x_Aᵀ L_xB + x_Aᵀ L_xx x_B,
    L = E_cas + Z · g + E_corr + Φ,   x = (κ_nr, κ_oo, c),   g_x x_A = −g_A

```cpp
auto h = hessian::df_sc_nevpt2(res, molecule, basis, classes, aux, h_core, b, l, solution, ncore,
                               ncas, batch, coordinates);
// h->hessian: k x k in the coordinates, exactly symmetric; h->asymmetry, h->translation;
// the parts, each k x k: h->e_cas, h->z_lxy, h->z_lx, h->z_third, h->z_koo,
// h->skeleton, h->hvp, h->rotation_tau, h->rotation_kappa;
// h->casscf (df_casscf's terms: lambda^x, its residuals); h->e_corr, h->max_zc,
// h->min_gap, h->zvector_residual, h->defect, h->mix_step
```

**One parametrization for every part**: the orbitals C(R) exp(K) along the
Löwdin connection C(R) = C (Cᵀ S(R) C)^−½, K = κ_nr + κ_oo antisymmetric, the
CI vector exp(A) c on the unit sphere, B's fitted index in `df_adjoint_xy`'s
gauge. Every part below is the second derivative of its term of L along
that one path, so their sum is eq. (hess); a part in another
parametrization would leave a term of second order in κ unaccounted. The
three terms of L, and in order what computes them:

1. **E_cas**: `df_casscf`, as is. Its λ^x is x's (κ_nr, c) block, eq. cp.
2. **t_x**: `df_nevpt2_tangents` over that λ^x, with κ_oo^x (eq. koo).
3. **E_corr + Φ's adjoint** (h̄, Λ) and the gradient's multipliers:
   `sc_adjoint`, `zc_multipliers`, `zc_adjoint` and `lagrangian_gradient`,
   `sc_gradient`'s calls with the adjoint kept.
4. **Σ ⟨t_A, HVP t_B⟩**: `sc_hessian_product` along each coordinate's t
   at the gradient's z^c, paired with every t_A (three gemvs per product).
5. **The Z terms** at the gradient's (Z, z): `df_casscf_z`, its lxy, I^x Z
   against λ^y in both orders, and Λᵀ (Z · ∂²g) Λ over λ^x.
6. **The skeleton**: `df_adjoint_xy` at E_corr + Φ's (h̄, Λ).
7. **The rotation cross terms and the κ_oo rows**, below, from nmo² stacks.

### E_corr + Φ: what the HVP leaves

Write N = E_corr + Φ as a function of I = (h_MO, B_MO) and c, and
O(M, κ) = κᵀ M + M κ, the one-index transform, so that I(R, K) =
e^−K I(R) e^K and t_A = I^A + O(I, κ^A) + c^A. The chain rule on N along the
path, x_AB dropped (eq. hess has none):

    N_AB^total = ⟨N_I, I^AB⟩                                 skeleton
               + ⟨t_A, N'' t_B⟩                               HVP
               + ⟨N_I, O(I^A, κ^B) + O(I^B, κ^A)⟩             τκ: ∂_A ∂_K e^−K I(R) e^K
               + ⟨N_I, ½ O(O(I, κ^A), κ^B) + (A ↔ B)⟩         κκ: e^−K I e^K's second order

The HVP is F = N(c/|c|)'s, the Rayleigh-normalized functional: its CI–CI
elements carry the sphere's −(u·v) N_c · c, which is exp(A) c's second
order, so no c term is left out.

**Both cross terms are nmo² Gram matrices.** For symmetric M, h̄ and Λ,
⟨h̄, κᵀ M + M κ⟩ = 2 ⟨κ, M h̄⟩, so

    ⟨N_I, O(M, κ)⟩ = ⟨κ, X(M)⟩,   X(M) = 2 (M_h h̄ + Σ_Q M_Q Λ_Q),

X(I) = X_N, the adjoint's source term. Per coordinate: X(t_A) (from t's h
and B, two gemms), Y_A = X(O(I, κ^A)) = κᵀ X_N + 2 h κ h̄ + 2 Σ_Q B_Q κ Λ_Q
(κ Λ_Q one strided-batched gemm over Λ's slabs, then one gemm over B), and
X(I^A) = X(t_A) − Y_A. Then

    τκ_AB = ⟨κ^B, X(I^A)⟩ + ⟨κ^A, X(I^B)⟩,   κκ_AB = ½ (⟨κ^B, Y_A⟩ + ⟨κ^A, Y_B⟩)

with κ the total, κ_nr + κ_oo.

### The κ_oo rows

`ThirdDerivative` and `df_casscf` cover the non-redundant pairs. The κ_oo
rows of E_cas's and of Z · g's second derivative follow from one
invariance: E_cas(C U, c) = E_cas(C, c) for every block-diagonal orthogonal
U over the core and the virtuals, at every geometry and every c (κ_oo has no
active index, so c does not move). And from one identity, BCH to second
order:

    exp(sP + tQ) = exp(sP) exp(tQ) exp(−½ st [P, Q]) + O(3),

so for any F(C exp(K)), with P non-redundant and Q a κ_oo rotation,

    ∂_s ∂_t F = ∂_s ∂_t F(C e^{sP} e^{tQ}) − ½ ∇F · [P, Q],   ∇F · X = d/dε F(C e^{εX}).

[P, Q] is non-redundant (core–active, core–virtual, active–virtual).

**E_cas's rows are zero.** E_cas(C e^{sP} e^{tQ}) does not depend on t, so
its first term vanishes, and ∇E_cas = g = 0 at the converged CASSCF; κ_oo
against κ_oo (e^{tQ + t'Q'} block-diagonal) and against R
(E_cas(R, C(R) e^{tQ}) = E_cas(R, C(R))) are zero outright, and against c
too. So `df_casscf` is E_cas's whole share.

**Z · g's rows are ½ ⟨[κ_nr, κ_oo], X_N⟩.** The orbital gradient is
covariant, G(C U) = Uᵀ G(C) U (the CI gradient invariant), so
Z · g(C' U) = ⟨U Z Uᵀ, G(C')⟩ and the t-derivative at t = 0 is
⟨[Q, Z], G⟩, [Q, Z] non-redundant. Its derivatives along R_A, κ_nr^A and c^A
add to ⟨[κ_oo^B, Z], G_A + G_κ κ_nr^A + G_c c^A⟩, and that bracket is the
CP equation's orbital rows, zero at λ^A; κ_oo against κ_oo is zero as
before. What is left is BCH's term on both mixed pairs,
−½ ∇(Z · g) · ([κ_nr^A, κ_oo^B] + [κ_nr^B, κ_oo^A]). The Z-vector equation
is L_K = 0 along every non-redundant X, ∇(Z · g) · X = −N_K · X =
−⟨X, X_N⟩ (E_cas's term is g = 0), and ⟨[P, Q], X⟩ = ⟨Q, [X, P]⟩:

    κ_oo row_AB = ½ (⟨κ_oo^B, [X_N, κ_nr^A]⟩ + ⟨κ_oo^A, [X_N, κ_nr^B]⟩).

N's own κ_oo terms are not separate: κ_oo is in t's κ, so the HVP, τκ and
κκ carry them. Exact up to the CP's and the Z-vector's residuals.

### Coordinates, and the empty perturbers

The Hessian is returned in k coordinates q_i = Σ_x Q_ix R_x (`coordinates`,
orthonormal rows; the Cartesian ones when empty), every bilinear part as
Q P Qᵀ, the HVP along each coordinate's own t_i = Σ_x Q_ix t_x. That is not
bookkeeping: an empty SC perturber (|N| ≤ 1e-14, by symmetry) that a
direction populates takes K6 at that direction's η, so the HVP is not
linear in the direction there, and a symmetry-breaking Cartesian t mixes
the symmetric block into the broken one. The test's symmetric case passes
the fixture's symmetry-adapted coordinates.

A coordinate breaks the symmetry when its product populates an empty
perturber (N^{ii} past `nevpt2::kEmptyPopulationFloor`) against the
**largest** population scale over the coordinates, not its own. The rows
are orthonormal, so every coordinate has the same size, and one that moves
nothing (a rigid translation, whose tangent is round-off) has a round-off
scale of its own, against which round-off populates everything: on
`nevpt2.toml`'s water in symmetry-adapted coordinates the translations
Tz and Tx read N^{ii} 1e-29 to 2e-22 against the vibrations' scale of 2e4,
and before #952 the pair (Tz, Tx) was refused on that. Every breaking
coordinate is taken again against each other such coordinate, the pair as each other's
`sc_hessian_product` partners (K6 at the mean η, element (i, j) from the
product along t_j partnered by t_i). A pair `class_energies_hvp` refuses
(its weighted ρ_τ past 1e-6: the energy is not a quadratic form on the
pair) is collected, and when any is, the call is Numerical naming every
refused pair with its message: refused, never reported. `defect` is the
largest accepted pair's ρ_τ.

### Mixing inactive pairs

SC-NEVPT2 is not invariant to rotations among the core or among the
virtual orbitals, and its energy is not smooth at symmetric geometries
(#894; Park 2019 notes the same caveats). That is known and is not this
call's to fix. Its job is to **not publish that non-smoothness silently**,
so SC is refused at such points by design, and PC is the method to use.

The case that showed it (#952) is `examples/nevpt2.toml`'s water (C2v,
cc-pVDZ / cc-pVDZ-JKFIT, CASSCF(4,4)). There the call refused no pair, yet
its B2 frequency was 14063 cm⁻¹ against PC's 4028 (the A1 ones within
0.6 cm⁻¹). No empty perturber is involved: the B2 vibration populates
none. Its virtuals 17 (a1) and 18 (b2) are 1.435e-3 Eh apart, and a B2
displacement couples them. The pair's 2 × 2 block [[ε_p, F t], [F t, ε_q]]
has gap √(g² + 4F²t²) and mixing angle tan 2θ = 2Ft/g, and κ_oo = F/g is
the tangent's own. The smallest inactive gap along the B2 vibration (from
the numerical `[hessian]`'s displaced gradients) follows that curve, with
κ_oo = 890 /bohr:

| t (bohr) | 0 | 1.25e-4 | 2.5e-4 | 5e-4 | 1e-3 | 4e-3 | 3e-2 |
|---|---|---|---|---|---|---|---|
| gap (Eh) | 1.435e-3 | 1.470e-3 | 1.571e-3 | 1.922e-3 | 2.932e-3 | 9.885e-3 | 1.716e-2 |
| √(1 + (2κt)²) g | | 1.470e-3 | 1.571e-3 | 1.921e-3 | 2.931e-3 | 1.03e-2 | |

The B2 element, from a ladder (rungs h, h/2, h/4) of the analytic
gradient, against the analytic 15.729 Eh/bohr² (PC: 1.2850):

| h (bohr) | D(h) | D(h/2) | D(h/4) |
|---|---|---|---|
| 0.03 | 1.303 | 1.333 | 1.405 |
| 0.004 | 1.461 | 1.557 | 2.425 |
| 0.0005 | 6.72 | 12.90 | 15.32 |

So the analytic element is the correct t → 0 limit: the ladder reaches it
only below 1e-4 bohr. Past the avoided crossing's half-width the pair has
rotated, and the curvature is PC's. A frequency from it describes a region
narrower than the mode's own zero-point amplitude.

The rule is #817's branch-switch convention, as PC's Schur lift uses it
(below, "The refusals"). Per coordinate i and core–core or virtual–virtual
pair (p, q), t_mix = g / 2|F^i_pq| = 1 / 2|κ_oo^i_pq| is the half-width
of the pair's avoided crossing: there 2θ is 45° and the gap √2 g. Any
coordinate with t_mix below 1e-3 bohr is refused, Numerical, naming the
coordinate, its pair, the gap and t_mix, and saying "use PC-NEVPT2". The
same reason applies: closer than that, the surface switches branch inside a
vibrational or finite-difference step, and one analytic Hessian does not
represent it. `mix_step` returns the smallest t_mix of an accepted call.
The check reads κ_oo from the tangents and ε from the adjoint, before the
product, so a refusal costs E_cas and the tangents.

| case | coordinates | smallest t_mix (bohr) | refused |
|---|---|---|---|
| water (`nevpt2.toml`) | Cartesian | 7.53e-4 (O y; H y, z: 1.5e-3 and 3.9e-3) | O y |
| water | symmetry-adapted | 5.62e-4 (the B2 vibration) | the B2 vibration |
| `h2co_skew_sto3g`, `nevpt2_hessian.toml` | Cartesian | 0.554 | no |
| `nh3_cs_sto3g` | the fixture's | 0.240 | no (its A″ pairs are, as above) |
| `h2co_neardeg_sto3g` | Cartesian | 0.0229 (κ_oo^x 21.8, gap 6.0e-3) | no |
| water without the B2 vibration | symmetry-adapted | 0.180 | no |

PC is not refused on the same pair. Its ladder of the B2 element is flat,
1.2850090393, 1.2850090426, 1.2850090413 at h = 0.004, 0.002, 0.001 and
1.2850090384, 1.2850090392, 1.2850090198 at 5e-4 down to 1.25e-4 (floors
2.1e-7 and 1.7e-6), on the analytic 1.2850090388. PC is invariant enough
to the pair's rotation there, and its 4028 cm⁻¹ stands.

**All-electron, single state.** The call takes no frozen core, and
`state_specific_electrons` refuses a state average (QD's reference),
InvalidConfig. Its other refusals are its parts', passed on as they come:
`df_nevpt2_tangents`' degenerate inactive pair (Numerical), the CP solve's
saddle (Divergence), z^c's degenerate pair and the gradient's gate
(Numerical). The tangents run before the gradient, so a degenerate pair
names the tangents.

### Memory

The parts run one after another, each freeing what it held. Held
throughout: t_i's h, B and c and κ and κ_nr (k (naux nmo² + 3 nmo² + n_det)),
E_cas's terms, and E_corr + Φ's (h̄, Λ) (naux nmo²). The HVP's directions
are chunked by an estimate of its per-direction memory (16 naux nmo² +
8 n_act⁶ + 8 n_det + 16 nmo² doubles) against half the device's free
memory; a pair takes two more directions and their partners' copies. The
rotation step adds B_MO and one naux nmo² scratch, and four k nmo² stacks.
naux nmo² past `int32_t` is Unsupported.

### Tests

`test/gtest/df_sc_nevpt2_hessian_gpu.cpp` (suite `HessianDfScNevpt2`), on
this tree's DF-CASSCF of `nevpt2_hessian/`'s fixtures (#803) through the
registry (‖g‖ < 1e-10), against their semi-numerical SC Hessians:

- `MatchesTheSkewFixture`: `h2co_skew_sto3g`, Cartesian: every element
  within 2× the fixture's bar; each part's largest element in bars (the
  visibility rule, > 10); CP residuals ≤ 1e-8, asymmetry ≤ 1e-9, the sum
  rule ≤ 1e-8, no accepted pair defect (no SC perturber is empty there).
- `RefusesTheSymmetricFixturesKinkBlock` (slow tier): `nh3_cs_sto3g` in the
  fixture's symmetry-adapted coordinates, **#816's amended acceptance**:
  the call is Numerical, and the pairs it names (a) contain each of the
  fixture's failing pairs (7, 8), (8, 10) and (8, 11) and (b) all lie in the
  A″ × A″ off-diagonal block; (c) every element not refused (the A″
  diagonal and everything outside the block) is within 2× the bar, from
  runs over the A′ block with each A″ coordinate added. Why amended: the
  block is #894's SC kink at the symmetric geometry, empty perturbers whose
  energy along a breaking direction is t² Q(direction), a second derivative
  along every line but no bilinear form, so the block's second derivative
  is ill-defined there and refusing all of it is the right output. The
  original gate ("exactly the fixture's failing pairs") cannot be met by
  #914's weighted rule: every A″ pair is refused on class Sir's perturber 2,
  and the defects of the fixture's failing pairs and of its passing ones
  overlap, so no threshold separates them (measured on the RTX 3080):

  | pair | fixture | raw ρ_τ | weighted defect |
  |---|---|---|---|
  | (7, 8) | failing | 1.440e-2 | 2.111e-4 |
  | (8, 10) | failing, asymmetric | 4.597e-2 | 2.797e-4 |
  | (8, 11) | failing | 1.107e-2 | 8.309e-5 |
  | (7, 9) | asymmetric | 3.497e-3 | 4.285e-5 |
  | (7, 10) | asymmetric | 3.158e-2 | 1.811e-4 |
  | (10, 11) | asymmetric | 3.491e-2 | 1.025e-4 |
  | (8, 9) | passes | 1.789e-2 | 2.326e-4 |
  | (9, 10) | passes | 2.808e-2 | 1.428e-4 |
  | (7, 11) | passes | 3.331e-3 | 2.357e-5 |
  | (9, 11) | passes | 6.828e-3 | 4.284e-5 |

  Taken without partners (each direction's own η, symmetrized), the block
  misses the fixture by 12.1 bars at (7, 8), 5.0 at (7, 10) and 2.2 at
  (8, 11), and by under 1 bar elsewhere: the elements do not single out the
  fixture's three either.
- `RefusesTheSymmetricWater` (#952): `nevpt2.toml`'s water
  (`symmetric_water.h`), Cartesian: Numerical, naming O y and the pair
  (17, 18), not O z, and saying "use PC-NEVPT2".
- `SymmetricWaterRefusesOnlyItsB2Vibration` (#952, slow tier): the same in
  symmetry-adapted coordinates refuses exactly the B2 vibration; without
  it, the rest is a Hessian (the translations' rows ≤ 1e-8), so a rigid
  translation breaks no symmetry.
- `DegenerateInactivePairIsRefused`: `df_nevpt2_tangents_gpu.cpp`'s C3v
  ammonia (6-31G, CAS(8,7), virtual e pairs 1e-13 Eh apart): the tangents'
  Numerical refusal, naming the pair and the coordinate, passed on.
- `Refusals`: coordinates that are not whole rows, two rows along one
  coordinate, and a two-root solution (a state average, QD's reference).

Not gated (implied by the fixture match): the match to #804's numerical
`[hessian]`, the reduction to `df_casscf`, the asymmetry and the sum rule
(still returned).

Measured on the RTX 3080 (`--preset default`) and the gfx1200
(`--preset hip`), alike to the digits printed: the skew fixture's worst
element 0.103 bars; the parts' largest elements in bars E_cas 1.5e7,
Z · ∂²g/∂R∂R 1.05e6, Z's L_Ax pair 8.6e5, Z's third derivative 4.4e5 (the
smallest), the κ_oo rows 6.6e5, the skeleton 1.21e9, the HVP 1.21e9 (the
two largely cancel), τκ 4.5e5, κκ 3.7e6; CP residual 1.4e-9, Z-vector
3.4e-11, asymmetry ≤ 3.3e-13, sum rule 2.0e-10, max |z^c| 4.3e-3, smallest
inactive gap 0.140; 3.4 s a call (CUDA), 4.3 s (HIP). On the symmetric
fixture the outside of the A″ × A″ block sits within 0.096 bars.

## The PC-NEVPT2 Hessian

`df_pc_nevpt2` is the same eq. (hess) for PC-NEVPT2 (#817), and the same
assembly as `df_sc_nevpt2` above, part for part, in the same
parametrization and coordinates:

```cpp
auto h = hessian::df_pc_nevpt2(res, molecule, basis, classes, aux, h_core, b, l, solution, ncore,
                               ncas, batch, coordinates, pc_floor);
// h->hessian and its parts as df_sc_nevpt2's; h->kept (per class), h->lift_steps
// (per coordinate and class, t* in bohr); h->e_corr, h->max_zc, h->min_gap,
// h->zvector_residual
```

What changes is PC's own code at three steps, and nothing else:

- **Step 3**: `pc_adjoint` in place of `sc_adjoint`, then `zc_multipliers`,
  `zc_adjoint` and `lagrangian_gradient`, which is `pc_gradient`'s chain
  with the adjoint kept.
- **Step 4**: `pc_hessian_product` along each coordinate's t, paired with
  every t_A. PC has no empty perturbers (a tuple that vanishes by symmetry
  has b = S x = 0 and moves as a smooth quadratic), so it takes no partners
  and no pair is taken again.
- `pc_floor`, `EnergyInputs::pc_floor`, goes to the adjoint and the product
  alike, so every part differentiates the same energy.

### The refusals

The call refuses, Numerical, where one analytic Hessian does not represent
the PC surface:

- **A class kept below 1e-10** of its metric's largest eigenvalue (#917):
  the second derivative is not attainable there. The check reads the
  adjoint's `PcClassEnergies::kept` before the Lagrangian is solved, names
  each such class, and says that a `pc_floor` of 1e-10 cuts it.
  `pc_class_energies_hvp` would refuse the same class.
- **A Schur lift within 1e-3 bohr.** `pc_hessian_product` reports, per
  direction and class, the step t* at which a lifted null mode reaches the
  cut (`../nevpt2/gradient/README.md`, "The Schur lift"). Here the
  direction is the coordinate's own t, its CI part the CP response c^B, so
  t* is in bohr. Any coordinate and class with t* < 1e-3 bohr is refused,
  every one named with its t*. This is a stated convention: closer than
  that, the PC surface switches branch inside a vibrational or
  finite-difference step.
- **A degenerate inactive pair** that a coordinate splits:
  `df_nevpt2_tangents`' `DegeneratePair` records (gap below 1e-6 Eh,
  |f^(x)_pq| above 1e-8), each named. κ_oo^x_pq is not a division the state
  determines there. PC is invariant to the rotation only at exact
  degeneracy, and the displacement lifts that. So the canonical orbitals PC
  reads are not differentiable along the coordinate. The check is
  conservative: it refuses on any Cartesian record, even when the caller's
  coordinates combine them so that f vanishes.

Its other refusals are `df_sc_nevpt2`'s, plus the energy's own refusals of
a class (no positive metric eigenvalue, or a pole), passed on.

**Each refusal is reached by a real input (#943).** None of #803's four
fixtures reaches the kept or the lift check: their smallest kept is 2.3e-6
and their smallest t* is 6.5 bohr (below). t* is a lift of round-off null
modes there, so it varies by run and by card, but it stays three or more
orders above the bar. Three small inputs reach the three checks, at the
default cut (gfx1200, `--preset hip`):

- **kept**: a C2v formaldehyde in STO-3G with CAS(6,6) over orbitals 5 to
  10. Sr's and Si's metrics carry real modes between the cut and 1e-10 of
  their largest, as PSB3's do (`../nevpt2/energy/README.md`, "The
  conditioning floor"), and keep 2.9e-12 and 3.8e-12. At a `pc_floor` of
  1e-10 they keep 2.5e-10 and 2.8e-10 and nothing is refused, as the
  message says. HF CAS(10,7) with its F 1s active also trips the check
  (Sr keeps 5.9e-13). It is not the test's input, because `df_casscf`
  refuses it first: its H̃ is singular ("A singular H̃", #978).
- **t***: `hcn_sto3g`'s geometry with CAS(6,6) over orbitals 4 to 9, the
  fixture's π space and the σ pair beside it. The fixture's CAS(4,4) holds
  only π orbitals. An even number of π electrons there makes only Σ and Δ
  configurations, so a bend's CP response c^B is 0 in the active space and
  lifts nothing. With σ active, an x or y coordinate's c^B mixes σ and π.
  That breaks the axial symmetry that keeps Sir's null modes null, so it
  lifts them, with t* = 2.2e-4 to 4.6e-4 bohr on every x and y coordinate.
  The z coordinates keep the symmetry and lift nothing. The lift is a
  symmetry's, not round-off's, so t* repeats to the digits printed. At a
  `pc_floor` of 1e-10 the same t* are about 5e-3 to 1e-2, and nothing is
  refused.
- **the degenerate pair**: the C3v ammonia below. PC's tangents record its
  e pairs (gaps 1e-13, f^(x) 5e-2 to 7e-2) where SC's refuse, so the
  branch is reached and is not shadowed by `df_nevpt2_tangents`' own
  refusal.

### κ_oo at a near-degenerate pair

The Hessian's design notes left one question open: do PC's second-order κ_oo
terms cancel κ_oo^x's 1/(ε_p − ε_q)? `h2co_neardeg_sto3g` measures it. Its
virtual pair is 6.0e-3 Eh apart at R0, and the fixture's ladder runs
through points where the gap is 5.6e-5. There:

| | `h2co_skew_sto3g` | `h2co_neardeg_sto3g` |
|---|---|---|
| max \|κ_oo^x\| virtual–virtual | — | 21.8 (core–core 0.72) |
| κκ part, largest element | 3.8e6 bars | 1.5e9 bars |
| ⟨t, HVP t⟩, largest element | 1.1e9 bars | 1.8e9 bars |
| the κ_oo rows | 6.6e5 bars | 2.2e6 bars |
| total, worst miss | 0.100 bars | 0.085 bars |

The κ_oo terms that grow as 1/gap² are the κκ rotation term and the
HVP's κ_oo–κ_oo block. Each is about 1e9 bars, and they cancel in the sum
to the fixture's bar, which was measured independently by re-converged
energies through a 5.6e-5 Eh gap. So at this gap the 1/gap terms cancel
to the energy's own scale, and **the near-degenerate pair is not
refused**. The cancellation costs digits: about nine of the parts' sixteen
go into it. A pair much closer than this one would lose the rest before it
reaches the 1e-6 Eh rule. That is untested, because no fixture has such a
pair. The test asserts every element within 2× the bar. A pair below
1e-6 Eh is the degenerate case above, and is refused.

### Tests

`test/gtest/df_pc_nevpt2_hessian_gpu.cpp` (suite `HessianDfPcNevpt2`),
`df_sc_nevpt2`'s harness, against each fixture's PC Hessian:

- `MatchesTheSkewFixture`: `h2co_skew_sto3g`, Cartesian. Every element is
  within 2× the bar, every part's largest element is above 10 bars (the
  visibility rule), and nothing is refused. The CP residual must be
  ≤ 1e-8, the asymmetry ≤ 1e-9, the sum rule ≤ 1e-8 and every t* ≥ 1e-3
  bohr.
- `MatchesTheSymmetricFixture`: `nh3_cs_sto3g` in its symmetry-adapted
  coordinates. Every element is within 2× the bar and nothing is refused.
  PC has no lifted null mode there (its dropped modes are the CAS(2,2)
  structural ones), so the "Schur-lift pairs refused" item of the issue
  has no case. The diagnostics are as above.
- `MatchesTheNearDegenerateFixture`: `h2co_neardeg_sto3g`. It prints
  κ_oo^x's size from `df_nevpt2_tangents` (no record below the
  degenerate-pair rule) and the parts in bars. Every element is within 2×
  the bar and nothing is refused (the κ_oo measurement above).
- `MatchesTheLinearFixture`: `hcn_sto3g`. Every element is within 2× the
  bar, and no coordinate's lift is refused.
- `SymmetricWaterIsNotRefused` (#952, slow tier): `nevpt2.toml`'s water in
  symmetry-adapted coordinates, where SC is refused: PC is not, and its B2
  element is within 2× the floor (2.1e-7) of the numerical ladder's
  1.2850090413.
- `DegenerateInactivePairIsRefused`: the C3v ammonia of the SC suite is
  Numerical, naming `hessian::df_pc_nevpt2` and each pair with its
  coordinate, its orbitals and its gap.
- `ThinKeptClassIsRefused` (#943): the C2v formaldehyde above is
  Numerical, naming Sr with its kept below 1e-10 and the `pc_floor` that
  cuts it.
- `NearbyBranchSwitchIsRefused` (#943): HCN CAS(6,6) above is Numerical,
  naming every x and y coordinate with class Sir and a t* below 1e-3 bohr,
  and no z coordinate.
- `ContinuousFamilyIsRefused` (#978, slow tier): HF CAS(10,7), the CASSCF
  the kept bullet above names, is a Divergence from `hessian::df_casscf`
  calling its H̃ singular ("A singular H̃").
- `Refusals`: `df_sc_nevpt2`'s input refusals.

Measured on the gfx1200 (`--preset hip`) and the RTX 3080 (`--preset
default`). The two cards agree to the digits printed except where a cell
gives two values, gfx1200 / RTX 3080:

| fixture | worst | asymmetry | sum rule | CP residual | smallest kept | smallest t* (bohr) | test (CASSCF included) |
|---|---|---|---|---|---|---|---|
| `h2co_skew_sto3g` | 0.100 bars | 5.7e-14 / 2.8e-14 | 1.9e-10 | 1.4e-9 | 2.9e-7 | 25 / 16 | 5.9 / 4.7 s |
| `nh3_cs_sto3g` | 0.113 | 1.4e-14 / 7.2e-15 | 3.8e-12 / 2.6e-12 | 4.9e-11 | 1.4e-2 | 61 / 52 | 4.2 / 2.9 s |
| `h2co_neardeg_sto3g` | 0.085 | 2.8e-14 / 1.4e-14 | 1.4e-10 | 3.0e-9 | 2.3e-6 | 6.5 / 24 | 6.9 / 5.3 s |
| `hcn_sto3g` | 0.118 | 1.4e-14 / 5.7e-14 | 1.3e-12 / 9.3e-13 | 3.5e-11 | 3.0e-5 | 30 / 13 | 4.1 / 3.1 s |

On the skew fixture the parts' largest elements are E_cas 1.5e7 bars,
Z · ∂²g/∂R∂R 1.0e6, Z's L_Ax pair 8.6e5, Z's third derivative 4.4e5 (the
smallest), the κ_oo rows 6.6e5, the skeleton 1.12e9, the HVP 1.12e9, τκ
4.5e5 and κκ 3.8e6. Z-vector residual 3.4e-11, max |z^c| 4.3e-3.

## The state-averaged CASSCF Hessian

`df_sacasscf` is `casscf_hessian.tex` eq. (sa) for root I of a converged
state average, every term from a merged part, no integrals of its own:

    d²E_I/dx dy = L_xy + L_xλ λ^y + λ^xᵀ L_λy + λ^xᵀ L_λλ λ^y   (+ eq. rot at equal weights),
    A λ^x = −∂_x g_SA,   A z = −∂E_I/∂λ

```cpp
auto t = hessian::df_sacasscf(res, molecule, basis, classes, aux, h_core, b, l, solution, weights,
                              root, ncore, ncas);
// t->hessian: exactly symmetric; t->asymmetry: what symmetrizing discarded;
// t->exy (E_I,xy), t->z_terms (L_xy's z part, L_xλ), t->lxlambda_lambda (L_xλ λ^y, (x, y)),
// t->root_hessian, t->third, t->rotation (zero at unequal weights);
// t->z, t->cp (λ^x), t->residuals
```

The inputs are `df_casscf`'s with a `Solution` of two roots or more, plus
the `weights` (a Solution does not carry them; the averaged γ at them must
be the solution's to 1e-8, or it is InvalidConfig) and the `root`. In order:

1. **A**: H̃ at the averaged RDMs and weights, built as `df_casscf` builds it.
2. **z**: `response::zvector` on g_I, root I's orbital gradient (`response::fock`
   at its own RDMs). Only the equations' CI and Θ rows are scaled in
   zvector's convention, not the unknowns, so its (z_o, z_c, Θ) **is** the
   energy's z, `ThirdDerivative`'s and `df_sacasscf_z`'s as it comes
   (`response_zvector_rotations_gpu.cpp` compares it to the fixture's directly).
   Θ is packed only where `ThirdDerivative::size()` has it: whether the
   rotations are parameters is that operator's rule, not a second one here.
3. **λ^x**: `response::zvectors` (Schur) on `df_derivatives`' SA right-hand
   sides, the orbital rows packed, the CI rows over 2 w_r and, at unequal
   weights, the rotation rows t_sr = H̃^x_sr (`hamiltonian`'s off-diagonal),
   as `../response/zvector/README.md` has it. The same row scaling leaves λ^x
   the energy's; it is packed (pairs, d_r per root, Θ) into
   `ThirdDerivative`'s layout, Λ. The solve carries `df_casscf`'s probe
   column, so a singular A is refused ("A singular H̃" above).
4. **The check**: `apply_coupled` at λ^x (Θ included), plus the right-hand
   sides, the rotation rows as y_θ + t; above 1e-6 in any column is a
   Divergence, as in `df_casscf`.
5. **λ^xᵀ (z · ∂²g_SA) λ^y**: `ThirdDerivative::create(z)`, `apply` over Λ,
   Λᵀ Y.
6. **λ^xᵀ ∂²E_I λ^y**, root I's own parameter Hessian in the same exp(K),
   exp(A) parametrization: a second H̃ at weights **one-hot on I**
   (`response::Hessian` at root I's Fock), its **unreduced** product
   `apply_coupled`, which solves nothing. `response::Hessian` is the exact
   Hessian of E(C exp(κ)) off a stationary point too (its README's finite
   difference), which root I's orbitals are not; the CI block 2 w_r Q (H₀ −
   E_r) and the coupling 2 w_r Q H′ c_r are exact at eigenvectors; the
   rotations' rows 2 (w_r − w_s)(c_s · H′ c_r + (E_s − E_r) Θ_sr). So the
   product's CI rows times 2 w_r (2 on root I, 0 elsewhere) and its Θ rows
   times 2 (w_r − w_s) are ∂²E_I λ^y in the energy's coordinates; Λᵀ of it.
   A root s ≠ I enters E_I only through Θ_sI: its d_s does not, to second
   order (H c_I = E_I c_I is orthogonal to it).
7. **L_xy's z part and L_xλ**: `df_sacasscf_z` at z; L_xλ λ^y = L_xλᵀ Λ, one
   gemm, added with its transpose (λ^xᵀ L_λy).
8. **E_I,xy**: `df_casscf_exy` at root I's RDMs and `df_derivatives`' F^[x]
   at one-hot weights.
9. **Eq. rot** at equal weights only: `mcscf::root_rotation` over the SA
   `df_derivatives` (step 3's) and λ^x's orbital part, root I's slice; its
   factor 2 is in it. At unequal weights the rotations are in z and λ^x and
   the sum is complete without it.
10. The sum, symmetrized; `asymmetry` is what that discarded.

Steps 1, 3 and 4 are `df_sacasscf_cp` (above), and run first, before z;
the order is otherwise as listed.

**Degeneracies are refused.** Two roots under 1e-8 hartree apart are
Numerical before anything runs. At equal weights `root_rotation` refuses a
mixing rate max_x |θ^x_sr| above 10 per bohr; at unequal weights that rate
is λ^x's own Θ, and the same bound is put on it (Numerical, naming the
coordinate). Neither threshold is new: both are `root_rotation`'s.

### Memory

The parts run one after another, each freeing what it held: the two H̃
(each its own `response::Hessian` and CI scratch), `df_derivatives` at the
SA weights (held to the end for eq. rot) and at one-hot weights, the CP
solve's scratch over 3 n_atom columns, the third-derivative operator, then
`df_sacasscf_z`'s parts (its derivative integrals, nmo² naux doubles per
coordinate in as many coordinates as fit, and 3 n_atom nmo² for h1 and τ)
and `df_casscf_exy`'s. Held throughout: Λ, z and the
right-hand sides (3 n_atom (size + pairs + n_det nroots) doubles). Each part
refuses its own memory, Unsupported, before it allocates.

### Tests

`test/gtest/df_sacasscf_hessian_gpu.cpp` (suite `HessianDfSaCasscf`), every
root of the four state-averaged cases of `test/gtest/data/casscf_hessian/`
(two roots at (0.5, 0.5) and (0.7, 0.3), three at equal weights and at
(0.5, 0.3, 0.2)), the orbitals and roots **loaded from the fixture**, z and
λ^x solved here: E_I,xy against `exy` to 1e-8; L_xy's z part against
`exy_z`, L_xλ λ^y against `lxl_lambda`, root I's term against
`root_hessian_lambda`, the third-derivative action against `third`, eq. rot
against `rot` (equal weights; zero at unequal) and the total against
`hessian`, each to 1e-7; the total against `relaxed_fd` to the design's 1e-6
(the issue's ~1e-5); z's and λ^x's orbital parts against `z_orb` and
`lambda_orb` to 1e-7 of their largest element; every CP residual to 1e-8, the
asymmetry to 1e-8, the sum rule to 1e-7. At equal weights the total less
eq. rot must miss `relaxed_fd` by more than 1e-2. `Refusals`: a root out of
range, one weight, a zero weight, the solution's weights swapped, one root,
and a degenerate pair (Numerical). `ContinuousFamilyIsRefused` (slow-gated,
11 to 19 s on the gfx1200): HF CAS(10,7), one orbital of the 2π pair active, at
two roots and equal weights. It is this tree's own DF-CASSCF through the
registry, and it must be a Divergence from `df_sacasscf` that calls A
singular, with ρ under its bar when the message names one ("A singular H̃").

Measured on the RTX 3080 (CUDA), worst over the ten roots, of max |H| 0.05
to 0.48: E_I,xy ≤ 6.1e-12 from `exy`, L_xy's z part ≤ 1.1e-11 from `exy_z`,
L_xλ λ^y ≤ 1.5e-10, root I's term ≤ 2.0e-10, the third-derivative action
≤ 5.1e-10, eq. rot ≤ 7.5e-11 and the total ≤ 5.1e-10 from `hessian`; the
total ≤ 2.4e-8 from `relaxed_fd`, the fixture's own analytic-vs-difference
gap (its `hessian_vs_relaxed_fd`, 2.4e-8 at worst). z_o ≤ 2.9e-10 and λ^x's
orbital part ≤ 4.9e-9 of their largest elements; CP residuals ≤ 4.9e-9
after 17 to 42 Schur iterations; asymmetry ≤ 3.8e-14, sum rule ≤ 9.0e-11.
Without eq. rot the equal-weight totals miss `relaxed_fd` by 4.9e-2 (two
roots) and 5.2e-2, 0.48 and 0.48 (three), #609's figures. 2.1 to 2.7 s a
root (the water in 6-31G, 9 coordinates), 5 to 6.4 s a case. The gfx1200
was not run. The comparison against the semi-numerical CASSCF `[hessian]`
the issue also asks for waits on that step (#610).

## Files

| File | Role |
|------|------|
| `hessian.cppm` | Module interface: `hessian::RhfResponse`, `hessian::df_rhf_response`, `hessian::RhfTwoElectron`, `hessian::df_rhf_two_electron`, `hessian::CasscfTwoElectron`, `hessian::df_casscf_two_electron`, `hessian::CasscfExy`, `hessian::df_casscf_exy`, `hessian::CasscfTerms`, `hessian::df_casscf`, `hessian::df_casscf_cp`, `hessian::DegeneratePair`, `hessian::df_nevpt2_tangents`, `hessian::QdNevpt2Tangents`, `hessian::df_qd_nevpt2_tangents`, `hessian::df_adjoint_xy`, `hessian::SaCasscfZTerms`, `hessian::df_sacasscf_z`, `hessian::CasscfZTerms`, `hessian::df_casscf_z`, `hessian::ScNevpt2Terms`, `hessian::df_sc_nevpt2`, `hessian::PcNevpt2Terms`, `hessian::df_pc_nevpt2`, `hessian::SaCasscfTerms`, `hessian::df_sacasscf`, `hessian::SaCasscfCp`, `hessian::df_sacasscf_cp`, `hessian::RhfTerms`, `hessian::df_rhf`, `hessian::rhf_dipole_derivative`, `hessian::infrared_intensities` |
| `df_rhf.cpp` | D and −W, the one-electron and nuclear-repulsion terms, the two parts, and the symmetrized sum |
| `df_rhf_response.cpp` | The MO blocks, the G(D_S^x) builds, B, the CPHF and the terms |
| `atom_block.cppm` | Internal partition: `atom_function_range` and `occupied_atom_block`, the cross term's atom block both two-electron parts build, or a batch of them from one walk |
| `df_rhf_two_electron.cpp` | The fitted densities' second-derivative contractions and the cross terms' atom blocks |
| `df_casscf_two_electron.cpp` | The CASSCF Γ̃ and Ŵ contractions and the J, K and active cross terms; with a rotation their product-rule derivative, the atom blocks over [C_o \| C K_o] |
| `df_casscf_exy.cpp` | E_xy: the one-electron and nuclear-repulsion explicit terms, F, and the connection's three terms over the MO-basis B and the τ^x stack; with a rotation the z_o part, F_L and E^(2)'s commutator terms |
| `state_specific.cppm` | Internal partition: `state_specific_electrons`, a one-root solution's inputs checked and its active electrons by spin, which `df_casscf` and `df_nevpt2_tangents` share; `state_averaged_electrons` and `kSaCasscfMaxRotation`, the same for a state average and its rotations' bound, which `df_sacasscf` and `df_qd_nevpt2_tangents` share; `probed_zvectors` |
| `df_casscf.cpp` | The state-specific CASSCF Hessian: E_λx, E_xy, λ^x by `zvectors` beside the probe column, the unreduced residual check, Λᵀ E_λ· and the symmetrized sum; the CP step alone (`df_casscf_cp`) |
| `df_nevpt2_tangents.cpp` | The NEVPT2 parameter response and t_x: per chunk of derivative integrals, W^x to B's layout, the one-index transforms by κ_nr, f^(x) by `builders_tangent`, κ_oo under the degenerate-pair rule and its transform |
| `df_adjoint_xy.cpp` | Σ h̄ h^xy + Σ Λ B^xy at a general MO adjoint: X, G, ⟨sym(G) μ^x, μ^y⟩ and the per-chunk N^y and ⟨μ^x, H^y⟩ over `df_integral_derivatives`' W^x, both by M^x's atom blocks, V^y, the Gram matrices and the second-derivative integrals at the adjoint's AO forms |
| `df_sc_nevpt2.cpp` | The SC-NEVPT2 Hessian: E_cas, t in the coordinates, E_corr + Φ's adjoint and the gradient's multipliers, the HVP and its partnered pairs, the Z terms, the skeleton, the rotation cross terms and κ_oo rows as Gram matrices of nmo² stacks, the symmetrized sum |
| `df_pc_nevpt2.cpp` | The PC-NEVPT2 Hessian: `df_sc_nevpt2.cpp`'s assembly over PC's adjoint and product, with its three refusals: a class kept below 1e-10, a coordinate's Schur-lift step below 1e-3 bohr, a degenerate inactive pair |
| `df_qd_nevpt2_tangents.cpp` | The QD-NEVPT2 parameter response and t_x over a state average: per chunk, κ_nr and the CP's rotations, the active Hamiltonian's and f^SA's change by `qd_builders_tangent`, Eq. qd-theta at the equal pairs, κ_oo under the degenerate-pair rule |
| `df_sacasscf.cpp` | A state-averaged root's Hessian, and `df_sacasscf_cp` (its λ^x alone) and `state_averaged_electrons`: z and λ^x by the Z-vectors, the residual check, root I's parameter Hessian by the one-hot unreduced product, the third-derivative action, the z terms, E_I,xy, eq. rot at equal weights and the symmetrized sum |
| `df_sacasscf_z.cpp` | A state average's z terms at given multipliers, root I's or a QD-NEVPT2 Lagrangian's: L_xy's z part (E_xy along K(z_o), and at z_c's transition RDMs by the ± polarization) and L_xλ (the right-hand side at the caller's weights, if any, + I^x z); `df_casscf_z`, the same at one root and given Z without the right-hand side, plus Λᵀ T(·, ·, Z) Λ |
| `infrared.cpp` | The dipole derivative from U^x, and the IR intensities |
| `CMakeLists.txt` | Build configuration |
