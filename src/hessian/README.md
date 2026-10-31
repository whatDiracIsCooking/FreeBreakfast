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
`infrared_intensities`. For DF-CASSCF, the first piece so far: the
**CASSCF two-electron explicit part**, `df_casscf_two_electron`
(`casscf_hessian.tex` eq. df2, below).

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
   and of L⁻¹ g^x = 2 Σ_i r^x_ii. R^x is never held whole: one **atom block**,
   R^x for the three coordinates x of atom A (3 naux n_occ² doubles), is
   built by one walk over the aux atoms. With X = `three_center_derivative`'s
   block of aux atom B (the derivative on m's centre), d(mn|P)/dA_k =
   −[m on A] X_mn − [n on A] X_nm + [P on A] (X_mn + X_nm), so with
   Y_ij = Σ_{m on A, n} C_mi C_nj X_mn,P and Z the same over every m,
   T^x_ij = −(Y_ij + Y_ji) + [P on A] (Z_ij + Z_ji) for the rows P on B:
   one strided-batched gemm X C_o over every (k, m), then per k two more
   (batched over j and over i, so both orders land without a transpose)
   accumulating straight into the block's rows. After the walk,
   (M^x c)_P = [P on A] Σ_Q M1[k][P][Q] c_Q + Σ_{Q on A} M1[k][Q][P] c_Q
   (`metric_derivative`'s M1, two gemms per k), one trsm by L over the whole
   block, and three gemvs for the Coulomb vectors.
4. **The contraction**, one pair of atom blocks (A, B ≤ A) at a time: the
   3 × 3 blocks −2 r_Aᵀ r_B (exchange) and t_Aᵀ t_B (Coulomb) by two gemms,
   downloaded and written to H_AB and H_BA. Block A is held across the inner
   loop and block B rebuilt for each pair, so n_atom (n_atom + 1) / 2 blocks
   are built, each one walk of `three_center_derivative` over every aux atom.
   That walk is where the time goes for a large molecule: the API gives every
   m for one aux atom, of which a block uses the rows m on A (and P on A).

### Memory

Held for the call: c (naux n_occ²). Then, in turn, the explicit part's Γ
(naux nao², the size of B), Y (naux n_occ nao), one naux² weight, d and D;
and the cross terms' M1 (3 naux²), one aux atom's X (3 nao² nblk, nblk the
largest atom's aux count) and two atom blocks (3 naux n_occ² each) with
their Coulomb vectors. The integral calls' scratch (the Cartesian Γ, X's
Cartesian block) is the workspace's. The call refuses (Unsupported, naming
the bytes) when c plus the larger of the two phases would not fit the
device's free memory; naux nao², 3 naux n_occ² and naux n_occ nao must fit
`int32_t`. At 500 AOs, 1500 fitting functions and 60 occupied orbitals the
atom blocks are 130 MB each and Γ is 3 GB.

### Tests

`test/gtest/df_rhf_hessian_two_electron_gpu.cpp` (suite
`HessianDfRhfTwoElectron`), against `test/gtest/data/rhf_hessian/`'s
`coulomb` and `exchange` (`tools/emit/rhf_hessian_reference.py`, PySCF's
`_partial_hess_ejk` ej and −ek, checked there against `partial_hess_elec`
and the sum rule), for the skew water and the skew formaldehyde in cc-pVDZ /
cc-pVDZ-JKFIT: each term to 1e-9 absolute, its asymmetry to 1e-12 and the
translational sum rule Σ_A H[A k][y] to 1e-9; and the refusals.

Measured on the RTX 3080 (CUDA): |H − PySCF| 6.2e-11 (Coulomb) and 1.8e-11
(exchange) on the water, 9.1e-11 and 2.2e-11 on the formaldehyde, of max |H|
8.0 / 1.0 and 17.6 / 1.7; asymmetry ≤ 1.8e-15, sum rule ≤ 2.2e-12; 1.2 s and
2.6 s a call. The same pieces with the cross terms built in NumPy from
PySCF's integrals (Cholesky solves) reach ej and −ek to 5e-11 and 2e-10, so
the JKFIT sets sit well inside the 1e-9 bar.

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

   The blocks are contracted one pair (A, B ≤ A) at a time, as in
   `df_rhf_two_electron`: n_atom (n_atom + 1) / 2 blocks built.

With n_core = n_occ (no active space) and γ = 0 the J and K pieces are
`df_rhf_two_electron`'s Coulomb and exchange cross terms.

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

## Files

| File | Role |
|------|------|
| `hessian.cppm` | Module interface: `hessian::RhfResponse`, `hessian::df_rhf_response`, `hessian::RhfTwoElectron`, `hessian::df_rhf_two_electron`, `hessian::CasscfTwoElectron`, `hessian::df_casscf_two_electron`, `hessian::RhfTerms`, `hessian::df_rhf`, `hessian::rhf_dipole_derivative`, `hessian::infrared_intensities` |
| `df_rhf.cpp` | D and −W, the one-electron and nuclear-repulsion terms, the two parts, and the symmetrized sum |
| `df_rhf_response.cpp` | The MO blocks, the G(D_S^x) builds, B, the CPHF and the terms |
| `atom_block.cppm` | Internal partition: `atom_function_range` and `occupied_atom_block`, the cross term's atom block both two-electron parts build |
| `df_rhf_two_electron.cpp` | The fitted densities' second-derivative contractions and the cross terms' atom blocks |
| `df_casscf_two_electron.cpp` | The CASSCF Γ̃ and Ŵ contractions and the J, K and active cross terms |
| `infrared.cpp` | The dipole derivative from U^x, and the IR intensities |
| `CMakeLists.txt` | Build configuration |
