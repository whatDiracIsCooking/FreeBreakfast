# Derivative

The per-coordinate derivatives of a DF-CASSCF energy at fixed parameters, on
the GPU, for every nuclear coordinate at once: the derivative generalized Fock
F^[x] and the CP-CASSCF right-hand sides E_λx, `casscf_hessian.tex` eqs.
(fock), (rhs-orb) and (rhs-ci). They are what the analytic Hessian's response
solve takes (E_λλ λ^x = −E_λx) and what its connection terms
2 Σ F^[x] τ^y read.

## Module

`free_breakfast.mcscf.derivative`, target `free_breakfast::mcscf::derivative`,
re-exported by `free_breakfast.mcscf`.

```cpp
Result<mcscf::Derivatives> mcscf::df_derivatives(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux,
    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
    const SphericalMatrixAO &c, int32_t ncore, int32_t ncas, int32_t nalpha, int32_t nbeta,
    const double *roots, std::span<const double> weights, const double *rotation = nullptr);
```

The inputs are explicit, not pipeline types, as `mcscf::df_gradient` takes
them: the molecule both bases were placed on, `primitive_pair_classes` of the
basis, the core Hamiltonian, B and L from `density_fit`, C (nao × nao, core |
active | virtual), and the roots as unit CI vectors on the device
(`ci::Sigma`'s `Layout::full`, root r at r n_det, n_det the
(nalpha, nbeta) strings' product) with their weights (each ≥ 0, summing to 1).

**The weights choose the energy.** E = Σ_r w_r ⟨c_r|H|c_r⟩, and everything
is its derivative:

| Call | Energy | Gives |
|---|---|---|
| one root, `{1}` | a state-specific CASSCF's | E_λx, the SS right-hand side |
| every root of an average at its weights | E_SA | ∂_x g_SA, the SA right-hand side at the averaged densities |
| the same roots, weights one-hot on I | E_I | ∂_x ∂_λ E_I, root I's own right-hand side (projected off every root) |

Nothing requires the parameters to be stationary; F^[x] at root I's RDMs is
the third row's `fock`.

**A rotation** K (an antisymmetric nao × nao on the device, MO indices C's
columns) adds `fock_rotated`, F^[x] at the K-rotated densities: a state
average's L_xy z_o part takes it at the SA weights and K = K(z_o) (the
fixture's `z_orb`), and hands it to `hessian::df_casscf_exy`'s rotation.
Without it (the default) nothing below "The rotated densities" runs, and
every other output is the same computation as before.

## Conventions

x = 3 A + k (atom A, axis k), always the **last** index. MO indices are C's
columns. The connection is first and the rotation second, C T(x) e^K
(eq. order), with τ^x = −½ S^x_MO (S^x_MO = Cᵀ S^x C). Every array is
column-major.

| Member | Where | Shape, element | Contents |
|---|---|---|---|
| `fock` | device | (nmo, nmo, n): (p, q, x) at p + nmo (q + nmo x) | F^[x]_pq = Σ_r h^x_pr D_rq + Σ (pr\|st)^x Γ_qrst, h^x and the fitted (pq\|rs)^x (the metric's derivative included), the unperturbed RDMs, **no** τ. Zero in the virtual columns |
| `rhs_orbital` | device | as `fock` | G^x_pq = ∂E_x/∂κ_pq = 2 (F̃^x_pq − F̃^x_qp) for the non-redundant pairs p > q (active–core, virtual–core, virtual–active), G^x_qp = −G^x_pq, **zero** on the core–core, active–active and virtual–virtual blocks; F̃^x = F^[x] + F[τ^x] |
| `rhs_ci` | device | (n_det, nroots, n): (I, r, x) at I + n_det (r + nroots x) | column r: 2 w_r Q H̃^x c_r, Q = 1 − Σ_s c_s c_sᵀ over every root given, H̃^x = H^[x] + H[τ^x] |
| `rhs_theta` | host | (nroots, nroots, n): (r, s, x) at r + nroots (s + nroots x) | γ^x_rs = ∂E_x/∂Θ_rs = 2 (w_s − w_r) H̃^x_rs, antisymmetric: the rotations among the roots' right-hand side, a parameter only at unequal weights |
| `hamiltonian` | host | as `rhs_theta` | H̃^x_rs = ⟨c_r\|H̃^x\|c_s⟩, symmetrized, its core energy included, V_nn^x not. Its diagonal is root r's fixed-parameter gradient less V_nn^x |
| `fock_rotated` | device, with a rotation only (`std::optional`) | as `fock` | F^[x] at the densities K D − D K and K on each index of Γ (one index virtual): d/dε F^[x] along C → C (1 + ε K) at fixed MO-basis RDMs, F's outer index held. **Every column** is filled |

These are `test/gtest/data/casscf_hessian/`'s `rhs_*` / `root_rhs_*` and
`fock_x` exactly (its README's "_orb / _ci / _theta" covector split): K
antisymmetric with C → C e^K and K_pq = κ_pq for p > q; Θ with dc_r gaining
Σ_s Θ_sr c_s. A covector pairs with a vector as ½ Σ G∘K + Σ_r ci_r · dc_r +
½ Σ γ∘Θ.

**The CI rows are the energy's derivative, not the Z-vector's convention.**
`rhs_ci` is 2 w_r Q H̃^x c_r, with the weight and the 2 in it.
[`../zvector/`](../zvector/README.md)'s CI rows are unweighted and unscaled
(PySCF's `gen_g_hop`), and its orbital vectors are over the packed pairs
(`response::pack_orbital_pairs` of `rhs_orbital`, which holds G_pq at p > q,
the order `response::Fock::gradient` packs): a caller feeding these to that
operator converts them there. The tex (section "Solving it") has why the
response term is then assembled as Λᵀ E_λ·, so the operator's scaling cancels
between the solve and its right-hand side.

## The terms

C's columns: core i, j; active t, u, v, w; any p, q, a. D_c = 2 C_c C_cᵀ,
D_a = C_a γ C_aᵀ, G[D] = J[D] − ½ K[D] over the fit, Γ 8-fold symmetrized
(the formula for F assumes it, and the energy sees only that part).

**F^[x]** is eq. (fock) at the derivative integrals:

    F^[x]_pi = 2 (F^I^x + F^A^x)_pi,      F^[x]_pt = Σ_u F^I^x_pu γ_ut + A^x_pt,
    F^I^x = Cᵀ (h^x + J^x[D_c] − ½ K^x[D_c]) C,   F^A^x = Cᵀ (J^x[D_a] − ½ K^x[D_a]) C,
    A^x_pt = Σ_uvw (pu|vw)^x Γ_tuvw

- h^x is `kinetic_derivative` + `potential_derivative`; J^x and K^x are two
  `df_jk_derivative` calls, over C_core and over the active density's natural
  factor X = C_a U diag(√(n/2)) (γ = U diag(n) Uᵀ by `diagonalize`, so
  D_a = 2 X Xᵀ). A natural occupation below −1e-8 is refused.
- A^x over the fit, (pu|vw)^x = Tˣ N T + T N Tˣ − T N Mˣ N T (T = (mn|P),
  N = M⁻¹), with c = N T = L⁻ᵀ B_MO the fitted pairs (`b_mo` over (active, all)
  then a trsm):

      A^x = Cᵀ V^x + Σ_uP c_P,pu R^x_P,tu,
      V^x_mt = Σ_nP d(mn|P)/dx Ỹ_nPt,   Ỹ_nPt = Σ_u C_nu Y_P,tu,   Y = c_aa Γ,
      R^x = (T^x_aa − M^x c_aa) Γ.

  V^x and T^x_aa come from one walk over the aux atoms
  (`three_center_derivative`'s block X[k][m][n][P], P on atom B), with
  d(mn|P)/dR_Xk = −[m on X] X_mn − [n on X] X_nm + [P on X] (X_mn + X_nm):
  per block and axis Q1 = Xᵀ Ỹ (a gemm over (P, n)), O[n] = X_nᵀ Ỹ_n (a
  strided-batched gemm over n), whose sums over all m and n go to x = 3B + k
  (the [P on X] term) and whose per-atom row and column slices, summed over
  the blocks, go to 3X + k with a minus sign after the walk; and
  H = X C_a (batched over m) contracted per orbital atom X with C_a's rows on
  X, E_X, so that T^x_aa's pair-symmetric part is −(E_X + E_Xᵀ) + [P on X]
  (E + Eᵀ). It is stored unsymmetrized as −2 E_X + 2 [P on X] E: every
  consumer contracts it with something symmetric in the pair (Γ, and the
  8-fold symmetrized (tu|vw)^x). M^x c_aa takes `metric_derivative`'s M1
  through its two centre terms, two gemms per atom and axis. No derivative
  tensor larger than one aux atom's block is held.

**F[τ^x] and H[τ^x]** are the same expressions at the integrals one-index
transformed by τ^x, which is d/dε at C → C (1 + ε τ^x) with the RDMs fixed
(the product rule on every C):

    F^I[τ] = τ F^I + F^I τ + Cᵀ G[dD_c] C,   dD_c = L_c (2C_c)ᵀ + (2C_c) L_cᵀ,   L_c = C τ_{·,core}
    F^A[τ] = τ F^A + F^A τ + Cᵀ G[dD_a] C,   dD_a = L_a (C_a γ)ᵀ + (C_a γ) L_aᵀ, L_a = C τ_{·,act}
    A[τ]_pt = (τ Q)_pt + Σ_avw (pa|vw) R1_avw,t + 2 Σ_uaw (pu|aw) R2_uaw,t,
    R1 = Σ_u τ_au Γ_tuvw,  R2 = Σ_v τ_av Γ_tuvw,  Q = F_act − F^I_act γ

with F, F^I, F^A `response::fock`'s and (pa|vw), (pu|aw) `casscf_integrals`'
ppaa and papa. The J and K of dD_c and dD_a are the block `df_coulomb` and
the block two-factor `df_exchange` over all 3 n_atom coordinates at once
(K[L Rᵀ] + its transpose). F̃^x is then assembled as F^[x] is, from
F^I~ = F^I^x + F^I[τ], F^A~ = F^A^x + F^A[τ] and A^x + A[τ].

**H̃^x** over the active space: E_core = Σ_i (h̃_ii + F^I~_ii) with
h̃ = h^x_MO + τ h_MO + h_MO τ; h_eff = F^I~'s active block; (tu|vw)~ =
(tu|vw)^x + Σ_a [τ_at (au|vw) + τ_au (ta|vw) + τ_av (tu|aw) + τ_aw (tu|va)]
(a strided-batched gemm over ppaa's (v, w) blocks), 8-fold symmetrized on the
host. One `ci::Sigma::set_integrals` and `apply` per coordinate gives H̃ c;
then Cᵀ (H̃ c) is `hamiltonian`, and H̃ c − C (Cᵀ H̃ c) scaled by 2 w_r is
`rhs_ci`.

## The rotated densities

The z_o part of a state-averaged Lagrangian is E_SA's derivative along the
orbital rotation C → C (1 + ε K) at fixed MO-basis RDMs
(`casscf_hessian.tex` §5.1). In the fixed basis that is every quantity at
the K-rotated densities K D − D K and K on each index of Γ, which have a
virtual index and so are not the closed-shell-core form `df_derivatives`
takes its RDMs in. In the rotating basis they are the unrotated densities,
and F^[x] moves only through its C's: the outer index p, which a caller
contracting with an MO-basis matrix X (τ^y here) sees as Σ F X along with
X's own rotation, so that the two together leave F^[x] Kᵀ; and every inner
C, by the product rule, as `mcscf::df_gradient` takes its z_o terms
(`../gradient/README.md`, "The orbital response"):

    F^[x]_rot = F^[x] Kᵀ + assemble(F^I_L, F^A_L, A_L),
    F^I_L = F^I^x K + Cᵀ G^x[dD_c] C,   dD_c = 2 (C̃_c C_cᵀ + C_c C̃_cᵀ),
    F^A_L = F^A^x K + Cᵀ G^x[dD_a] C,   dD_a = 2 (F̃ Fᵀ + F F̃ᵀ),

C̃ = C K, G^x = J^x − ½ K^x, D_a = 2 F Fᵀ over γ's natural factor
F = C_a U diag(√(n/2)) and F̃ = C̃_a U diag(√(n/2)), and `assemble` the
columns as F^[x]'s (core 2 (F^I + F^A), active F^I γ + A).

- **G^x of an indefinite density.** `df_jk_derivative` takes a closed-shell
  2 P Pᵀ. With P± = X ± s Y the two densities differ by exactly
  4 s (X Yᵀ + Y Xᵀ), so G^x[2 (X Yᵀ + Y Xᵀ)] = (G^x(P₊) − G^x(P₋)) / (2 s):
  an identity of the bilinear form, not a difference quotient, and
  s = |X| / |Y| (two `nrm2`s) keeps both densities the size of the
  unperturbed one, so nothing cancels catastrophically. Two calls for the
  core and two for the active density: four more walks of
  `df_jk_derivative` in all.
- **A_L**, the active 2-RDM term with its inner u, v, w rotated:

      A_L = Cᵀ V^x[Ỹ_L] + c_paᵀ R_L + c_pãᵀ R^x,
      Ỹ_L = C̃_a Y + C_a Y_L,   Y_L = c′_aa Γ,   c′_aa = c_ãa + c_aã,
      R_L = (T′_L − M^x c′_aa) Γ,

  c_pã the fitted pairs with ã = C̃'s active column (`b_mo` over [C | C̃_a],
  then the trsm). The walk over the aux atoms that makes V^x and T′^x for
  A^x is one lambda over a list of (Y, C′) terms for Ỹ and of (C_h, C_e)
  terms for T′; A^x's own walk is one term of each, and A_L's a second walk
  with two of each ((Y, C̃_a), (Y_L, C_a); (C_a, C̃_a), (C̃_a, C_a)). M^x c′_aa
  is the same per-atom pair of gemms as M^x c_aa.

`McscfDerivative`'s tests are those of the unrotated path; the rotated F^[x]
is tested through `hessian::df_casscf_exy`'s connection term against the
fixture's L_xy z part (`../../hessian/README.md`, "L_xy's z_o part").

## Memory

Every coordinate at once: about 15 stacks of 3 n_atom nao² doubles at the
peak (the AO derivative stacks, their MO blocks, the response densities' J
and K, F̃, the outputs), beside 3 n_atom (nao ncas³ + 3 naux ncas²), one aux
atom's derivative block (3 nao² nblk) and the derivative calls' own scratch.
That estimate past the device's free memory is Unsupported before anything is
allocated. Water in cc-pVDZ (24 functions, 9 coordinates) is under a
megabyte; 500 functions and 20 atoms would be about 1 GB of stacks. Nothing
splits the coordinates into batches yet. A rotation adds six more stacks
(the rotated J^x and K^x, their sum, F^I_L, F^A_L and `fock_rotated`), C K
and [C | C̃_a], the rotated pairs (naux ncas nao) and the second walk's
3 n_atom naux ncas², counted in the same estimate.

## Tests

`test/gtest/mcscf_derivative_gpu.cpp` (suite `McscfDerivative`), against
`test/gtest/data/casscf_hessian/` (`tools/emit/casscf_hessian_reference.py`),
the orbitals and roots **loaded from the fixture** (PySCF's own solutions of
three of the cases are saddle points; the fixture's are the minima): the SS
water in 6-31G and in cc-pVDZ (d shells, one active), and the state averages
of two roots at (0.5, 0.5) and (0.7, 0.3) and three at (0.5, 0.3, 0.2), each
case's averaged `rhs_*` and, per root, its `fock_x` and its own `root_rhs_*`
(one-hot weights), to 1e-9 absolute; every output's translational sum rule
to 1e-9; for one root ⟨c|H̃^x|c⟩ + V_nn^x against the fixture's gradient to
1e-8; and the refusals.

Measured on the gfx1200 (HIP): F^[x] ≤ 8.6e-14 and `rhs_orbital` ≤ 2.2e-13
of max |·| ≈ 2; `rhs_ci` ≤ 4e-14 in 6-31G and 2.6e-10 in cc-pVDZ (of 0.16);
`rhs_theta` ≤ 1.1e-14; the gradient ≤ 7e-14; 0.3–0.9 s a case, the setup
included.

Three things about the fixture the test handles, so a later reader need not
rediscover them:

- **AO order.** The fixture's C rows are in PySCF's order, each atom's
  shells sorted by l; this tree keeps the basis file's order, so 6-31G's split
  SP shells put 2p before 3s here and after it there. The test maps the rows
  (and divides by the norms N_i, as every PySCF-orbital test here does).
- **The roots are not quite singlets.** h2o_631g's root carries a 6e-7
  quintet component, and the fixture's CI covectors are over the singlet
  space, so the test compares the singlet parts of `rhs_ci` (S² by
  `test::host_spin_square`). The raw difference is that component times
  |H̃^x|, 1.1e-7; the singlet parts agree to 1.6e-13.
- `fock_x` is stored **x first**, (n, nmo, nmo), unlike the per-root
  covectors.

## Files

| File | Role |
|------|------|
| `derivative.cppm` | Module interface: `Derivatives`, `df_derivatives` |
| `derivative.cpp` | Its implementation: host code over BLAS and the calls above, no kernels |
| `CMakeLists.txt` | The `free_breakfast.mcscf.derivative` target |
