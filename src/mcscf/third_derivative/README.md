# Third derivative

The third derivative of a state-averaged DF-CASSCF energy in its parameters,
contracted once with a fixed vector z, on the GPU: for each vector v of a
block, the covector a ↦ T(a, v, z). It is the λ^xᵀ (z · ∂²g_SA/∂λ∂λ) λ^y term
of a state-averaged nuclear Hessian (`.temp/hessian/casscf_hessian.tex` §5.1,
eq. (sa)'s fourth bullet), which only exists as an action, and the same
object an NEVPT2 Hessian's Lagrangian needs with its own multipliers.

## Module

`free_breakfast.mcscf.third_derivative`, target
`free_breakfast::mcscf::third_derivative`, re-exported by
`free_breakfast.mcscf`. Everything is in `namespace free_breakfast::mcscf`.

```cpp
class mcscf::ThirdDerivative {
public:
  static Result<ThirdDerivative> create(
      DeviceResources &res, const SphericalMatrixAO &h, const Buffers::device<double> &b,
      int32_t naux, const SphericalMatrixAO &c, int32_t ncore, int32_t ncas,
      const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta, const double *roots,
      std::span<const double> weights, const double *z);
  Status apply(DeviceResources &res, int32_t k, const double *v, double *y); // Y = T(., V, z)
  Status apply_integrals(DeviceResources &res, int32_t k, const double *h1, const double *w1,
                         double *y); // Y = d/de E''(., z) at the integrals moved along (h1, W)
  int32_t size() const noexcept;
};

Status mcscf::df_integral_derivatives(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux,
    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
    const SphericalMatrixAO &c, double *h1, double *w1); // I^x, every coordinate
```

`h` is the AO core Hamiltonian and `b` the fitted factor, naux × nao² as
`DensityFitted::b` lays it out (the same arguments as the density-fitted
`response::fock`); `c` the orbitals, core, active and virtual, nmo = nao.
`roots` is dim × nroots on the device, dim = alpha.size() · beta.size(),
root-major as `ci::Roots::vectors`: unit, orthogonal, and eigenvectors of the
active-space H over these integrals, as a converged SA-CASSCF's are. `alpha`
and `beta` must outlive the operator (its sigmas and RDM read their tables in
place); nothing else is kept. `z` is one column of `size()`.

## The parameters, and why the parametrization matters

E_SA(λ) = Σ_r w_r ⟨c_r(λ)|H(λ)|c_r(λ)⟩ with the orbitals C exp(K) and the
roots exp(A) c_r, both exponentials of their full generators at the
reference:

- K antisymmetric over the non-redundant pairs, κ_pq = K_pq for p > q, in
  [`../../response/README.md`](../../response/README.md)'s packed order;
- A = Σ_r (d_r c_rᵀ − c_r d_rᵀ) + Σ_{r,s} c_s Θ_sr c_rᵀ, d_r root r's change
  off every root and Θ antisymmetric, the rotations among the roots.

A column (vector or covector) is `size()` doubles: the packed κ, then d_r per
root (dim each, root-major), then, **at unequal weights only**, Θ_sr for
r < s in (r, s) lexicographic order (c_r gains Θ_sr c_s). At equal weights
E_SA is invariant to those rotations and they are not parameters; weights
equal for some pairs and not others are refused. A covector's parts are
∂/∂κ_pq, ∂/∂d_r (projected off every root) and ∂/∂Θ_sr, so **the pairing is
the plain dot product**, and a Hessian's third-derivative term is Λᵀ Y over
the columns `apply` writes for Λ. This is
`test/gtest/data/casscf_hessian/`'s layout (`_orb`, `_ci`, `_theta`) packed.

A third derivative, unlike a gradient or a Hessian at a stationary point,
depends on the parametrization at second order: another second-order
retraction (C (1 + K + K²/2 + …) truncated, a normalized c + d) changes
T(a, v, z) by E_λλ times its curvature. **Differencing a Hessian-vector product
does not give T either** on any pair with an orbital part: the tree's
products (`response::Hessian`, `RelaxedHessian`) re-centre the rotation at
the point they are applied at, so away from the reference they are the
Hessian of another parametrization, and they equal E_λλ only where g = 0.
Every term of a Hessian has to use the same parametrization; the fixture's
is this one, and so is everything this operator's callers assemble against it.

**The CI convention is the energy's.** A vector's d_r is root r's first-order
change, and a covector's CI rows are ∂E/∂d_r as written, the fixture's
convention: not `mcscf::zvector`'s unweighted, unscaled equations
(`../zvector/README.md`). A caller feeding `zvector`'s solutions in checks
them against that README's convention first; this operator was tested on the
fixture's z and λ^x only.

## What it computes

By Faà di Bruno over φ(λ) = (exp K, exp(A) c) and the energy E(U, c) as a
polynomial in a general (not orthogonal) U and in c,

    T(a, v, z) = E'''[φ_a, φ_v, φ_z] + E''[φ_av, φ_z] + E''[φ_az, φ_v] + E''[φ_a, φ_vz] + E'[φ_avz],

with φ_a = (K_a, A_a c), φ_av = (½(K_aK_v + K_vK_a), ½(A_aA_v + A_vA_a) c) and
φ_avz the symmetrized triple products over 6. Each E-derivative is a
covector in (U, c) space evaluated at its other arguments, then pulled back
through the map that is linear in a:

| Covector | Orbital part (nmo × nmo, occupied columns) | CI part, per root |
|---|---|---|
| G1 = E′ | 2F at the averaged densities | 2 w_r H c_r |
| G2[(B, Δ)] = E″[·, (B, Δ)] | 2 [F⁽¹⁾[B] + F at the transition densities of Δ] | 2 w_r (H′[B] c_r + H Δ_r) |
| G3[(B, Δ), (B′, Δ′)] | 2 [F⁽²⁾[B, B′] + F⁽¹⁾[B](Δ′) + F⁽¹⁾[B′](Δ) + F(Δ, Δ′)] | 2 w_r (H″[B, B′] c_r + H′[B] Δ′_r + H′[B′] Δ_r) |

F⁽ᵏ⁾ is the generalized Fock matrix with its last three integral indices
transformed by k of the generators (h B D for the one-electron part, nothing
at k = 2), H′ and H″ the active-space Hamiltonian at the integrals'
first and second derivatives along the generators (E_core, h_eff and
(tu|vw) all moving), and the transition densities are the symmetric
Σ_r w_r (T(a_r, b_r) + T(b_r, a_r)), core block weighted by Σ w_r 2 a_r·b_r.
G1 and G2 along z are built once in `create`; per column `apply` builds
G2 along v and N_v = G3[φ_v, φ_z] + G2[φ_vz] (one covector: both pair with
φ_a), then pulls all four back:

- orbital: N_v + G2[φ_z]·(−½)(K_v ·  + · K_v) + G2[φ_v]·(−½)(K_z · + · K_z) +
  G1·(⅙ Σ of the six orders of K_v, K_z around it), antisymmetrized and packed;
- CI: Σ M·A_a V over four pairs (M, V), with V = c, A_v c, A_z c and A_vz c,
  giving ∂/∂d_s = Σ [M (CᵀV)ᵀ − V (CᵀM)ᵀ] off every root and ∂/∂Θ =
  X − Xᵀ with X = Σ (CᵀM)(CᵀV)ᵀ.

**Over density fitting.** Every two-electron term is over the MO factor
L_Q = Cᵀ B_Q C (nmo² naux, built once): F_mp = Σ_{r,Q} L_Q,mr Y_Q,pr with
Y = Γ P, P a pair tensor over the occupied orbitals. A generator X enters
as L X (one strided-batched gemm over Q, its occupied columns) and
P(X) = Xᵀ L + L X; two generators as P(X, X′) = Xᵀ L X′ + X′ᵀ L X. Γ is the
full occupied 2-RDM (core included, 8-fold symmetric), so each Y is one gemm
(nocc² × nocc² by nocc² × naux) and each Fock term one gemm
(nmo × nocc naux by its transpose). The sigmas' integrals are
G = Σ P₁ P₂ᵀ over the occupied orbitals, read back and folded into E_core,
h_eff and (tu|vw) on the host.

**Per column**: two right transforms and three pair transforms of L, six Y
gemms, six Fock gemms, six transition RDM12s (three symmetric transition
densities by polarization), two sigma integral builds and six sigmas of the
nroots roots, and O(nmo³) and O(dim nroots²) of products for the pullbacks.
The work is one column at a time, each synchronizing: the RDMs and the
sigmas' integrals pass through the host.

## For a Hessian: the entry point

A state-averaged Hessian, root I, eq. (sa):

1. its z (`mcscf::zvector` at root I; at unequal weights with Θ), and the
   3N responses λ^x (`mcscf::zvectors`), each in this layout;
2. `ThirdDerivative::create(..., z_I)`, then `apply` over the block Λ;
3. the term is Λᵀ Y, symmetric in (x, y) to rounding.

An NEVPT2 Hessian creates it over the same reference with its own
multipliers as z, and pairs with its own responses: nothing here is
specific to E_SA's own multipliers.

## I^x z: the Hessian along z at the derivative integrals

L_xλ, the SA Hessian's mixed Lagrangian row, is root I's CP right-hand
side plus I^x z: z · ∂_x ∂_λ g_SA, the exact SA Hessian of E_x(λ) (the
energy at the derivative integrals, in this same exp(K)/exp(A)
parametrization) applied to z (`casscf_hessian.tex` eq. (sa)). The
integrals enter E_SA linearly, so I^x z is the derivative of the Hessian
action along z, E''[φ_a, φ_z] + E'[φ_az], along the integrals moved by I^x:

    I^x z (a) = d/de { G2[φ_z] · φ_a + G1 · φ_az },   φ_az = (½(K_aK_z + K_zK_a), ½(A_aA_z + A_zA_a) c)

with G1 and G2[φ_z] the covectors of the table above at the integrals
h + e h1, L + e W. Every quantity `create` built along z (L, P_0, P(K_z),
Y_0, Y_b, the averaged and z densities, A_z c) is reused; nothing new is
needed of the RDMs, because the densities do not depend on the integrals.

**The integrals' direction.** `apply_integrals` takes the direction as the
core Hamiltonian's h1 (nmo × nmo) and the MO factor's W (nmo² naux, one
symmetric nmo × nmo per Q), the two-electron integrals moving along
(pq|rs)' = Σ_Q (W_Q,pq L_Q,rs + L_Q,pq W_Q,rs). Not every two-electron
direction is one factor, but the fitted derivative is:
(pq|rs)^x = T^x N Tᵀ + T N T^xᵀ − T N M^x N Tᵀ = U^x Lᵀ + L U^xᵀ with
U^x = T^x L_M⁻ᵀ − ½ L L_M⁻¹ M^x L_M⁻ᵀ over MO pairs (L_M the metric's
Cholesky factor, N = M⁻¹). The connection τ^x = −½ S^x_MO moves L as
τL + Lτ and h as τh + hτ, so `df_integral_derivatives` writes, per coordinate,

    h1^x = Cᵀ h^x C + τ^x h_MO + h_MO τ^x,     W^x = U^x + τ^x L + L τ^x,

the fixture's I^x exactly. T^x is walked one aux atom at a time
(`three_center_derivative`; with E_A = Σ_{m on A, n} C_mp X_mn,P C_nq the MO
derivative is S + Sᵀ over the pair, S^(A,k) = −E_A + [P on A] Σ_A′ E_A′),
symmetrized by one batched gemm against the identity, then W = (S + Sᵀ) L_M⁻ᵀ
by a trsm; M^x from `metric_derivative` through two trsms per coordinate.
Its output is every coordinate at once, 3 n_atom nmo² naux doubles beside
3 n_atom nmo² for h1: about 1.4 MB for the water in 6-31G (9 coordinates),
but 3 GB per coordinate at 500 functions over 1500 fitting functions.

**Per column** of `apply_integrals`, with W_o W's occupied columns, W_oo its
occupied pairs and P_W(K_z) = K_zᵀ W + W K_z over the occupied pair:

- orbital: G1^x = 2 [W_o Y_0 + L_o Γ_0 W_oo + h1 D_0] and
  G2^x = 2 [W_o Y_b + L_o (Γ_0 P_W(K_z) + Γ_z W_oo) + (W K_z) Y_0 +
  (L K_z) Γ_0 W_oo + (h1 K_z) D_0 + h1 D_z], then G2^x − ½(G1^x K_z + K_z G1^x)
  antisymmetrized and packed;
- CI: c_1^x = 2 w H^x c and c_z^x = 2 w (H^x[K_z] c + H^x A_z c), H^x the
  active-space Hamiltonian over (h1, W_oo P_0ᵀ + P_0 W_ooᵀ) and H^x[K_z] its
  one-index transform by K_z (two `set_integrals`, three sigmas), pulled
  back as the pairs (c_z^x − ½ A_z c_1^x, c) and (½ c_1^x, A_z c).

The first pair's −½ A_z c_1^x is where the roots stop
being eigenvectors: at the derivative integrals c_1^x has components along
the other roots (⟨c_s|H^x|c_r⟩ ≠ 0 for r ≠ s), so A_z c_1^x carries the
inter-root term −Σ_s w H̃^x_rs z_s of the exact CI–CI block, and its Θ part
the rotations among the roots at unequal weights. Nothing assumes
stationarity: `response::Hessian` and `RelaxedHessian::apply_coupled` cannot
build this (one B, ppaa/papa and the Fock at C; stationarity), which is why it
lives here. Per column it is three Y gemms, ten Fock gemms, two Hamiltonians and three
sigmas of the nroots roots; the water's nine columns take 0.02 s on the 3080,
`df_integral_derivatives` 0.03 to 0.08 s.

## Not done

- **The core's Γ is explicit.** The occupied 2-RDM is nocc⁴ doubles and each
  Y costs nocc⁴ naux: right for the active spaces and molecules a CASSCF
  Hessian is run on, and the first thing to replace (by the core's J/K
  structure, as `response::fock` does) when nocc grows past ~50.
- **The columns are not batched.** A block's sigmas and RDMs run a column
  at a time, `apply_integrals`' too (its integral directions are built for
  every coordinate at once).
- **`df_integral_derivatives` holds every coordinate's W at once**,
  3 n_atom nmo² naux doubles: batching the coordinates is the first thing to
  do for a molecule past a few hundred functions.
- **Integral-direct.** Density-fitted only, as the CASSCF gradient is.
- **Partly equal weights**, where some rotations among the roots are
  parameters and some are not, are refused.

## Tests

`test/gtest/mcscf_third_derivative_gpu.cpp` (`McscfThirdDerivative`) against
`test/gtest/data/casscf_hessian/` (`tools/emit/casscf_hessian_reference.py`):
the stretched water's SA-CAS(4,4)/6-31G over cc-pVDZ-JKFIT at two roots
(0.5, 0.5) and (0.7, 0.3) and three equal and (0.5, 0.3, 0.2), every root's
z, from the fixture's orbitals and roots (rows into this basis's order and
normalisation; PySCF's two-root solutions are saddle points, and the fixture's
minima are what is compared). Λᵀ T Λ against `third`, a mixed fourth-order
difference of the exact gradient of E_SA in this parametrization, to 1e-7
absolute (5.2e-10 seen at worst, of max |third| 5e-3 to 5e-2); u₀ᵀ T(·, u₁, z)
of the fixture's two random vectors against `third_random` to 1e-7 absolute
(9e-9 seen, of 0.15 to 3.4); |T − Tᵀ| to 1e-12 of max |T| (2.4e-15 seen);
and the refusals. Both misses are the fixture's differences' own error: a
numpy prototype of these formulas, run against the same fixture before the
port, lands on the same values. One root's operator and its 9 columns take
under 0.1 s on the gfx1200.

`test/gtest/mcscf_integral_derivative_gpu.cpp` (`McscfIntegralDerivative`)
against the same fixture, the same four cases, every root's z: I^x z
(`df_integral_derivatives`, then `apply_integrals`) part by part against
`lxl_z` (the CI part's singlet component, the fixture's covectors being over
the singlet space), L_xλ = root I's CP right-hand side (`df_derivatives` at
one-hot weights) + I^x z, contracted with λ^y, against `lxl_lambda`, both to
1e-10 absolute (3.2e-13 seen at worst, of max |lxl_z| 6e-3 to 0.14 and
|lxl_lambda| 0.26 to 0.8); the translational sum rule over the atoms to 1e-9
(1.8e-15 seen); and the refusals.

## Files

| File | Role |
|------|------|
| `third_derivative.cppm` | Module interface: `ThirdDerivative` |
| `third_derivative.cpp` | The precomputation along z, the per-column covectors and their pullbacks, the occupied RDMs and the sigmas' integrals |
| `integral_derivative.cpp` | `apply_integrals` (I^x z) and `df_integral_derivatives` (h1^x and W^x, the connection included) |
| `CMakeLists.txt` | The module `free_breakfast.mcscf.third_derivative` |
