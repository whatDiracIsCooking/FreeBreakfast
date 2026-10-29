# Hessian

The exact orbital-orbital Hessian of a (state-averaged) CASSCF wavefunction at
fixed CI, over the packed non-redundant pairs, on the GPU: its product with a
block of vectors, as a `LinearOperator`, and its diagonal for the
preconditioner. What `mcscf/solve`'s augmented Hessian (#251) solves over.

## Module

`free_breakfast.response.hessian`, target `free_breakfast::response::hessian`,
re-exported by `free_breakfast.response`.

```cpp
class response::Hessian {
public:
  // Integral-direct.
  static Result<response::Hessian> create(
      DeviceResources &res, const BasisSet &basis, const SphericalMatrixAO &h,
      const std::shared_ptr<const Buffers::device<double>> &u,
      std::span<CoulombPairClass> j_classes, std::span<OrderedPairClass> k_classes,
      const SphericalMatrixAO &c, int32_t ncore, int32_t ncas,
      const Buffers::device<double> &ppaa, const Buffers::device<double> &papa,
      const response::Fock &fock, double tau_j = 1e-14, double tau_k = 1e-14);
  // Density-fitted.
  static Result<response::Hessian> create(
      DeviceResources &res, const SphericalMatrixAO &h, const Buffers::device<double> &b,
      int32_t naux, const SphericalMatrixAO &c, int32_t ncore, int32_t ncas,
      const Buffers::device<double> &ppaa, const Buffers::device<double> &papa,
      const response::Fock &fock);

  Status apply(DeviceResources &res, int32_t k, const double *x, double *y); // Y = H X
  int32_t size() const noexcept;                                             // the pair count
  const Buffers::device<double> &diagonal() const noexcept;                  // diag(H)
};
```

The integral-direct overload's arguments through `k_classes` are
`response::fock`'s; the density-fitted one takes B in place of the basis, U
and the classes, as the density-fitted `response::fock` does: `b` is
`DensityFitted::b`, naux × nmo² over the spherical AOs. In both, `fock`
is `response::fock`'s result over the same C, ncore and ncas, and `ppaa` and
`papa` are `casscf_integrals`' over the same C (both empty for ncas = 0, the
RHF orbital Hessian), each fitted or not as the caller chose: H is wholly
density-fitted only when they are. Its γ and
Γ are the weighted ones, so state averaging needs nothing here. H is the
second derivative of E(C exp(κ(x))) in the gradient's coordinate:
E = E₀ + g·x + ½ xᵀHx + …, so it is **twice** PySCF's `gen_g_hop` `h_op`
and `h_diag`, as `response::Fock::gradient` is twice its g. Everything is in
[`../README.md`](../README.md)'s packed order. `apply` models `recipes`'
`LinearOperator` (`static_assert`ed), so `davidson_solve` takes it as is.

create reads its arguments and keeps none of them but the two spans of
classes, or B's storage (by pointer: the buffer may be moved, not freed),
which must outlive the operator: an iteration may rebuild the orbitals and
integrals while the Hessian of the previous ones is still in use.

**One operator, two kinds of J and K.** Everything but the J and K of a
density is shared: the MO intermediates, H_act, the product's gemms and the
diagonal's assembly. Which J and K the operator builds is a private member
holding either the classes, their screens and three Cartesian matrices, or
B's storage and two nmo × nmo matrices; the create called picks it.

## What it computes

`gen_g_hop`'s h_op, term by term, with D the 1-RDM (2 on the core diagonal, γ
on the active block), h = Cᵀ h C, V^c = F^I − h, V^a = F^A and κ = κ(x) the
antisymmetric unpack. X₂ is built, and H x = 2 (X₂ − X₂ᵀ) packed:

| Term of X₂ | Rows, columns | From |
|---|---|---|
| h κ D | all, occupied | two gemms (D's occupied block only) |
| −½ (F + Fᵀ) κ | all | one gemm |
| 2 κ[core, rot] (V^c + V^a)[rot, :] | core, all | one gemm |
| γ (κ[act, :] V^c + C_actᵀ (2 J − K)[D₃] C) | active, all | gemms over the J and K below |
| H_act vec(κ[:, act]) | all, active | one gemv |
| C_coreᵀ (2 J − K)[2 D₃ + D₄] C_rot | core, active and virtual | gemms over the J and K below |

rot is [ncore, nmo). H_act[(p, u), (r, v)] = Σ_wx (pr\|wx) Γ_wxuv +
(pw\|rx) (Γ_uwxv + Γ_wuxv) is gen_g_hop's `hdm2`, an (nmo ncas)² matrix
built once in create: for each v, one strided-batched gemm over r of ppaa's
(or papa's) rows (p, r) against Γ's columns, so the product's (pu|rv)-type
contraction is a single gemv. D₃ = C_core κ[core, rot] C_rotᵀ and
D₄ = C_act γ κ[act, :] Cᵀ, each plus its transpose, are `update_jk_in_ah`'s
transition densities.

**The two J and two K per product cannot be avoided.** The core–virtual block
of H holds 4 (ai\|bj) − (ab\|ij) − (aj\|bi) over core i, j and virtual a, b,
and the core–active one integrals with two core and two non-active indices:
none of them is in ppaa or papa, which carry two active indices, and no MO
matrix contracts them. So H x is not "gemms plus a geam" for ncore > 0: like
PySCF, each product builds J and K of D₃ and of 2 D₃ + D₄ through `coulomb`
and `exchange`, integral-direct, over U C. ncore = 0 builds none.

**The diagonal** is gen_g_hop's h_diag, assembled on the host in O(pairs)
from diagonals read back once: of h, F, V^c and V^c + V^a; of H_act; ppaa's
and papa's core rows ((qu\|qv) and (qq\|uv) for core q); and (pp\|ii),
(pi\|pi) over every p and core i. Integral-direct, those are the diagonals of
Cᵀ J[C_i C_iᵀ] C and Cᵀ K[C_i C_iᵀ] C, one J and one K per core orbital in
create; density-fitted, see below.

**The prune cap.** `coulomb_pair_classes` prunes for |D_cd| ≤ 2. C_i C_iᵀ is
built as s C_i C_iᵀ, s = min(1, 2 / m_i²), m_i = max_µ |(U C)_µi|, as
`casscf_integrals` holds it; a transition density's max |D| is read back by
iamax per build (one synchronize each) and scaled the same way. J and K are
screened at s τ and scaled by 1 / s.

**Screening makes H symmetric only to 1e-13 of max |H|, and K's screen is
why.** A screened J or K skips what falls under τ for *this* density, so the
screened H x is not exactly linear in x and xᵀHy ≠ yᵀHx. Measured on the five
cc-pVDZ cases below over the gradient and eight random unit x, building the
operator once per (τ_j, τ_k), as max |xᵀHy − yᵀHx| / (max |diag H| ‖x‖ ‖y‖):

| Source | H₂O | N₂ | H₂CO SA2 | H₂CO SA2 weighted | H₂O AVAS |
|---|---|---|---|---|---|
| neither (τ_j = τ_k = 0) | 1.6e-16 | 4.3e-16 | 1.3e-16 | 1.8e-16 | 2.3e-16 |
| J only (τ_j = 1e-14) | 1.5e-16 | 3.9e-16 | 3.7e-16 | 2.8e-16 | 2.6e-15 |
| K only (τ_k = 1e-14) | 1.0e-13 | 1.8e-14 | 4.5e-15 | 7.5e-15 | 7.3e-14 |
| both, the defaults | 1.0e-13 | 1.8e-14 | 4.6e-15 | 7.5e-15 | 7.3e-14 |
| both at 1e-12 | 2.7e-11 | 4.4e-12 | 1.0e-12 | 1.0e-12 | 2.5e-11 |
| both at 1e-10 | 5.2e-9 | 4.4e-10 | 2.0e-10 | 3.6e-10 | 2.1e-9 |

So it is **K's** screen, which reads the density through each block's factor
and each group's reach as well as through max |D|, and it grows with τ_k: the
coefficient of τ_k max |diag H| ‖x‖ ‖y‖ is 0.5 to 10 at 1e-14 and 2 to 50 at
1e-10. J's at 1e-14 is near rounding (2.6e-15 at worst); at τ_j = 1e-12 alone it
reaches 6e-14, under a tenth of K's at the same τ. The **prune cap**
is not a source: J and K screen s D at s τ, so the scale cancels in every
skip test, and building with s = 1 throughout (outside the classes' |D| ≤ 2,
for densities up to 2.7 here) gives the same 1e-13 level. The **per-core-orbital builds in
create** are not one either: they feed only the diagonal, which `apply` never
reads. In absolute terms the defaults part xᵀHy and yᵀHx by at most 7.8e-13
(H₂O AVAS), which is past 1e-12 of the smallest |xᵀHy| (9e-4); that is why a
bar relative to |xᵀHy| failed.

Making the screened product exactly linear would mean screening K by a bound
that does not depend on the density, a change to `exchange` (and a looser or
costlier screen) rather than to this operator, so the screening is kept and
**the bar is absolute: |xᵀHy − yᵀHx| ≤ 100 τ max |H| ‖x‖ ‖y‖**, 1e-12 at the
default τ, as the STO-3G check already had it. Absolute, because every skipped
term of a K is under τ whatever the vectors (`exchange`'s guarantee), so the
error does not shrink with |xᵀHy|; of max |H|, because that is the scale the
skipped integrals are contracted into H at; ‖x‖ ‖y‖, because the products are
of x and y. 100 is the measured coefficient's worst (10 at 1e-14, 50 at 1e-10)
with room, and the tests take max |diag H|, which is at most max |H|.

**It does not matter to an augmented-Hessian Davidson** (#251), which assumes
a symmetric operator. `davidson_solve` over [[0, gᵀ], [g, H]] (preconditioned
by 1 / (θ − diag), from e₀, to 1e-10) took the same iterations at τ = 0 and at
the defaults (H₂O cc-pVDZ 15, N₂ 12), with the lowest eigenvalue the same to
2e-15 and the step x / x₀ to 2.7e-12 relative: the asymmetry (3e-13 and 4e-13
absolute there) is far below the 1e-10 residual it converges to.

Scratch per product (κ, X₂, the gemms' intermediates, two n_cart² densities,
an n_cart × nmo product, and the n_cart × nocc factor R that only the
density-fitted K reads) is carved from `res.device_workspace()`, which
neither `coulomb` nor `exchange` takes. The members are what lives across
products: U C, h, F + Fᵀ, V^c, V^c + V^a, D's occupied block, γ, H_act, the
diagonal, and the three Cartesian matrices J and K read and write.

## Density fitting

PySCF's `approx_hessian`: the same H, with every J and K from B, the
fitted factor with (mn\|ls) ≈ Σ_Q B_Q,mn B_Q,ls. Spherical throughout: no U C
(the operator's "U C" is C), no screen, and **no prune cap**: `df_coulomb`
and `df_exchange` have no |D| bound to hold, so the cap's `norm<inf>`
read-back of each transition density (one synchronize each, two per product)
is not done, and a product synchronizes nothing.

**The product.** J of D₃ and of D′ = 2 D₃ + D₄ is `df_coulomb` of the density
as the integral-direct path forms it. K is #380's two-factor `df_exchange`,
from factors and never from D: D₃ = L R₃ᵀ + R₃ Lᵀ with L = C_core and
R₃ = C_rot κ[core, rot]ᵀ, and D₄ = C_act R₄ᵀ + R₄ C_actᵀ with
R₄ = C (γ κ[act, :])ᵀ, so D′ = L′ Rᵀ + R L′ᵀ with L′ = [C_core C_act], C's
first nocc columns as they lie, and R = [2 R₃, R₄]. Each K is then **one**
call, K[L Rᵀ], and K = K[L Rᵀ] + K[L Rᵀ]ᵀ by a `geam`: rank ncore for D₃ and
nocc for D′, not two calls for D′'s two terms. R₃ and R₄ are two small gemms
over κ the product already holds.

**The diagonal**'s (pp\|ii) and (pi\|pi) over every p and core i, with no J
or K per core orbital: L_Q,ip = (Q\|ip), B transformed over C_core and C,
naux · ncore · nmo doubles. B is half-transformed ncore columns of C at a
time, X_Q,pm = Σ_n B_Q,nm C_n,p (one batched gemm over m); the first block is
C_core's, and one gemm over C completes it to L. Each block also gives
(Q\|pp) for its p by a batched gemv, so (Q\|pp) costs no more scratch than L
does. Then (pp\|ii) = Σ_Q (Q\|pp) (Q\|ii) is one gemm against L's (i, i)
columns, and (pi\|pi) = Σ_Q L_Q,ip² a batch of naux-long dots per core i.

**Scratch.** create carves, past the integral-direct path's, the half
transform and L (naux · ncore · nmo each), (Q\|pp) (naux · nmo) and the two
nmo × ncore results from `res.device_workspace()`, calling no DF routine
while they live. A product's scratch is the integral-direct one's, but
`df_coulomb` and `df_exchange` take the
workspace (γ, and X and Y of naux · nocc · nmo each), so the density-fitted
`apply` allocates its scratch per call instead of carving it. The members
are the integral-direct ones but the three Cartesian matrices, which are two
nmo × nmo J and K here.

**Symmetry.** With no screen, the density-fitted H is symmetric to rounding,
as the integral-direct one is at τ = 0 (the table above): over the five
cc-pVDZ cases, max |xᵀHy − yᵀHx| / (max |diag H| ‖x‖ ‖y‖) is at most 9e-17.
The tests hold it to the integral-direct bar, 100 τ at the default τ, which
it passes by over four orders of magnitude.

**Accuracy** is the fit's, and depends on the set. Against the
integral-direct H, with ppaa, papa and the Fock fitted from the same B, an
element of H x differs by at most 1.2e-4 ‖x‖ and of the diagonal by 4.7e-4
over aug-cc-pVDZ-acCD-6-6, where max |diag H| is 73 to 95. cc-pVDZ-JKFIT,
built for occupied pairs, gives 2.8e-3 and 6.2e-3 (N₂): H, like ppaa and
papa, reads virtual-orbital pairs.

## Not done

- **Frozen orbitals**: every core orbital rotates, as in `fock`.
- **Batched J and K**: a block's 2k J and 2k K are built one at a time.
  Batching the integral-direct builds is an open kernel question (#200 was
  repurposed to the fitted CIS sigma, whose block costs one pass).

## Tests

`test/gtest/response_hessian_gpu.cpp`, against `test/gtest/data/casscf/casscf.json`
(PySCF's `mc1step.gen_g_hop`, doubled), at each case's recorded start and set
up as `response_fock_gpu.cpp` does: H₂O cc-pVDZ CAS(4,4) [2, 3, 5, 6], N₂ cc-pVDZ
CAS(6,6), H₂CO cc-pVDZ CAS(4,3) at [0.5, 0.5] and [0.7, 0.3], and H₂O cc-pVDZ
by AVAS.

- the diagonal summed over each pair of orbital sets against PySCF's, to 1e-7
  (seen: 2e-10);
- gᵀHg and \|Hg\| against PySCF's, to 1e-7 relative (seen: 1.3e-9);
- H x against a central finite difference of `response::fock`'s gradient along
  two random unit x, the RDMs fixed, to 1e-6 of \|Hx\| (seen: 9e-12). The
  gradient at C exp(tκ) is the derivative after the rotation, so the
  difference is taken of G′ + t [κ, G′] / 2 packed, G′ = 2 (F′ − F′ᵀ) over
  every pair: d/dY E(C exp(tκ + Y)) by exp(K + Y) = exp(K) exp(Y − [K, Y]/2 +
  …). Without that term the slope is H x + [κ, G]/2, which is not small off a
  stationary point;
- symmetry, xᵀHy = yᵀHx over the gradient and the random x, of the
  operator at the default τ, to 1e-12 (100 τ) of max |diag H| ‖x‖ ‖y‖ (above;
  seen: 1e-13);
- on H₂O STO-3G, H e_i for every unit vector, in one block, against the
  diagonal to 1e-7 (seen: 3e-14), symmetric to 1e-12 of max |H|, and the
  finite difference, both at a two-root frontier CAS(2,2) weighted
  [0.6, 0.4] (every kind of pair) and at ncas = 0 at and off the SCF;
- the refusals;
- the density-fitted H over aug-cc-pVDZ-acCD-6-6, at each case's CASCI orbitals and
  RDMs, over ppaa, papa and a Fock (the density-fitted `fock`) contracted
  from the same B on the host: H x along the gradient and the random x, and
  the whole diagonal, against a host product from those MO integrals to 1e-12
  of max |diag H| (seen: 6e-15). The host product is H x analytically, the t
  derivative of the finite difference's corrected gradient: (pq\|rs) and h
  rotated to first order in κ, through the generalized Fock, which is linear
  in them at fixed RDMs, plus [κ, G] / 2. Then symmetry at the integral-direct
  bar (seen: 9e-17), and against the integral-direct H within the fit's error:
  3e-4 ‖x‖ per element of H x and 1.5e-3 per element of the diagonal (seen:
  1.2e-4 and 4.7e-4);
- on H₂O STO-3G density-fitted over cc-pVDZ-JKFIT, every H e_i against the host product and
  the diagonal, and symmetric, at the CAS(2,2) above (ncore 4 of 7 orbitals,
  so the diagonal's last block of C is narrower than ncore) and at ncas = 0,
  at and off the SCF;
- the density-fitted overload's refusals.
