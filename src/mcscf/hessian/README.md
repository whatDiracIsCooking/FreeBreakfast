# Hessian

The exact orbital-orbital Hessian of a (state-averaged) CASSCF wavefunction at
fixed CI, over the packed non-redundant pairs, on the GPU: its product with a
block of vectors, as a `LinearOperator`, and its diagonal for the
preconditioner. What the optimizer's augmented Hessian (#251) solves over.

## Module

`free_breakfast.mcscf.hessian`, target `free_breakfast::mcscf::hessian`,
re-exported by `free_breakfast.mcscf`.

```cpp
class OrbitalHessian {
public:
  static Result<OrbitalHessian> create(
      DeviceResources &res, const BasisSet &basis, const SphericalMatrixAO &h,
      const std::shared_ptr<const Buffers::device<double>> &u,
      std::span<CoulombPairClass> j_classes, std::span<OrderedPairClass> k_classes,
      const SphericalMatrixAO &c, int32_t ncore, int32_t ncas,
      const Buffers::device<double> &ppaa, const Buffers::device<double> &papa,
      const CasscfFock &fock, double tau_j = 1e-14, double tau_k = 1e-14);

  Status apply(DeviceResources &res, int32_t k, const double *x, double *y); // Y = H X
  int32_t size() const noexcept;                                             // the pair count
  const Buffers::device<double> &diagonal() const noexcept;                  // diag(H)
};
```

The arguments through `k_classes` are `casscf_fock`'s; `fock` is its result
over the same C, ncore and ncas, and `ppaa` and `papa` are `casscf_integrals`'
over the same C (both empty for ncas = 0, the RHF orbital Hessian). Its γ and
Γ are the weighted ones, so state averaging needs nothing here. H is the
second derivative of E(C exp(κ(x))) in the gradient's coordinate:
E = E₀ + g·x + ½ xᵀHx + …, so it is **twice** PySCF's `gen_g_hop` `h_op`
and `h_diag`, as `CasscfFock::gradient` is twice its g. Everything is in
[`../README.md`](../README.md)'s packed order. `apply` models `recipes`'
`LinearOperator` (`static_assert`ed), so `davidson_solve` takes it as is.

create reads its arguments and keeps none of them but the two spans of
classes, which must outlive the operator: an iteration may rebuild the
orbitals and integrals while the Hessian of the previous ones is still in use.

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
(pi\|pi) over every p and core i, the diagonals of Cᵀ J[C_i C_iᵀ] C and
Cᵀ K[C_i C_iᵀ] C, one J and one K per core orbital in create.

**The prune cap.** `coulomb_pair_classes` prunes for |D_cd| ≤ 2. C_i C_iᵀ is
built as s C_i C_iᵀ, s = min(1, 2 / m_i²), m_i = max_µ |(U C)_µi|, as
`casscf_integrals` holds it; a transition density's max |D| is read back by
iamax per build (one synchronize each) and scaled the same way. J and K are
screened at s τ and scaled by 1 / s.

**Screening breaks symmetry at the 1e-13 level.** J and K skip what falls
under τ whatever the density, so the screened H is not exactly linear: at the
default τ = 1e-14, xᵀHy and yᵀHx part by up to 5e-13 over random unit x, y on
the cc-pVDZ cases, which is past 1e-12 of |xᵀHy| when that is small. With J
and K unscreened they agree to 5e-14 of it. A Davidson over H does not notice
either.

Scratch per product (κ, X₂, the gemms' intermediates, two n_cart² densities
and an n_cart × nmo product) is carved from `res.device_workspace()`, which
neither `coulomb` nor `exchange` takes. The members are what lives across
products: U C, h, F + Fᵀ, V^c, V^c + V^a, D's occupied block, γ, H_act, the
diagonal, and the three Cartesian matrices J and K read and write.

## Not done

- **Frozen orbitals**: every core orbital rotates, as in `fock`.
- **Batched J and K** (#200): a block's 2k J and 2k K are built one at a time.
- **Density fitting** (#177): would replace both the per-product J and K and
  the per-core-orbital builds of the diagonal.

## Tests

`test/gtest/mcscf_hessian_gpu.cpp`, against `test/gtest/data/casscf/casscf.json`
(PySCF's `mc1step.gen_g_hop`, doubled), at each case's recorded start and set
up as `mcscf_fock_gpu.cpp` does: H₂O cc-pVDZ CAS(4,4) [2, 3, 5, 6], N₂ cc-pVDZ
CAS(6,6), H₂CO cc-pVDZ CAS(4,3) at [0.5, 0.5] and [0.7, 0.3], and H₂O cc-pVDZ
by AVAS.

- the diagonal summed over each pair of orbital sets against PySCF's, to 1e-7
  (seen: 2e-10);
- gᵀHg and \|Hg\| against PySCF's, to 1e-7 relative (seen: 1.3e-9);
- H x against a central finite difference of `casscf_fock`'s gradient along
  two random unit x, the RDMs fixed, to 1e-6 of \|Hx\| (seen: 9e-12). The
  gradient at C exp(tκ) is the derivative after the rotation, so the
  difference is taken of G′ + t [κ, G′] / 2 packed, G′ = 2 (F′ − F′ᵀ) over
  every pair: d/dY E(C exp(tκ + Y)) by exp(K + Y) = exp(K) exp(Y − [K, Y]/2 +
  …). Without that term the slope is H x + [κ, G]/2, which is not small off a
  stationary point;
- symmetry, xᵀHy = yᵀHx over the gradient and the random x, to 1e-12 of
  |xᵀHy|, with J and K unscreened (above);
- on H₂O STO-3G, H e_i for every unit vector, in one block, against the
  diagonal to 1e-7 (seen: 3e-14), symmetric to 1e-12 of max |H|, and the
  finite difference, both at a two-root frontier CAS(2,2) weighted
  [0.6, 0.4] (every kind of pair) and at ncas = 0 at and off the Scf;
- the refusals.
