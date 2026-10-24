# Active space

The Hamiltonian of an active space of orbitals C, a core count and an active
count, built **integral-direct** from `coulomb` and `exchange`: no AO
two-electron integral is formed or stored. It is what a CASCI diagonalizes,
and what a CASSCF rebuilds once per macro-iteration; FCI is `ncore = 0`,
`ncas = nmo`.

## Module

`free_breakfast.integrals.active_space`, target
`free_breakfast::integrals::active_space`, re-exported by
`free_breakfast.integrals`. One function and its product:

```cpp
struct ActiveSpaceHamiltonian {
  double e_core;                  // core energy, nuclear repulsion included (hartree)
  int32_t ncas;                   // n_act
  Buffers::device<double> h_eff;  // n_act x n_act, column-major
  Buffers::device<double> eri;    // (tu|vw) at t + n_act (u + n_act (v + n_act w)), all n_act^4
};

Result<ActiveSpaceHamiltonian> active_space_hamiltonian(
    DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
    const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
    std::span<CoulombPairClass> j_classes, std::span<OrderedPairClass> k_classes,
    const SphericalMatrixAO &c, int32_t ncore, int32_t ncas,
    double tau_j = 1e-14, double tau_k = 1e-13);
```

Columns `[0, ncore)` of C are the core, `[ncore, ncore + ncas)` the active
space; a caller choosing other orbitals reorders C's columns first. The J and
K classes are the caller's, built once per basis, so a CASSCF loop builds none.
Classes through d only, as J and K.

## How it is built

With P = C_core C_coreᵀ (one spin's core density) and U C the orbitals over the
Cartesian AOs the kernels evaluate:

| Quantity | From |
|---|---|
| E_core = E_nuc + 2 tr(P h) + tr(P G), G = 2 J[P] − K[P] | one `coulomb` and one `exchange` of P (skipped for `ncore = 0`) and two dots |
| h_eff = C_actᵀ h C_act + (U C_act)ᵀ G (U C_act) | the bare part over the spherical AOs, G's over the Cartesian ones |
| (tu\|vw) = (U C_act)ᵀ J[D_vw] (U C_act) | one `coulomb` per pair v <= w, written into the (v, w) and (w, v) blocks |

**Each pair density is symmetrized**, D_vw = (C_v C_wᵀ + C_w C_vᵀ) / 2:
`coulomb` requires a symmetric density, and the antisymmetric part's J is zero,
so the symmetric part is the whole answer. n_act (n_act + 1) / 2 J builds in
all.

**The prune cap.** `coulomb_pair_classes` prunes primitive pairs assuming
|D_cd| <= 2 (`kPruneDensityCap`, `coulomb/pair_class/pair_class_launch.cu`),
and a pair density can exceed that. With m_v = max_µ |(U C)_µv|, every
|(D_vw)_cd| <= m_v m_w, so each J is built of s D_vw, s = min(1, 2 / (m_v m_w)),
screened at s `tau_j`, and scaled by 1 / s: J's absolute screen guarantee
holds for D_vw itself. The m_v are read back to the host once.

**`tau_j` defaults to 1e-14, not `coulomb`'s 1e-10.** At 1e-10, (tu|vw) and
(vw|tu), two different J builds, part by 1.1e-10 on H₂O/cc-pVDZ, past
1e-12 max|(tu|vw)|, and E_core moves 4.7e-10 from PySCF's. At 1e-13 N₂/cc-pVDZ
still parts them by up to 9e-13, past the bar on some runs (J adds atomically,
so its last bits vary run to run); at 1e-14, six runs out of six pass. `tau_k` is
`exchange`'s own default.

Scratch (U C, gemmwich's intermediate, a block, the spherical P) is carved
from `res.device_workspace()`, which neither `coulomb` nor `exchange` takes;
the three Cartesian matrices they read and write are their own type, allocated
per call.

## Not done

- **Batched J** (#200): one call over every pair density, once it lands.
- **Density fitting** (#177): (tu|vw) = Σ_P B_tu^P B_vw^P is far cheaper than
  n_act² J builds.

## Tests

`test/gtest/active_space_gpu.cpp`, against PySCF's CASCI through
`test/gtest/data/active_space/` (`tools/emit_active_integrals_reference.py`),
over H₂O in STO-3G and cc-pVDZ and N₂ in cc-pVDZ, each at `tau = 0` and the
defaults: E_core and the sign-invariant h_eff_tt, (tt|uu) and (tu|tu) to 1e-10;
the RHF energy given back by an active space spanning occupied and virtual
orbitals to 1e-10 of `Scf::energy`; the 8-fold symmetry to 1e-12 max; the cap,
on orbitals scaled past it, pruned against unpruned classes to 1e-12 max; and
the refusals. Its Scf runs with `tau_j = 0`: at the default, J's screen moves
N₂'s RHF energy by 2.6e-10 Eh.
