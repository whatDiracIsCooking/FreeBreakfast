# Active space

The Hamiltonian of an active space of orbitals C, a core count and an active
count, built **integral-direct** from `coulomb` and `exchange`: no AO
two-electron integral is formed or stored. It is what a CASCI diagonalizes,
and what a CASSCF rebuilds once per macro-iteration; FCI is `ncore = 0`,
`ncas = nmo`. Beside it, the (pq|uv) and (pu|qv) over every orbital that a
CASSCF's generalized Fock and orbital Hessian contract.

## Module

`free_breakfast.integrals.active_space`, target
`free_breakfast::integrals::active_space`, re-exported by
`free_breakfast.integrals`. Two functions and their products:

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

struct CasscfIntegrals {
  ActiveSpaceHamiltonian hamiltonian; // as active_space_hamiltonian gives it
  Buffers::device<double> ppaa;       // (pq|uv) at p + nmo (q + nmo (u + n_act v))
  Buffers::device<double> papa;       // (pu|qv) at p + nmo (q + nmo (u + n_act v))
};

Result<CasscfIntegrals> casscf_integrals(/* active_space_hamiltonian's arguments, but tau_k = 1e-14 */);
```

Columns `[0, ncore)` of C are the core, `[ncore, ncore + ncas)` the active
space; a caller choosing other orbitals reorders C's columns first. The J and
K classes are the caller's, built once per basis, so a CASSCF loop builds none.
Classes through f only, as J and K.

**ppaa and papa** are nmo² n_act² doubles each, one nmo × nmo column-major
block in (p, q) per active pair (u, v), the blocks at nmo² (u + n_act v). So a
micro iteration's contraction over (u, v), as the generalized Fock's
Σ_vwx (pv|wx) Γ_qvwx, is one gemm against an nmo² × n_act² matrix, and a block
is an nmo × nmo matrix to gemm with. p and q count all of C's columns, u and v
the active ones from 0 (column ncore + u), as (tu|vw) does. ppaa is symmetric
in p, q and in u, v; papa's block (v, u) is its block (u, v) transposed. They
are PySCF's `mc_ao2mo` ppaa and papa with papa's indices in (p, q, u, v)
order rather than (p, u, q, v).

## How it is built

With P = C_core C_coreᵀ (one spin's core density) and U C the orbitals over the
Cartesian AOs the kernels evaluate:

| Quantity | From |
|---|---|
| E_core = E_nuc + 2 tr(P h) + tr(P G), G = 2 J[P] − K[P] | one `coulomb` and one `exchange` of P (skipped for `ncore = 0`) and two dots |
| h_eff = C_actᵀ h C_act + (U C_act)ᵀ G (U C_act) | the bare part over the spherical AOs, G's over the Cartesian ones |
| (tu\|vw) = (U C_act)ᵀ J[D_vw] (U C_act) | one `coulomb` per pair v <= w, written into the (v, w) and (w, v) blocks |
| (pq\|uv) = (U C)ᵀ J[D_uv] (U C) | the same J as (tu\|vw), contracted with all of C too: no J past the Hamiltonian's |
| (pu\|qv) = (U C)ᵀ K[C_u C_vᵀ] (U C) | one `exchange` of D_uv per pair u <= v, and one of its antisymmetric part per u < v |

**Each pair density is symmetrized**, D_vw = (C_v C_wᵀ + C_w C_vᵀ) / 2:
`coulomb` requires a symmetric density, and the antisymmetric part's J is zero,
so the symmetric part is the whole answer. n_act (n_act + 1) / 2 J builds in
all.

**K is not symmetric in its pair, so (pu|qv) takes both parts.** C_u C_vᵀ is
D_uv plus A_uv = (C_u C_vᵀ − C_v C_uᵀ) / 2, and K[D]ᵀ = K[Dᵀ] makes
K[C_v C_uᵀ] = K[D_uv] − K[A_uv]: the pairs u <= v suffice, each block (u, v)
the projected sum and block (v, u) the projected difference. `exchange` takes
D_uv as `Symmetric` and A_uv as `Antisymmetric`, the split its README
describes; A_uu = 0 is skipped.

**Builds per call.** `active_space_hamiltonian`: n_act (n_act + 1) / 2 J, plus
one J and one K of the core for ncore > 0. `casscf_integrals`: the same J,
and n_act² K past the core's: n_act (n_act + 1) / 2 of the D_uv and
n_act (n_act − 1) / 2 of the A_uv.

**The prune cap.** `coulomb_pair_classes` prunes primitive pairs assuming
|D_cd| <= 2 (`kPruneDensityCap`, `coulomb/pair_class/pair_class_launch.cu`),
and a pair density can exceed that. With m_v = max_µ |(U C)_µv|, every
|(D_vw)_cd| <= m_v m_w, so each J is built of s D_vw, s = min(1, 2 / (m_v m_w)),
screened at s `tau_j`, and scaled by 1 / s: J's absolute screen guarantee
holds for D_vw itself. K is capped the same way (`exchange`'s README states
the |D| <= 2 its pruning prototype assumed): each `exchange` runs on s D_vw or
s A_vw, |(A_vw)_cd| <= m_v m_w too, screened at s `tau_k` and scaled by 1 / s.
The m_v are read back to the host once.

**`tau_j` defaults to 1e-14, not `coulomb`'s 1e-10.** At 1e-10, (tu|vw) and
(vw|tu), two different J builds, part by 1.1e-10 on H₂O/cc-pVDZ, past
1e-12 max|(tu|vw)|, and E_core moves 4.7e-10 from PySCF's. At 1e-13 N₂/cc-pVDZ
still parts them by up to 9e-13, past the bar on some runs (J adds atomically,
so its last bits vary run to run); at 1e-14, six runs out of six pass. `tau_k` is
`exchange`'s own default for `active_space_hamiltonian`, and 1e-14 for
`casscf_integrals`: papa's active block is (tu|vw) built by K, and at 1e-13
K's screen parts it from the Hamiltonian's J-built (tu|vw) by up to 5.8e-12
on H₂O/cc-pVDZ and N₂/cc-pVDZ, 7.6 times past 1e-12 max|(tu|vw)|.

Scratch (U C, gemmwich's intermediate, a block, the spherical P; for
`casscf_integrals` U C and the intermediate over all nmo columns and an
nmo × nmo block) is carved
from `res.device_workspace()`, which neither `coulomb` nor `exchange` takes;
the three Cartesian matrices they read and write are their own type, allocated
per call.

## Not done

- **Batched J** (#200): one call over every pair density, once it lands.
- **Density fitting** (#177): (tu|vw) = Σ_P B_tu^P B_vw^P is far cheaper than
  n_act² J builds.

## Tests

`test/gtest/active_space_gpu.cpp`, against PySCF's CASCI through
`test/gtest/data/active_space/` (`tools/emit/active_integrals_reference.py`),
over H₂O in STO-3G and cc-pVDZ and N₂ in cc-pVDZ, each at `tau = 0` and the
defaults: E_core and the sign-invariant h_eff_tt, (tt|uu) and (tu|tu) to 1e-10;
the RHF energy given back by an active space spanning occupied and virtual
orbitals to 1e-10 of `ScfOutput::energy`; the 8-fold symmetry to 1e-12 max; the cap,
on orbitals scaled past it, pruned against unpruned classes to 1e-12 max; and
the refusals. `casscf_integrals` likewise, over the same cases and taus: the
sign-invariant (pp|uu) and (pu|pu) against PySCF's `CASSCF.ao2mo` ppaa and
papa to 1e-10, each summed over p's degenerate set and u's (p ranges over the
whole basis, so N₂'s π and δ sets outside the active space rotate freely; the
sum is invariant, and is the element itself for single orbitals); ppaa's
active block, and papa's read as (pu|qv) at p = t, q = v, against
`active_space_hamiltonian`'s (tu|vw) to 1e-12 max, so papa's K is checked
against an independent J; ppaa's (pq|uv) = (qp|uv) = (pq|vu) and papa's
(pu|qv) = (qv|pu) to 1e-12 max; the cap, on the scaled orbitals, ppaa on
pruned classes against unpruned ones and papa against 8⁴ times the unscaled
orbitals' (K's classes are unpruned, so its cap shows as exact scaling), to
1e-12 max; and the refusals. Its Scf step runs with `tau_j = 0`: at the default, J's screen moves
N₂'s RHF energy by 2.6e-10 Eh.
