# Fock

The energy, the Fock matrices and the orbital gradient of a (state-averaged)
CASSCF wavefunction at fixed orbitals C and fixed weighted RDMs γ and Γ, on the
GPU. What a CASSCF macro iteration evaluates after its CI, and what the orbital
Hessian is then built from.

## Module

`free_breakfast.mcscf.fock`, target `free_breakfast::mcscf::fock`, re-exported
by `free_breakfast.mcscf`.

```cpp
struct mcscf::Fock {
  double energy;                        // E, the weighted energy, E_nuc included (hartree)
  Buffers::device<double> fock_inactive; // F^I, nmo x nmo, MO basis
  Buffers::device<double> fock_active;   // F^A, nmo x nmo, MO basis
  Buffers::device<double> fock;          // the generalized Fock F, nmo x nmo
  Buffers::device<double> gradient;      // g over the packed pairs
  Buffers::device<double> gamma;         // γ_tu at t + ncas u
  Buffers::device<double> gamma2;        // Γ_tuvw at t + ncas (u + ncas (v + ncas w))
};

Result<mcscf::Fock> mcscf::fock(
    DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
    const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
    std::span<CoulombPairClass> j_classes, std::span<OrderedPairClass> k_classes,
    const SphericalMatrixAO &c, int32_t ncore, int32_t ncas,
    const Buffers::device<double> &ppaa, std::span<const double> gamma,
    std::span<const double> gamma2, double tau_j = 1e-14, double tau_k = 1e-14);

Status pack_orbital_pairs(DeviceResources &res, int32_t nmo, int32_t ncore, int32_t ncas,
                          const double *full, double *packed);
Status unpack_orbital_pairs(DeviceResources &res, int32_t nmo, int32_t ncore, int32_t ncas,
                            const double *packed, double *kappa);
```

The arguments through `k_classes` are `active_space_hamiltonian`'s, built once
per basis; `ppaa` is `casscf_integrals`' (pq|uv) over the same C (empty for
ncas = 0), and γ and Γ are `ci::Rdm::rdm12`'s, row-major, read back to the host
(the energy and the prune cap read them there). Every matrix
is column-major, (p, q) at p + nmo q. The packed order is
[`../README.md`](../README.md)'s. The tau defaults are `casscf_integrals`',
so F^I's and F^A's screens are the integrals' own.

## What it computes

With P_c = C_core C_coreᵀ and P_a = C_act γ C_actᵀ:

| Quantity | From |
|---|---|
| F^I = Cᵀ h C + Cᵀ (2 J[P_c] − K[P_c]) C | one `coulomb` and one `exchange` of P_c (none for ncore = 0) |
| F^A = Cᵀ (J[P_a] − ½ K[P_a]) C | one `coulomb` and one `exchange` of P_a (none for ncas = 0) |
| F_pq, q core | 2 (F^I + F^A)_pq: one `geam` |
| F_pt, t active | Σ_v F^I_pv γ_vt + Σ_uvw (pu\|vw) Γ_tuvw: two gemms |
| F_pq, q virtual | 0 |
| g_pq = 2 (F_pq − F_qp) | one `geam` over all of F, then the two packing copies |
| E = E_core + Σ_tu h_eff,tu γ_tu + ½ Σ_tuvw (tu\|vw) Γ_tuvw | off F^I's and F's diagonals, on the host |

This is Helgaker's generalized Fock, F_pq = Σ_r h_pr D_rq + Σ_rst (pr\|st) d_qrst
over C's orbitals, written out for a CASSCF's block-diagonal densities; it is
PySCF's `mc1step.gen_g_hop` `g` with column q for orbital q, and its gradient
twice gen_g_hop's (the derivative of E itself, not of E / 2).

**(pu|vw) is one gemm.** ppaa's block (v, w) is (pq|vw) over every p, q, so
the active u are ncas contiguous columns, q = ncore + u, of each block. One 2D
copy gathers them into an nmo × ncas³ matrix (pu|vw) at
p + nmo (u + ncas (v + ncas w)), and Γ in the same t-fastest order is an
ncas × ncas³ matrix: F_act += (pu|vw) Γᵀ is a single gemm with K = ncas³. The
gathered matrix is nmo ncas³ doubles of workspace, PySCF's `paaa`.

**E_core and h_eff are read off F^I.** E_core = E_nuc + Σ_i (h_ii + F^I_ii)
over the core, and h_eff is F^I's active block: exactly the definitions
`active_space_hamiltonian` builds them by, from the same J and K of P_c. (tu|vw)
is the active-row block of the (pu|vw) above, so its contraction with Γ is the
active diagonal of F's second term: Σ_t F_tt = Σ_tu F^I_tu γ_ut +
Σ_tuvw (tu|vw) Γ_tuvw, and E = E_core + ½ (Σ F^I_aa γ + Σ_t F_tt). No J or K is
built past F^I's and F^A's. It takes `casscf_integrals`' ppaa rather than the
whole `CasscfIntegrals` so that ncas = 0, which `casscf_integrals` refuses,
still has an energy: the caller passes an empty buffer.

**ncas = 0 is the RHF limit.** F^A = 0, F = 2 F^I in the core columns and 0
past them, so g_ai = 4 F^I_ai over virtual a and occupied i: the RHF gradient,
zero at a converged SCF, and E is the RHF energy.

**The prune cap.** `coulomb_pair_classes` prunes primitive pairs assuming
|D_cd| ≤ 2, and P_a can exceed it: with m_t = max_µ |(U C)_µt|,
|(P_a)_cd| ≤ Σ_tu |γ_tu| m_t m_u. So J and K of P_a are built on s P_a,
s = min(1, 2 / that bound), screened at s tau and scaled by 1 / s, as
`casscf_integrals` holds the cap for each pair density. The m_t are read back
once. P_c is not scaled, as `active_space_hamiltonian`'s is not.

Scratch (U C, gemmwich's intermediate, one nmo × nmo block that ends as
2 (F − Fᵀ), and the nmo ncas³ gathered (pu|vw)) is carved from
`res.device_workspace()`, which neither `coulomb` nor `exchange` takes; the
three Cartesian matrices they read and write are their own type, allocated per
call. The result's seven members are its own buffers: the orbital Hessian reads
all of them again with no J or K.

## Not done

- **The macro loop** (#251) builds on this and on `../hessian/`'s orbital
  Hessian.
- **Frozen orbitals**: every core orbital rotates.
- **Density fitting** (#177), as for `active_space`.

## Tests

`test/gtest/mcscf_fock_gpu.cpp`, against `test/gtest/data/casscf/casscf.json`
(PySCF's `mc1step.gen_g_hop`, `tools/emit_casscf_reference.py`), at each case's
recorded start: H₂O cc-pVDZ CAS(4,4) from RHF orbitals [2, 3, 5, 6], N₂ cc-pVDZ
CAS(6,6) frontier, H₂CO cc-pVDZ CAS(4,3) at two singlets weighted [0.5, 0.5]
and [0.7, 0.3], and H₂O cc-pVDZ by AVAS (O 2p, H 1s). The test reaches each
start through `pipeline.active_space` and the `Casci` step, over an SCF with J
unscreened, and takes the RDMs from `ci::Rdm::rdm12` over the Casci step's roots; the
orbitals pair with PySCF's by their energies, checked to 1e-6.

- the energy against PySCF's CASCI energy, against the Casci step's roots
  weighted, and against E_core + Σ h_eff γ + ½ Σ (tu|vw) Γ from
  `active_space_hamiltonian` over the same RDMs, each to 1e-9;
- the gradient's sqrt(Σ g_pq²) per pair of orbital sets (the element itself
  for two single orbitals; N₂'s π pairs rotate freely within themselves) and
  |g|, to 1e-7;
- g · x against a central finite difference of that independent energy along
  three random unit x (C exp(±ε κ(x)) through `skew_eig` and `skew_expm`,
  Richardson-extrapolated over ε = 2e-3 and 1e-3, the RDMs fixed), to 1e-6 of
  max(|g · x|, |g|);
- the RHF limit, ncas = 0 on H₂O cc-pVDZ: |g_ai| < 1e-6 at the converged SCF
  and E its energy to 1e-9; off it, g = 4 F^I_ai and F = 2 F^I in the core
  columns, and g against a finite difference of the energy;
- the packed order element by element, and unpack as its antisymmetric
  inverse, over empty and non-empty spaces;
- the refusals.
