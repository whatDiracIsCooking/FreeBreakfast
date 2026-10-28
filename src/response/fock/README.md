# Fock

The energy, the Fock matrices and the orbital gradient of a (state-averaged)
CASSCF wavefunction at fixed orbitals C and fixed weighted RDMs γ and Γ, on the
GPU. What a CASSCF macro iteration evaluates after its CI, and what the orbital
Hessian is then built from.

## Module

`free_breakfast.response.fock`, target `free_breakfast::response::fock`,
re-exported by `free_breakfast.response`.

```cpp
struct response::Fock {
  double energy;                        // E, the weighted energy, E_nuc included (hartree)
  Buffers::device<double> fock_inactive; // F^I, nmo x nmo, MO basis
  Buffers::device<double> fock_active;   // F^A, nmo x nmo, MO basis
  Buffers::device<double> fock;          // the generalized Fock F, nmo x nmo
  Buffers::device<double> gradient;      // g over the packed pairs
  Buffers::device<double> gamma;         // γ_tu at t + ncas u
  Buffers::device<double> gamma2;        // Γ_tuvw at t + ncas (u + ncas (v + ncas w))
};

// Integral-direct.
Result<response::Fock> response::fock(
    DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
    const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
    std::span<CoulombPairClass> j_classes, std::span<OrderedPairClass> k_classes,
    const SphericalMatrixAO &c, int32_t ncore, int32_t ncas,
    const Buffers::device<double> &ppaa, const Buffers::device<double> &papa,
    std::span<const double> gamma, std::span<const double> gamma2,
    double tau_j = 1e-14, double tau_k = 1e-14);

// Density-fitted.
Result<response::Fock> response::fock(
    DeviceResources &res, const Molecule &molecule, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, int32_t naux, const SphericalMatrixAO &c,
    int32_t ncore, int32_t ncas, const Buffers::device<double> &ppaa,
    const Buffers::device<double> &papa, std::span<const double> gamma,
    std::span<const double> gamma2);
```

The integral-direct overload's arguments through `k_classes` are
`active_space_hamiltonian`'s, built once per basis. The density-fitted one
takes B in their place: `b` is `DensityFitted::b`, naux × nmo²
over the spherical AOs, `[Q][n][m]` as `df_coulomb` reads it. In both, `ppaa`
and `papa` are `casscf_integrals`' (pq|uv) and (pu|qv) over the same C (both
empty for ncas = 0), and γ and Γ are `ci::Rdm::rdm12`'s, row-major, read back
to the host (the energy reads them there). Every matrix is column-major, (p, q)
at p + nmo q. The gradient is packed by
[`../orbital_pairs/`](../orbital_pairs/README.md), in
[`../README.md`](../README.md)'s order. The tau
defaults are `casscf_integrals`', so F^I's screens are the integrals' own.

**The two differ only in how F^I is built.** F^A, the generalized Fock, the
gradient and the energy are one shared code path over F^I, ppaa, papa, γ
and Γ.

## What it computes

With P_c = C_core C_coreᵀ and P_a = C_act γ C_actᵀ:

| Quantity | From |
|---|---|
| F^I = Cᵀ h C + Cᵀ (2 J[P_c] − K[P_c]) C | one `coulomb` and one `exchange` of P_c, or one `df_coulomb` of 2 P_c and one `df_exchange` of C_core (none for ncore = 0) |
| F^A_pq = Σ_tu γ_tu [(pq\|tu) − ½ (pt\|qu)] | two gemvs, over ppaa and papa (none for ncas = 0) |
| F_pq, q core | 2 (F^I + F^A)_pq: one `geam` |
| F_pt, t active | Σ_v F^I_pv γ_vt + Σ_uvw (pu\|vw) Γ_tuvw: two gemms |
| F_pq, q virtual | 0 |
| g_pq = 2 (F_pq − F_qp) | one `geam` over all of F, then `pack_orbital_pairs`' two copies |
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

**F^A is two gemvs over ppaa and papa.** Cᵀ (J[P_a] − ½ K[P_a]) C with
P_a = C_act γ C_actᵀ is Σ_tu γ_tu [(pq|tu) − ½ (pt|qu)], and ppaa's block
(t, u) is (pq|tu) over every p, q, papa's (pt|qu): as nmo² × ncas² matrices,
column t + ncas u block (t, u), each is one gemv against γ in the same
t-fastest order. nmo² ncas² multiply-adds each, against the AO J and K of
P_a this replaced (and the |D| ≤ 2 prune cap that P_a needed, with its one
read-back of the active orbitals): F^A now costs nothing past the integrals
the generalized Fock and the Hessian already read, and is exactly as screened
as they are. Against that AO route F^A agrees to 2.4e-13 (H₂O cc-pVDZ
CAS(4,4), both at τ = 1e-14).

**E_core and h_eff are read off F^I.** E_core = E_nuc + Σ_i (h_ii + F^I_ii)
over the core, and h_eff is F^I's active block: exactly the definitions
`active_space_hamiltonian` builds them by, from the same J and K of P_c. (tu|vw)
is the active-row block of the (pu|vw) above, so its contraction with Γ is the
active diagonal of F's second term: Σ_t F_tt = Σ_tu F^I_tu γ_ut +
Σ_tuvw (tu|vw) Γ_tuvw, and E = E_core + ½ (Σ F^I_aa γ + Σ_t F_tt). No J or K is
built past F^I's. It takes `casscf_integrals`' ppaa rather than the
whole `CasscfIntegrals` so that ncas = 0, which `casscf_integrals` refuses,
still has an energy: the caller passes an empty buffer.

**ncas = 0 is the RHF limit.** F^A = 0, F = 2 F^I in the core columns and 0
past them, so g_ai = 4 F^I_ai over virtual a and occupied i: the RHF gradient,
zero at a converged SCF, and E is the RHF energy.

**The density-fitted F^I.** `df_coulomb` reads any D, so J is of
D = 2 P_c; `df_exchange` builds K of D = 2 C Cᵀ from the factor C, so C_core
gives K[2 P_c] without forming P_c for it. 2 J[P_c] − K[P_c] is then
J[D] − ½ K[D]. Both are spherical: B is over the spherical AOs, so there is no
U, no screen and no prune cap. Whether the whole result is density-fitted is
the caller's: ppaa and papa are taken as given, fitted or not.

Scratch: integral-direct, U C, gemmwich's intermediate, one nmo × nmo block
that ends as 2 (F − Fᵀ), and the nmo ncas³ gathered (pu|vw), carved from
`res.device_workspace()`, which neither `coulomb` nor `exchange` takes; the
three Cartesian matrices they read and write are their own type, allocated per
call. Density-fitted, three nmo × nmo matrices per call (2 P_c, J and K, two
of them reused as gemmwich's intermediate and the projected G), since
`df_coulomb` and `df_exchange` take the workspace; the block and the gathered
(pu|vw) are carved from it after them. The result's seven members are its own
buffers: the orbital Hessian reads all of them again with no J or K.

## Not done

- **Frozen orbitals**: every core orbital rotates.
- **Density-fitted ppaa and papa**: the density-fitted overload takes them
  as given; building them from B is `mcscf::solve`'s DF path (#385).

## Tests

`test/gtest/response_fock_gpu.cpp`, against `test/gtest/data/casscf/casscf.json`
(PySCF's `mc1step.gen_g_hop`, `tools/emit/casscf_reference.py`), at each case's
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
- F^A from ppaa and papa against the AO route it replaced,
  Cᵀ (J[P_a] − ½ K[P_a]) C by one `coulomb` and one `exchange` at the old
  prune cap, on H₂O cc-pVDZ CAS(4,4), to 1e-10 (2.4e-13 measured);
- the density-fitted overload, over cc-pVDZ-JKFIT, at each case's CASCI
  orbitals and RDMs and over ppaa and papa contracted from the same B on the
  host: the energy, F^I, F^A, F and g against a host contraction of B to
  1e-10, and against the integral-direct `fock` within the fit's error
  (1e-3 in the energy, 3e-3 per element; 2.8e-4 and 1.2e-3 measured at worst);
- its RHF limit, ncas = 0 at a DF-RHF over the same set converged by the Scf
  step: E its energy to 1e-9 and |g_ai| < 1e-6;
- the refusals, of both overloads.
