# qd_nevpt2_nac golden data

References for the QD-NEVPT2 derivative coupling d_PQ = ⟨Ψ_P|∇Ψ_Q⟩ (Park
2020, Eqs. 45–54), SC and PC. No file here is hand-edited.

- `brute_force/` (#744, below): the first-order wave functions built
  explicitly in the full determinant space, the PT2 transition density
  ⟨Ψ⁽¹⁾_M|E_xy|N⟩, and finite differences of the **whole** coupling and of
  d^PT2 alone. Written by `tools/emit/qd_nevpt2_nac_reference.py` (PySCF).
- `coupling/` (#743, below): central differences of the interstate coupling
  h_PQ (its reference and PT2 parts apart) and of the zeroth-order coupling
  d^NEV + d^CAS, through the tree's own DF-SA-CASSCF and QD-NEVPT2. Written
  by `free-breakfast-qd-nevpt2-gradient --coupling`
  (`tools/qd_nevpt2_gradient/`). `test/gtest/nevpt2_numerical_heff_gpu.cpp`
  checks that tool's library's in-process ∇H_eff (#783) against
  `h2co_skew_ccpvdz_sa3`'s `{sc,pc}_h`, within twice the committed error bar,
  each pair up to the sign of its two eigenvectors.

## `brute_force/` — the brute-force first-order wave functions (#744)

Only these references can see d^PT2 = Σ_MN T^PQ_MN ⟨Ψ⁽¹⁾_M|∇N⟩ (Park
Eq. 49, T^PQ = ½(R_P R_Qᵀ − R_Q R_Pᵀ)): no zeroth-order overlap sees Ψ⁽¹⁾.
They are built on `../qd_nevpt2/`'s three brute-force cases, the only ones
whose full determinant space (every orbital) is small enough.

### Cases

The geometry, basis (cc-pVDZ-JKFIT fitting), active space and weights of
each are `../qd_nevpt2/qd_nevpt2.json`'s.

| case | roots | core, active, n_mo | determinants | steps (bohr) | components | H_eff gaps SC (PC), hartree | inactive gaps (core, virtual) |
|---|---|---|---:|---|---|---|---|
| `h6_631g_sa2` | 2 | 2, 3, 12 | 48 400 | 1e-3, 5e-4, 2.5e-4 | 3 directions | 0.2907 (0.2898) | 0.28, 0.023 |
| `h8_sto3g_sa3` | 3 | 2, 4, 8 | 4 900 | 5e-4, 2.5e-4, 1.25e-4 | 3 directions | 0.7512, 0.0152 (0.7509, 0.0139) | 0.039, 0.10 |
| `h8_ring_sto3g_sa3` | 3 | 1, 4, 8 | 4 900 | 6e-5, 3e-5, 1.5e-5 | 3 directions and all 24 Cartesian | 0.1416, 4.7e-3 (0.1400, 6.0e-3) | —, 1.8e-3 |

The directions are `../qd_nevpt2_gradient/`'s per case (seed 696,
translation and rotation about the centroid projected out, orthonormal),
copied into each file and checked to be free of rigid motion. The steps are
that directory's, except the ring's, halved from 1e-4: its SC pair (1, 2)
(H_eff states 4.7 mEh apart, |d| up to 8.9) needs the smaller step for its
Richardson error, and its noise (below) rules out a smaller one still.

### The steps

1. **Ψ⁽¹⁾_M, SC and PC**, every root M, as a full-space determinant vector,
   class by class from the amplitudes `qd_nevpt2_reference.py`'s brute-force
   H_eff validation uses:
   - SC: per class and external occupation pattern t,
     −P_t V|M⟩ / (⟨P_t V M|H_D|P_t V M⟩/N_t − E_M⁽⁰⁾), V|M⟩ on t being H|M⟩ on
     t, over the patterns whose norm N_t passes `_norm_to_energy`'s guard;
   - PC: per class and tuple, −Q (Qᵀ(H_D − E_M⁽⁰⁾)Q)⁻¹ QᵀV|M⟩, Q the
     orthonormalized span of the class's spin-free E_pq E_rs on |M⟩ (the PC
     cut and gap rule; `pc_brute_couplings`).

   **Self-checks**: ⟨N|H|Ψ⁽¹⁾_M⟩ per class equals `couplings[N, M, k]`
   (`pc_couplings` for PC) and its sum `heff2[N, M]`, within `BRUTE_FORCE`
   (1e-9), and ⟨N|Ψ⁽¹⁾_M⟩ = 0. Run on the fixture's state and again on the
   polished one (below). Measured: at most 6.3e-15 (SC) and 5.4e-14 (PC),
   overlaps at most 2.1e-16.
2. **The PT2 transition density** D^(MN)_xy = ⟨Ψ⁽¹⁾_M|E_xy|N⟩, E_xy spin
   summed, for every ordered pair (M, N), diagonal included, over all n_mo
   orbitals: on **the `../qd_nevpt2/` fixture's own state**. Its CI vectors
   are checked equal to the committed `ci.npy`, and its SC and PC H_eff to
   the committed ones (both differences 0.0 on every case). So a consumer starting
   from that fixture's integrals and CI vectors has the same input. Outside
   the (t,i), (a,t) and (a,i) blocks it is zero, to 8.3e-15 of its largest
   element (`OFF_BLOCK`, 1e-14): E_xy|N⟩ lies outside the first-order
   interacting space everywhere else.
3. **The whole coupling, by central differences** (Park Eq. 45). With
   Ψ̃_P = Σ_M (|M⟩ + |Ψ⁽¹⁾_M⟩) R_MP, Ψ⁽⁰⁾_P = Σ_M |M⟩ R_MP and
   ⟨Ψ⁽⁰⁾_P(R)|Ψ⁽¹⁾_Q(R)⟩ = 0 at every geometry,

       d_PQ = d/dR Z_PQ + d/dR ½(Y_PQ − Y_QP)
       Z_PQ(R) = ⟨Ψ⁽⁰⁾_P(R₀)|Ψ⁽⁰⁾_Q(R)⟩    (d^NEV + d^CAS, Eqs. 47 + 48)
       Y_PQ(R) = ⟨Ψ⁽¹⁾_P(R₀)|Ψ⁽⁰⁾_Q(R)⟩    (½(Y − Yᵀ): d^PT2, Eq. 49)

   Ψ⁽¹⁾ is built **only at R₀**. Each cross-geometry overlap is a full-space
   determinant overlap, `fci.addons.overlap` over the MO cross overlap
   C(R₀)ᵀ S_AO(R₀, R) C(R) from `gto.intor_cross`, so the basis functions
   moving with the atoms are in it: this is the **total** coupling (the CSF
   term included, `use_etfs = false`), not translation-invariant, so no
   translational sum is checked.

### Each point

- **The reference.** The differences need a stationary point, since an
  overlap feels a residual orbital gradient to first order. From the
  fixture's orbitals, the DF SA-CASSCF (`casscf_reference`'s mc1step) is
  polished by Newton steps over the exact orbital Hessian, the CI re-solved
  densely, until |g| stops falling (the best point kept, at most 20 steps,
  under `POLISH_BAR`, 1e-11). Reached: 1.0e-14, 2.4e-15, 3.8e-15. Over it the
  roots (signed below), the core and virtuals canonicalized over the
  state-averaged Fock, SC and PC's H_eff, their eigenvectors R and Ψ⁽¹⁾ are
  rebuilt. The polished and fixture states differ by the fixture's own
  convergence (`../qd_nevpt2/` does not polish these cases), which is why
  part 2 is written over the fixture and part 3 over the polished state.
- **A displaced point** carries the reference's orbitals, Löwdin
  orthonormalized in the displaced overlap, re-converges the DF SA-CASSCF at
  the same weights and fitting set, and polishes it the same way (worst |g|
  anywhere: 3.1e-14, 9.7e-15, 1.0e-14). Then the same QD-NEVPT2.
- **Matching and phases.** Each displaced root is matched to the reference's
  by the largest |⟨M(R₀)|N(R)⟩| (a permutation, each within `GUARD` = 1e-3
  of 1) and signed so that overlap is positive. H_eff(R) is carried into that
  matched, signed basis, and its eigenvectors are matched to R(R₀) by
  R(R₀)ᵀR(R) and signed the same way. Worst over every point: 1 − |CI
  overlap| 4.3e-6, 1 − |R overlap| 2.2e-6 (`h8_sto3g_sa3`). PC's cut counts
  are checked unchanged at every point.

### The phase rule

At R₀ each CI vector is signed by `casscf_nac_reference.py`'s rule (its
first coefficient, row-major over (α string, β string), within 1e-8 of its
largest is positive; `reference.ci` holds them), and each column of R has
its largest component positive (`pipeline/nevpt2`'s rule; `reference.sc_vectors`,
`pc_vectors`). d_PQ changes sign with Ψ_P's or Ψ_Q's phase, so a consumer
whose phases differ multiplies d_PQ by s_P s_Q. Its s_P can be read off the
overlap of its own root vectors with `reference.ci`, carried through R.
Only P < Q is written: d_QP = −d_PQ. The bra is P, the ket Q, as in
`mcscf::df_nac`.

### The error bar

Per component (direction or Cartesian), scheme, term (`zeroth`, `pt2`,
`total`) and pair:

- `central`: D(h) = (F(+h) − F(−h)) / 2h at each step, F the term's overlap;
- `richardson`: (4 D(h/4) − D(h/2)) / 3 over the two finest, **the number to
  test against**;
- `error_bar`: |R(fine) − R(coarse)| + `noise` / 2h_finest, #696's
  convention, the noise per scheme, term and pair.

**The noise.** The arithmetic is deterministic (one thread; a rerun writes
the same bytes), so the noise is each point's own convergence error: the
polish stops near |g| ~ 1e-14, and the zeroth-order overlap feels the
orbitals' residual through R, which H_eff's near-degeneracy amplifies. It is
measured by re-converging the reference geometry from three other starts
(the orbitals of the three coarsest +h points carried back) and taking each
overlap's largest distance from the reference's own (Z from 1, ½(Y − Yᵀ)
from 0). `noise` is three times that (a point error e moves a Richardson
value by up to 1.5 e/h), at least 1e-14:

| case | largest measured, zeroth SC (PC) | largest measured, pt2 SC (PC) | pair it is in |
|---|---:|---:|---|
| `h6_631g_sa2` | 1.5e-15 (1.3e-15) | 7.9e-17 (2.3e-16) | (0, 1) |
| `h8_sto3g_sa3` | 1.7e-13 (8.0e-13) | 1.7e-16 (3.0e-16) | (1, 2) |
| `h8_ring_sto3g_sa3` | 8.7e-13 (3.9e-13) | 3.3e-18 (3.4e-17) | (1, 2) |

Both 8-hydrogen cases' noisy pair is their near-degenerate H_eff pair. Ψ⁽¹⁾
is fixed at R₀, so d^PT2's noise is round-off.

**Checks before writing**, every case, scheme, term and component, or nothing
is written:

- every error bar under `ERROR_BAR` = **2e-7**, a fifth of the design's
  ~1e-6 gate. Not #696's 1e-7: on the ring's SC pair (1, 2) no step brings
  both its Richardson error and its noise term under 1e-7 (at 1e-4 its
  largest bar is 8.2e-7, at 1.5e-5 the noise term alone is 1e-7);
- the ladder ratio (D(h) − D(h/2)) / (D(h/2) − D(h/4)) in (3.2, 4.8) wherever
  the finer rung difference clears both 1e-7 and 5 noise / h_finest;
- the self-checks, the off-block zero and the guard above.

**Measured** (per family; the largest error bar of `zeroth` / `pt2` /
`total`; the ladder ratios checked):

| case, family | SC error bars | SC ratios | PC error bars | PC ratios |
|---|---|---|---|---|
| `h6_631g_sa2` | 9.5e-8 / 2.0e-11 / 9.5e-8 | 3.34–4.00 (5) | 4.1e-11 / 2.1e-11 / 6.1e-11 | 4.00 (1) |
| `h8_sto3g_sa3` | 5.5e-9 / 4.2e-11 / 5.6e-9 | 4.00 (10) | 1.6e-8 / 4.4e-11 / 1.6e-8 | 4.00 (10) |
| `h8_ring_sto3g_sa3`, directions | 1.3e-7 / 3.4e-10 / 1.3e-7 | 3.96–4.00 (20) | 7.2e-8 / 3.4e-10 / 7.2e-8 | 3.97–4.13 (10) |
| `h8_ring_sto3g_sa3`, Cartesian | 1.7e-7 / 3.3e-10 / 1.7e-7 | 3.85–4.03 (116) | 7.4e-8 / 3.3e-10 / 7.5e-8 | 3.90–4.16 (40) |

Every error bar above 1e-7 is on the ring's SC pair (1, 2). A consumer's bar
is twice the component's error bar (#744's "How results are compared"), so
at most 3.4e-7 there, and 2e-7 or less elsewhere. d^PT2 never clears the
ladder floor: its steps change it by less than 1e-7, so its error bar (the
Richardson distance and round-off) is the evidence.

### Visibility

Each separately coded term's largest |·| over the case's components and its
bar. For d^PT2 and d^NEV + d^CAS (`zeroth`), the bar is twice the error bar,
#749's test, and the ratio is the largest per component. For D^(MN)'s blocks,
the quantity is the antisymmetric ½(D^(MN) − D^(NM)) #746 tests, against
1e-10 of that combination's largest element. One pair per case and scheme
(`visibility` in each file has every pair); the (a,t) and (a,i) bars are
the (t,i) one:

| case | pair | scheme | d^NEV + d^CAS | d^PT2 | D (t,i) | D (a,t) | D (a,i) |
|---|---|---|---|---|---|---|---|
| `h6_631g_sa2` | (0, 1) | SC | 0.114 / 1.9e-7 (8.0e7) | 2.2e-3 / 4.0e-11 (5.5e7) | 1.0e-2 / 1.0e-12 (1e10) | 4.8e-3 (4.7e9) | 6.0e-3 (5.8e9) |
| `h6_631g_sa2` | (0, 1) | PC | 0.108 / 8.2e-11 (1.4e9) | 5.3e-3 / 4.1e-11 (1.3e8) | 2.8e-2 / 2.8e-12 (1e10) | 5.5e-3 (2.0e9) | 1.5e-2 (5.5e9) |
| `h8_sto3g_sa3` | (1, 2) | SC | 9.83 / 1.1e-8 (2.1e9) | 1.2e-3 / 8.3e-11 (1.5e7) | 3.4e-3 / 3.4e-13 (1e10) | 2.9e-3 (8.6e9) | 8.3e-5 (2.4e8) |
| `h8_sto3g_sa3` | (1, 2) | PC | 9.27 / 3.2e-8 (4.7e8) | 2.7e-3 / 8.8e-11 (3.1e7) | 3.7e-3 / 3.7e-13 (1e10) | 2.6e-3 (7.1e9) | 4.1e-5 (1.1e8) |
| `h8_ring_sto3g_sa3` | (0, 2) | SC | 0.311 / 6.7e-8 (1.6e8) | 5.5e-4 / 6.7e-10 (8.3e5) | 1.2e-4 / 7.4e-13 (1.7e8) | 7.4e-3 (1e10) | 1.8e-5 (2.5e7) |
| `h8_ring_sto3g_sa3` | (1, 2) | PC | 7.57 / 1.5e-7 (7.3e7) | 5.9e-4 / 6.7e-10 (8.8e5) | 1.1e-5 / 8.4e-13 (1.3e7) | 8.4e-3 (1e10) | 1.1e-3 (1.3e9) |

Every term of every pair clears 10× its bar on every case, by at least 5e5;
`visibility` in each file has every pair. d^PT2 is 7e-5 to 5e-2 of the whole
coupling, so a test against the total alone would miss it; that is why
`pt2` is written separately.

### The cross-check against #743

The zeroth-order part `zeroth` is #743's d^NEV + d^CAS, and
`h8_ring_sto3g_sa3` is a case of both, along the same three directions.
The two differ in the program (PySCF here, the tree's own pipeline there),
not in the quantity: the same fitting set, weights and SA-canonical
orbitals. **Done** (#743, `coupling/h8_ring_sto3g_sa3.json`'s
`brute_force_cross_check`): every direction and pair agrees within the two
error bars combined, SC within 6.4e-8 (0.10 of the bars) and PC within
1.5e-8 (0.06), below.

### Files

`brute_force/<case>.json` and `brute_force/<case>/{sc,pc}_pt2_density.npy`.

| `.npy` | shape | what |
|---|---|---|
| `sc_pt2_density`, `pc_pt2_density` | (n, n, n_mo, n_mo) | [M, N, x, y] = ⟨Ψ⁽¹⁾_M\|E_xy\|N⟩, `'<f8'`, Fortran order (the layout `src/io/npy` reads); the MOs are the fixture's SA-canonical ones, core, active, virtual |

| JSON key | meaning |
|---|---|
| `name`, `basis`, `aux_basis`, `atoms` (bohr), `ncore`, `ncas`, `nelecas`, `nmo`, `weights` | the case |
| `steps`, `seed`, `directions` | above; `directions` 3 × 3N, atom-major |
| `pairs` | the pairs P < Q, the order of every per-pair list |
| `noise`, `noise_measured` | per scheme and term, per pair (`total`'s noise is `zeroth`'s plus `pt2`'s) |
| `fixture` | step 1's checks and the density's off-block maximum on the fixture's state, and the CI and H_eff agreement with `../qd_nevpt2/` |
| `reference` | the polished R₀: `gradient_norm`, `hessian_least_eigenvalue`, `casci_energies`, `ci` (signed), `inactive_gaps`, `psi1` (step 1's checks there), per scheme `heff` (symmetrized), `energies`, `vectors` (R, column P), `heff_gaps` |
| `directional`, `full` | per scheme, per term a list over components (direction, or atom-major Cartesian) of `central` (per step, per pair), `richardson`, `error_bar`, `richardson_minus_finest` (per pair) and the checked `ratios`; `full` null but on the ring |
| `displaced` | per family and component: `gradient_norm` and per scheme `z` and `y`, per step the [−h, +h] pair of n × n matrices, so every difference can be redone |
| `guard`, `checks`, `visibility` | above |
| `tolerances`, `provenance` | the bars, PySCF and numpy versions, threads, the command |

### Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/emit/qd_nevpt2_nac_reference.py [case ...]
```

writes the cases named, or all three, each on one thread: a few minutes
for all three as three processes, most of it the ring's 162 points (under
a second each). Two full runs wrote
identical files. `test/pytest/test_qd_nevpt2_nac_reference.py` checks the
files' invariants on the push gate, and reruns `h8_sto3g_sa3`'s density and
one point in its slow tier.

## `coupling/` — h_PQ and the zeroth-order coupling, through the tree (#743)

These are the two parts of the QD-NEVPT2 coupling a zeroth-order overlap can
see, SC and PC, for every pair P < Q of H_eff's eigenstates. Each is a
central difference through this tree's own chain: `[casscf]` + DF `[eri]`
(cc-pVDZ-JKFIT) + `[nevpt2] quasi_degenerate`, rerun as one nested
`Pipeline` at every point. That is the chain and the guard of
`../qd_nevpt2_gradient/` (its README has the matching and the continuity
guard), with two changes: every point is **polished** (below), and the
quantities differenced are not energies. d^PT2 needs Ψ⁽¹⁾ and is
`brute_force/`'s.

### Cases

The geometry, basis, active space and weights of each are
`../qd_nevpt2/qd_nevpt2.json`'s. Gaps are in hartree. The inactive gaps are
the smallest between two canonical core, or virtual, energies at the
reference (`guard.inactive_gap` has the smallest at any point).

| case | roots, weights | steps (bohr) | components | points | root gaps | H_eff gaps SC (PC) | inactive gaps (core, virtual) | ‖g‖ reference, worst |
|---|---|---|---|---:|---|---|---|---|
| `h2co_skew_ccpvdz_sa3` | 3, equal | 1e-3, 5e-4, 2.5e-4 | 3 directions and all 12 Cartesian | 96 | 0.1080, 0.1938 | 0.1545, 0.1986 (0.1546, 0.1985) | 0.072, 7.6e-3 | 1.0e-12, 2.8e-12 |
| `h2co_skew_ccpvdz_sa3_weighted` | 3, 0.4/0.4/0.2 | 2.5e-4, 1.25e-4, 6.25e-5 | 3 directions | 24 | 0.1465, 0.3048 | 0.1555, 0.2029 (0.1566, 0.1907) | 2.6e-3, 3.6e-3 | 7.7e-13, 2.9e-12 |
| `h8_ring_sto3g_sa3` | 3, equal | 6e-5, 3e-5, 1.5e-5 | 3 directions | 24 | 0.1031, 0.0217 | 0.1416, 4.7e-3 (0.1400, 6.0e-3) | —, 1.8e-3 | 2.1e-14, 2.8e-12 |

The points include six for the noise (below). The directions are
`../qd_nevpt2_gradient/`'s three per case (seed 696, translation and rotation
projected out), and `brute_force/`'s.

- **Steps.** Skew H₂CO's are `../qd_nevpt2_gradient/`'s. The ring's are
  #744's: its SC pair (1, 2) needs them. The weighted case's are a quarter of
  skew H₂CO's. Its SC overlaps along the second direction curve on a length
  of ~1e-3 bohr (its two closest core orbitals are 2.6e-3 apart), so at
  1e-3 one ladder ratio of `sc_zeroth` falls to 0.7.
- **`h2co_ccpvdz_sa2` is not a case.** Its 1A1 and 1A2 roots couple only
  along an A2 displacement, and H₂CO has none, so its data would be zeros.

### The quantities

The reference's eigenvectors R (column P, `reference.sc_heff_vectors`,
`pc_heff_vectors`) are held fixed. Each displaced root is matched to the
reference's by `mcscf::state_overlaps` and signed so that its overlap is
positive. H_eff(R) is carried into that matched, signed basis, and so is
each displaced eigenvector R(R), which is matched to R(R₀) and signed the
same way. Per pair, the families differenced are:

| family | quantity at R | its derivative |
|---|---|---|
| `{sc,pc}_h_reference` | Σ_r R_rP R_rQ E_r(R) | h_PQ's reference part, Σ_r ρ_r ∇E_r |
| `{sc,pc}_h_pt2` | R_Pᵀ (H_eff(R) − diag E(R)) R_Q | h_PQ's PT2 part |
| `{sc,pc}_h` | R_Pᵀ H_eff(R) R_Q | h_PQ |
| `{sc,pc}_zeroth` | Σ_MN R_MP(R₀) R_NQ(R) S_MN(R₀, R) | d^NEV + d^CAS (Park 2020 Eqs. 47 + 48) |
| `casscf_zeroth` | S_IJ(R₀, R), I < J roots | the roots' total d_IJ, QD off |

S comes from the full AO cross overlap, so the overlaps are **total**
couplings, CSF term included (`use_etfs = false`). They are not
translation-invariant; h is. The derived terms are formed from the
Richardson values and the error bars:

- `d_nev` = h / (E_Q − E_P), with bar h's / |E_Q − E_P|;
- `d_cas` = Σ_{M<N} (R_MP R_NQ − R_MQ R_NP) d_MN over `casscf_zeroth`, with
  bar Σ |coefficient| bar_MN.

The phases follow the reference's roots as the solver left them (the file
does not store them), and P is the bra, as in `mcscf::df_nac`. A consumer
whose phases differ multiplies by s_P s_Q. Only P < Q is written, and
`pairs` gives the order.

### Each point, and the polish

The reference is the RHF and DF-SA-CASSCF of `../qd_nevpt2_gradient/`, and
every displaced point starts from the reference's orbitals, Löwdin-carried.
**The polish.** An overlap, or an off-diagonal element of H_eff, feels a
residual ‖g‖ to first order. An eigenvalue feels it only through the roots'
energies. At the gradient oracle's ‖g‖ < 1e-10 the noise was far too large:
on skew H₂CO a point at ‖g‖ ~3e-11 moved PC's H_eff off the diagonal by
4e-10 hartree and `pc_zeroth` by 2e-9, which is a bar of 1e-5. So in this
mode each point (the reference too) is rerun from its own converged
orbitals, asking for ‖g‖ < 3e-12 within 8 macro iterations, at most 10
times, and the lowest ‖g‖ is kept. A rerun that does not converge keeps the
point it started from. Measured: every point reached 2.9e-12 or less, over
94, 22 and 19 reruns (`guard.polish_passes`).

### The error bar

As `../qd_nevpt2_gradient/`'s, per component, family and pair: `central`,
`richardson` (**the number to test against**), and `error_bar` =
|R(fine) − R(coarse)| + `noise` / 2h_finest.

**The noise.** It is measured per family. The reference geometry is
re-converged (and polished) from the orbitals of the three coarsest +h
points. `noise.measured` is the family's largest distance from the
reference's own value. `noise` is three times that, as in #744, but never
below a floor:

- **h families:** the case's energy noise from `../qd_nevpt2_gradient/`
  (1e-11 for H₂CO, 1e-12 for the ring).
- **Overlaps:** 3e-12. Two full runs differed by up to 5.3e-9 in skew H₂CO's
  `casscf_zeroth` at 2h = 5e-4, a point error of 2.6e-12 that the restarts
  (1.4e-13) did not show.
- **Above the floor, set per case where the data shows a larger point
  error.** These errors repeat from run to run, so they are not round-off,
  and the restarts do not see them:
  - **skew H₂CO, SC and PC `zeroth`: 2e-10.** The Cartesian ladders of
    `sc_zeroth`, pair (0, 2), along components 8 and 11 (D = −0.0142168,
    −0.0142168, −0.0142167) are flat to the noise at every step tried, and
    at a quarter of the steps the same happens along component 1. These are
    point errors of ~1e-10.
  - **ring, SC `zeroth` 1.3e-11 and PC `zeroth` 6e-12.** Here the point
    error is round-off, but each backend repeats its own (#1111, "Across
    backends" below). The noise is three times the largest point
    difference between any two of ten runs, five per card: 4.3e-12 (SC)
    and 2.0e-12 (PC).
  - **weighted, PC `zeroth`: 3e-10.** One point along direction 2 at
    h = 1.25e-4 sits 2.2e-10 off its neighbours.

| case | h: noise (measured) | `sc_zeroth` | `pc_zeroth` | `casscf_zeroth` |
|---|---|---|---|---|
| `h2co_skew_ccpvdz_sa3` | 1e-11 (≤ 5.2e-13) | 2e-10 (2.1e-13) | 2e-10 (2.9e-12) | 3e-12 (4.0e-13) |
| `h2co_skew_ccpvdz_sa3_weighted` | 1e-11 (≤ 1.5e-13) | 3.6e-12 (1.2e-12) | 3e-10 (5.9e-13) | 3e-12 (2.6e-13) |
| `h8_ring_sto3g_sa3` | 1e-12 (≤ 3.8e-15) | 1.3e-11 (4.9e-13) | 6e-12 (5.6e-13) | 3e-12 (7.4e-14) |

**Checks before writing.** Every case, family and component must pass, or
nothing is written:

- every error bar under 5e-6, a bound and not a target (the measured bars
  are below);
- the ladder ratio in (3.2, 4.8) wherever |D(2h) − D(h)| clears both 1e-7
  and 5 noise / h;
- `casscf_zeroth` within its error bar of `mcscf::df_nac`'s analytic total
  d_IJ (`use_etfs = false`) on every case;
- `zeroth` within the three bars summed of `d_nev + d_cas`;
- in full mode, each h family's translational sum within twice its largest
  error bar;
- the guard of `../qd_nevpt2_gradient/`;
- and, where `brute_force/` has the case, the cross-check below.

Richardson against the finest D is recorded (`richardson_minus_finest`) but
not bounded. The ring's `sc_zeroth` reaches 2.6e-5 there (#744's too), and
the error bar, not the finest step, is the claim.

**Measured** (largest error bar per family, directional / full; h families
`h_reference` / `h_pt2` / `h`):

| case | `casscf_zeroth` | SC h | SC `zeroth` | PC h | PC `zeroth` |
|---|---|---|---|---|---|
| `h2co_skew_ccpvdz_sa3` | 8.9e-9 / 7.3e-9 | 2.0e-8, 2.2e-8, 2.2e-8 / 2.0e-8, 7.5e-8, 7.5e-8 | 4.1e-7 / 5.6e-7 | 2.0e-8, 3.3e-8, 3.3e-8 / 2.0e-8, 3.6e-8, 3.6e-8 | 4.6e-7 / 4.8e-7 |
| `h2co_skew_ccpvdz_sa3_weighted` | 3.5e-8 | 8.0e-8, 8.4e-8, 8.3e-8 | 5.2e-8 | 8.1e-8, 3.6e-7, 3.6e-7 | 3.9e-6 |
| `h8_ring_sto3g_sa3` | 1.1e-7 | 3.4e-8, 3.6e-8, 3.6e-8 | 5.2e-7 | 3.4e-8, 3.4e-8, 3.4e-8 | 2.3e-7 |

- **The Richardson part of each bar**, |R(fine) − R(coarse)|, is at most
  1.6e-7 (skew H₂CO), 1.5e-6 (weighted, PC `zeroth`) and 8.2e-8 (ring). Per
  component it is never more than 0.39 of the consumer's bar, twice the
  error bar (#743's "How results are compared").
- **Above #744's data gate of 2e-7:**
  - skew H₂CO's SC and PC `zeroth` (up to 5.6e-7), from their 2e-10 noise;
  - the weighted case's PC `zeroth` (3.9e-6) and PC h (3.6e-7). The
    weighted case's PC point error, over the small steps its SC needs, is
    the cause.
  - the ring's SC and PC `zeroth` (5.2e-7, 2.3e-7), from the noise that
    covers both backends (below).

  So a consumer's bar there is up to 7.8e-6 (weighted PC `zeroth`) and
  1.1e-6 elsewhere. Everything else is under 2e-7.
- **Ladder ratios** where the floor allows: 3.93–4.02 on 8 directional and
  12 Cartesian components of skew H₂CO, 3.97–4.00 on 7 of the ring. The
  weighted case's rungs all sit under its floor, so its error bar is the
  evidence.

### Checks against independent code

- **QD off, against `mcscf::df_nac`** (`checks.casscf_zeroth.analytic`).
  `casscf_zeroth` is the finite difference of the roots' own overlaps, and
  the analytic total d_IJ comes from `df_nac` at the reference, every pair
  (Z-vector residuals ≤ 9.6e-11). Agreement:
  - skew H₂CO directions within 7.7e-10 (0.11 of the error bar), Cartesian
    within 9.5e-10 (0.13);
  - weighted within 1.1e-8 (0.32);
  - ring within 4.0e-9 (0.04).

  This checks the differencing, the matching and the phase alignment
  against code that is already gated. The h difference is not compared: with
  QD off, this oracle's h is the derivative of a diagonal matrix in the
  matched basis, not `df_nac`'s h_IJ.
- **Against `brute_force/`** (`brute_force_cross_check`, the ring). `zeroth`
  along the same three directions matches #744's PySCF full-space overlap
  differences within the two error bars combined: SC within 6.4e-8 (0.10 of
  the bars), PC within 1.5e-8 (0.06). The directions are identical (miss
  0.0). The pair signs, s_P s_Q read off the data (sum over components of
  ours × theirs), are [1, 1, 1] on the committed run and consistent
  (product +1) on every run. They differ from run to run because the
  solver's CI phases do.
- **Internal:** `zeroth` against `d_nev + d_cas` within 2.9e-9 on the H₂CO
  cases and 2.6e-8 on the ring (at most 0.007 of the bars summed), which
  checks Eqs. 47 + 48's split.
- **Full 3N translational sum of h** (skew H₂CO): SC within 2.9e-8 (0.20 of
  twice its family's largest error bar), PC within 3.9e-9 (0.05). The
  reference parts are within 2.4e-11. Against twice the case's largest
  committed error bar (5.6e-7) the worst is 0.03.
- **Guard** (as `../qd_nevpt2_gradient/`): the worst prediction is 0.94 h²
  (skew H₂CO), the active subspace 1.2e-6, CI 2.3e-5 and u_K 2.1e-8, all
  against 1e-3. Every PC cut is unchanged.

### Reproducibility

Not byte for byte: the tree's GPU sums reassociate. Three full runs on the
RX 9060 XT (HIP) were compared, the committed one and two earlier ones of
the same code, one of them before the overlap floor was raised. The largest
difference of any Richardson value from the committed one (phases aligned
per pair) was:

| case | difference | of its error bar |
|---|---:|---:|
| `h2co_skew_ccpvdz_sa3` | 5.0e-8 (`pc_d_nev`) | 0.86 |
| `h2co_skew_ccpvdz_sa3_weighted` | 2.6e-7 (`pc_d_nev`) | 0.42 |
| `h8_ring_sto3g_sa3` | 8.9e-8 (`sc_zeroth`) | 0.69 |

That table is the bars of the time, before the ring's overlap noise was
raised (below).

#### Across backends (#1111)

The ring's first record was made on the RX 9060 XT (HIP). On the RTX 3080
(CUDA), `sc_zeroth` missed it by up to 1.41 of its bar, at pair (1, 2),
while runs on one card agreed to within 0.40. Ten runs of one build were
compared, five per card (diagnosis runs). With the old bars:

| `sc_zeroth`, pair (1, 2), direction | 0 | 1 | 2 |
|---|---:|---:|---:|
| CUDA mean − HIP mean, in bars | +0.99 | −1.05 | −0.37 |
| the same, in standard errors | 8.3 | 18.1 | 7.3 |
| spread within CUDA, in bars | 0.59 | 0.26 | 0.24 |
| spread within HIP, in bars | 0.12 | 0.19 | 0.15 |

So the offset is **systematic**, not either card's spread. Nothing else
shows one. In every other family and pair, the two cards' means differ by
at most 0.08 of a bar, at `pc_zeroth`, whose spread on each card is up to
0.66.

**The source, point by point.** Each displaced point's `sc_zeroth` (1, 2)
differs between the cards' means by up to 3.6e-12. That is δh₁₂ /
(E₂ − E₁) at every point, to within the cards' spread, with δh₁₂ the
difference of the same point's `sc_h` and E₂ − E₁ = 4.7e-3 Eh: an error
in H_eff's off-diagonal rotates the displaced eigenvectors R(R) by
δh / gap. Up to 1.6e-14 Eh of δh₁₂ against a within-card spread of
7e-15. It is all in `sc_h_pt2`; `sc_h_reference`, the CASSCF energies,
differ by under 2e-15.

- **Not the CASSCF point.** The cards' means of `casscf_zeroth` agree to
  1.1e-13 at every point, within its run-to-run spread of 3.2e-13. The
  CASSCF energies agree to 1.5e-14.
- **The class is Srsi, ket 2.** Per class, H_eff's (1, 2) from
  `Nevpt2Output::qd_couplings` was printed at each point (three runs per
  card, temporary instrumentation, not committed). Srsi carries 89% of the
  difference: it differs by up to 1.9e-14 between the cards, against a
  within-card spread of 1.1e-15. Every other class is under 2e-15. Srsi's
  element is 5.5e-6 Eh, so this is 3.4e-9 of it.
- Within Srsi, C₁₂ carries it, at 3.3e-14 against a 1.9e-15 spread. C₁₂
  uses ket 2's denominators Δ_t + H²_t / N²_t. C₂₁ differs by only 5.5e-15.
  The transition norm Σ_t T^t₁₂ agrees to 9e-16, within its spread. So the
  numerators are not the cause. Ket 2's Srsi Rayleigh quotients are: small
  norms make η = H / N sensitive to rounding in H and N (the same
  conditioning as "Conditioning" in `src/nevpt2/energy/README.md`).
- **Why it repeats on each card.** The within-card spread is 17 times
  smaller. So the rounding is mostly deterministic: the order of a reduction
  that one backend's library always takes and the other's does not. Atomics
  would spread it within a card too.

**What it is, and the bar.** It is round-off, conditioned by the gap 4.7e-3
and ket 2's small Srsi norms. It is not a bug, and no change in the tree
removes it: one backend's reduction order is not the other's. So the bar
carries it:

- The noise is three times the largest point difference between any two of
  the ten runs, across or within cards: 4.3e-12 SC and 2.0e-12 PC, so
  1.3e-11 and 6e-12. A point error e moves a Richardson value by up to
  1.5 e / h, and the bar carries noise / 2h.
- This assumes no backend's own error is larger than the difference
  between the two. Only that difference can be measured.
- PC shows no offset, but its spread within one card, 2.0e-12, already
  passed what the 3e-12 floor covers (1e-12). It gets the same rule.

The SC `zeroth` bar becomes 5.2e-7, and PC's becomes 2.3e-7. Both are inside
the 5e-6 bound and inside every check before writing.

**Against those bars** (the committed record is a HIP run made with them):

- the ten runs: CUDA's within 0.39 of a bar and HIP's within 0.05 on
  `sc_zeroth`, and both within 0.22 on `pc_zeroth`;
- a CUDA regeneration (the acceptance run): within 0.30, at `sc_zeroth`
  (1, 2).

`../qd_nevpt2_gradient/`'s ring shows a smaller CUDA–HIP offset, 0.78 of a
bar (#1107), inside its bar.

### Visibility

Each separately coded term's largest |·| over the case's components, then
the consumer's bar (twice the largest error bar), and in brackets the
largest per-component |Richardson| / (2 × error bar). The terms are h_PQ's
reference part Σ_r ρ_r ∇E_r, its PT2 part, d^NEV = h_PQ / (E_Q − E_P) and
d^CAS. Every pair, both schemes:

| case | pair | scheme | h reference part | h PT2 part | d^NEV | d^CAS |
|---|---|---|---|---|---|---|
| `h2co_skew_ccpvdz_sa3` | (0, 1) | SC | 2.8e-5 / 4e-8 (6.9e2) | 1.7e-3 / 4.6e-8 (3.9e4) | 1.1e-2 / 3e-7 (3.9e4) | 0.27 / 1.4e-8 (2.1e7) |
| `h2co_skew_ccpvdz_sa3` | (0, 1) | PC | 9.7e-5 / 4e-8 (2.4e3) | 6.2e-4 / 4.1e-8 (1.6e4) | 4.1e-3 / 2.7e-7 (1.6e4) | 0.27 / 1.5e-8 (2.1e7) |
| `h2co_skew_ccpvdz_sa3` | (0, 2) | SC | 5.5e-3 / 4e-8 (1.4e5) | 4.3e-2 / 1.5e-7 (8.9e5) | 0.12 / 4.3e-7 (8.8e5) | 0.72 / 1.8e-8 (5.6e7) |
| `h2co_skew_ccpvdz_sa3` | (0, 2) | PC | 9.3e-3 / 4e-8 (2.3e5) | 7.1e-2 / 4.1e-8 (1.8e6) | 0.2 / 1.2e-7 (1.8e6) | 0.72 / 1.8e-8 (5.6e7) |
| `h2co_skew_ccpvdz_sa3` | (1, 2) | SC | 2.5e-4 / 4e-8 (6.2e3) | 9.0e-3 / 5.7e-8 (1.9e5) | 4.5e-2 / 2.9e-7 (1.9e5) | 0.65 / 1.5e-8 (5.3e7) |
| `h2co_skew_ccpvdz_sa3` | (1, 2) | PC | 2.5e-4 / 4e-8 (6.2e3) | 8.9e-3 / 7.2e-8 (2.2e5) | 4.5e-2 / 3.6e-7 (2.2e5) | 0.65 / 1.5e-8 (5.3e7) |
| `h2co_skew_ccpvdz_sa3_weighted` | (0, 1) | SC | 2.7e-5 / 1.6e-7 (1.7e2) | 1.3e-3 / 1.7e-7 (7.7e3) | 8.0e-3 / 1.1e-6 (7.5e3) | 2.5e-2 / 5.6e-8 (4.5e5) |
| `h2co_skew_ccpvdz_sa3_weighted` | (0, 1) | PC | 2.4e-5 / 1.6e-7 (1.5e2) | 2.2e-4 / 3.3e-7 (1.1e3) | 1.4e-3 / 2.1e-6 (1.3e3) | 2.4e-2 / 5.7e-8 (4.2e5) |
| `h2co_skew_ccpvdz_sa3_weighted` | (0, 2) | SC | 2.1e-2 / 1.6e-7 (1.3e5) | 4.6e-2 / 1.7e-7 (2.8e5) | 0.19 / 4.7e-7 (4.1e5) | 0.25 / 7.1e-8 (5.2e6) |
| `h2co_skew_ccpvdz_sa3_weighted` | (0, 2) | PC | 2.9e-2 / 1.6e-7 (1.8e5) | 3.7e-2 / 1.7e-7 (2.3e5) | 0.19 / 4.9e-7 (4.0e5) | 0.25 / 7.1e-8 (5.2e6) |
| `h2co_skew_ccpvdz_sa3_weighted` | (1, 2) | SC | 7.0e-5 / 1.6e-7 (4.4e2) | 2.4e-3 / 1.7e-7 (1.5e4) | 1.2e-2 / 8.1e-7 (1.5e4) | 4.7e-2 / 5.9e-8 (8.7e5) |
| `h2co_skew_ccpvdz_sa3_weighted` | (1, 2) | PC | 8.9e-5 / 1.6e-7 (5.5e2) | 2.7e-3 / 7.3e-7 (9.7e3) | 1.4e-2 / 3.8e-6 (9.2e3) | 4.7e-2 / 6.0e-8 (8.5e5) |
| `h8_ring_sto3g_sa3` | (0, 1) | SC | 7.5e-2 / 6.7e-8 (1.1e6) | 1.9e-2 / 6.9e-8 (2.8e5) | 0.39 / 4.9e-7 (8.4e5) | 0.36 / 2.2e-7 (1.6e6) |
| `h8_ring_sto3g_sa3` | (0, 1) | PC | 7.2e-2 / 6.7e-8 (1.1e6) | 1.8e-2 / 6.7e-8 (2.7e5) | 0.38 / 4.8e-7 (8.0e5) | 0.36 / 2.2e-7 (1.6e6) |
| `h8_ring_sto3g_sa3` | (0, 2) | SC | 5.9e-3 / 6.7e-8 (8.8e4) | 8.2e-2 / 7.2e-8 (1.2e6) | 0.55 / 4.9e-7 (1.2e6) | 0.74 / 2.6e-7 (2.9e6) |
| `h8_ring_sto3g_sa3` | (0, 2) | PC | 4.9e-3 / 6.7e-8 (7.3e4) | 8.1e-2 / 6.7e-8 (1.2e6) | 0.55 / 4.6e-7 (1.2e6) | 0.72 / 2.5e-7 (2.9e6) |
| `h8_ring_sto3g_sa3` | (1, 2) | SC | 1.5e-2 / 6.7e-8 (2.2e5) | 5.0e-2 / 6.8e-8 (7.4e5) | 10 / 1.4e-5 (7.1e5) | 1.9 / 2.5e-7 (7.7e6) |
| `h8_ring_sto3g_sa3` | (1, 2) | PC | 1.3e-2 / 6.7e-8 (2.0e5) | 5.8e-2 / 6.7e-8 (8.7e5) | 9.5 / 1.1e-5 (8.5e5) | 1.9 / 2.5e-7 (7.8e6) |

**Every term of every pair clears 10× its bar on every case**, by at least
1.5e2. The weakest is the reference part of the weighted case's pair
(0, 1). The QD-off `casscf_zeroth` clears it by at least 5e5
(`visibility[].casscf_zeroth`). No new-case issue is needed.

### Files

`coupling/<case>.json`:

| Key | Meaning |
|---|---|
| `name`, `basis`, `aux_basis`, `atoms` (bohr), `ncore`, `ncas`, `nelecas`, `active`, `weights`, `schemes` | the case |
| `steps`, `energy_noise`, `noise` (per family; `noise.measured` the restarts') | above |
| `pairs` | P < Q, the order of every per-pair list |
| `reference` | the polished R₀: `casscf_energies`, `casscf_gaps`, `sc_energies`, `pc_energies`, `sc_heff`, `pc_heff` (n × n, the roots' basis), `sc_heff_vectors`, `pc_heff_vectors` (R, column P at [P n, (P + 1) n)), `sc_heff_gaps`, `pc_heff_gaps`, `gradient_norm`, `casscf_gradient`, `inactive_gaps`, `pc_dropped`, `agreement` (with `../qd_nevpt2/`), and `nac`: per pair `mcscf::df_nac`'s `total`, `numerator`, `csf` (n_atom × 3), `iterations`, `residual` |
| `seed`, `directions` | 3 × 3N, atom-major |
| `directional`, `full` (skew H₂CO only) | per family a list over pairs of `central` (per step), `richardson`, `error_bar` (one entry per direction, or n_atom × 3 rows), and `analytic` for `casscf_zeroth`; the derived `{sc,pc}_d_nev`, `{sc,pc}_d_cas` (`richardson`, `error_bar`); `checks`; `displaced`, per component and step the [−h, +h] pair of every family's values and ‖g‖, so every difference can be redone |
| `guard` | `../qd_nevpt2_gradient/`'s, with `polish_passes` |
| `visibility` | per pair, `casscf_zeroth` and per scheme each term's `max`, `bar`, `best`, `visible` |
| `brute_force_cross_check` | the ring only, above |

### Regenerating

From a HIP or CUDA build, on a GPU, under the card's lock
(`with_gpu_card`, `devtools/lib.sh`):

```sh
build-hip/tools/qd_nevpt2_gradient/free-breakfast-qd-nevpt2-gradient \
    test/gtest/data --coupling [case ...]
```

writes the cases named, or all three, in under two minutes on the RX 9060 XT.
A refused case writes nothing and fails the run, naming the check.
