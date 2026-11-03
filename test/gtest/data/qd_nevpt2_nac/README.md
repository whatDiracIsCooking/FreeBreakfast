# qd_nevpt2_nac golden data

References for the QD-NEVPT2 derivative coupling d_PQ = ⟨Ψ_P|∇Ψ_Q⟩ (Park
2020, Eqs. 45–54), SC and PC. No file here is hand-edited.

- `brute_force/` (#744, below): the first-order wave functions built
  explicitly in the full determinant space, the PT2 transition density
  ⟨Ψ⁽¹⁾_M|E_xy|N⟩, and finite differences of the **whole** coupling and of
  d^PT2 alone. Written by `tools/emit/qd_nevpt2_nac_reference.py` (PySCF).
- The zeroth-order coupling references, h_PQ and d^NEV + d^CAS, are #743's
  (`tools/qd_nevpt2_gradient/`), written beside these.

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
**Pending**: #743's data is not yet written. When it is, `zeroth` must agree
with it per component within the two error bars combined. The two differ in
the program (PySCF here, the tree's own pipeline there), not in the
quantity: the same fitting set, weights and SA-canonical orbitals.

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
