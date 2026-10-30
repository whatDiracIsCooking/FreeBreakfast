# `test/gtest/data/`

Golden reference data the gtests read, through `FBF_TEST_DATA_DIR`, an absolute
path compiled into the test binary. These are copies, never hand-edited: each
file names the tool that wrote it and the source it came from.

| Directory | What | Written by |
|---|---|---|
| `one_electron/` | Cartesian basis shells, the nuclei, libcint's one-electron matrices (`int1e_ovlp`, `int1e_kin`, `int1e_nuc`) and PySCF's `hcore` for H₂O in STO-3G and cc-pVDZ, the H₁₀ chain in cc-pVDZ and CO in cc-pVTZ (f on both atoms, the bond off every axis), recorded by PySCF. CO's four arrays are split upstream into two pieces, `co_ccpvtz_j_st` (S, T) and `co_ccpvtz_j_vh` (V, hcore), under FelizJueves's 512 KiB hook; the tool reassembles them as its `oracle.manifest.one_e_fixture` does, refusing a piece whose basis, geometry or shared arrays differ from `co_ccpvtz_j`'s at all | `tools/extract_integral_golden.py`, from FelizJueves's `test/oracle/fixtures/` |
| `coulomb/` | The same bases and nuclei, an AO density (a Hilbert matrix, `density_recipe`) and libcint's Coulomb matrix `vj` of it, for the same four molecules, recorded by PySCF | `tools/extract_integral_golden.py`, from the same fixtures |
| `exchange/` | The same bases, nuclei and density as `coulomb/`, and libcint's exchange matrix `vk` of it (PySCF `get_jk`, `vk_il = Σ_jk (ij\|kl) D_jk`), for the same four molecules, recorded by PySCF. The tool refuses to write one unless its `_k` fixture's basis, geometry and density equal the `_j` fixture's exactly, so a K test reads the J test's density, and unless `vk` is symmetric to 1e-14 · max\|vk\|. Beside them, `h2o_sto3g_nonsymmetric.json` and `h2o_ccpvdz_nonsymmetric.json`: the same shells and nuclei, a random **non-symmetric** density (uniform on [-1, 1), numpy `default_rng(194)`), and PySCF's `get_jk(hermi=0)` of it, `vj` and `vk`, which no FelizJueves fixture carries; `nonsymmetric_jk_golden.py` refuses to write one unless its PySCF molecule has the golden's shells and reproduces the golden's own `vk` to 1e-13 · max\|vk\|, and both arrays match an explicit einsum over `int2e` | `tools/extract_integral_golden.py`, from FelizJueves's `_k` fixtures; the `_nonsymmetric` files `tools/emit/nonsymmetric_jk_golden.py`, which runs PySCF |
| `overlap_mixed/` | Two Cartesian bases, `rows` and `cols`, and libcint's cross-basis overlap `int1e_ovlp_cross` between them (PySCF `gto.intor_cross`): H₂O, rows STO-3G, columns 6-31G or cc-pVDZ at the same geometry, or 6-31G at another. The tool refuses one whose shape is not the two AO counts | `tools/extract_integral_golden.py`, from FelizJueves's `h2o_sto3g_x_*` fixtures and their row fixture `h2o_sto3g_j` |
| `record_bound/` | K's dressed record bound, FelizJueves's host `ao.k.record_bounds`, of every record of every ordered shell pair of H₂O/cc-pVDZ and the H₁₀ chain, the shells read from `exchange/` | `tools/emit/record_bound_golden.py`, which imports FelizJueves's `src/` (numpy, for its reference Boys) |
| `e_table/` | The rolled K kernel's E table, FelizJueves's host `launch_kblock.e_table` (each record's undressed `E^{la lb}`, nonzero entries), of every record of every ordered shell pair of H₂O/cc-pVDZ and CO/cc-pVTZ (through f), with each record's inputs, the shells read from `exchange/`. FelizJueves checking itself: the table has no libcint counterpart | `tools/emit/e_table_golden.py`, which imports FelizJueves's `src/` |
| `scf/` | `rhf.json`: PySCF's closed-shell RHF (spherical, `conv_tol` 1e-12) for H₂O in STO-3G and cc-pVDZ and NH₄⁺ in cc-pVDZ, run over this tree's own `data/basis` files: total energy, nuclear repulsion, orbital energies, geometry in bohr; `rohf.json`: PySCF's ROHF (`conv_tol` 1e-12, `conv_tol_grad` 1e-9) for OH in STO-3G and cc-pVDZ, NH₂, triplet CH₂ and H₂O⁺ in cc-pVDZ and triplet CH₂ in cc-pVTZ (f shells), the same fields plus `spin`, `multiplicity`, `n_alpha` and `n_beta`, orbital energies those of PySCF's Roothaan effective Fock | `tools/emit/scf_reference.py`, which runs PySCF |
| `density_fitting/` | libcint's spherical Coulomb metric `int2c2e` (P\|Q) and three-centre `int3c2e` (mn\|P) for H₂O in cc-pVDZ / cc-pVDZ-JKFIT (orbitals to d, aux to f) and cc-pVTZ / cc-pVTZ-JKFIT (O's f orbitals and g fitting functions, so (ff\|g)), PySCF's DF `vj` and `vk` of one RHF density in the cc-pVDZ case, that RHF's orbitals and PySCF's DF MO integrals over them (the ovov, oooo and vvvv blocks of (pq\|rs)), and PySCF's DF-RHF energies and orbital energies for H₂O and NH₄⁺ in cc-pVDZ / cc-pVDZ-JKFIT and H₂O in cc-pVTZ / cc-pVTZ-JKFIT, and in `df_rohf.json` PySCF's DF-ROHF for `scf/rohf.json`'s cases but the STO-3G one, over the matching JKFIT set, all over this tree's own `data/basis` and `data/aux_basis` files: `.npy` arrays and the manifest `density_fitting.json` | `tools/emit/df_reference.py`, which runs PySCF |
| `df_high_aux/` | The top of the DF aux ladder, which no carried fitting set reaches: libcint's `int2c2e_sph` (P\|Q) and `int3c2e_sph` (mn\|P) over a **hand-built** basis — one single-primitive shell per l on two skew centres, orbitals to f and auxiliary to i — the smallest basis reaching every emitted class, all 28 metric (lP ≤ lQ ≤ 6) and all 70 eri3 (la ≤ lb ≤ 3, lP ≤ 6), each by a one-centre pair and a two-centre one. Coefficients are the primitives' own normalisers, so both sides are unit-normalised and the test needs no norm factor. `.npy` arrays, `int3c2e` split along the aux index, and the manifest `df_high_aux.json`, which carries both bases shell by shell so the test builds exactly these | `tools/emit/df_high_aux_reference.py`, which runs PySCF |
| `cis/` | `cis.json`: PySCF's closed-shell RHF (`conv_tol` 1e-12) and `TDA` (CIS), over this tree's own `data/basis` and `data/aux_basis` files: the lowest five singlets and triplets and the singlets' oscillator strengths for H₂O in STO-3G, 6-31G and cc-pVDZ (exact integrals) and in cc-pVDZ / cc-pVDZ-JKFIT (density-fitted), the lowest three singlets of HF in cc-pVTZ (f shells, energies only), and the lowest five singlets and their oscillator strengths of benzene in cc-pVDZ, density-fitted and exact; every root checked against a dense diagonalization of the explicit A matrix; and `<case>_int1e_r.npy`, the spherical AO dipole for H₂O in STO-3G and cc-pVDZ | `tools/emit/cis_reference.py`, which runs PySCF |
| `localization/` | `pipek_mezey.json`: PySCF's Mulliken Pipek-Mezey, p = 2 and 4, of the RHF orbitals of `scf/`'s three cases: the functional and every localized orbital's population on every atom; `ibo.json`: the same for PySCF's IBOs, plus the IAO partial charges; `rohf.json`: both, per occupied space, of the DF-ROHF orbitals of H₂O⁺ and triplet CH₂ in cc-pVDZ / cc-pVDZ-JKFIT (`density_fitting/df_rohf.json`'s cases) | `tools/emit/localization_reference.py`, which runs PySCF |
| `active_space/` | `active_space.json`: PySCF's closed-shell RHF (`conv_tol` 1e-14, `conv_tol_grad` 1e-11) for H₂O in STO-3G and cc-pVDZ and N₂ in cc-pVDZ, over this tree's own `data/basis` files, and PySCF CASCI's active-space Hamiltonian over a chosen core and active set: E_core, h_eff_tt, (tt\|uu) and (tu\|tu) | `tools/emit/active_integrals_reference.py`, which runs PySCF |
| `casci/` | `casci.json`: PySCF's closed-shell RHF (`conv_tol` 1e-14, `conv_tol_grad` 1e-11) for H₂O in STO-3G and cc-pVDZ and N₂ in cc-pVDZ, over this tree's own `data/basis` files, and PySCF `mcscf.CASCI` over its orbitals, H₂O CAS(4,4) (frontier, and an explicit set in cc-pVDZ) and N₂ CAS(6,6), and `mcscf.avas`'s picks over MINAO (H₂O O 2p + H 1s, N₂ 2s + 2p, and N₂ 2p at threshold 0.5, in cc-pVDZ): the lowest three singlets' energies, ⟨S²⟩ and natural occupations; and over PySCF's DF-ROHF (cc-pVDZ-JKFIT) orbitals, NH₂'s doublet CAS(5,5) and CH₂'s triplet CAS(4,4), and `mcscf.avas`'s full-valence picks over them (NH₂ CAS(7,6), CH₂ CAS(6,6)) with their active orbitals up to sign | `tools/emit/casci_reference.py`, which runs PySCF |
| `ci/` | `direct_spin1.json`: active-space integrals (E_core, h_eff, (tu\|vw)) from PySCF's CASCI over its own RHF in its own basis library, and `fci.direct_spin1`'s lowest 12 roots over them, each with ⟨S²⟩ and its spin-summed 1-RDM, and state averages (`rdm12`: CI vectors with their weighted `make_rdm12` 1- and 2-RDM), for H₂O STO-3G CAS(4,4), H₂O 6-31G CAS(6,6), N₂ cc-pVDZ CAS(6,6) and CH₂O STO-3G CAS(4,4) | `tools/emit/fci_reference.py`, which runs PySCF |
| `fci/` | `fci.json`: PySCF's closed-shell RHF (`conv_tol` 1e-14, `conv_tol_grad` 1e-11) for H₂O in STO-3G and 6-31G and N₂ in STO-3G, over this tree's own `data/basis` files, and PySCF `fci.FCI` over every orbital: the lowest three singlets' energies, ⟨S²⟩ and natural occupations | `tools/emit/fci_step_reference.py`, which runs PySCF |
| `casscf/` | `casscf.json`: PySCF's closed-shell RHF (`conv_tol` 1e-14, `conv_tol_grad` 1e-11) over this tree's own `data/basis` files, and PySCF `mcscf.CASSCF` (`conv_tol` 1e-12, `conv_tol_grad` 1e-7, gated against `mc2step` from the same start, then polished to \|g\| < 1e-10) for H₂O CAS(4,4) from the RHF orbitals [2, 3, 5, 6], N₂ CAS(6,6) frontier, H₂CO CAS(4,3) frontier at two singlets weighted [0.5, 0.5] and [0.7, 0.3], and H₂O AVAS O 2p + H 1s, all cc-pVDZ (and in `casscf_df.json` alone, NH₂'s doublet CAS(5,5) and CH₂'s triplet CAS(6,6) from PySCF's DF-ROHF orbitals): at each case's start, the CASCI energies and the sign-invariant parts of `mc1step.gen_g_hop`'s gradient and orbital Hessian (doubled); converged, each root's energy, the weighted energy, each space's orbital energies and the natural occupations. **Every test starts at the case's recorded start (`active` or `avas`), not at a frontier selection** | `tools/emit/casscf_reference.py`, which runs PySCF |
| `nevpt2/` | `nevpt2_df.json`: PySCF's strongly contracted NEVPT2 (`mrpt.NEVPT`, density-fitted throughout) per root, on the polished DF (SA-)CASSCF orbitals of each `casscf/casscf_df.json` case but N₂'s, from the same starts and over the same `data/aux_basis/cc-pVDZ-JKFIT.json`: each root's eight classes' norms and energies, their sum and the total, plus the CASSCF energies it sits on and the active-rotation invariance check; and `<case>/<array>.npy`: nevpt2-gpu-demo's five density-fitted N₂ goldens (the CI vector, dm1/dm2, the integral blocks, the fitted B blocks, the external four-index blocks and the class norms and energies), with PySCF's dm3, f3ca and f3ac added for the CAS(4,4) and CAS(8,8) cases, described in `nevpt2/README.md` | `tools/emit/nevpt2_reference.py`, which runs PySCF; `tools/emit/nevpt2_golden.py`, from the demo's goldens and PySCF |

## `one_electron/*.json`, `coulomb/*.json` and `exchange/*.json`

| Key | Meaning |
|---|---|
| `source` | The FelizJueves commit and fixture this was copied from, and for a one-electron file reassembled from pieces, the pieces (`pieces`) |
| `provenance` | The fixture's own: PySCF, libcint and numpy versions, date, the molecule as PySCF was given it, the basis name |
| `nao` | Cartesian AO count |
| `atoms` | The nuclei, in PySCF's atom order: `symbol`, `charge` (Z), `coord` (bohr) |
| `shells` | One per contraction, in AO order: `l`, `centre` (bohr), `exponents`, `coefficients`. The coefficients multiply **unnormalised** Cartesian primitives `x^a y^b z^c exp(-alpha r^2)`, so they go into a `Shell` as given, with no `primitive_normalized` |
| `arrays` | `{name: {shape, values}}`, `values` row-major in AO order |
| `density_recipe` | `coulomb/` and `exchange/` only: how the fixture made its density |

An `exchange/*_nonsymmetric.json` file has the same keys but for `source`,
which names the exchange golden its shells were copied from and the PySCF
call, and `provenance`, the PySCF and numpy versions, the molecule, the basis
name and `cart`; its `arrays` are `density`, `vj` and `vk`. Regenerate it with
`uv run --group tools tools/emit/nonsymmetric_jk_golden.py`, after the tool
below.

Regenerate after bumping `deps/FelizJueves`:

```sh
tools/extract_integral_golden.py
```

## `overlap_mixed/*.json`

| Key | Meaning |
|---|---|
| `source` | The FelizJueves commit, the cross fixture, and the row fixture (`rows_fixture`) |
| `provenance` | The cross fixture's own, the columns' molecule and basis |
| `rows`, `cols` | Each basis as `nao` and `shells`, the shells as in `one_electron/` |
| `int1e_ovlp_cross` | `{shape, values}`: `rows.nao` x `cols.nao`, row-major, each basis's AO order |

Written by the same `tools/extract_integral_golden.py` run.

## `record_bound/*.json`

| Key | Meaning |
|---|---|
| `source` | The FelizJueves commit, the reference call, and the `exchange/` file whose shells it used |
| `n_shells`, `n_records` | The basis's shell count, and the records over every ordered shell pair |
| `pairs` | One per ordered shell pair: `a` and `b`, the first and second shell's index in basis order, and `records`, each `[zeta, bound]` in FelizJueves's record order. A test matches a record by its two shells and its zeta |

Regenerate after `tools/extract_integral_golden.py`:

```sh
uv run --group tools tools/emit/record_bound_golden.py
```

## `e_table/*.json`

| Key | Meaning |
|---|---|
| `source` | The FelizJueves commit, the reference call, and the `exchange/` file whose shells it used |
| `n_shells`, `n_records` | The basis's shell count, and the records over every ordered shell pair |
| `classes` | One per class `(la, lb)` reached: `width`, a record's table length, and `odd`, per entry `1` where its Hermite `\|t\|` is odd: the ket's table (`e_table(..., sign=True)`) is the unsigned one with exactly those negated, which the tool asserts of every record |
| `pairs` | One per ordered shell pair: `a` and `b`, the shells' index in basis order, `la` and `lb`, `records`, each `[zeta, X_PA, Y_PA, Z_PA, X_PB, Y_PB, Z_PB]` (P − A and P − B as `mmd.hermite.e_block` forms them) in FelizJueves's record order, and `e`, each record's unsigned table. A test uploads the records as they are, so it compares the table alone |

Regenerate after `tools/extract_integral_golden.py`:

```sh
uv run --group tools tools/emit/e_table_golden.py
```

## `active_space/active_space.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `cases` | One per (molecule, basis): `name`, `basis`, `charge`, `atoms` (`symbol`, `z`, `coord` in bohr), `nao` (spherical), `electrons`, `energy` (the RHF total, hartree), `mo_energies` (ascending), and the active space below |
| `ncore`, `ncas`, `orbitals` | The canonical orbitals, by index in ascending energy, in CASCI's column order: `orbitals[:ncore]` the core, the next `ncas` the active space, then the rest. A test reorders its own Scf's columns the same way |
| `e_core` | `get_h1eff`'s constant: the core energy, nuclear repulsion included |
| `h_eff_diagonal` | h_eff_tt, one per active orbital |
| `coulomb`, `exchange` | (tt\|uu) and (tu\|tu), row t, column u |

Only elements no orbital's sign changes are recorded. Each active space keeps
every degenerate set whole and holds at most one (the tool refuses otherwise),
so a rotation within that set is a symmetry operation and leaves them unchanged
too: N₂'s core takes the 1π_u pair and leaves 3σ_g, below it, to the active
space. Regenerate with:

```sh
uv run --group tools tools/emit/active_integrals_reference.py
```

## `scf/rhf.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `cases` | One per (molecule, basis): `name`, `basis` (the builtin name `resolve_basis` takes), `charge`, `atoms` (`symbol`, `z`, `coord` in bohr), `nao` (spherical), `electrons`, `energy` and `nuclear_repulsion` (hartree), `mo_energies` (hartree, ascending) |

Only quantities invariant under the AO normalisation are recorded, since
PySCF's spherical AOs and this tree's are normalised differently. PySCF is
handed the basis from `data/basis/*.json` itself, not its own library's copy, so
the two sides compute over one basis. Regenerate with:

```sh
uv run --group tools tools/emit/scf_reference.py
```

## `density_fitting/`

`density_fitting.json` is the manifest: it names every `.npy` beside it and
what it holds. Each `.npy` is `'<f8'` in **Fortran order**, the layout
`src/io/npy` reads, so a file loads into an `NpyArray` (first axis fastest)
with no reordering, and with a plain `np.load` in Python.

| Key | Meaning |
|---|---|
| `provenance` | PySCF, libcint (PySCF's `cint.h`) and numpy versions, `threads` (BLAS and OpenMP pinned to it), the command that wrote it. No date or host, so a rerun is byte-identical |
| `integrals` | One per case: `name`, `basis` and `aux_basis` (this tree's builtin names), `atoms` (`symbol`, `z`, `coord` in bohr), `nao` and `naux` (spherical), `shells` and `aux_shells`, `int2c2e`, `int3c2e`, and on `h2o_ccpvdz` also `jk` |
| `shells`, `aux_shells` | One per PySCF shell, in AO order: `atom` (index into `atoms`), `l`, `nctr`. A shell of `nctr` contractions holds `nctr` × (2l + 1) AOs, each contraction's m = −l..l together, in the BSE file's column order: the order this tree's reader gives by making each coefficient column a shell of its own. Within a shell, m runs −l..l (p: x, y, z), the libcint order `spherical_transform_matrix` follows |
| `int2c2e` | The file of (P\|Q): `naux` × `naux` |
| `int3c2e` | The pieces of (mn\|P), each `{file, aux: [begin, end)}`, in P order |
| `jk` | `density` (D, `nao` × `nao`), `vj` and `vk` (each `nao` × `nao`), `occupied` (5) and `df_rhf_energy` |
| `df_rhf` | One per DF-RHF: `name`, `basis`, `aux_basis`, `charge`, `atoms`, `nao`, `naux`, `electrons`, `energy` and `nuclear_repulsion` (hartree), `mo_energies` (hartree, ascending) |

**The (mn\|P) layout.** Each `int3c2e` piece is an array of shape
`(nao, nao, end − begin)`, element `[m, n, P − begin]` = (mn\|P), so the
logical order is `[m][n][P]` with **m fastest and P slowest**: (mn\|P) sits at
flat offset `m + nao · (n + nao · (P − begin))`. That is PySCF's native
`df.incore.aux_e2(..., aosym="s1")` layout, an F-contiguous
`(nao, nao, naux)`, written as it comes: each P is one column-major
`nao` × `nao` matrix, symmetric in m and n. A piece is a contiguous run of
those slabs. The tensor is split along P into the fewest balanced pieces that
each fit under the 512 KiB limit, as FelizJueves's metric fixture is split by
rows: `h2o_ccpvdz` (24 × 24 × 116) in two, `h2o_ccpvtz` (58 × 58 × 139) in
eight. Concatenating the pieces along the last axis in manifest order gives
the whole tensor.

**The density.** D = 2 C Cᵀ, C the five occupied orbitals of PySCF's own
converged DF-RHF (`conv_tol` 1e-12) of the `h2o_ccpvdz` case, at that case's
geometry and over the same fitting set: symmetric positive semidefinite of
rank 5, so a test can factor it back into occupied orbitals. `vj` and `vk` are
`RHF(mol).density_fit(auxbasis=...).get_jk(mol, D, hermi=1)`:
J_mn = Σ (mn\|P) (P\|Q)⁻¹ (Q\|ls) D_ls and
K_mk = Σ (mn\|P) (P\|Q)⁻¹ (Q\|ks) D_ns. The tool checks both against those sums
over the very `int2c2e` and `int3c2e` it writes, to 1e-10 relative.

**Normalisation.** Every orbital and fitting function is PySCF's: unit
self-overlap (the tool asserts `int1e_ovlp`'s diagonal is 1 to 1e-12 in both
bases). A test whose function i is N_i times PySCF's scales an array by N_i per
index (and D by 1 / (N_m N_n)), N_i read from its own overlap diagonal, as
`sad/` says. Energies and orbital energies need no such care.

**Geometries.** The integrals and `jk` use FelizJueves's `h2o_ccpvdz_j` water
(O at the origin, H at (1.43, 0.21, −1.07) and (−1.38, 0.35, −1.12) bohr: in no
symmetry plane, so every component of every shell is reached), the molecule of
its cc-pVTZ-JKFIT metric fixtures. `df_rhf` uses `scf/rhf.json`'s geometries,
so its energies sit beside the exact-integral ones there (the DF error is
2.1e-5 Eh for H₂O and −1.05e-6 Eh for NH₄⁺ in cc-pVDZ).

Regenerate with (one thread; the output is byte-identical run to run):

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/df_reference.py
```

## `df_high_aux/`

`df_high_aux.json` is the manifest, and the arrays are `'<f8'` in Fortran
order and split along P exactly as `density_fitting/`'s are. What differs is
the **basis**, and why there is a second directory at all: the two fitting
sets this tree carries stop at aux g, so `density_fitting/` reaches `(g|g)`
and `(ff|g)` and no further, while the emitted kernels go to `(i|i)` and
`(ff|i)` (`kMetricMaxMomentum`, `kThreeCenterAuxMaxMomentum`). No carried set
has an i fitting shell, so those classes have no library basis to be run
over.

So this one is hand-built: **one single-primitive shell per l**, orbitals
l = 0..3 and auxiliary l = 0..6, on each of two skew centres (bohr). That is
the smallest basis reaching every class both families emit — all 28 metric
`lP ≤ lQ ≤ 6` and all 70 eri3 `la ≤ lb ≤ 3` against `lP ≤ 6` — and it reaches
each of them twice over, once with the pair on one centre (so `P − A` and
`P − B` vanish and the Boys argument is 0) and once with a real separation.
`df_high_aux_gpu.cpp` checks that coverage as a case of its own.

| Key | Meaning |
|---|---|
| `source` | libcint's version (PySCF's `cint.h`) and the two intors used |
| `what` | One line on what the basis is and why it is not a library set |
| `atoms` | The two centres: `symbol`, `z`, `coord` in bohr. Nothing here is a nuclear integral, so the symbols only keep PySCF from sharing one basis between the centres |
| `nao`, `naux` | Spherical counts, 32 and 98 |
| `orbital_max_l`, `aux_max_l` | 3 and 6, the two ladders' limits |
| `orbital_shells`, `aux_shells` | One per shell, in AO order (centre-major, l ascending — PySCF's order and this tree's for the same `Shell` list): `atom`, `l`, `centre`, `exponent`, `coefficient`. The test builds its `BasisSet`s from these, so the two sides cannot drift |
| `int2c2e` | The file of (P\|Q): 98 × 98 |
| `int3c2e` | The pieces of (mn\|P), each `{file, aux: [begin, end)}`, in P order |

**Normalisation, and why this fixture needs no norm factor.** Each
`coefficient` is the primitive's own normaliser `N(alpha, l)` — the one
`ShellStamp::primitive_normalized` applies — so a `Shell` built from it has
unit self-overlap, matching PySCF's own unit-normalised functions. Both
arrays are then comparable element for element, unlike
`df_three_center_gpu.cpp`'s, which divides by its carried basis's function
norms. The test asserts the self-overlaps are 1, so a wrong coefficient in
the manifest fails there rather than as a mysterious tolerance miss.

Regenerate with (one thread; the output is byte-identical run to run):

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/emit/df_high_aux_reference.py
```

## `cis/cis.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, threads, the TDA's Davidson `tda_conv_tol` (a residual norm) and `tda_lindep`, the command |
| `cases` | One per (molecule, basis, integrals): `name`, `basis`, `aux_basis` (`null` for exact integrals, else the fitting set both the RHF and the TDA use), `atoms` (`symbol`, `z`, `coord` in bohr), `nao` (spherical), `occupied`, `electrons`, `rhf_energy`, `nuclear_repulsion`, `singlets`, `triplets` (where computed) and `int1e_r` (where written) |
| `energies` | Excitation energies (hartree), ascending, PySCF's Davidson roots |
| `groups` | The roots as indices into `energies`, split into degenerate groups (1e-6 hartree apart at most; the gap to the next group is checked over 1e-4) |
| `oscillator_strengths` | Singlets only, and not HF's: one per **group**, the group's summed length-gauge f = ⅔ ω Σₓ \|⟨0\|x\|n⟩\|², ⟨0\|x\|n⟩ = √2 Σᵢₐ xᵢₐ Xᵢₐ with X unit-normalised (PySCF's convention). A triplet's is zero by spin and is not recorded |
| `dense_max_error`, `dense_max_oscillator_error` | The largest difference between PySCF's roots and the dense A's lowest eigenvalues, and between PySCF's group f and the one from the dense eigenvectors |
| `int1e_r` | The `.npy` beside it: libcint's `int1e_r_sph`, (3, nao, nao) in x, y, z order, origin at 0, '<f8' Fortran order, PySCF's spherical AO order (the order `density_fitting/` uses) |

**What a test can check invariantly.** Benzene's E states and HF's Π come
in degenerate pairs, whose individual eigenvectors, and so whose individual
oscillator strengths, are any rotation within the pair: only the pair's sum
is defined. So every root list is recorded in whole groups, and f per group;
"the lowest five" of benzene is six roots (two dark singles, the bright E₁ᵤ
pair, a dark pair), and HF's "lowest three" is the Π pair and a Σ. A test compares each
energy by index and each group's summed f. The transition dipole is origin
independent (occupied and virtual orbitals are S-orthogonal), so a test may
use `int1e_r` at origin 0 whatever origin PySCF used internally. A dark
state's recorded f is rounding (1e-25 or below), so compare it absolutely.

**The cross-check.** The issue named a private tree's
`h2o_{sto3g,631g}_cis_*.json` as an independent check; it is not reachable
here, so the emitter checks itself instead: for every case and spin it
builds A = δ(εₐ − εᵢ) + 2(ia\|jb) − (ij\|ab) (singlet; the triplet drops
2(ia\|jb)) from PySCF's own MO integrals (exact `ao2mo`, or the DF
object's `ao2mo` over the same fitting set), diagonalizes it densely, and
refuses to write unless PySCF's roots equal its lowest eigenvalues to 1e-8
(measured: 3e-14 or better everywhere) and the group f agree to 1e-6
(measured: 7e-8 at benzene's bright E₁ᵤ pair, 2e-9 or better elsewhere). The
Davidson run is sized from the dense spectrum, three roots past the wanted
ones and then to a group boundary, because a run that cuts a degenerate pair
stalls; the roots themselves are PySCF's.

**Tolerance.** The Davidson residual is converged to 1e-7, so the energies
are PySCF's to ~1e-14, and the oscillator strengths to ~1e-7 (they are
quadratic in the vectors, which are good to the residual over the gap).
DF moves the H₂O energies by up to ~1e-5 and benzene's by up to ~5e-5
against the exact ones: a DF sigma is held to the DF case, never to the exact one.

Regenerate with (one thread; byte-identical run to run, about two minutes,
nearly all benzene):

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/cis_reference.py
```

## `localization/pipek_mezey.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `cases` | One per (molecule, basis), as in `scf/rhf.json`: `name`, `basis`, `charge`, `atoms` (`symbol`, `z`, `coord` in bohr), `electrons`, and `localized` |
| `localized` | One per exponent: `exponent` (2 or 4), `functional` (sum_i sum_A (Q^A_ii)^p), and `populations`, Q^A_ii with one row per localized orbital, in no particular order, and one column per atom, in `atoms` order |

The orbitals themselves are not recorded: PySCF's AOs are normalised and
ordered differently from this tree's, and an atom's population is invariant
under any transformation within its AOs, so the populations are what both
codes can compare. PySCF's optimizer stops at any stationary point, which from
a symmetric molecule's canonical orbitals is often a saddle; the tool
alternates it with PySCF's Jacobi stability check until no uphill rotation is
left. Regenerate with:

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/localization_reference.py
```

## `localization/ibo.json`

Written by the same run, from the same RHF orbitals: PySCF's IAOs over MINAO
(`lo.iao`, its table being `data/basis/minao.json` bit for bit),
Loewdin-orthonormalized, and its Pipek-Mezey over their populations
(`lo.ibo.PipekMezey`), driven to a maximum as above.

| Key | Meaning |
|---|---|
| `provenance`, `cases` | As in `pipek_mezey.json`, plus `iao_charges` per case |
| `iao_charges` | Z_A - 2 sum_i sum_{r in A} (A^T S C)_ri^2 per atom, in `atoms` order: the IAO partial charges of Knizia's eq 3 |
| `localized` | One per exponent: `exponent`, `functional` (sum_i sum_A of the IBO's IAO population on A, to the p), and `populations`, each IBO's *Mulliken* population on each atom, as in `pipek_mezey.json`, for the convention-free match |

## `localization/rohf.json`

Written by the same run: PySCF's DF-ROHF, as `tools/emit/df_reference.py`
runs it (`conv_tol` 1e-12, internally stable, over the same orbital and
fitting sets), of a doublet (H₂O⁺) and a triplet (CH₂), whose doubly occupied
(`mo_occ` 2) and singly occupied (`mo_occ` 1) orbitals are localized each
within its own space, by both methods, driven to a maximum as above. The IAOs
are those of every occupied orbital, so they span both spaces.

| Key | Meaning |
|---|---|
| `provenance` | As in `pipek_mezey.json` |
| `cases` | One per molecule: `name`, `basis`, `aux_basis`, `charge`, `multiplicity`, `atoms`, `electrons`, `n_alpha`, `n_beta`, `pipek_mezey`, `ibo` and `iao_charges` |
| `pipek_mezey`, `ibo` | One per exponent: `exponent`, then `doubly` and `singly`, each with that space's `functional` (over Mulliken populations for `pipek_mezey`, IAO ones for `ibo`) and `populations` (Mulliken, one row per localized orbital, as in `pipek_mezey.json`) |
| `iao_charges` | Z_A minus twice the doubly and once the singly occupied orbitals' IAO population on A, per atom |

The run also rewrites `pipek_mezey.json` and `ibo.json`, whose values move
in the last digit or two from run to run (BLAS reductions are not pinned
here); a change that only adds cases keeps the committed ones.

## `ci/direct_spin1.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `cases` | One per active space: `name`, `basis` (PySCF's library name), `atoms` (`symbol`, `z`, `coord` in bohr), `n_orbitals`, `n_alpha`, `n_beta`, `e_core` (hartree), `h1` (`{shape, values}`, n × n, row-major), `eri` (`{shape, values}`, n⁴, row-major (t, u, v, w), chemists' (tu\|vw)), `roots`, and `rdm12` |
| `roots` | The lowest 12 of the M_s = 0 sector, ascending: `energy` (hartree, `e_core` included), `s2` (⟨S²⟩), `degenerate` (another root lies within 1e-8, so this root's 1-RDM depends on the solver's rotation within the level), and `rdm1` (`{shape, values}`, the spin-summed γ_tu, n × n) |
| `rdm12` | State averages: the ground state at weight 1, and for H₂O STO-3G CAS(4,4) also its two lowest singlets at [0.7, 0.3]. Each has `roots` (indices into `roots`, none degenerate), `weights`, `vectors` (`{shape, values}`, one row per root, alpha-major like `direct_spin1`'s), `rdm1` (Σ_r w_r γ^(r), n × n), `rdm2` (Σ_r w_r of `make_rdm12`'s Γ_pqrs = ⟨p† r† s q⟩ spin summed, n⁴ row-major) and `rdm2_from_rdm12s` (the same from `make_rdm12s` summed over the spins; the tool asserts the two agree to 1e-12). A test reads the RDMs of exactly these vectors, so no solver's sign or rotation enters |

Everything is in PySCF's own MO basis, so nothing from this tree's AOs enters
and there is no ordering or sign to reconcile. The roots are `direct_spin1`'s
with no spin constraint, so the triplets' M_s = 0 components are among them;
every case has three singlets within its 12, and the 12th root is not split
from the 13th. Water's STO-3G has only 7 MOs, so its CAS(6,6) is in 6-31G.
CH₂O is the case chosen for a triplet below the second singlet (T₁ under S₁,
n → π*), and the tool asserts it has one; the other three have one too. The
tool also checks the roots against a dense diagonalisation of the sector.
Regenerate with:

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/fci_reference.py
```

## `casci/casci.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `cases` | One per active space: `name`, `basis` (this tree's builtin name), `charge`, `atoms` (`symbol`, `z`, `coord` in bohr), `electrons`, `ncas`, `nelecas`, `ncore`, `active` (the 0-based MO indices, or `null` for the frontier orbitals), `avas` (`labels` and `threshold` of an AVAS case, whose `ncas` and `nelecas` are `mcscf.avas`'s, or `null`), `rhf_energy`, `mo_energies` (ascending), `e_core` (hartree, nuclear repulsion included), and `roots`; an ROHF case adds `scf` (`"rohf"`), `multiplicity` and `aux_basis` (`name`, `file`, its `sha256`, `naux`), and its `rhf_energy` and `mo_energies` are the DF-ROHF's (the Roothaan effective Fock's eigenvalues); the ROHF AVAS cases add `active_overlaps`, below |
| `active_overlaps` | Per active orbital, in column order, \|⟨SCF orbital k\|active orbital⟩\| over every k (nmo values): PySCF's AVAS orbitals as columns of R over the SCF's, up to sign |
| `roots` | The lowest three singlets (one for N₂'s AVAS CAS(10,8), whose excited roots `ci::solve` does not converge at its defaults), or for a case with a `multiplicity` (2S + 1, absent for a singlet: H₂O cc-pVDZ's triplet and the ROHF cases) the lowest three roots of that S in the M_s = S sector, ascending: `energy` (hartree, total), `s2` (⟨S²⟩), `degenerate` (another root of the sector lies within 1e-8), and `natural_occupations` (the spin-summed 1-RDM's eigenvalues, descending) |

**The ROHF cases** (`"scf": "rohf"`: NH₂ cc-pVDZ CAS(5,5) at multiplicity 2,
CH₂ ³B₁ cc-pVDZ CAS(4,4) at 3, both frontier) are over PySCF's
density-fitted ROHF at the case's multiplicity (`conv_tol` 1e-14,
`conv_tol_grad` 1e-10), fitted over this tree's
`data/aux_basis/cc-pVDZ-JKFIT.json` as parsed shells, which is the open shell
the Scf step runs. The tool checks the ROHF's occupation is aufbau, its core
doubly occupied and every singly occupied orbital active (an index case's
only). The CASCI itself
is exact: the orbitals are handed to an unfitted ROHF object, so its core J
and K and its active integrals are, as the Casci step builds them.

**The ROHF AVAS cases** (`nh2_ccpvdz_rohf_avas`, N 2s + 2p + H 1s, CAS(7,6);
`ch2_ccpvdz_rohf_avas`, C 2s + 2p + H 1s, CAS(6,6); both at threshold 0.2)
run `mcscf.avas` over the same DF-ROHFs at its default `openshell_option` 2:
the occupied orbitals it projects are the first n_alpha, singly occupied
ones among them, and nelecas is N − 2 ncore. They record `active_overlaps`,
which no orbital's sign moves; the tool refuses it when two SCF orbitals, or
two canonical active ones, lie within 1e-6, since a rotation within the pair
would move it.

The 1-RDM itself is not recorded: it depends on the signs of the active
orbitals, which the two codes pick independently, and its eigenvalues do not.
A degenerate root's 1-RDM depends on the solver's rotation within the level
too, so a test skips its occupations. No active space splits a degenerate set
of orbitals; the tool refuses one that does. An AVAS case runs PySCF's
`mcscf.avas` over this tree's `data/basis/minao.json`, and the tool refuses
one with a projection weight within 1e-3 of its threshold. The roots are `direct_spin1`'s
lowest 12 of the M_s = S sector, checked against a dense diagonalisation of
it, with the roots of spin S kept. Regenerate with:

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/casci_reference.py
```

A full rerun moves the recorded cases in their last digits (threaded BLAS);
`--append` keeps the file's cases byte for byte and computes only the ones
the tool's `CASES` adds, which is how the ROHF cases, and the ROHF AVAS
ones, were written.

## `fci/fci.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `cases` | One per molecule: `name`, `basis` (this tree's builtin name), `charge`, `atoms` (`symbol`, `z`, `coord` in bohr), `electrons`, `nmo`, `determinants` (C(nmo, electrons / 2)²), `rhf_energy`, and `roots` |
| `roots` | The lowest three singlets, or for a case with a `multiplicity` (2S + 1, absent for a singlet: H₂O STO-3G's triplet) the lowest roots of that S in the M_s = S sector, ascending: `energy` (hartree, total), `s2` (⟨S²⟩), `degenerate` (another of the lowest 12 roots of the sector lies within 1e-8), and `natural_occupations` (the spin-summed 1-RDM's eigenvalues, descending) |

As in `casci/`, the 1-RDM itself is not recorded, and a degenerate root's
occupations are skipped by a test. Unlike a CASCI's, an FCI energy and its
natural occupations do not depend on the orbitals at all, only on the basis.
The roots are the lowest 12 of the M_s = 0 sector, with the singlets kept: by a
dense diagonalisation up to 20,000 determinants, and above that by
`direct_spin1`'s Davidson, run twice from PySCF's guesses spread by differently
seeded noise, the two agreeing to 1e-10. Regenerate with:

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/fci_step_reference.py
```

## `casscf/casscf.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it, and `threads` (BLAS and OpenMP are pinned to it) |
| `cases` | One per CASSCF: `name`, `basis` (this tree's builtin name), `charge`, `atoms` (`symbol`, `z`, `coord` in bohr), `electrons`, `nmo`, `ncas`, `nelecas`, `ncore`, the start (`active` and `avas`, below), `weights` (one per root, the lowest singlets), `rhf_energy`, `mo_energies` (the RHF's, ascending), `casci` and `casscf` |
| `active` | The start, as 0-based indices into the RHF orbitals in ascending energy, as `sort_mo` takes them: the core is the remaining occupied orbitals, lowest first, then these in the given order, then the remaining virtuals. `null` for the frontier orbitals (the `ncas` above the lowest `ncore`) or an AVAS start |
| `avas` | `labels` and `threshold` of an AVAS start (PySCF `mcscf.avas` over `data/basis/minao.json`, its three spaces canonicalized), or `null` |
| `casci` | At the start, before any rotation: `orbital_energies` (the RHF Fock's diagonal over the start's columns, core, active, virtual, which is how a test matches its own orbitals to these), `energies` (each root's CASCI energy, by a dense diagonalisation of the M_s = 0 sector, singlets kept), `energy` (weighted), `pairs`, `gradient_norm` (\|g\|), `g_hessian_g` (gᵀHg) and `hessian_g_norm` (\|Hg\|) |
| `pairs` | One per pair of orbital sets (P, Q) holding a non-redundant rotation, P in a later space than Q, in gen_g_hop's packed order: `p` and `q` (the column indices of each set; one column unless a degenerate set), `gradient` (sqrt Σ g_pq² over p in P, q in Q) and `hessian_diagonal` (Σ H_pq,pq over the same) |
| `casscf` | Converged, at the polished orbitals: `energies` (each root's total energy), `energy` (weighted), `orbital_energies` (`core`, `active`, `virtual`: the eigenvalues, ascending, of the state-averaged Fock `mc.get_fock` within each space), `natural_occupations` (the weighted 1-RDM's eigenvalues, descending), and `gradient_norm` (\|g\| there, below 1e-10) |

**The starts deviate from issue #244**, which asked for the frontier
selection throughout. The tool keeps only a start from which `mc1step` and
`mc2step` reach the same minimum (the weighted energy to 1e-9, `mc.converged`
set by both), run after run at 8 threads. Water's frontier CAS(4,4) failed:
its active space has to trade the 1b1 lone pair for the 1b2 O–H bond, so
`mc2step` stalls or, at two roots, converges 6 mEh up. So water starts from
the σ O–H orbitals [2, 3, 5, 6] (1b2, 3a1, 4a1, 2b2). N₂'s frontier CAS(6,6)
at two roots landed 20 mEh up in about one run in three. LiH's CAS(2,4) has a
zero orbital-Hessian eigenvalue at its minimum. So the two-root cases are
formaldehyde's frontier CAS(4,3) (π, n, π*: the ground state and the n → π*
singlet). The later tests (generalized Fock and gradient, Hessian, the SA
loop and the Casscf step) must start at each case's recorded `active` or
`avas`, not at a frontier selection of their own.

g and H are the derivatives of E(C exp(κ)) at fixed RDMs, κ_pq = x_pq and
κ_qp = −x_pq for each packed pair p > q:
E = E₀ + g·x + ½ xᵀHx + …, so g_pq = 2(F_pq − F_qp) with F the generalized
Fock. PySCF's `gen_g_hop` returns half of each (g, `h_op` and `h_diag`); the
tool doubles them, and checks g and H against central finite differences of
the energy and h_diag against H's own diagonal. Only quantities invariant
under each orbital's sign are recorded, and under a rotation within a
degenerate set (N₂'s π pairs), which is why a set is the unit in `pairs`.
The tool refuses a degenerate set that straddles two spaces.

PySCF's solvers run with fcisolver `conv_tol` 1e-14 and `conv_tol_residual`
1e-9, `ah_start_tol` 1e-10 and `ah_conv_tol` 1e-14. At its defaults `mc1step`
stalls near \|g\| = 3e-7 and never sets `mc.converged` at `conv_tol_grad`
1e-7. A run that stalls just above it is restarted from where it stopped, at
most twice. At 1e-7 the orbitals, and so the orbital energies, still move by
about 5e-7 from run to run, and `mc2step` does not converge at 1e-8. So the
converged results are taken after a polish: Newton steps over the exact
orbital Hessian, with the CI re-solved densely at every step, to \|g\| < 1e-10.
The tool checks that the polish leaves the weighted energy within 1e-9 of
`mc1step`'s, and that H's least eigenvalue there is above 1e-3. Each root's
energy is not stationary, so `mc1step`'s and `mc2step`'s roots agree only to
a few 1e-8 (3.4e-8 for formaldehyde); the polished ones do not depend on that.

The output is not bit-for-bit reproducible: threaded reductions differ from
run to run. Over 13 runs, every stored value agreed with the others to within
2.2e-10 (the largest drift, in N₂'s converged orbital energies; energies
and natural occupations moved by at most 3e-12). The stated tolerance is 1e-9: a regeneration that
moves any value by more than that is a change, not noise. Regenerate with:

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/casscf_reference.py
```

## `nevpt2/nevpt2_df.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF, block2 and numpy versions, `threads` (1: BLAS and OpenMP pinned to it), the command that wrote it, and `casscf`, the file whose cases these are. No date or host, so a rerun is byte-identical |
| `invariance` | The check the tool makes of every root: `check` (what was compared), `tolerance` (1e-10) and `max_difference` (the largest disagreement over all cases and roots, 3.1e-15); and the same of PC, `pc_check` and `pc_max_difference` (1.8e-14) |
| `block2` | The gates on the PC fields: `sc_tolerance` (1e-10, block2's SC against PySCF's per class, and PC Sijrs against SC Sijrs), `pc_tau` (1e-13) and `pc_min_gap` (1e3), the cut and gap every PC solve must clear |
| `cases` | One per `casscf_df.json` case but `n2_ccpvdz` (below), by the same `name`: `ncas`, `nelecas`, `ncore`, `weights`, `aux_basis` (`name`, `file`, its `sha256`, `naux`), `casscf_energies` (each root's energy at the polished orbitals) and `casscf_energy` (weighted), and `roots` |
| `roots` | One per state, lowest first: `casci_energy` (the root's energy over the polished orbitals, the zeroth order), `classes`, `e_corr` (their energies' sum, the NEVPT2 correction) and `energy` (`casci_energy` + `e_corr`), and `pc_class_energies` (eight `name`/`energy` pairs in the same order, block2's PC-NEVPT2) and `e_pc_total` (their sum) |
| `classes` | Eight, in PySCF's summation order: `name` (PySCF's: `Sr`, `Si`, `Sijrs`, `Sijr`, `Srsi`, `Srs`, `Sij`, `Sir`), `excitation` (the change in active-electron count PySCF's log gives it, primed for the two semi-internal classes), `norm` (the class's perturbers' summed squared norm) and `energy` (its second-order contribution). A class can be empty to rounding (H₂CO's excited root's `Sij`: norm ±1e-17, energy `-0.0`), so compare norms and energies absolutely |

**The cases and the orbitals.** The tool imports `casscf_reference.py`'s
cases, its molecule and fitting-set parsing (this tree's
`data/aux_basis/cc-pVDZ-JKFIT.json` as parsed shells, never PySCF's library
`cc-pvdz-jkfit`, with the aux molecule checked against the file shell for
shell), its start selection and its CASSCF settings, runs the DF `mc1step`
from the same start and polishes it by the same Newton steps to
\|g\| < 1e-10, and refuses a run whose polished energies are more than 1e-9
(that file's stated drift) from `casscf_df.json`'s. NEVPT2 then runs over a
multi-root CASCI on those orbitals, as PySCF's
`examples/mrpt/41-for_state_average.py` does (`mcscf.CASCI`, `nroots` the
case's state count, `fix_spin_(ss=0)`), then `mrpt.NEVPT(casci, root=r)` per
root, each from a deep copy of the CI (`NEVPT.kernel` writes its
canonicalized vector back into the shared list).

**Everything is fitted.** The CASCI is made from the exact RHF's converged
orbitals wrapped as a DF-RHF over the same fitting set (`mf.density_fit`,
not re-run), so it is PySCF's DFCASCI and NEVPT takes its DF branch
(`dfnevpt2._ERIS`): the MO integrals, the core Fock and the canonicalizing
Fock all over the fitted integrals. NEVPT falls back to exact integrals
silently when it judges memory short; the tool counts its `_ERIS` calls and
refuses a root that did not take the DF branch.

**The CI.** The CASCI's own spin-penalised Davidson stalls short of the
eigenvectors (N₂: residual 8e-7, `fcisolver.converged` unset, `conv_tol_residual`
ignored), and NEVPT2 is first order in the CI vector, so it moved N₂'s
correction by 1.1e-6. The tool swaps in the dense singlets of the same
active-space Hamiltonian (as `casscf_reference.py`'s polish takes them),
checking each against the Davidson root it replaces (overlap above
1 − 1e-8, energies within 1e-10, and within 1e-10 of the polish's).

**Canonicalization, and the invariance.** Per root, `NEVPT.kernel`
canonicalizes the core and virtuals over **that root's** Fock (its own
1-RDM, not the state average's) and rotates the active space to the root's
natural orbitals (`cas_natorb=True`). SC-NEVPT2 is invariant to that active
rotation, and a test or a C++ step may rely on it and leave the active
orbitals alone: the tool runs every root a second time with core and
virtuals canonicalized over the root's Fock and the active orbitals
unrotated (`cas_natorb=False`) and fails emission unless every class's norm
and energy and `e_corr` agree to 1e-10. They agree to 3.1e-15.

**PC, from block2.** PySCF has no PC-NEVPT2, so the PC fields are block2's
(`pyblock2.icmr.icnevpt2_full.WickICNEVPT2`, the `tools` group's `block2`,
also the source of `nevpt2/<case>/`'s PC goldens), run on exactly the state
the C++ step computes from: the root's CI vector, its core and virtuals
canonicalized over its own Fock, its active orbitals **unrotated**, and
PySCF's own fitted integrals for those orbitals (`dfnevpt2._ERIS`) injected
in place of block2's transform, `canonicalized = True` so block2 takes the
orbitals and energies as given. Two block2 facts the tool works around:
`eri_helper.init_eris` always transforms the exact 4-index ERIs (so handed a
DFCASCI it would compute a mixed answer), and `kernel(eris=...)` raises in
0.5.4 (its `teris` timer is bound only on the build path), so the fitted
integrals go in by replacing `init_eris` for the call. Three gates, each
failing emission: block2's SC (`WickSCNEVPT2`) on the same state equals
PySCF's per class to 1e-10 (it does to 4.3e-15), so block2 sees the state the
SC fields were recorded on; PC Sijrs, with no active index, equals SC Sijrs
to the same; and every per-tuple solve's matrix has a gap of at least 1e3 at
1e-13 of its largest singular value, `nevpt2::pc_class_energies`' own
refusal rule, so the energy does not depend on where a cut fell (block2's
lstsq cuts lower, at eps · d, inside the structural null modes' noise, which
carries no energy). **PC is invariant to the active rotation too**, and is
checked like SC: block2's PC again on each root's natural orbitals, every
class to 1e-10. They agree to 1.8e-14. The C++ energy layer measures the
same thing on its own (`Nevpt2PcInvariance` in `nevpt2_energy_gpu.cpp`).

**No N₂: degenerate virtuals.** SC-NEVPT2 is **not** invariant to a rotation
inside a set of degenerate core or virtual orbitals, and canonicalization
leaves that rotation arbitrary. N₂'s canonical virtuals hold six exactly
degenerate π* pairs, and its Srs class, which contracts per virtual pair,
depends on how each is split. Measured on PySCF's own N₂ DF-CASSCF(6,6), CI
and integrals fixed: one pair rotated by 0.1, 0.3 and π/4 moves Srs by
−3.2e-8, −6.2e-7 and −4.6e-6; random relative rotations of all six, by up to
5.0e-6; the same rotation of every pair, by nothing (7e-14). Two PySCF runs
landed at opposite ends of that band, and the C++ step, whose diagonalizer
splits each pair by 1e-15-level noise in its own orbitals, scattered over it
run to run (−5e-8 to −4.7e-6 off the value this file used to carry) while
its CASSCF matched to 1e-12. So PySCF's N₂ answer is defined only to ~5e-6,
and the case is left out (`EXCLUDED` in the tool) rather than tested at a
tolerance that large. **The invariance check above covers only rotations of
the active space**, so it could not see this. The tool now refuses any root
whose canonical core or virtual energies have two within 1e-6 hartree
(`DEGENERATE`), so a case like N₂'s cannot come back silently. The four
cases left (single-root, AVAS, and SA2 at both weightings) are C₂ᵥ, with no
degenerate irreducible representation.

**Tolerance.** Within this file the numbers are PySCF's to rounding: the
CI is exact, the orbital gradient is below 1e-10, and the output is
byte-identical run to run. A test that runs its own CASSCF does not inherit
that. The CASSCF energy is stationary in the orbitals; NEVPT2 is not, so it
moves to first order in whatever separates the test's orbitals from these.
Measured by running NEVPT2 at `mc1step`'s own unpolished orbitals
(\|g\| from 2e-7 to 1.5e-6, `conv_tol_grad` 1e-7) against the polished ones:

| Case | \|g\| at `mc1step` | CASSCF energy moved | NEVPT2 correction moved |
|---|---|---|---|
| `h2o_ccpvdz` | 1.5e-6 | 2.7e-13 | 1.3e-9 |
| `n2_ccpvdz` | 2.2e-7 | 1.4e-14 | 4.7e-6 |
| `h2co_ccpvdz_sa2` | 1.0e-6 | 2.7e-8 (per root) | 4.9e-8 |
| `h2co_ccpvdz_sa2_weighted` | 6.5e-7 | 2.7e-8 (per root) | 4.3e-8 |

So a test's tolerance is set by how closely its CASSCF reaches these
orbitals (its orbital gradient against PySCF's, and its per-root energies,
which are not stationary either), not by the 1e-9 the CASSCF energy alone
would suggest. (N₂'s row is from before its exclusion, and its 4.7e-6 lies
inside the degenerate-pair band above, so it need not be the orbitals'
doing at all.)
Regenerate with (one thread; the output is byte-identical run to run):

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/nevpt2_reference.py
```
