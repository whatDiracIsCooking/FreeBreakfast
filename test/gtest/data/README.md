# `test/gtest/data/`

Golden reference data the gtests read, through `FBF_TEST_DATA_DIR`, an absolute
path compiled into the test binary. These are copies, never hand-edited: each
file names the tool that wrote it and the source it came from.

| Directory | What | Written by |
|---|---|---|
| `one_electron/` | Cartesian basis shells, the nuclei, libcint's one-electron matrices (`int1e_ovlp`, `int1e_kin`, `int1e_nuc`) and PySCF's `hcore` for H₂O in STO-3G and cc-pVDZ, recorded by PySCF | `tools/extract_integral_golden.py`, from FelizJueves's `test/oracle/fixtures/` |
| `coulomb/` | The same bases and nuclei, an AO density (a Hilbert matrix, `density_recipe`) and libcint's Coulomb matrix `vj` of it, for H₂O in STO-3G and cc-pVDZ, recorded by PySCF | `tools/extract_integral_golden.py`, from the same fixtures |
| `exchange/` | The same bases, nuclei and density as `coulomb/`, and libcint's exchange matrix `vk` of it (PySCF `get_jk`, `vk_il = Σ_jk (ij\|kl) D_jk`), for H₂O in STO-3G and cc-pVDZ, recorded by PySCF. The tool refuses to write one unless its `_k` fixture's basis, geometry and density equal the `_j` fixture's exactly, so a K test reads the J test's density, and unless `vk` is symmetric to 1e-14 · max\|vk\| | `tools/extract_integral_golden.py`, from FelizJueves's `h2o_{sto3g,ccpvdz}_k` fixtures |
| `overlap_mixed/` | Two Cartesian bases, `rows` and `cols`, and libcint's cross-basis overlap `int1e_ovlp_cross` between them (PySCF `gto.intor_cross`): H₂O, rows STO-3G, columns 6-31G or cc-pVDZ at the same geometry, or 6-31G at another. The tool refuses one whose shape is not the two AO counts | `tools/extract_integral_golden.py`, from FelizJueves's `h2o_sto3g_x_*` fixtures and their row fixture `h2o_sto3g_j` |
| `record_bound/` | K's dressed record bound, FelizJueves's host `ao.k.record_bounds`, of every record of every ordered shell pair of H₂O/cc-pVDZ and the H₁₀ chain, the shells read from `exchange/` | `tools/emit_record_bound_golden.py`, which imports FelizJueves's `src/` (numpy, for its reference Boys) |
| `scf/` | `rhf.json`: PySCF's closed-shell RHF (spherical, `conv_tol` 1e-12) for H₂O in STO-3G and cc-pVDZ and NH₄⁺ in cc-pVDZ, run over this tree's own `data/basis` files: total energy, nuclear repulsion, orbital energies, geometry in bohr | `tools/emit_scf_reference.py`, which runs PySCF |
| `localization/` | `pipek_mezey.json`: PySCF's Mulliken Pipek-Mezey, p = 2 and 4, of the RHF orbitals of `scf/`'s three cases: the functional and every localized orbital's population on every atom; `ibo.json`: the same for PySCF's IBOs, plus the IAO partial charges | `tools/emit_localization_reference.py`, which runs PySCF |
| `active_space/` | `active_space.json`: PySCF's closed-shell RHF (`conv_tol` 1e-14, `conv_tol_grad` 1e-11) for H₂O in STO-3G and cc-pVDZ and N₂ in cc-pVDZ, over this tree's own `data/basis` files, and PySCF CASCI's active-space Hamiltonian over a chosen core and active set: E_core, h_eff_tt, (tt\|uu) and (tu\|tu) | `tools/emit_active_integrals_reference.py`, which runs PySCF |
| `sad/` | `atomic_scf.json`: PySCF's spherically averaged atomic RHF (`AtomSphAverageRHF`) for H–Ar and Zn in STO-3G and H, He, C, O, Ne in cc-pVDZ, each with its basis in BSE form | `tools/emit_sad_reference.py`, which runs PySCF |
| `casci/` | `casci.json`: PySCF's closed-shell RHF (`conv_tol` 1e-14, `conv_tol_grad` 1e-11) for H₂O in STO-3G and cc-pVDZ and N₂ in cc-pVDZ, over this tree's own `data/basis` files, and PySCF `mcscf.CASCI` over its orbitals, H₂O CAS(4,4) (frontier, and an explicit set in cc-pVDZ) and N₂ CAS(6,6), and `mcscf.avas`'s picks over MINAO (H₂O O 2p + H 1s, N₂ 2s + 2p, and N₂ 2p at threshold 0.5, in cc-pVDZ): the lowest three singlets' energies, ⟨S²⟩ and natural occupations | `tools/emit_casci_reference.py`, which runs PySCF |
| `ci/` | `direct_spin1.json`: active-space integrals (E_core, h_eff, (tu\|vw)) from PySCF's CASCI over its own RHF in its own basis library, and `fci.direct_spin1`'s lowest 12 roots over them, each with ⟨S²⟩ and its spin-summed 1-RDM, for H₂O STO-3G CAS(4,4), H₂O 6-31G CAS(6,6), N₂ cc-pVDZ CAS(6,6) and CH₂O STO-3G CAS(4,4) | `tools/emit_fci_reference.py`, which runs PySCF |
| `fci/` | `fci.json`: PySCF's closed-shell RHF (`conv_tol` 1e-14, `conv_tol_grad` 1e-11) for H₂O in STO-3G and 6-31G and N₂ in STO-3G, over this tree's own `data/basis` files, and PySCF `fci.FCI` over every orbital: the lowest three singlets' energies, ⟨S²⟩ and natural occupations | `tools/emit_fci_step_reference.py`, which runs PySCF |
| `casscf/` | `casscf.json`: PySCF's closed-shell RHF (`conv_tol` 1e-14, `conv_tol_grad` 1e-11) over this tree's own `data/basis` files, and PySCF `mcscf.CASSCF` (`conv_tol` 1e-12, `conv_tol_grad` 1e-7, gated against `mc2step` from the same start, then polished to \|g\| < 1e-10) for H₂O CAS(4,4) from the RHF orbitals [2, 3, 5, 6], N₂ CAS(6,6) frontier, H₂CO CAS(4,3) frontier at two singlets weighted [0.5, 0.5] and [0.7, 0.3], and H₂O AVAS O 2p + H 1s, all cc-pVDZ: at each case's start, the CASCI energies and the sign-invariant parts of `mc1step.gen_g_hop`'s gradient and orbital Hessian (doubled); converged, each root's energy, the weighted energy, each space's orbital energies and the natural occupations. **Every test starts at the case's recorded start (`active` or `avas`), not at a frontier selection** | `tools/emit_casscf_reference.py`, which runs PySCF |

## `one_electron/*.json`, `coulomb/*.json` and `exchange/*.json`

| Key | Meaning |
|---|---|
| `source` | The FelizJueves commit and fixture this was copied from |
| `provenance` | The fixture's own: PySCF, libcint and numpy versions, date, the molecule as PySCF was given it, the basis name |
| `nao` | Cartesian AO count |
| `atoms` | The nuclei, in PySCF's atom order: `symbol`, `charge` (Z), `coord` (bohr) |
| `shells` | One per contraction, in AO order: `l`, `centre` (bohr), `exponents`, `coefficients`. The coefficients multiply **unnormalised** Cartesian primitives `x^a y^b z^c exp(-alpha r^2)`, so they go into a `Shell` as given, with no `primitive_normalized` |
| `arrays` | `{name: {shape, values}}`, `values` row-major in AO order |
| `density_recipe` | `coulomb/` and `exchange/` only: how the fixture made its density |

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
uv run --group tools tools/emit_record_bound_golden.py
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
uv run --group tools tools/emit_active_integrals_reference.py
```

## `sad/atomic_scf.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `bases` | `{name: {elements: {Z: {electron_shells}}}}` in Basis Set Exchange JSON form: raw coefficients over primitive-normalised Gaussians, so a test reads it with `parse_basis` and `primitive_normalized` as a production run would |
| `atoms` | One per (basis, element): `basis`, `z`, `symbol`, `energy` (hartree, for a reader; no test compares it), and `density_blocks` |
| `density_blocks` | Per l, the spin-summed density over that momentum's shells for one m (the same for every m, zero between different m), over PySCF's **unit-normalised** spherical AOs. A test divides element (i, j) by N_i N_j, N_i the norm of this tree's function i |

Every element is one whose per-l electron counts in PySCF's
`NRSRHF_CONFIGURATION` equal the plain Madelung fill `src/sad` uses. Regenerate
with:

```sh
uv run --group tools tools/emit_sad_reference.py
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
uv run --group tools tools/emit_scf_reference.py
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
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit_localization_reference.py
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

## `ci/direct_spin1.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `cases` | One per active space: `name`, `basis` (PySCF's library name), `atoms` (`symbol`, `z`, `coord` in bohr), `n_orbitals`, `n_alpha`, `n_beta`, `e_core` (hartree), `h1` (`{shape, values}`, n × n, row-major), `eri` (`{shape, values}`, n⁴, row-major (t, u, v, w), chemists' (tu\|vw)), and `roots` |
| `roots` | The lowest 12 of the M_s = 0 sector, ascending: `energy` (hartree, `e_core` included), `s2` (⟨S²⟩), `degenerate` (another root lies within 1e-8, so this root's 1-RDM depends on the solver's rotation within the level), and `rdm1` (`{shape, values}`, the spin-summed γ_tu, n × n) |

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
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit_fci_reference.py
```

## `casci/casci.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `cases` | One per active space: `name`, `basis` (this tree's builtin name), `charge`, `atoms` (`symbol`, `z`, `coord` in bohr), `electrons`, `ncas`, `nelecas`, `ncore`, `active` (the 0-based MO indices, or `null` for the frontier orbitals), `avas` (`labels` and `threshold` of an AVAS case, whose `ncas` and `nelecas` are `mcscf.avas`'s, or `null`), `rhf_energy`, `mo_energies` (ascending), `e_core` (hartree, nuclear repulsion included), and `roots` |
| `roots` | The lowest three singlets (one for N₂'s AVAS CAS(10,8), whose excited roots `ci_solve` does not converge at its defaults), ascending: `energy` (hartree, total), `s2` (⟨S²⟩), `degenerate` (another root of the sector lies within 1e-8), and `natural_occupations` (the spin-summed 1-RDM's eigenvalues, descending) |

The 1-RDM itself is not recorded: it depends on the signs of the active
orbitals, which the two codes pick independently, and its eigenvalues do not.
A degenerate root's 1-RDM depends on the solver's rotation within the level
too, so a test skips its occupations. No active space splits a degenerate set
of orbitals; the tool refuses one that does. An AVAS case runs PySCF's
`mcscf.avas` over this tree's `data/basis/minao.json`, and the tool refuses
one with a projection weight within 1e-3 of its threshold. The roots are `direct_spin1`'s
lowest 12 of the M_s = 0 sector, checked against a dense diagonalisation of
it, with the singlets kept. Regenerate with:

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit_casci_reference.py
```

## `fci/fci.json`

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `cases` | One per molecule: `name`, `basis` (this tree's builtin name), `charge`, `atoms` (`symbol`, `z`, `coord` in bohr), `electrons`, `nmo`, `determinants` (C(nmo, electrons / 2)²), `rhf_energy`, and `roots` |
| `roots` | The lowest three singlets, ascending: `energy` (hartree, total), `s2` (⟨S²⟩), `degenerate` (another of the lowest 12 roots of the sector lies within 1e-8), and `natural_occupations` (the spin-summed 1-RDM's eigenvalues, descending) |

As in `casci/`, the 1-RDM itself is not recorded, and a degenerate root's
occupations are skipped by a test. Unlike a CASCI's, an FCI energy and its
natural occupations do not depend on the orbitals at all, only on the basis.
The roots are the lowest 12 of the M_s = 0 sector, with the singlets kept: by a
dense diagonalisation up to 20,000 determinants, and above that by
`direct_spin1`'s Davidson, run twice from PySCF's guesses spread by differently
seeded noise, the two agreeing to 1e-10. Regenerate with:

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit_fci_step_reference.py
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
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit_casscf_reference.py
```
