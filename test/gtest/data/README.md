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
