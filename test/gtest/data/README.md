# `test/gtest/data/`

Golden reference data the gtests read, through `FBF_TEST_DATA_DIR`, an absolute
path compiled into the test binary. These are copies, never hand-edited: each
file names the tool that wrote it and the source it came from.

| Directory | What | Written by |
|---|---|---|
| `one_electron/` | Cartesian basis shells, the nuclei, libcint's one-electron matrices (`int1e_ovlp`, `int1e_kin`, `int1e_nuc`) and PySCF's `hcore` for H₂O in STO-3G and cc-pVDZ, recorded by PySCF | `tools/extract_integral_golden.py`, from FelizJueves's `test/oracle/fixtures/` |
| `coulomb/` | The same bases and nuclei, an AO density (a Hilbert matrix, `density_recipe`) and libcint's Coulomb matrix `vj` of it, for H₂O in STO-3G and cc-pVDZ, recorded by PySCF | `tools/extract_integral_golden.py`, from the same fixtures |
| `exchange/` | The same bases, nuclei and density as `coulomb/`, and libcint's exchange matrix `vk` of it (PySCF `get_jk`, `vk_il = Σ_jk (ij\|kl) D_jk`), for H₂O in STO-3G and cc-pVDZ, recorded by PySCF. The tool refuses to write one unless its `_k` fixture's basis, geometry and density equal the `_j` fixture's exactly, so a K test reads the J test's density, and unless `vk` is symmetric to 1e-14 · max\|vk\| | `tools/extract_integral_golden.py`, from FelizJueves's `h2o_{sto3g,ccpvdz}_k` fixtures |
| `record_bound/` | K's dressed record bound, FelizJueves's host `ao.k.record_bounds`, of every record of every ordered shell pair of H₂O/cc-pVDZ and the H₁₀ chain, the shells read from `exchange/` | `tools/emit_record_bound_golden.py`, which imports FelizJueves's `src/` (numpy, for its reference Boys) |
| `scf/` | `rhf.json`: PySCF's closed-shell RHF (spherical, `conv_tol` 1e-12) for H₂O in STO-3G and cc-pVDZ and NH₄⁺ in cc-pVDZ, run over this tree's own `data/basis` files: total energy, nuclear repulsion, orbital energies, geometry in bohr | `tools/emit_scf_reference.py`, which runs PySCF |
| `localization/` | `pipek_mezey.json`: PySCF's Mulliken Pipek-Mezey, p = 2 and 4, of the RHF orbitals of `scf/`'s three cases: the functional and every localized orbital's population on every atom | `tools/emit_localization_reference.py`, which runs PySCF |
| `sad/` | `atomic_scf.json`: PySCF's spherically averaged atomic RHF (`AtomSphAverageRHF`) for H–Ar and Zn in STO-3G and H, He, C, O, Ne in cc-pVDZ, each with its basis in BSE form | `tools/emit_sad_reference.py`, which runs PySCF |

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
