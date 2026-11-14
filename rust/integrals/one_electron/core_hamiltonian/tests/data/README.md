# The core Hamiltonian's reference data

PySCF's spherical AO core Hamiltonian H = T + V (`int1e_kin` +
`int1e_nuc`, point nuclei) for the three fixtures
`tests/core_hamiltonian.rs` compares against, so
the tests need no PySCF. The molecules are the overlap's
(`rust/integrals/one_electron/overlap/tests/data/`), copied byte for byte.

| File | What |
|---|---|
| `h2o.xyz`, `co.xyz` | The molecules, in Angstrom: water, and CO with the bond off every axis |
| `h2o_sto-3g.npy`, `h2o_cc-pvdz.npy`, `co_cc-pvtz.npy` | H, float64, `nao` × `nao` (7, 24, 60), in the spherical AO order the basis's layout follows |
| `reference.py` | The script that wrote them: the overlap's, with `int1e_kin` + `int1e_nuc` for `int1e_ovlp` |

## What the reference pins, and why

The same three conventions the overlap's reference pins, for the same
reason (its `README.md` has each in full), so the comparison measures the
integrals and nothing else:

- **The same basis digits**: the BSE JSON of `data/basis/` that
  `fb_io_basis::resolve_basis` compiles in, converted with
  `tools/atomic_cd/eri_quartet.py`'s `basis_from_bse`.
- **The same coordinates**: Angstrom converted to bohr with CODATA 2018's
  Bohr radius, `fb_core_constants::BOHR_PER_ANGSTROM`'s, and passed in bohr.
  The nuclei sit at the atoms, so they move with it.
- **The same contraction normalization**: H = D H_pyscf D, D_ii the
  published contraction's norm, since the basis normalizes each primitive
  but never a contraction.

## A new reference

From the checkout root (PySCF is in the `tools` dependency group; inside the
devcontainer, set `UV_PROJECT_ENVIRONMENT` to a scratch path first, so the
group does not install into the image's `/opt/venv`):

```bash
uv run --group tools python \
    rust/integrals/one_electron/core_hamiltonian/tests/data/reference.py \
    mol.xyz cc-pvdz mol_cc-pvdz.npy
```
