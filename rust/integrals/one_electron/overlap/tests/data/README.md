# The overlap's reference data

PySCF's spherical AO overlap S for the three fixtures `tests/overlap.rs`
compares against, so the tests need no PySCF. Copied from
`experimental/rust/s_overlap/data` (byte for byte, but for `reference.py`'s
paths).

| File | What |
|---|---|
| `h2o.xyz`, `co.xyz` | The molecules, in Angstrom: water, and CO with the bond off every axis |
| `h2o_sto-3g.npy`, `h2o_cc-pvdz.npy`, `co_cc-pvtz.npy` | S, float64, `nao` × `nao` (7, 24, 60), in the spherical AO order the basis's layout follows |
| `reference.py` | The script that wrote them |

## What the reference pins, and why

The comparison is meant to measure the integrals and nothing else, so
`reference.py` matches three conventions to what the basis crates read:

- **The same basis digits.** It loads the BSE JSON of `data/basis/` that
  `fb_io_basis::resolve_basis` compiles in, found by the same folded name,
  and converts it with `tools/atomic_cd/eri_quartet.py`'s `basis_from_bse`.
  PySCF's own copy of a named set can differ.
- **The same coordinates.** It converts Angstrom to bohr itself with CODATA
  2018's Bohr radius, `fb_core_constants::BOHR_PER_ANGSTROM`'s, and passes
  bohr. PySCF's own conversion uses CODATA 2010's, about 2e-10 apart, enough
  to move S past the bar.
- **The same contraction normalization.** The basis normalizes each
  primitive (`BasisStamp::primitive_normalized`) but never renormalizes a
  contraction, so an AO's norm is that of the published coefficients.
  PySCF rescales every contraction to exactly 1. So S = D S_pyscf D, with
  D_ii the published contraction's norm, which the script computes from
  PySCF's own primitives (`gto_norm`, `gaussian_int`) and applies before
  writing.

diag(S) is therefore not 1: it is off by 7e-11 in STO-3G and by up to
1.3e-6 in cc-pVDZ and cc-pVTZ.

## A new reference

From the checkout root (PySCF is in the `tools` dependency group; inside the
devcontainer, set `UV_PROJECT_ENVIRONMENT` to a scratch path first, so the
group does not install into the image's `/opt/venv`):

```bash
uv run --group tools python \
    rust/integrals/one_electron/overlap/tests/data/reference.py \
    mol.xyz cc-pvdz mol_cc-pvdz.npy
```
