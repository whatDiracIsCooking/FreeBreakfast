# rust/s_overlap — the AO overlap S from Rust, against PySCF

The spherical AO overlap matrix S of a molecule in a basis, built on the GPU by
free_breakfast's own pipeline steps and driven from Rust. It is checked against
PySCF at 1e-12 relative to max |S|, and agrees to about 1e-15.

## Layers

| Layer | File | What |
|---|---|---|
| C ABI | `ffi/fb_overlap.h`, `ffi/fb_overlap.cpp` | `fb_overlap(res, xyz, basis, &buf, &nao)`: the Molecule, BasisSet, SphericalTransform, PrimitivePairClasses and Overlap steps, the same chain a TOML run walks. Built on `../gemm`'s handles (`fb_handles.hpp`), so S comes back as an ordinary `fb_buffer`. A step's refusal (missing file, unknown basis, element the basis lacks) is `FB_FAILED`, with its message in `fb_last_error()` |
| Rust | `src/lib.rs` | `overlap(&gpu, xyz, basis) -> Result<Matrix>`: S as an `fb_gemm::Matrix` on the caller's `Gpu`, so it feeds `fb_gemm::gemm` directly. Declares `fb_overlap` by hand; keep it in step with the header |
| | `src/npy.rs` | Just enough .npy to read a 2-D `<f8` array, C or Fortran order, into column-major |
| Demo | `src/main.rs` | `overlap-demo XYZ BASIS [REFERENCE.npy]`: size, symmetry and diagonal of S; with a reference, the largest difference from it, failing past the bar |
| Reference | `data/reference.py` | PySCF's S for the same molecule and basis |
| Fixtures | `data/*.xyz`, `data/*.npy` | H2O in STO-3G and cc-pVDZ, CO in cc-pVTZ (f on both atoms, bond off every axis): what `cargo test` compares against, so the tests need no PySCF |

## What the reference pins, and why

The comparison is meant to measure the integrals and nothing else, so
`data/reference.py` matches three conventions to what free_breakfast reads:

- **The same basis digits.** It loads the BSE JSON the binary compiles in
  (`data/basis/`), found by `resolve_basis`'s folded name, and converts it with
  `tools/atomic_cd/eri_quartet.py`'s `basis_from_bse`. PySCF's own copy of
  a named set can differ.
- **The same coordinates.** It converts Angstrom to bohr itself with the tree's
  CODATA 2018 Bohr radius and passes bohr. PySCF's own conversion uses CODATA
  2010, about 2e-10 apart, enough to move S past the bar.
- **The same contraction normalization.** free_breakfast normalizes each
  primitive (`BasisStamp::primitive_normalized`) but **never renormalizes a
  contraction**, so an AO's norm is that of the published coefficients. PySCF
  rescales every contraction to exactly 1. So S_fb = D·S_pyscf·D, with D_ii
  the published contraction's norm. The script computes D from PySCF's own
  primitives (`gto_norm`, `gaussian_int`, the sum PySCF itself normalizes by)
  and writes D·S·D.

The third one is more than bookkeeping. **diag(S) is not 1 in free_breakfast:**
it is off by 7e-11 in STO-3G and by up to 1.3e-6 in cc-pVDZ and cc-pVTZ, whose
published general contractions are rounded further from unit norm. SCF energies
and orbitals do not depend on how the AOs are scaled. Anything that reads
diag(S) as 1, or compares AO-basis matrices to PySCF's element by element, does.

## Measured (RTX 3080, CUDA, 2026-11-10)

| Molecule | Basis | nao | max \|S − S_ref\| | max \|diag(S) − 1\| |
|---|---|---|---|---|
| H2O | STO-3G | 7 | 4.4e-16 | 7.0e-11 |
| H2O | cc-pVDZ | 24 | 4.4e-16 | 1.0e-6 |
| CO | cc-pVTZ | 60 | 1.1e-15 | 1.3e-6 |

## Build and run

As for `../gemm` (whose README has the Rust toolchain note and the link
mechanics), one CMake target builds both C ABIs and both link records:

```bash
cmake --preset experimental
cmake --build build-experimental --target free_breakfast_rust_s_overlap_linkprobe
cd experimental/rust/s_overlap
cargo test --release                       # needs a GPU; the fixtures need no PySCF
cargo run --release --bin overlap-demo -- data/co.xyz cc-pvtz data/co_cc-pvtz.npy
```

A new reference (from the checkout root; PySCF is in the `tools` group):

```bash
uv run --group tools python experimental/rust/s_overlap/data/reference.py \
    mol.xyz cc-pvdz mol_cc-pvdz.npy
```

Inside the devcontainer, set `UV_PROJECT_ENVIRONMENT` to a scratch path first,
so the `tools` group does not install into the image's `/opt/venv`.

Nothing in ctest or either gate runs cargo. This directory is checked by hand only.
