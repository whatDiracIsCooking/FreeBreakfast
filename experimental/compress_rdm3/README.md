# `compress_rdm3/`

Input data for experiments on compressing or screening the 3-RDM that
`nevpt2::rdms` builds: one complete SC-NEVPT2 case for a molecule with no
symmetry, in the layout of `test/gtest/data/nevpt2/` (that README has every
array's shape and convention), plus its dm3, f3ca and f3ac.

No CMake target yet: this directory holds a generator and its output only.
`design.md` has what the case says about dm3, two exact ways to compress it,
and why **neither is worth building against `src/nevpt2` as it stands** —
the layer's cost is one n⁶ · ndet gemm that the first cannot reach and the
second would replace with more arithmetic.

## The case

`glycolaldehyde_ccpvdz_cas1010_casscf_df/`, written by `generate.py`.

| | |
|---|---|
| molecule | glycolaldehyde, O=C–C–O torsion 30°, OH out of plane: **C1**, not optimised |
| basis / aux | cc-pVDZ / cc-pVDZ-JKFIT (cc-pVDZ-RI only for the MP2 starting orbitals) |
| reference | DF-CASSCF(10,10) from DF-MP2 natural orbitals, then a DF-CASCI on its orbitals, canonicalised as `NEVPT.kernel` does |
| sizes | ncore 11, ncas 10, nvirt 55, naux 372, 63 504 determinants |
| natural occupations | 1.9807 1.9785 1.9782 1.9768 1.9433 0.0573 0.0238 0.0219 0.0211 0.0184 |
| E(CASSCF) | −227.9355599023 |
| SC-NEVPT2 E_corr | −0.460138480123 (PySCF DF-NEVPT2) |

Started from RHF orbitals, the 1-step CASSCF drifts: dE is still about −2e-6
at macro iteration 50, 18 mEh above this minimum. From MP2 natural orbitals it
converges in 18 iterations.

## Arrays

The 27 arrays of the tree's goldens (`ci`, `dm1`–`dm3`, `f3ca`, `f3ac`,
`h1e`, `h2e`, `e_core`, `e_virt`, every `h1e_v_*` / `h2e_v_*` block, `cvcv`,
`B_aa`, `B_ca`, `B_va`, `B_cv`, `class_norms`, `class_energies`,
`e_nevpt2_total`), plus `e_casci`. Each is a `'<f8'` Fortran-order `.npy`, as
`tools/emit/nevpt2_golden.py` writes them.

**There is no PC-NEVPT2 reference** (`pc_class_energies`, `e_pc_total`): the
tree's goldens take theirs from block2, which the generator does not run.

## Checks before writing

`generate.py` rebuilds the eight SC classes from the very arrays it saves
(PySCF's `Sr` … `Sir` on the same eris and RDMs) and writes nothing unless:

- their sum matches a fresh `NEVPT.kernel` on the same CASCI (measured 1.4e-15);
- `B_aa B_aa` reproduces (tu|vw) (8.9e-16) and `B_cv B_cv` reproduces `cvcv` (6.9e-17);
- ‖ci‖ = 1 (2.0e-15) and tr dm3 = N³ (2.8e-12).

## Exempt from the file-size limit

dm3, f3ca and f3ac are 8 MB each, over the tree's 3 MB per-file limit, so
`.pre-commit-config.yaml` excludes this directory from
`check-added-large-files`. Every other array fits (`cvcv` is the largest, at
2.9 MB). The whole directory is 38 MB.

## Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    experimental/compress_rdm3/generate.py [--threads N]
```

It takes about 30 s at 12 threads. Only the default of one thread reruns bit
for bit; the numbers above came from `--threads 12`.
