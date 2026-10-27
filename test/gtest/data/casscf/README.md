# `casscf/`

Two files, both written by `tools/emit/casscf_reference.py` and never
hand-edited:

- `casscf.json`: the exact CASSCF references. Its keys, its starts and its
  tolerances are documented in `../README.md` (`casscf/casscf.json`).
- `casscf_df.json`: the same five cases density-fitted, below.

## `casscf_df.json`

The five cases of `casscf.json` (H₂O CAS(4,4) from `active` [2, 3, 5, 6], N₂
CAS(6,6) frontier, H₂CO CAS(4,3) at two singlets weighted [0.5, 0.5] and
[0.7, 0.3], H₂O AVAS O 2p + H 1s, all cc-pVDZ) **from the same starts**: the
same exact RHF (`conv_tol` 1e-14, `conv_tol_grad` 1e-11) and the same
`active` or AVAS pick, so the starting orbitals are the exact cases' own. Only
the CASSCF is fitted: `mcscf.CASSCF(mf, ...).density_fit(auxbasis=...)`, which
puts every two-electron quantity of the CASSCF over the fitted integrals:
the core's J and K, the active-space integrals, the orbital gradient and
Hessian, and the state-averaged Fock.

Each case is gated, polished and recorded as the exact ones are: `mc1step` at
`conv_tol` 1e-12 and `conv_tol_grad` 1e-7, `mc.converged` set and checked
against `mc2step` from the same start (the weighted energy to 1e-9), then
Newton steps over the (fitted) orbital Hessian with the CI re-solved densely,
to \|g\| < 1e-10, with H's least eigenvalue there above 1e-3. At the start the
fitted g and H are checked against finite differences of the fitted energy.

### The fitting set

**`data/aux_basis/cc-pVDZ-JKFIT.json`, this tree's own file, not PySCF's
library copy.** The tool parses it into explicit shells (as
`tools/emit/df_reference.py` does) and hands those to `density_fit`, then
checks that the auxiliary molecule PySCF fitted over has exactly the shells
and exponents the file gives. `auxbasis="cc-pvdz-jkfit"` as a string would
load PySCF's own copy, and a test comparing at 1e-8 would then be comparing
two fitting sets. Each case records which file it used, as `aux_basis`.

The JKFIT set fits everything here, the active-space integrals included, as
`density_fit` does with one `auxbasis`. The fitting error against
`casscf.json` is 2e-5 Eh (water) to 2.6e-4 Eh (N₂) in the weighted energy;
these references are of the fitted energy, not of the exact one.

### Keys

Every case has the keys of a `casscf.json` case (`name`, `basis`, `charge`,
`atoms`, `electrons`, `nmo`, `ncas`, `nelecas`, `ncore`, `active`, `avas`,
`weights`, `rhf_energy`, `mo_energies`, `casci`, `casscf`), with the same
meaning, over the fitted integrals: `rhf_energy` and `mo_energies` are the
exact RHF's, which is the start; `casci` is the fitted CASCI at it; `casscf`
is the fitted, polished minimum (`energies` per root, `energy` weighted,
`orbital_energies` of the fitted state-averaged Fock, `natural_occupations`
descending, `gradient_norm` below 1e-10). One key more:

| Key | Meaning |
|---|---|
| `aux_basis` | `name` (`cc-pVDZ-JKFIT`), `file` (its path from the repository root), `sha256` (of that file, when written) and `naux` (the case's fitting functions, spherical) |

### Drift

Like `casscf.json`, the output is not bit-for-bit reproducible (BLAS and
OpenMP at 8 threads, threaded reductions). Over 6 runs every stored value
agreed with the first to within 1.8e-10 (N₂'s converged active orbital
energies); energies moved by at most 2.6e-12 and natural occupations by at
most 4.3e-12. The stated tolerance is 1e-9: a regeneration that moves any
value by more than that is a change, not noise.

## Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/emit/casscf_reference.py --only df
```

writes `casscf_df.json` alone (about 30 s at 8 threads); `--only exact`
writes `casscf.json` alone, and no `--only` writes both. Regenerating one
leaves the other's bytes alone, so a change to the fitted cases does not
rewrite the exact references' provenance or noise.
