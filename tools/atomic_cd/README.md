# `tools/atomic_cd/`

Generates a **density-fitting (auxiliary) basis for any orbital basis**, one
element at a time, from nothing but the orbital basis itself: no training
set, no fitted exponents. It is the atomic Cholesky family of methods
(aCD/acCD, Aquilante, Lindh and coworkers) in the form Lehtola made
automatic, including his 2023 contraction step.

Everything here is Python over PySCF's integrals. There is no C++
counterpart: `src/integrals/eri/density_fitting` fits over a ready-made set
(`data/aux_basis/`) and does not generate one.

## The pipeline

| Step | Module | What it does |
|---|---|---|
| 1. Uncontract | `eri_quartet.primitives` | The orbital basis becomes its distinct `(l, exponent)` primitives; contraction coefficients play no role in which products a fit can need. |
| 2. Pool (opt-in screening) | `eri_quartet.screen_pairs` | Every primitive pair is a candidate, or, with `--screen`, only the pairs a pivoted Cholesky of the atom's own primitive ERIs (aCD) pivots on. |
| 3. Select (acCD) | `compact` | Per target l, each pair coupling to l (Gaunt rule) gives one pure spherical Gaussian candidate (Lehtola's Eq. 16 effective exponent); a pivoted Cholesky of their unit-diagonal Coulomb metric keeps the pivots. |
| 4. Contract | `contract` | Per l, the pivots are contracted against the atom's own orbital products: the singular vectors of J = (mu nu\|P) V^-1/2 with s^2 >= epsilon, one general contraction per l (Lehtola 2023, after Kallay's natural auxiliary functions). |
| (optional) Prune | `compact.pruned_max_l` | Drops target l above Lehtola's Eq. 9 cap, max(2 l_occ, l_occ + l_obs + l_inc). |

`cholesky.pivoted_cholesky` is the one decomposition both Cholesky stages
run. `df_check` measures a result; `cli` drives it all.

## Running it

PySCF is in the opt-in `tools` dependency group (`pyproject.toml`), not
`dev`. From the repo root, and inside the devcontainer with
`UV_PROJECT_ENVIRONMENT=$PWD/.venv` so the group does not land in the
image's `/opt/venv` (`tools/README.md`):

```sh
uv run --group tools python -m tools.atomic_cd.cli C cc-pVTZ --df-check
uv run --group tools python -m tools.atomic_cd.cli C cc-pVTZ --bse-out c.json
uv run --group tools python -m tools.atomic_cd.cli C label --bse-in orbital.json
```

`--bse-out` writes Basis Set Exchange JSON, spherical, general-contracted,
coefficients over normalized primitives -- the format `src/io/basis` reads.
`--df-check` prints the one-centre ERI fitting error and DF-RHF minus
conventional RHF for the built-in molecules (CO, CH4) that contain the
element.

## Options and their defaults

| Option | Default | Why |
|---|---|---|
| `--contract` / `--no-contract` | on | About 40% fewer functions at equal or better accuracy (table below). |
| `--contract-threshold EPS` | `1e-5` | Lehtola's "large" epsilon. Absolute, on eigenvalues of J^T J, so it bites harder for heavy atoms; tested on C/H/O only. |
| `--threshold TAU` | `1e-7` contracted, `1e-4` not | Both Cholesky stages. Tight under contraction, which then sizes the set (Lehtola 2023's recipe, measured better per function than contracting a 1e-4 set); without it, 1e-7 would mean ~400 functions on C/cc-pVTZ. |
| `--screen` | off | Never shrinks the contracted set, and cost accuracy where the difference was above noise (C/cc-pVTZ CO: 1.0e-6 Eh against 1.1e-7). Buys ~28% fewer primitives and ~18% faster (mu nu\|P) in PySCF, paid back on every run. A speed knob. |
| `--prune-l [L_INC]` | off; bare flag `1` | Fine for global-fit HF/MP2 energies, but it gives up the threshold's integral-error guarantee for the dropped l (below). |
| `--max-l L` | `2 * l_obs` | Every l the atom's shells reach, as the papers do. The C++ fitting kernels stop at g, so a set meant for them needs `--max-l 4` (or pruning) today. |

Non-positive thresholds, negative `--max-l`/`--prune-l`, and
`--contract-threshold` with `--no-contract` are rejected by name. A
negative exponent-form value needs `=`: `--threshold=-1e-6`.

## How good is it

C/H/O with cc-pVTZ, functions per C atom; dE is E(DF-RHF) - E(RHF):

| Aux set | C functions | one-centre error | dE CO (Eh) | dE CH4 (Eh) |
|---|---|---|---|---|
| default (contracted) | 190 | 3.8e-6 | +1.1e-7 | +2.6e-7 |
| `--no-contract` | 309 | 7.2e-5 | +2.4e-7 | +3.2e-7 |
| cc-pVTZ-JKFIT | 79 | 1.6e-2 | +1.3e-4 | -1.9e-6 |

So the default sets are about JKFIT's size at cc-pVDZ (71 functions on C
against 70) and 2.4x at cc-pVTZ, and on CO one order (cc-pVDZ) to three
orders (cc-pVTZ) more accurate. That is the unbiased-basis trade the aCD
papers describe: no method-specific fitting, tunable accuracy, more
functions as the orbital basis grows. At cc-pVDZ the CO error floors near 3e-5 at any threshold --
the one-centre-product limit, since no one-centre function exactly
represents a two-centre orbital product. Generation is under a second even
for cc-pV5Z (0.7 s, ~600 MB); `--df-check` at 5Z is a minute, nearly all
of it PySCF's conventional reference RHF.

**Pruning trades integrals for energies.** On C/cc-pVTZ (uncontracted),
`--prune-l` raised the one-centre ERI error ~170x while moving the DF-RHF
energies by under 1e-6 Eh at l_inc = 1 (about 1e-5 at l_inc = 0). That is Eq. 9 working as designed for HF and
MP2, but integrals involving virtual products -- a CASSCF orbital Hessian,
CCSD's particle-particle terms -- have not been checked.

## Things that were derived or decided here, not copied

- **The acCD metric is a closed form.** For same-centre normalized
  r^l exp(-a r^2) Y_lm, (a|b)/sqrt((a|a)(b|b)) = sech(ln(a/b)/2)^(l+1/2): a
  function of log-exponent spacing alone. It matches PySCF's `int2c2e` to
  1e-15 at l = 0 and 3e-13 at l = 10, where it is the more accurate.
- **Cholesky vectors are not coefficients.** A vector holds Coulomb
  overlaps (phi_k|psi_J); the basis is the pivots. An earlier version wrote
  the vectors as contractions: it fit the isolated atom 10-100x better and
  CO ~10^4x worse (5e-3 Eh). An atom-only check cannot see this, which is
  why `df_check` runs molecules.
- **Eq. 16 is kept on measurement**: against plain a+b exponents it was as
  good or better on CO/CH4 from cc-pVDZ to cc-pVQZ and a tie at cc-pV5Z.
- **Contraction's per-l block structure was checked, not assumed**: W
  couples only equal (l, m) (off-block ~5e-12 relative) and every m gives
  the same block (5e-13), so one component per l is exact.
- **The ERIs use full eight-fold symmetry**: packed lower triangle of the
  unique-pair matrix (`eri_quartet.PackedSymmetric`), read only through a
  diagonal, a row gather and lower-row blocks.

## Tests

`test/pytest/test_atomic_cd.py`, which skips without the `tools` group (and
so on the push gate):

```sh
uv run --group tools pytest -rs test/pytest/test_atomic_cd.py
```

## Not done

- **One element per run.** A fitting set is one document over many
  elements, with a `name` for `resolve_aux_basis`; that wrapper, and
  regeneration commands in `data/aux_basis/README.md`, are still to come.
- **Unchecked against the C++ side**: whether `src/io/basis` and the DF
  kernels accept a general-contracted auxiliary set, and the l <= g cap
  they need.
- **Tested on C, H and O only.** Heavy elements, ECP bases (pruning's l_occ
  assumes all-electron) and correlated or CASSCF energies are open.

## References

1. N. H. F. Beebe and J. Linderberg, "Simplifications in the generation and
   transformation of two-electron integrals in molecular calculations",
   *Int. J. Quantum Chem.* **12**, 683 (1977). -- Cholesky decomposition of
   the ERI matrix; the bound `cholesky.py` relies on.
2. H. Koch, A. Sanchez de Meras and T. B. Pedersen, "Reduced scaling in
   electronic structure calculations using Cholesky decompositions",
   *J. Chem. Phys.* **118**, 9481 (2003), doi:10.1063/1.1578621. -- Full
   molecular CD; arbitrary accuracy, but molecule-specific.
3. F. Aquilante, R. Lindh and T. B. Pedersen, "Unbiased auxiliary basis sets
   for accurate two-electron integral approximations", *J. Chem. Phys.*
   **127**, 114107 (2007), doi:10.1063/1.2777146. -- Atomic CD as an
   auxiliary basis.
4. F. Aquilante, L. Gagliardi, T. B. Pedersen and R. Lindh, "Atomic
   Cholesky decompositions: a route to unbiased auxiliary basis sets for
   density fitting approximation with tunable accuracy and efficiency",
   *J. Chem. Phys.* **130**, 154107 (2009), doi:10.1063/1.3116784. -- aCD
   and acCD.
5. S. Lehtola, "Straightforward and accurate automatic auxiliary basis set
   generation for molecular calculations with atomic orbital basis sets",
   *J. Chem. Theory Comput.* **17**, 6886 (2021), arXiv:2106.11081. -- The
   procedure `compact` and `eri_quartet.screen_pairs` follow: primitive
   pair candidates, Eq. 16 effective exponents, unit-normalized metric.
6. S. Lehtola, "Automatic generation of accurate and cost-efficient
   auxiliary basis sets", *J. Chem. Theory Comput.* **19**, 6242 (2023),
   arXiv:2306.11039. -- SVD contraction (`contract`) and Eq. 9 pruning
   (`compact.pruned_max_l`).
7. M. Kallay, "A systematic way for the cost reduction of density fitting
   methods", *J. Chem. Phys.* **141**, 244113 (2014),
   doi:10.1063/1.4905005. -- Natural auxiliary functions, the SVD behind
   `contract`.
8. B. I. Dunlap, "Robust and variational fitting: removing the four-center
   integrals from center stage in quantum chemistry", *J. Mol. Struct.
   THEOCHEM* **529**, 37 (2000), doi:10.1016/S0166-1280(00)00528-5. --
   Coulomb-metric fitting errors are second order, why energies tolerate
   what integrals do not.
9. R. Yang, A. P. Rendell and M. J. Frisch, "Automatically generated
   Coulomb fitting basis sets: design and accuracy for systems containing H
   to Kr", *J. Chem. Phys.* **127**, 074102 (2007). -- The l_occ ranges
   Eq. 9 uses.
10. G. L. Stoychev, A. A. Auer and F. Neese, "Automatic generation of
    auxiliary basis sets", *J. Chem. Theory Comput.* **13**, 554 (2017),
    doi:10.1021/acs.jctc.6b01041. -- AutoAux, the main alternative.
11. F. Weigend, "A fully direct RI-HF algorithm: implementation, optimised
    auxiliary basis sets, demonstration of accuracy and efficiency",
    *Phys. Chem. Chem. Phys.* **4**, 4285 (2002), doi:10.1039/B204199P. --
    The JKFIT sets in `data/aux_basis/` that the table above compares
    against.
