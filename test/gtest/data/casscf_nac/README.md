# `casscf_nac/`

`casscf_nac.json`, written by `tools/emit/casscf_nac_reference.py` and never
hand-edited: PySCF's analytic nonadiabatic (derivative) couplings between the
roots of a density-fitted state-averaged CASSCF, `pyscf.nac.sacasscf` put
over `df.grad.sacasscf`, with the pieces they are built from, the references
the DF SA-CASSCF coupling tests of milestone 18 read (#659–#665).

## Cases

All cc-pVDZ over `data/aux_basis/cc-pVDZ-JKFIT.json`, CAS(4,3) frontier
(at the planar geometry 1b1 π, 2b2 n and 2b1 π*), singlets, equal weights:

| `name` | Molecule | ncore | Roots, weights | `from` |
|---|---|---|---|---|
| `h2co_ccpvdz_sa2` | H₂CO, C₂ᵥ (`casscf_reference.py`'s) | 6 | 2, [0.5, 0.5] | `h2co_ccpvdz_sa2` |
| `h2co_skew_ccpvdz_sa3` | H₂CO pyramidalized and twisted, C₁ | 6 | 3, [1/3, 1/3, 1/3] | — |

`from` names the `../casscf/casscf_df.json` case with the same molecule,
start, active space and fitting set; the tool checks its energies against
that file's (each root to 1e-6, the weighted energy to 1e-9). It is also
`../casscf_gradient/`'s `h2co_ccpvdz_sa2`. At its C₂ᵥ geometry many
coupling components vanish by symmetry; the skewed three-root case has none
that do, and has pairs other than (0, 1). PySCF refuses unequal weights in
SA gradients and couplings, so none are carried.

## How each case is made

1. The exact RHF (`conv_tol` 1e-14), whose orbitals are the start.
2. The DF-SA-CASSCF over a DF-RHF `_scf` sharing its `with_df`, from mc1step
   and from mc2step, each polished to |g| < 1e-10, put in canonical orbitals
   and its CI re-solved densely, exactly as in `../casscf_gradient/` (whose
   README has the detail). The two minima must agree to 1e-9; mc1step's is
   written.
3. The CI signs fixed (below).
4. For every ordered pair, PySCF's couplings, its Lagrange solve at
   `conv_rtol` 1e-12 (PySCF's default is 1e-7).

### The DF route

PySCF 2.14's `NonAdiabaticCouplings` derives from the **non-DF**
`grad.sacasscf.Gradients`. The tool's `DFNonAdiabaticCouplings` derives from
it and from `df.grad.sacasscf.Gradients`: the NAC class's methods (transition
RDMs in place of a root's, core and nuclear terms removed, the CSF term), the
DF class's Lagrange terms, and a kernel that, as the DF kernel does, makes
the Hellmann–Feynman gradient the DF-CASSCF one. Two checks show the patch is
not wrong (`checks`): the fitted couplings approach PySCF's exact ones as the
fitting set grows, and the total coupling matches a finite difference of the
fitted states' overlaps.

### Signs

d_IJ changes sign with the phase of either state. After the dense re-solve,
each CI vector is signed so that its **first coefficient, in row-major
(alpha string, beta string) order, of magnitude within 1e-8 of its largest,
is positive**. The tool refuses a vector with a coefficient between 1e-8 and
1e-4 short of its largest, where the rule would be fragile. Orbital signs are
PySCF's, as in `../casscf_gradient/`. A test in its own orbitals aligns
orbital signs by `orbitals`, re-signs its CI vectors by the same rule, or
compares d_IJ s_I s_J (s the sign that aligns each of its CI vectors with
`ci`). `h`, `total`, `etf`, `csf`, `hellmann_feynman`, `lagrange`, the
transition RDMs, `b` and `z` all flip with s_I s_J.

### Which state is the bra

PySCF's docstring says `kernel(state=(ket, bra))` is ⟨bra|∂ket⟩. The
overlap finite difference shows the opposite: `kernel(state=(I, J))` is
⟨I|∂J/∂R⟩ = h_IJ / (E_J − E_I) + d^CSF_IJ. Every pair here is written as
`bra` I, `ket` J with that meaning, from `state=(I, J)`.

## Keys

Per case:

| Key | Meaning |
|---|---|
| `name`, `from`, `basis`, `charge`, `electrons` | As in `../casscf/` |
| `aux_basis` | `name`, `file` (from the repository root), `sha256`, `naux` |
| `atoms` | `symbol`, `z`, `coord` (bohr), in row order |
| `nao`, `nmo`, `ncore`, `ncas`, `nelecas` | As in `../casscf_gradient/` |
| `weights`, `energies`, `energy` | One weight per root; each root's energy; the weighted energy |
| `orbitals` | `core_energies`, `virtual_energies`, `natural_occupations`, as in `../casscf_gradient/` |
| `ci` | Each root's CI vector, n_alpha-strings × n_beta-strings in PySCF's `cistring` order, unit norm, signed as above |
| `pairs` | The non-redundant (p, q), p > q, of `b` and `z`, PySCF's packed (row-major) order, as in `../casscf_gradient/` |
| `couplings` | One record per ordered pair (I, J), I ≠ J, sorted by (I, J); below |
| `checks` | What the tool measured; below |

Per ordered pair (`couplings[i]`), every vector n_atom × 3, row A, column k
the derivative along R_Ak, bohr⁻¹ (or hartree/bohr for `h` and its halves):

| Key | Meaning |
|---|---|
| `bra`, `ket` | I and J |
| `energy_gap` | E_J − E_I (hartree) |
| `total` | d_IJ = ⟨I\|∂J/∂R⟩, the CSF term included (`use_etfs=False`) |
| `etf` | d_IJ without the CSF term (`use_etfs=True`), h_IJ / (E_J − E_I); translation-invariant |
| `csf` | d^CSF_IJ = `total` − `etf`: PySCF's `nac_csf`, the half-derivative AO overlap contracted with the antisymmetrized AO transition 1-RDM (not multiplied by the gap) |
| `h` | h_IJ, the `mult_ediff=True`, `use_etfs=True` numerator; symmetric in I, J |
| `hellmann_feynman` | h_IJ's Hellmann–Feynman half: PySCF's `grad_elec_active` (the DF-CASSCF electronic gradient less its core part, no nuclear repulsion) at the symmetrized transition RDMs `sym_1`, `sym_2` |
| `lagrange` | h_IJ's multiplier half, z's orbital and CI terms (`get_LdotJnuc`); `hellmann_feynman` + `lagrange` = `h` |
| `rdm` | The active-space transition RDMs, PySCF's convention: `trans_1[p][q]` = ⟨I\|q†p\|J⟩, `trans_2` = ⟨I\|p†r†sq\|J⟩ flat row-major over (p, q, r, s), not symmetric (`direct_spin1.trans_rdm12(c_I, c_J)`); `sym_1` = ½(γ + γᵀ), `sym_2` = ½(Γ_pqrs + Γ_qpsr), the densities the Hellmann–Feynman half is taken at |
| `b` | The transition gradient before the Z-vector: `orbital`, `gen_g_hop_active`'s orbital gradient at `sym_1`, `sym_2` (PySCF's convention, half this tree's g), over `pairs`; `ci_norm`, the norm of its CI part, zero at a converged state average |
| `z` | The multipliers, A z = −b, A the state-averaged Hessian projected off the state-averaged CI space (as in `../casscf_gradient/`): `orbital` over `pairs`, `ci` one matrix per root, shaped as `ci`, and `residual` ‖A z + b‖ |

So `total` = `h` / `energy_gap` + `csf`, `total`(I, J) = −`total`(J, I) and
`h`(I, J) = `h`(J, I). The orbital multipliers' packing, and the factor of
two between PySCF's and this tree's g and H, are as `../casscf_gradient/`
describes.

## Checks

Every case records, and the tool asserts:

| `checks` key | Asserted below | Measured (this file) |
|---|---|---|
| `rerun` (a fresh coupling object vs `total`) | 1e-9 | 0 |
| `antisymmetry` (\|d_IJ + d_JI\| for `total` and `etf`, \|h_IJ − h_JI\|) | 1e-8 | ≤ 1.6e-13 |
| `pieces` (HF + Lagrange vs `h`, `etf` + `csf` vs `total`) | 1e-10 | ≤ 1.2e-16 |
| `etf_translation` (max over k of \|Σ_A etf[A][k]\|) | 1e-8 | ≤ 2.9e-13 |
| `finite_difference` (`total` along one random unit displacement vs a five-point difference, step 5e-3 bohr, of ⟨I(R₀)\|J(R)⟩, the displaced states re-converged from Löwdin-carried orbitals and each phased to its own root) | 1e-6 | ≤ 5.4e-8 |
| `aux_ladder` (max \|d^fit − d^exact\| over every pair, against PySCF's exact `NonAdiabaticCouplings`, for cc-pVDZ-, cc-pVTZ- and cc-pVQZ-JKFIT, each fit's own minimum, signs aligned per pair) | each below the last, the last 1e-4 | sa2 1.7e-5, 4.5e-6, 1.2e-7; sa3 8.1e-4, 1.7e-4, 7.8e-6 |

## Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/emit/casscf_nac_reference.py
```

About 7 minutes. BLAS and OpenMP run at one thread and no date or host is
written, so a rerun on the same PySCF and numpy rewrites the file byte for
byte. `test/pytest/test_casscf_nac_reference.py` checks the invariants with
the standard library alone.
