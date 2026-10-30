# CIS

Configuration interaction singles (the Tamm–Dancoff approximation) over an RHF
reference: the singlet and triplet excitations, as the lowest roots of the A
matrix A_ia,jb = δ_ij δ_ab (ε_a − ε_i) + its two-electron part, solved by
`davidson_solve` (`../recipes/davidson/`) over a sigma that applies A without
forming it. The pipeline's `CisStep` (`../pipeline/cis/`) runs that solve
over an SCF's products.

Everything it exports is in **`namespace free_breakfast::cis`**, spelled bare
inside it — `cis::Preconditioner`, `cis::guess`, `cis::Sigma`, `cis::DfSigma` — and the `.cu`
launchers its bridge headers declare are in **`free_breakfast::cis::device`**,
bare too (`device::precondition`). CIS is a *method*, and a method's
vocabulary (`diagonal`, `guess`, `sigma`) recurs once per method, so it takes a
namespace rather than a `Cis`/`cis_` prefix on each name. `docs/style.md`
"Namespaces, and when a name needs a prefix" has the rule and the cases where a
prefix still survives — filenames, message text and gtest suite names among them.

An excitation vector has n_ov = n_occ · n_virt elements, element ia at
i · n_virt + a: row-major (i, a), i over the occupied orbitals and a over the
virtual ones. A block of vectors is column-major with leading dimension n_ov,
as `davidson_solve` hands it over.

## Sub-modules

There is no aggregate module: each directory is its own module and target,
imported and linked on its own, and `CMakeLists.txt` is a plain
`add_subdirectory` list.

| Directory | Module | Role |
|-----------|--------|------|
| `precondition/` | `free_breakfast.cis.precondition` | The orbital-energy diagonal Δε_ia = ε_a − ε_i (`cis::diagonal`), the unit-vector Davidson start on its lowest entries (`cis::guess`), and the diagonal correction r / (θ − Δε) as a `DavidsonPreconditioner` (`cis::Preconditioner`), with an optional exact-block correction over clusters of near-degenerate Δε (`Preconditioner::cluster`), all on the device. |
| `ris_basis/` | `free_breakfast.cis.ris_basis` | The RIS minimal auxiliary basis (`cis::ris_basis`): one uncontracted shell per l on each atom, of exponent θ / R_A² over the Ghosh radius, `s` on H and `s`, `Sp` or `Spd` (`cis::RisShells`) on a heavy atom, as a spherical `BasisSet` over a molecule's atoms. Host-only. |
| `sigma/` | `free_breakfast.cis.sigma` | The method's vocabulary, `cis::Spin` (`Singlet`, `Triplet`) and `cis::Options` (`n_states`, `spin`, the `DavidsonOptions`, `cluster_gap`), and `cis::Sigma`, the integral-direct σ = A b as a `LinearOperator<…, double>`: per column, the spherical transition density C_occ X C_virtᵀ, its Cartesian form, `nonsymmetric_jk`'s G or K, and σ = Δε ∘ b + w · C_occᵀ R C_virt; and `cis::DfSigma`, the density-fitted one over B's MO blocks B_vo, B_oo and B_vv, a whole block in a dgmm, two gemms (Coulomb) and two strided-batched gemms (exchange). |

## Tests

Device, in `test/gtest/cis_<name>_gpu.cpp`, linked into
`free_breakfast_tests_gpu`. The golden data, PySCF's RHF + TDA over the A
matrix it builds explicitly, is `test/gtest/data/cis/`, written by
`tools/emit/cis_reference.py`.
Host-only modules (`ris_basis/`) are tested in `test/gtest/cis_<name>_host.cpp`,
linked into `free_breakfast_tests_host`.
