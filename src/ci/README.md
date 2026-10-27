# CI

Determinant configuration interaction over plain arrays. CASCI, CASSCF and FCI
all call it, so it never sees a basis, a molecule or an `ScfOutput`: what it takes is
a number of orbitals, electron counts and, for sigma, integrals as arrays. It
depends only on `core`, and on `recipes` for the `LinearOperator` concept the
GPU sigma models.
`ci.cppm` (`free_breakfast.ci`) re-exports every sub-module, so one import
covers the layer, and each sub-module stays importable on its own.

Everything it exports is in **`namespace free_breakfast::ci`**, spelled bare
inside it — `ci::Sigma`, `ci::StringSet`, `ci::solve` — and the `.cu`
launchers its bridge headers declare are in **`free_breakfast::ci::device`**,
bare too (`device::olsen`, `device::gather_d`). CI is a *method*, and a
method's vocabulary (`rdm1`, `solve`, `diagonal`, `pack`) recurs once per
method, so it takes a namespace rather than a `Ci`/`ci_` prefix on each name.
`docs/style.md` "Namespaces, and when a name needs a prefix" has the rule and
the cases where a prefix still survives — filenames, message text and gtest
suite names among them.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `strings/` | `free_breakfast.ci.strings` | The occupation strings of one spin (`ci::StringSet`): enumeration, lexical addressing onto [0, C(n, N)), and the single-replacement lists for E_pq. Host-side. |
| `sigma/` | `free_breakfast.ci.sigma` | σ = Hc and diag(H) over two `ci::StringSet`s, from E_core, h_eff and (tu\|vw) as plain arrays: `ci::Sigma` on the device, a `LinearOperator` over blocks of CI vectors, with the 1-RDM over the same tables. |
| `solver/` | `free_breakfast.ci.solver` | `ci::solve`: the lowest roots of one spin through `davidson_solve` over any sigma that is a `LinearOperator`, spin held by a penalty λ (S² − S(S+1)), with ⟨S²⟩ per root; and S² on a vector, as `ci::SpinSquare` on the device. |
| `rdm/` | `free_breakfast.ci.rdm` | The state-averaged spin-summed γ and 2-RDM Γ_tuvw of CI vectors (`ci::Rdm::rdm12`, on the device), in PySCF's `make_rdm12` convention. One vector's γ is `ci::Sigma::rdm1`. |

A determinant is a pair of strings, alpha and beta, each from its own
`ci::StringSet`. The strings are in PySCF's `cistring` order, so a CI vector laid
out alpha-major over two sets matches `fci.direct_spin1`'s element for element.

## Tests

Host, in `test/gtest/ci_<name>_host.cpp`, linked into `free_breakfast_tests_host`;
device, in `test/gtest/ci_<name>_gpu.cpp`, linked into `free_breakfast_tests_gpu`.
The golden data, PySCF's `fci.direct_spin1` over its own active-space
integrals and its `make_rdm12` over stored CI vectors, is `test/gtest/data/ci/`,
written by `tools/emit/fci_reference.py`.

The host references the device is checked against, `host_sigma`,
`host_diagonal`, `host_rdm1`, `host_rdm12` and `host_spin_square`, are plain
loops written for clarity, in `test/ci_oracles/`
(`free_breakfast.test.ci_oracles`, namespace `free_breakfast::test`). Only
the test binaries link them; no calculation runs them.

## Files

| File | Role |
|------|------|
| `ci.cppm` | Aggregate module interface; `export import`s every sub-module |
| `CMakeLists.txt` | Build configuration: `add_subdirectory` per module, and the `free_breakfast.ci` module target |
