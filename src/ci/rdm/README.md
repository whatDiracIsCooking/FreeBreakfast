# RDM

The spin-summed one-particle reduced density matrix of a CI vector, on the
host: the reference `CiSigma::rdm1` (`../sigma/`), the device form a
calculation uses, is tested against. The 2-RDM is CASSCF's and comes with it.

## Module

`free_breakfast.ci.rdm`, also re-exported by `free_breakfast.ci`; target
`free_breakfast::ci::rdm`.

| Name | Role |
|---|---|
| `rdm1(alpha, beta, c, gamma)` | γ_tu = ⟨c\| E_tu \|c⟩, n × n row-major; `Status` |

It refuses (`InvalidConfig`, nothing written) two `StringSet`s over different
orbital counts, or a span of the wrong size.

## Design

γ_tu = Σ_σ ⟨c\| a†_tσ a_uσ \|c⟩ is PySCF's `make_rdm1`. Each replacement
a†_p a_q \|K⟩ = sign \|J⟩ in either of K's two lists adds c_J · sign · c_K to
γ_pq, so one pass over the lists builds it, O(dim · N (n − N + 1)). The trace
is N_α + N_β for a unit vector, from the diagonal entries p = q alone.

## Tests

`test/gtest/ci_solver_gpu.cpp` checks it on `ci_solve`'s roots against
`direct_spin1`'s `make_rdm1` to 1e-8, with the trace; see `../solver/`.
`test/gtest/ci_sigma_gpu.cpp` checks `CiSigma::rdm1` against it.

## Files

| File | Role |
|------|------|
| `rdm.cppm` | Module interface: `rdm1` |
| `rdm.cpp` | The pass over the replacement lists |
| `CMakeLists.txt` | The module `free_breakfast.ci.rdm` |
