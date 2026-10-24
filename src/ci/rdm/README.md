# RDM

The spin-summed reduced density matrices of CI vectors: the 1-RDM of one vector
on the host, the reference `CiSigma::rdm1` (`../sigma/`), its device form, is
tested against; and the state-averaged 1- and 2-RDM of several, on the device,
with a host reference that is its oracle. The 2-RDM is CASSCF's: its
generalized Fock, gradient and energy read γ and Γ from `rdm12`.

## Module

`free_breakfast.ci.rdm`, also re-exported by `free_breakfast.ci`; target
`free_breakfast::ci::rdm`.

| Name | Role |
|---|---|
| `rdm1(alpha, beta, c, gamma)` | γ_tu = ⟨c\| E_tu \|c⟩, n × n row-major; `Status` |
| `rdm12(res, alpha, beta, vectors, weights, gamma, gamma2)` | On the device: γ = Σ_r w_r γ^(r) and Γ_tuvw = Σ_r w_r ⟨c_r\| E_tu E_vw − δ_uv E_tw \|c_r⟩, n⁴ row-major; `Status` |
| `host_rdm12(alpha, beta, vectors, weights, gamma, gamma2)` | The same on the host, written for clarity: `rdm12`'s oracle |

`vectors` is on the host, root r at [r · dim, (r + 1) · dim), which is what
`CiRoots::vectors` holds, and γ and Γ come back on the host. A single state is
`weights = {1}`.

`rdm1` refuses (`InvalidConfig`, nothing written) two `StringSet`s over
different orbital counts, or a span of the wrong size. Both `rdm12`s refuse the
same, and also weights of the wrong count for the vectors (none at all
included), a negative or NaN weight, and weights not summing to 1 within 1e-12;
the device one also refuses more determinants than an `int32_t` counts.

## Design

γ_tu = Σ_σ ⟨c\| a†_tσ a_uσ \|c⟩ is PySCF's `make_rdm1`. Each replacement
a†_p a_q \|K⟩ = sign \|J⟩ in either of K's two lists adds c_J · sign · c_K to
γ_pq, so one pass over the lists builds it, O(dim · N (n − N + 1)). The trace
is N_α + N_β for a unit vector, from the diagonal entries p = q alone.

Γ is PySCF's `make_rdm12` convention, Γ_pqrs = ⟨a†_p a†_r a_s a_q⟩ summed over
the spins, which is ⟨E_pq E_rs⟩ − δ_qr ⟨E_ps⟩. Inserting a resolution of the
identity over the determinants, ⟨c\| E_tu E_vw \|c⟩ = Σ_K D_{tu,K} D_{wv,K}
with the one-replacement intermediate D_{tu,K} = ⟨c\| E_tu \|K⟩ (for c real,
⟨K\| E_vw \|c⟩ = D_{wv,K}). D is gathered over each determinant's own lists:
a†_p a_q \|K⟩ = s \|J⟩ is ⟨J\| E_pq \|K⟩ = s, so K's links reach every c_J
that D_K takes.

On the device that is one thread per determinant K, writing D as a
count × n² column-major block (adjacent threads, adjacent rows, so every write
is coalesced) for one chunk of determinants at a time, 64 MiB of D a chunk.
Then M += w_r Dᵀ D and γ += w_r Dᵀ c are two gemms per chunk, accumulated over
every chunk and root, and on the host Γ_tuvw = M_{tu,wv} − δ_uv γ_tw. A
thread owns its row and nothing is atomic, so the result does not depend on
the run. The vectors, both link tables, M, γ and D are carved from
`res.device_workspace()`; cost is O(Σ_r dim · n⁴) in the gemms, M is n⁴
doubles (128 MiB at the strings' 64-orbital limit).

`rdm12` keeps its own gather rather than reusing `CiSigma`'s, whose
`rdm1` it shares the idea with: `CiSigma` packs D over v ≥ w as
D_vw + D_wv, which is all σ and γ need (both contract it with something
symmetric in the pair) but loses Γ, whose q and v in Σ (pv\|wx) Γ_qvwx are
not symmetric. So its link tables carry the unpacked pair p · n + q, and it
takes only the two `StringSet`s: no `CiSigma` and no integrals.

The host reference builds the same d_K per determinant and adds the outer
product w_r d_tu d_wv, O(Σ_r dim · n⁴) scalar work, and takes γ from `rdm1`,
so its γ is `rdm1`'s exactly.

## Tests

`test/gtest/ci_solver_gpu.cpp` checks `rdm1` on `ci_solve`'s roots against
`direct_spin1`'s `make_rdm1` to 1e-8, with the trace; see `../solver/`.
`test/gtest/ci_sigma_gpu.cpp` checks `CiSigma::rdm1` against it.

`test/gtest/ci_rdm_gpu.cpp` checks `rdm12` over the CI vectors stored in
`test/gtest/data/ci/direct_spin1.json`: each case's ground state, and H₂O
STO-3G CAS(4,4)'s two lowest singlets at [0.7, 0.3]. The device γ and Γ match
`host_rdm12` to 1e-12, both match PySCF's `make_rdm12` and `make_rdm12s` summed
over the spins to 1e-8, and γ matches Σ_r w_r `rdm1` to 1e-12 (and, for one state, `CiSigma::rdm1`
to 1e-12). Random unit
vectors over seven sectors (open shells, an empty and a full spin, one
orbital, N_α < N_β) at three weights check the device against the host again;
a slow case (`MatchesHostAcrossChunks`, skipped under
`FREE_BREAKFAST_FAST_TESTS`) does it over 108 900 determinants, two chunks of
D. Each checks Γ_tuvw = Γ_vwtu, Σ_v Γ_tuvv = (N − 1) γ_tu and
Σ_tu Γ_ttuu = N (N − 1) to 1e-10. The refusals are checked on both, with
γ and Γ left untouched.

## Files

| File | Role |
|------|------|
| `rdm.cppm` | Module interface: `rdm1`, `rdm12`, `host_rdm12` |
| `rdm.cpp` | The passes over the replacement lists, and `rdm12`'s uploads, gemms and Γ's assembly |
| `ci_rdm_bridge.h` | The gather's launcher, shared by the module (GMF) and the `.cu` |
| `ci_rdm.cu` | The gather of D, one thread per determinant |
| `CMakeLists.txt` | The device library `free_breakfast.ci.rdm.device` and the module `free_breakfast.ci.rdm` |
