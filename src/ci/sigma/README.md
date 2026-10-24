# Sigma

σ = Hc and diag(H) for a determinant CI vector: `CiSigma` on the device, as a
`LinearOperator` (`recipes.linear_operator`) over a block of CI vectors, with
the 1-RDM over the same tables, and `host_sigma` / `host_diagonal` on the
host. The host pair is written for
clarity, not speed: it is the reference the GPU sigma is tested against.

## Module

`free_breakfast.ci.sigma`, also re-exported by `free_breakfast.ci`; target
`free_breakfast::ci::sigma`.

| Name | Role |
|---|---|
| `CiSigma::create(res, alpha, beta)` | Upload the strings and replacement lists; `Result<CiSigma>` |
| `CiSigma::set_integrals(res, e_core, h_eff, eri)` | Upload an H over them, replacing any before; `Status` |
| `CiSigma::apply(res, k, x, y)` | Y = H X for size() × k device blocks: the `LinearOperator` member; `Status` |
| `CiSigma::diagonal(res, diagonal)` | diag(H) into size() device doubles; `Status` |
| `CiSigma::rdm1(res, c, gamma)` | The 1-RDM γ_tu = ⟨c\|E_tu\|c⟩ of one device vector into n × n device doubles; `Status` |
| `CiSigma::size()` | alpha.size() · beta.size(), the operator's n |
| `host_sigma(alpha, beta, e_core, h_eff, eri, c, sigma)` | σ = Hc; `Status` |
| `host_diagonal(alpha, beta, e_core, h_eff, eri, diagonal)` | H_II for every determinant; `Status` |

All take the integrals as host arrays: `h_eff` n × n and `eri` n⁴, row-major,
(tu\|vw) in chemists' order, real with the usual symmetry. They refuse
(`InvalidConfig`, nothing written) two `StringSet`s over different orbital
counts, or any span of the wrong size. `CiSigma` also refuses no orbitals, more
determinants (or rows of a block) than an `int32_t` counts, another device than
`create`'s, and `apply` or `diagonal` before `set_integrals`.

## Design

**The layout is `direct_spin1`'s.** Determinant (I_α, I_β) is element
I_α · beta.size() + I_β, and the strings are in PySCF's order (see
`../strings/`), so a vector compares with PySCF's element for element.

**H is written with the one-electron part absorbed.**
H = E_core + Σ k_tw E_tw + ½ Σ (tu\|vw) E_tu E_vw, with
k_tw = h_tw − ½ Σ_u (tu\|uw) taking the −δ_uv E_tw of the two-electron term.

**σ is gathered and scattered per determinant K.** K's α list (K_β held) and
β list (K_α held) are every a†_t a_u that takes K somewhere, so:

1. D_K[vw] = (E_vw c)_K = Σ_I ⟨I\|E_wv\|K⟩ c_I, gathered over both lists;
2. σ_K += Σ k_vw D_K[vw];
3. G_K[tu] = ½ Σ (tu\|vw) D_K[vw];
4. σ_I += ⟨I\|E_tu\|K⟩ G_K[tu], scattered over both lists.

E_tu = E^α_tu + E^β_tu in both the gather and the scatter, so the α–α, β–β and
α–β terms all come through the replacement lists with nothing spin-specific.
Cost is O(dim · n⁴), scratch O(n²): no vector of D's is held.

**diag(H) is Slater–Condon's diagonal rule**, over the occupied spin orbitals α
ascending then β ascending: E_core + Σ_i h_ii + Σ_{i<j} [(ii\|jj) − δ_σ (ij\|ji)],
summed in that order. The order is documented because the test's dense H
sums its diagonal the same way, and the two are compared for equality.

**The GPU sigma is the same algorithm, regrouped.** `CiSigma::apply` takes
each (vector, determinant) pair of the block as a *row*, and runs:

1. a gather, one thread per row, of D_K over K's α and β lists;
2. one gemm per chunk of rows, G = D [½ (tu\|vw) | k_vw], whose last column is
   the one-electron term;
3. a gather, one thread per row, of σ_I = E_core c_I + G_I[k] + Σ s · G_J[tu]
   over I's *own* lists, since ⟨I\|E_qp\|J⟩ = ⟨J\|E_pq\|I⟩.

Step 3 turns the host's scatter into a gather, so no thread writes another's
element: nothing is atomic and a run's result does not depend on scheduling.
D and G are packed over the pairs v ≥ w, n(n+1)/2 of them, because (tu\|vw) is
symmetric in each pair; D's packed entry is D[vw] + D[wv]. The scratch is
`res.device_workspace()`: G for the whole block, k · dim · (n(n+1)/2 + 1)
doubles (540 MB at CAS(12,12), k = 1), and D for one 64 MiB chunk of rows. G
is whole because step 3 reads any J's row.

**Tables once, integrals as often as needed.** `create` uploads the two sets'
strings and their lists (target, packed pair, sign per replacement), once per
(n, N_α, N_β); `set_integrals` uploads the packed contraction matrix and the
diagonal's terms, and a CASSCF can call it again each macro-iteration. The
gemm passes host scalars, so `res.blas()` must be in host pointer mode, its
default and what `davidson_solve` holds.

**The 1-RDM is the first gather and a gemv.** γ_vw = Σ_K c_K (E_vw c)_K =
Σ_K c_K D_K[vw], so `rdm1` runs apply's step 1 over each chunk of
determinants and then one gemv, Dᵀ c, into the packed sums, summed across
chunks by the gemv's β. A packed entry is D[vw] + D[wv], so the sum is 2 γ_vw
off the diagonal, and a small kernel unpacks it, halving those. It reads
only the link tables, so it needs no `set_integrals`, and its scratch is one
chunk of D plus n(n+1)/2 doubles. The 2-RDM, Dᵀ D, is the same gather with a
gemm.

**diag(H) on the device** is host_diagonal's rule in host_diagonal's order, one
thread per determinant.

## Tests

`test/gtest/ci_sigma_host.cpp` builds a dense H by the Slater–Condon rules
directly on determinants as spin-orbital bit patterns, sharing no code with the
replacement lists. Against it:

- σ on random unit vectors to 1e-12, for random integrals over eight shapes
  (open and closed shells, an empty spin, a full one, a single orbital) and
  for every golden case in `test/gtest/data/ci/`;
- diag(H) exactly;
- the dense H's lowest eigenvalues (Householder, then bisection) against
  `direct_spin1`'s roots to 1e-10, for every golden case;
- the refusals.

`test/gtest/ci_sigma_gpu.cpp` checks `CiSigma` against the host pair: σ on
random blocks of one to four vectors, each column to 1e-12 ‖σ‖, and diag(H)
to 1e-12, and `rdm1` on random unit vectors against `ci.rdm`'s `rdm1` to
1e-12, for every golden case, for random integrals at CAS(8,8),
CAS(10,10) (a block spanning two chunks) and seven shapes with N_α ≠ N_β (an
empty spin, a full one and a single orbital among them), and at CAS(12,12) under `slow_test.h`; `rdm1` also at CAS(12,12) in the fast tier, over
eight chunks; then the refusals.

## Files

| File | Role |
|------|------|
| `sigma.cppm` | Module interface: `CiSigma`, `host_sigma`, `host_diagonal` |
| `sigma.cpp` | The host gather–contract–scatter and diagonal; `CiSigma`'s uploads, chunked gemm and gemv, and launches |
| `ci_sigma_bridge.h` | The launchers' declarations, shared by `sigma.cpp`'s GMF and the `.cu` |
| `ci_sigma.cu` | The two gathers, the diagonal kernel and the 1-RDM's unpacking |
| `CMakeLists.txt` | The device library `free_breakfast.ci.sigma.device` and the module `free_breakfast.ci.sigma` |
