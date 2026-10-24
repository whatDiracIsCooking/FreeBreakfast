# Sigma

σ = Hc and diag(H) for a determinant CI vector, on the host. Written for
clarity, not speed: it is the reference the GPU sigma is tested against.

## Module

`free_breakfast.ci.sigma`, also re-exported by `free_breakfast.ci`; target
`free_breakfast::ci::sigma`.

| Name | Role |
|---|---|
| `host_sigma(alpha, beta, e_core, h_eff, eri, c, sigma)` | σ = Hc; `Status` |
| `host_diagonal(alpha, beta, e_core, h_eff, eri, diagonal)` | H_II for every determinant; `Status` |

Both take the integrals as plain arrays: `h_eff` n × n and `eri` n⁴, row-major,
(tu\|vw) in chemists' order, real with the usual symmetry. They refuse
(`InvalidConfig`, nothing written) two `StringSet`s over different orbital
counts, or any span of the wrong size.

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

## Files

| File | Role |
|------|------|
| `sigma.cppm` | Module interface: `host_sigma`, `host_diagonal` |
| `sigma.cpp` | The gather–contract–scatter, and the diagonal |
| `CMakeLists.txt` | The module `free_breakfast.ci.sigma` |
