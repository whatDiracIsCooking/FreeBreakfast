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
| `CiSigma::apply(res, k, x, y, layout = full)` | Y = H X for size(layout) × k device blocks: the `LinearOperator` member; `Status` |
| `CiSigma::apply_fp32(res, k, x, y, layout = full)` | The same Y = H X over the same double blocks, D, G and the gemm in float, to ~1e-7 relative; `ci_solve`'s first stage; `Status` |
| `CiSigma::diagonal(res, diagonal)` | diag(H) into size() device doubles; `Status` |
| `CiSigma::rdm1(res, c, gamma)` | The 1-RDM γ_tu = ⟨c\|E_tu\|c⟩ of one device vector into n × n device doubles; `Status` |
| `CiSigma::size(layout = full)` | The operator's n: alpha.size() · beta.size(), or the symmetric layout's alpha.size() (alpha.size() + 1) / 2 |
| `CiLayout` | `full`, every determinant; or `symmetric`, the packed triangle of a vector symmetric under the swap of the two strings |
| `ci_unpack(res, n_strings, k, x, y)` | k symmetric-layout device vectors into full ones; `Status` |
| `host_sigma(alpha, beta, e_core, h_eff, eri, c, sigma)` | σ = Hc; `Status` |
| `host_diagonal(alpha, beta, e_core, h_eff, eri, diagonal)` | H_II for every determinant; `Status` |

All take the integrals as host arrays: `h_eff` n × n and `eri` n⁴, row-major,
(tu\|vw) in chemists' order, real with the usual symmetry. They refuse
(`InvalidConfig`, nothing written) two `StringSet`s over different orbital
counts, or any span of the wrong size. `CiSigma` also refuses no orbitals, more
determinants than an `int32_t` counts, a negative block, another device than
`create`'s, `apply`, `apply_fp32` or `diagonal` before `set_integrals`, and the
symmetric layout over two different sets.

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
`res.device_workspace()`: G for one vector's rows, rows · (P + 1) with
P = n(n+1)/2, and D for one 64 MiB chunk of them. G spans the whole vector
because step 3 reads any J's row, but only J's in I's own vector, so `apply`
runs the three steps one vector of the block at a time and the scratch does
not grow with k: a whole block's G would be 60 GB for `ci_solve`'s six first
guesses at CAS(14,14). The rows are every determinant in the full layout and
the triangle's in the symmetric one (below), so G is, in double:

| | rows, full / symmetric | G, full | G, symmetric |
|---|---|---|---|
| CAS(12,12) | 853,776 / 427,350 | 540 MB | 270 MB |
| CAS(14,14) | 11,778,624 / 5,891,028 | 10.0 GB | 5.0 GB |

and half that again in float (`apply_fp32`).

**The symmetric layout halves the work when N_α = N_β.** Over one
`StringSet` for both spins, swapping the two strings of every determinant,
(I_α, I_β) → (I_β, I_α), commutes with H: E_tu = E^α_tu + E^β_tu is
spin-free, and the replacement lists carry each spin's own sign and nothing
across, so the matrix the lists build is unchanged by the swap. Every
eigenvector is then symmetric or antisymmetric, c(I_α, I_β) = ±c(I_β, I_α),
and in the M_s = 0 sector the parity is (−1)^S: a singlet's is symmetric,
which is what `fci.direct_spin0` uses. For a symmetric c, D_K = (E c)_K, and so
G_K and σ_K, are symmetric under the swap too, so only the determinants
K_α ≥ K_β need computing. `CiLayout::symmetric` holds just those, m (m + 1)/2
of the m² for a set of m strings, K at triangle element K_α (K_α + 1)/2 + K_β,
each off-diagonal one scaled by √2. The scaling makes
the packing an isometry of the symmetric subspace, so the plain dot product,
and with it H's symmetry, are kept, and `davidson_solve` runs on packed vectors
as it stands (`../solver/` has why that, and not full-length vectors with only
sigma's work skipped).

In that layout the first gather reads a link to J below the triangle from its
mirror, with the 1/√2 of whichever element it reads; the gemm runs over the
triangle's rows alone; and the second gather reads G_J of such a J from its
mirror's row, unscaled, then scales σ_I by √2 off the diagonal. So D, the gemm
and G all halve, and nothing is atomic still. Only the symmetric parity is
implemented: `ci_solve` wants singlets, and the antisymmetric (odd S) states of
M_s = 0 are what M_s = 1 solves for. The triangle's index map is
`ci_symmetric.h`, shared with `ci.solver`'s S². `ci_unpack` takes packed
vectors back to the full layout, which is what `rdm1` and `CiRdm::rdm12` read.

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

**`apply_fp32` is `apply` with D, G and the contraction in float.** The
vectors stay double; `set_integrals` uploads the contraction matrix a second
time, rounded to float (P (P + 1) floats beside the doubles), and `apply_fp32`
runs the same three steps with float D and G and `gemm<float>`. The gemm is
~85% of an FP64 sigma (CAS(12,12) on the RTX 3080, by nsys), and both cards
here are FP64-poor (the 3080 runs FP64 at 1/64 of its FP32 rate), so a column
is 6–7× cheaper there and ~3.5× on the gfx1200. Halving D and G also halves the gathers'
traffic and G's scratch, dim · (P + 1) floats. The gathers stay one thread per
row, deterministic and free of atomics. D accumulates in float, because an
entry takes at most two terms, one link of each spin (E_vw and E_wv cannot
both act on one string), so double accumulation would change one rounding at
most. σ's gather sums G in double. The error is then the three roundings,
D's, the contraction's and G's, adding in quadrature (the blocked FMA gemm
adds little of its own): 1–2 u, ~1e-7 relative of ‖σ − E_core c‖, flat in
P, on the RTX 3080 over the test's shapes and on both cards over N₂'s
CAS(10,10) and CAS(12,12) ground states (`bench/ci_sigma_precision`). That is no use for a converged
root, which needs a residual of 1e-12; it is what `ci_solve` uses for its
first iterations (see `../solver/`).

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
CAS(10,10), 11 orbitals over (5, 4) (two vectors, each spanning two chunks)
and seven shapes with N_α ≠ N_β (an
empty spin, a full one and a single orbital among them), and at CAS(12,12) under `slow_test.h`; `rdm1` also at CAS(12,12) in the fast tier, over
eight chunks; then the refusals. `apply_fp32` is checked over the same golden
cases and random shapes (and CAS(12,12) under `slow_test.h`), each column to
4 sqrt(P + 3) u of ‖σ − E_core c‖, u = 2⁻²⁴, a bound derived in the test from
the float roundings; the measured error is 1–2 u, and an error under 1e-12
fails too, so the double path cannot stand in for it. The symmetric layout,
`apply` and `apply_fp32`, is checked on random swap-symmetric blocks, packed,
against `host_sigma`'s σ packed, to the same two tolerances, for every golden
case with N_α = N_β and seven shapes over one set (CAS(8,8), CAS(10,10), a
single orbital, an empty set and a full one among them), and at CAS(12,12),
four chunks of triangle rows in double, under `slow_test.h`; `ci_unpack` of
each packed block gives the block back; and the layout is refused over two
sets with the same number of strings, (4, 1) and (4, 3).

## Files

| File | Role |
|------|------|
| `sigma.cppm` | Module interface: `CiLayout`, `ci_unpack`, `CiSigma`, `host_sigma`, `host_diagonal` |
| `sigma.cpp` | The host gather–contract–scatter and diagonal; `CiSigma`'s uploads, chunked gemm (double or float) and gemv, and launches; `ci_unpack`'s launch |
| `ci_sigma_bridge.h` | The launchers' declarations, shared by `sigma.cpp`'s GMF and the `.cu` |
| `ci_symmetric.h` | The symmetric layout's triangle index map and √2, for device code here and in `../solver/` |
| `ci_sigma.cu` | The two gathers (over double or float D and G, in either layout), the diagonal kernel, the 1-RDM's unpacking and `ci_unpack` |
| `CMakeLists.txt` | The device library `free_breakfast.ci.sigma.device` and the module `free_breakfast.ci.sigma` |
