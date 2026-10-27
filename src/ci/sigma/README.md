# Sigma

σ = Hc and diag(H) for a determinant CI vector: `ci::Sigma` on the device, as a
`LinearOperator` (`recipes.linear_operator`) over a block of CI vectors, with
the 1-RDM over the same tables. The host references it is tested against,
`host_sigma` and `host_diagonal`, are in `test/ci_oracles/`: written for
clarity, not speed, and linked by the tests alone.

## Module

`free_breakfast.ci.sigma`, also re-exported by `free_breakfast.ci`; target
`free_breakfast::ci::sigma`.

| Name | Role |
|---|---|
| `ci::Sigma::create(res, alpha, beta)` | Upload the strings and replacement lists; `Result<ci::Sigma>` |
| `ci::Sigma::set_integrals(res, e_core, h_eff, eri)` | Upload an H over them, replacing any before; `Status` |
| `ci::Sigma::apply(res, k, x, y, layout = full)` | Y = H X for size(layout) × k device blocks: the `LinearOperator` member; `Status` |
| `ci::Sigma::apply_fp32(res, k, x, y, layout = full)` | The same Y = H X over the same double blocks, D, G and the gemm in float, to ~1e-7 relative; `ci::solve`'s first stage; `Status` |
| `ci::Sigma::set_workspace_limit(bytes)` | Bound `apply`'s and `apply_fp32`'s device workspace; past it G is computed a block of strings at a time (below) |
| `ci::Sigma::diagonal(res, diagonal)` | diag(H) into size() device doubles; `Status` |
| `ci::Sigma::exchange()` | The device (ij\|ji), n × n, that `ci::solve`'s spin-averaged diagonal takes; null before `set_integrals` |
| `ci::Sigma::contraction()` | The device ½ (tu\|vw) and k_vw, packed over the pairs, that `apply` contracts and `ci::solve`'s P-space reads its Slater–Condon elements from; null before `set_integrals` |
| `ci::Sigma::rdm1(res, c, gamma, layout)` | The 1-RDM γ_tu = ⟨c\|E_tu\|c⟩ of one device vector, full or packed (`layout`, default full), into n × n device doubles; `Status` |
| `ci::Sigma::size(layout = full)` | The operator's n: alpha.size() · beta.size(), or the symmetric layout's alpha.size() (alpha.size() + 1) / 2 |
| `ci::Layout` | `full`, every determinant; or `symmetric`, the packed triangle of a vector symmetric under the swap of the two strings |
| `ci::unpack(res, n_strings, k, x, y)` | k symmetric-layout device vectors into full ones; `Status` |
| `ci::pack(res, n_strings, k, x, y)` | k full-layout device vectors into symmetric-layout ones, `ci::unpack`'s inverse on swap-symmetric vectors (its adjoint: any other goes to its symmetric part); `ci::solve`'s start; `Status` |

`set_integrals` takes the integrals as host arrays: `h_eff` n × n and `eri`
n⁴, row-major, (tu\|vw) in chemists' order, real with the usual symmetry.
`ci::Sigma` refuses (`InvalidConfig`, nothing written) two `ci::StringSet`s over
different orbital counts, any span of the wrong size, no orbitals, more
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

**σ is gathered and scattered per determinant K**, which is `host_sigma`'s
loop. K's α list (K_β held) and
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

**The GPU sigma is the same algorithm, regrouped.** `ci::Sigma::apply` takes
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
not grow with k: a whole block's G would be 60 GB for `ci::solve`'s six first
guesses at CAS(14,14). The rows are every determinant in the full layout and
the triangle's in the symmetric one (below), so G is, in double:

| | rows, full / symmetric | G, full | G, symmetric |
|---|---|---|---|
| CAS(12,12) | 853,776 / 427,350 | 540 MB | 270 MB |
| CAS(14,14) | 11,778,624 / 5,891,028 | 10.0 GB | 5.0 GB |

and half that again in float (`apply_fp32`).

**Past a workspace limit, G goes a block of strings at a time.** At
CAS(14,14) the symmetric G is 5 GB beside ~5 GB of Davidson subspace, past the
RTX 3080's 10 GB, so `set_workspace_limit` bounds the scratch. While G for the
whole vector and a chunk of D fit the limit nothing changes, to the bit (and
unset, the limit is unbounded). Past it, step 3 is split by which of I's links
a G row serves. Row I = (a, b) reads G at

- (a, b′) over its **β links**, the same α string, and
- (a′, b) over its **α links**, the same β string.

So the G of an *α block*, the rows whose α string is in [first, first + count)
against every β string, holds every G that the β links of those same rows
read; and that of a *β block* every G the α links of its rows read. In the
full layout `apply` makes two passes: α blocks, each running steps 1 and 2 over
its rows and then adding E_core c_I, G_I's one-electron term and the β-link
terms to its own rows; then β blocks, each adding the α-link terms. σ is
zeroed first and every block adds, each thread still owning its row, so
nothing is atomic and the result does not depend on the run; it differs from
the unblocked σ only in the order the same terms are summed, and in the gemm's
own blocking over a shorter m: ~1e-16 relative in double, and in float up
to float rounding, ~1e-7, where the gemm picks another kernel for the shape.
The cost is the first gather and the gemm twice over (each pass covers every determinant once), and G's
scratch is one block's.

**In the symmetric layout one pass does.** Take a block of strings S of the
one set and hold G over the *unpacked* determinants (s, x), s ∈ S, x any,
|S| · m rows (the gather reads c through the mirror, as for any J below the
triangle). A triangle row (a, b), a ≥ b:

- its β links read G(a, b′): held when a ∈ S;
- its α links read G(a′, b) = G(b, a′) by the swap symmetry of G: held when
  b ∈ S.

So the α block and the β block over S are the same rows up to the mirror, and
one block serves both halves: each block adds the β-link part (with E_core c
and the one-electron term) to the triangle rows with a ∈ S and the α-link part
to those with b ∈ S. Walking the blocks over all m strings gives every row
each part exactly once. The gathers and the gemm then run over m² rows, the
unpacked vector's, twice the triangle's t = m (m + 1)/2, as the full layout's
two passes run over 2 m²: blocking costs the symmetric layout's halving and no
more. (A block also holds both G(s, x) and G(x, s) when s, x ∈ S; skipping the
mirrored copies would trim the work by |S|² / 2 a block, at the cost of a
ragged block.)

The block is the most strings whose G and D fit the limit, at least one. While
bounded, a call first grows the workspace to the lesser of the limit and
`apply`'s unblocked double scratch, so `ci::solve`'s `apply_fp32` stage and its
`apply` stage carve one allocation, and the limit holds at the peak: a growth
from the float workspace to the double one would hold both for a moment.
`pipeline/fci` sets the limit to whatever its memory estimate leaves free.

On the RTX 3080, at N₂ cc-pVDZ CAS(12,12) (78 pairs) a sigma column costs, per
`bench/ci_sigma_blocked`, with the workspace it takes:

| Layout, precision | Unbounded | Under 200 MiB | Under 100 MiB |
|---|---|---|---|
| full, FP64 | 38.4 ms, 579 MiB | 73.0 ms | 72.3 ms |
| full, FP32 | 5.3 ms, 321 MiB | 11.4 ms | 10.7 ms |
| symmetric, FP64 | 18.2 ms, 322 MiB | 38.5 ms | 37.4 ms |
| symmetric, FP32 | 3.0 ms, 193 MiB | 3.0 ms (fits unblocked) | 6.9 ms |

(the workspace under a limit is the limit: the call grows it there first).
Blocking costs the predicted 2×, and nothing more as the blocks shrink. So blocking is
off unless asked for: unset, `apply` is the single pass it was. N₂ cc-pVDZ
CAS(14,14), three singlets, which wants a 5 GB G unblocked and aborts on the
workspace growth on that card, converges in `ci::solve`'s symmetric layout
under a 3000 MiB limit in 217 s, at a card-wide peak of 9,475 MiB, 767 of it
held before the run.

**The symmetric layout halves the work when N_α = N_β.** Over one
`ci::StringSet` for both spins, swapping the two strings of every determinant,
(I_α, I_β) → (I_β, I_α), commutes with H: E_tu = E^α_tu + E^β_tu is
spin-free, and the replacement lists carry each spin's own sign and nothing
across, so the matrix the lists build is unchanged by the swap. Every
eigenvector is then symmetric or antisymmetric, c(I_α, I_β) = ±c(I_β, I_α),
and in the M_s = 0 sector the parity is (−1)^S: a singlet's is symmetric,
which is what `fci.direct_spin0` uses. For a symmetric c, D_K = (E c)_K, and so
G_K and σ_K, are symmetric under the swap too, so only the determinants
K_α ≥ K_β need computing. `ci::Layout::symmetric` holds just those, m (m + 1)/2
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
implemented: `ci::solve` wants singlets, and the antisymmetric (odd S) states of
M_s = 0 are what M_s = 1 solves for. The triangle's index map is
`ci_symmetric.h`, shared with `ci.solver`'s S² and `ci.rdm`'s gather.
`ci::unpack` takes packed vectors back to the full layout; `rdm1` and
`ci::Rdm::rdm12` read either.

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

In the symmetric layout D_K is D of K's mirror (E_vw commutes with the swap
and c is symmetric), so Σ_K c_K D_K is Σ over the triangle of w_K c_K D_K, w_K
2 off the diagonal and 1 on it. The gather walks the triangle's rows with
`scale_off_diagonal`, which multiplies each off-diagonal row by √2, and the
gemv takes the packed c, whose off-diagonal elements carry the other √2; the
sum is then the full layout's, over half the rows.

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
root, which needs a residual of 1e-12; it is what `ci::solve` uses for its
first iterations (see `../solver/`).

**diag(H) on the device** is that rule in that order (`host_diagonal`'s too),
one thread per determinant.

## Tests

`test/gtest/ci_sigma_host.cpp` checks the host pair. It builds a dense H by the Slater–Condon rules
directly on determinants as spin-orbital bit patterns, sharing no code with the
replacement lists. Against it:

- σ on random unit vectors to 1e-12, for random integrals over eight shapes
  (open and closed shells, an empty spin, a full one, a single orbital) and
  for every golden case in `test/gtest/data/ci/`;
- diag(H) exactly;
- the dense H's lowest eigenvalues (Householder, then bisection) against
  `direct_spin1`'s roots to 1e-10, for every golden case;
- the refusals.

`test/gtest/ci_sigma_gpu.cpp` checks `ci::Sigma` against the host pair: σ on
random blocks of one to four vectors, each column to 1e-12 ‖σ‖, and diag(H)
to 1e-12, and `rdm1` on random unit vectors against `host_rdm1` to
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
four chunks of triangle rows in double, under `slow_test.h`; `ci::unpack` of
each packed block gives the block back; `ci::pack` of random swap-symmetric
full blocks over 1 to 924 strings is the host's packing and `ci::unpack` of it
the block, each to 1e-15, and of blocks that are not symmetric it is the
packing of their symmetric part; and the layout is refused over two
sets with the same number of strings, (4, 1) and (4, 3), by `apply`,
`apply_fp32` and `rdm1`. In double, `rdm1` of each packed column also matches
the full layout's on the unpacked one and `host_rdm1` to 1e-12 of
max |γ|; `test/gtest/ci_rdm_gpu.cpp` checks it again on unit vectors.
Blocked, `apply` under a forced limit equals the unblocked result to 1e-13
relative per column, and `apply_fp32` to its own float bound (the float gemm's
rounding depends on the row count it is handed, and cuBLAS on the RTX 3080
differs by ~1e-7 between the blocked and unblocked shapes of some full-layout
cases), in both layouts, on every golden case and
the random shapes (and swap-symmetric ones), at one string a block and at a
third of the unblocked scratch (a ragged last block); at CAS(12,12), under
`slow_test.h`, with blocks spanning several chunks of D. A limit at the
unblocked scratch gives the unbounded result bit for bit.

## Files

| File | Role |
|------|------|
| `sigma.cppm` | Module interface: `ci::Layout`, `ci::unpack`, `ci::pack`, `ci::Sigma` |
| `sigma.cpp` | `ci::Sigma`'s uploads, chunked (and, past the limit, blocked) gemm in double or float, gemv, and launches; `ci::unpack`'s and `ci::pack`'s launches |
| `ci_sigma_bridge.h` | The launchers' declarations, shared by `sigma.cpp`'s GMF and the `.cu` |
| `ci_symmetric.h` | The symmetric layout's triangle index map and √2, for device code here and in `../solver/` |
| `ci_sigma.cu` | The two gathers (over double or float D and G, in either layout, whole or by string blocks), the diagonal kernel, the 1-RDM's unpacking, `ci::unpack` and `ci::pack` |
| `CMakeLists.txt` | The device library `free_breakfast.ci.sigma.device` and the module `free_breakfast.ci.sigma` |
