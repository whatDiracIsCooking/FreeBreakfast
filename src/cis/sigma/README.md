# Sigma

The CIS operator σ = A b over the n_ov excitation vectors of an RHF
reference, as a `LinearOperator<…, double>` for `davidson_solve`,
integral-direct (`cis::Sigma`) or density-fitted in the MO basis
(`cis::DfSigma`), and the method's small vocabulary: the spin and the knobs of
a solve.

## Module

`free_breakfast.cis.sigma`; target `free_breakfast::cis::sigma`. It launches
nothing of its own: BLAS, `recipes.matrix_ao`'s spherical bridge,
[`nonsymmetric_jk`](../../integrals/eri/integral_direct/nonsymmetric_jk/README.md)
and [`b_mo`](../../integrals/eri/density_fitting/mo/README.md), so there is no
device library.

| Name | Role |
|---|---|
| `cis::Spin` | `Singlet` or `Triplet` (the Ms = 0 component) |
| `cis::Options` | `n_states` (5), `spin` (`Singlet`), `davidson`, a `DavidsonOptions` at `residual_tolerance = 1e-6`, and `cluster_gap` (0: off), the preconditioner's (`../precondition/`) |
| `cis::Sigma::create(res, basis, u, j_classes, k_classes, c, n_occ, delta, spin)` | `Result<Sigma>` over the spherical RHF orbitals `c`, the first `n_occ` occupied, and `cis::diagonal`'s device `delta`; allocates every buffer `apply` writes |
| `Sigma::apply(res, k, x, y)` | `Status`: Y = A X for the n_ov × k column-major device blocks, one column at a time |
| `Sigma::size()` | n_ov = n_occ · n_virt |
| `cis::DfSigma::create(res, b, naux, c, n_occ, delta, spin, max_block)` | `Result<DfSigma>` over the same orbitals and `delta`, and `pipeline.eri`'s `DensityFitted` B; builds the MO blocks and grows the workspace for blocks of `max_block` columns |
| `DfSigma::apply(res, k, x, y)` | `Status`: Y = A X for the whole block, `max_block` columns per pass |
| `DfSigma::size()` | n_ov |

```cpp
const auto &direct = std::get<fb::IntegralDirect>(eri.eri);     // pipeline.eri
FBF_TRY(fb::cis::diagonal(res, n_occ, n_virt, eps.data(), delta.data()));
auto sigma = FBF_TRY(fb::cis::Sigma::create(res, basis, transform.u, direct.j_classes,
                                            direct.k_classes, scf.c, n_occ, delta.data(),
                                            options.spin));
const fb::cis::Preconditioner precondition{res, sigma.size(), delta.data(), options.n_states};
FBF_TRY(fb::davidson_solve<double>(res, sigma, precondition, sigma.size(), options.n_states,
                                   max_subspace, start.data(), guess_count, values.data(),
                                   vectors.data(), options.davidson));
```

## Design

An excitation vector's element ia is at i · n_virt + a, so a column of the
block, read column-major as an n_virt × n_occ matrix, is Xᵀ. With C_occ and
C_virt the occupied and virtual columns of C, a singlet's A is
Δε_ia δ_ij δ_ab + 2 (ia|jb) − (ij|ab) and a triplet's drops 2 (ia|jb), so

  σ = Δε ∘ b + w · C_occᵀ R[D] C_virt, D = C_occ X C_virtᵀ,

with R = G = J − K/2 and w = 2 for a singlet and R = K and w = −1 for a
triplet, J_mn = Σ (mn|ls) D_ls and K_mn = Σ D_ls (ml|ns), the convention
`nonsymmetric_jk` computes. R is not symmetric (D is not), and the order
matters: (ij|ab) X_jb is Σ_mn C_mi K_mn C_na, C_occ on K's row index.

Per column, all on `res.stream()`:

1. `half = C_virt B` (n_ao × n_occ), then `D = C_occ halfᵀ`: two `gemm`s.
2. `D` onto the Cartesian layout, U D Uᵀ (`MatrixAOPair::spherical_to_cartesian`).
3. `G` and `K` of that Cartesian `D` (`nonsymmetric_jk`).
4. `R` back, Uᵀ R U (`MatrixAOPair::cartesian_to_spherical`).
5. `σ = Δε ∘ b` (`dgmm`), `half = Rᵀ C_occ`, then `σ += w C_virtᵀ half`: the
   second `gemm` accumulates into σ with β = 1, so there is no separate
   `axpy`.

**gemm, not gemmwich.** The issue named `gemmwich` for steps 1 and 5, but
`gemmwich` is a sandwich with one X on both sides (Xᵀ F X), and D and the
projection have C_occ on one side and C_virt on the other. Through `gemmwich`
they would need b padded into an n_mo × n_mo matrix and the ov block cut back
out of an n_mo × n_mo product; two packed `gemm`s each do the same products
without either copy. Since R is not symmetric, `half = Rᵀ C_occ` is what makes
step 5 C_occᵀ R C_virt and not its transpose.

**Columns in sequence.** J's and K's classes hold one build's scratch, so
every column waits for the last; the density-fitted sigma below takes the
whole block at once.

**The memory.** D, R and the other of G and K are held across
`nonsymmetric_jk`, which takes the device workspace, so they are the sigma's
own (`docs/style.md` "Scratch memory"): two `MatrixAOPair`s (D and R, each
with its own conversion scratch), one `CartesianMatrixAO` and the
n_ao × n_occ `half`, about 3 n_cart² + 3 n_sph² + n_sph · n_occ doubles. `create`
also grows the workspace to `nonsymmetric_jk`'s one n_cart² block, so no
`apply`, not even the first, allocates anything of the sigma's or grows the
workspace. `coulomb`'s and `exchange`'s own scratch is their classes',
allocated once per basis, so an `apply` takes nothing from the device's pool:
the test holds a second `apply` of four columns to exactly that, and to
leaving the workspace where it was.

**The screen.** `nonsymmetric_jk`'s defaults, `tau_j = 1e-10` and
`tau_k = 1e-13`, and no knob. J's default is the coarse one: its README
measures G at 9e-12 of max|G| from libcint on a dense *random* D, the worst
case for a per-term bound. That error enters σ as a perturbation of A of the
same relative size, so it moves an eigenvalue by about 1e-11 of ||A||, six
orders below the 1e-5 bar the physics is held to and five below the residual
`Options::davidson` converges to; and a CIS transition density, a product of
two orbital blocks, decays like the orbitals and is far from the random one.
Tightening J to 3e-12 buys digits nothing reads, at the cost of more J work
per column. The tests run at these defaults: symmetry ⟨b1, A b2⟩ = ⟨b2, A b1⟩
to 1e-12 of the largest |σ| times n_ov, and both references to 1e-5. The
screen does show in the assembled A: because it is per density, column j's
(ia|jb) and column i's are screened apart. HF / cc-pVTZ's full A, assembled
once while this was written, is symmetric to 1.4e-11 of max|A| (the same on
both cards), the size the J README predicts, and its three singlets were within
1e-5 of the reference; H₂O / STO-3G's, which the test assembles, to 1e-15, held to
1e-10.

**Not checked:** that |(D + Dᵀ)/2| <= 2, the cap J's classes were pruned for,
as `nonsymmetric_jk` does not check it. For a unit b, |D_mn| is at most the
norm of C_occ's row m times C_virt's row n; normalized orbitals keep that well
inside it, and Davidson's vectors are unit columns.

**`Options`.** `davidson_solve`'s `residual_tolerance` is relative,
||A x − θ x|| <= tol · max(|θ|, ||H||), and the private tree's was an absolute
1e-5 on the residual norm. Excitation energies and the subspace's largest Ritz
value are both of order one hartree, so `1e-6` relative is a residual of about
1e-6 hartree, a little tighter than the absolute 1e-5: an energy good to the
residual squared over the gap, ~1e-11, and a vector, and with it an oscillator
strength, to ~1e-6 over the gap. `n_states = 5` is what `cis.json` records per
spin.

## The density-fitted sigma

`cis::DfSigma` is the same A over `pipeline.eri`'s `DensityFitted` B, with
(pq|rs) ≈ Σ_Q B_Q,pq B_Q,rs, in the MO basis. B is fitted in the AOs, so
`create` transforms it once per SCF with `b_mo` over C's occupied and
virtual columns into three blocks, every one Q fastest:

| Block | `b_mo` rows, cols | Element (Q, p, q) at | Read as |
|---|---|---|---|
| B_vo | virt, occ | Q + naux (a + n_virt i) | the naux × n_ov matrix whose column is excitation ia |
| B_oo | occ, occ | Q + naux (i + n_occ j) | the (naux n_occ) × n_occ matrix with row (Q, i) |
| B_vv | virt, virt | Q + naux (a′ + n_virt a) | per a′, the naux × n_virt slice at Q + naux a′, leading dimension naux n_virt |

Then an `apply` takes the whole n_ov × k block X, a column read as the
n_virt × n_occ matrix Xᵀ (so the block, columns side by side, is the
n_virt × (n_occ k) matrix at x):

1. `σ = Δε ∘ X` (`dgmm`, the whole block);
2. a singlet's Coulomb term, `T = B_vo X` and `σ += 2 B_voᵀ T` (two `gemm`s,
   T naux × k);
3. the exchange term, σ_ja′ −= Σ_ib (ji|a′b) X_ib, in two strided-batched
   gemms through W, naux n_ov k doubles:
   - batched over a′: W_a′ = B_vv(:, a′, :) X, naux × (n_occ k), so that
     W's element (Q, i, c, a′) is at Q + naux (i + n_occ (c + k a′));
   - batched over the columns c: σ_c −= W_cᵀ B_oo, with W_c the
     (naux n_occ) × n_virt matrix at naux n_occ c whose rows (Q, i) are
     adjacent, contracted against B_oo's.

So four BLAS calls past the `dgmm` (two for a triplet), however wide the block.
**Why two batches and not the issue's one over Q.** With Q the fast index of
every block, which is how `b_mo` and `density_fit` leave it, B_oo^Q and
B_vv^Q are matrices whose row stride is naux, which BLAS cannot take. A
batch over Q would need every block transposed to Q-slowest first. Contracting
Q together with an MO index needs no transpose, since those two are adjacent.
The batch then runs over a′ in the first product and over the columns in the
second.

**The memory.** B_vo, B_oo and B_vv are the sigma's own, naux (n_ov + n_occ² +
n_virt²) doubles, with B_vv the dominant one (benzene / cc-pVDZ over
cc-pVDZ-JKFIT, naux 558: 39 MB of B_vv in 49 MB in all). T and W are scratch from the device workspace,
which `create` grows for blocks of `max_block` columns, so no `apply`
allocates or grows anything. `create` adds up the blocks and the workspace's
growth, which is the larger of `b_mo`'s half transform and `apply`'s regions,
and returns `Unsupported` naming the bytes when they are past the device's
free memory. It does that before it allocates anything, since an allocation
past the free memory aborts under `CreatePolicy`. A wider block is run in
slices of `max_block`. `CisStep` creates it at the guess's column count, the
widest block `davidson_solve` applies. Chunking the exchange term over Q, so
that B_vv need not be held whole, is the follow-up if a system needs it.

**No screen.** Every product is dense, and the fit is the approximation. The
roots are held to PySCF's DF TDA over the same aux set, never to the
integral-direct ones (`cis_step_gpu.cpp`).

**What the block buys.** `bench/cis_sigma_df.cpp` times an apply of k
columns over H₂O / cc-pVDZ's RHF (n_ov 95, naux 116 over cc-pVDZ-JKFIT), each
card held, 20 reps after two warm-ups (2026-10-29):

| k | RTX 3080 direct | RTX 3080 DF | RX 9060 XT direct | RX 9060 XT DF |
|---|---|---|---|---|
| 1 | 65.3 ms | 0.19 ms | 104.8 ms | 0.072 ms |
| 4 | 258.8 ms | 0.28 ms | 395.9 ms | 0.076 ms |
| 8 | 517.2 ms | 0.30 ms | 791.3 ms | 0.062 ms |

The direct operator is linear in k, one `nonsymmetric_jk` build per column.
The fitted one is flat at this size: its four BLAS calls are launch-bound
whatever k is.

## Tests

`test/gtest/cis_sigma_gpu.cpp`, over the setup steps' basis and
`pipeline.eri`'s integral-direct classes:

- `CisSigmaSpin.IsSymmetric/*`: ⟨b1, A b2⟩ = ⟨b2, A b1⟩ for random b on
  H₂O / cc-pVDZ (d shells, so the spherical bridge) with synthetic C (uniform
  on [−0.5, 0.5)) and ascending synthetic ε, both spins;
- `CisSigmaSpin.BlockEqualsColumns/*`: a block of four against the same four
  columns applied one call each;
- `CisSigma.SecondApplyAllocatesNothing`: a second apply of four columns
  takes nothing from the device pool (see "The memory"), and leaves the
  workspace unchanged;
- `CisSigma.Refusals`;
- `CisSigmaSpin.DfIsSymmetric/*` and `CisSigmaSpin.DfBlockEqualsColumns/*`:
  the same two over a cc-pVDZ-JKFIT fit, the block of five over a sigma
  created for four, so it runs as a pass of four and a pass of one;
- `CisSigma.DfSecondApplyAllocatesNothing`: a second apply takes nothing from
  the device pool and leaves the workspace unchanged;
- `CisSigma.DfRefusals`;
- `CisSigma.H2oSto3gMatchesReference`: the engine's RHF, A assembled column
  by column (the identity block), checked symmetric and diagonalized, against
  `data/cis/cis.json`'s five singlets and five triplets to 1e-5;
- `CisSigma.HfCcPvtzSingletsMatchReference`: HF / cc-pVTZ's three singlets
  (f shells, the Π pair recorded as one group), not from the full A: its 195
  columns of three f-shell builds each take ~200 s on the RTX 3080 and ~400 s
  on the RX 9060 XT, past ctest's 120 s backstop. So `davidson_solve` finds
  them over the same sigma from `cis::guess`'s start with
  `cis::Preconditioner`, at `cis::Options`' tolerance, to 1e-5. Slow, under
  `FBF_SKIP_IF_FAST_TESTS()`; H₂O / STO-3G stands for the family in the fast
  set.

## Files

| File | Role |
|------|------|
| `sigma.cppm` | Module interface: `cis::Spin`, `cis::Options`, `cis::Sigma` |
| `sigma.cpp` | `Sigma`: the checks, the buffers, and the five steps of a column |
| `df_sigma.cpp` | `DfSigma`: the checks, the memory check, the MO blocks, and the block's products |
| `CMakeLists.txt` | The module library |
