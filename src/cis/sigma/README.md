# Sigma

The integral-direct CIS operator σ = A b over the n_ov excitation vectors of
an RHF reference, as a `LinearOperator<…, double>` for `davidson_solve`, and
the method's small vocabulary: the spin and the knobs of a solve.

## Module

`free_breakfast.cis.sigma`; target `free_breakfast::cis::sigma`. It launches
nothing of its own: BLAS, `recipes.matrix_ao`'s spherical bridge and
[`nonsymmetric_jk`](../../integrals/eri/integral_direct/nonsymmetric_jk/README.md),
so there is no device library.

| Name | Role |
|---|---|
| `cis::Spin` | `Singlet` or `Triplet` (the Ms = 0 component) |
| `cis::Options` | `n_states` (5), `spin` (`Singlet`) and `davidson`, a `DavidsonOptions` at `residual_tolerance = 1e-6` |
| `cis::Sigma::create(res, basis, u, j_classes, k_classes, c, n_occ, delta, spin)` | `Result<Sigma>` over the spherical RHF orbitals `c`, the first `n_occ` occupied, and `cis::diagonal`'s device `delta`; allocates every buffer `apply` writes |
| `Sigma::apply(res, k, x, y)` | `Status`: Y = A X for the n_ov × k column-major device blocks, one column at a time |
| `Sigma::size()` | n_ov = n_occ · n_virt |

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
every column waits for the last; the density-fitted sigma (#200) takes the
whole block at once.

**The memory.** D, R and the other of G and K are held across
`nonsymmetric_jk`, which takes the device workspace, so they are the sigma's
own (`docs/style.md` "Scratch memory"): two `MatrixAOPair`s (D and R, each
with its own conversion scratch), one `CartesianMatrixAO` and the
n_ao × n_occ `half`, about 3 n_cart² + 3 n_sph² + n_sph · n_occ doubles. `create`
also grows the workspace to `nonsymmetric_jk`'s one n_cart² block, so no
`apply`, not even the first, allocates anything of the sigma's or grows the
workspace.

What an `apply` still draws is `coulomb`'s and `exchange`'s own: each build
takes a few small buffers from the device's stream-ordered pool (`exchange`'s
decay fields, max bound, shell centres and offsets, `coulomb`'s row map; about
2 KB on H₂O / cc-pVDZ) and returns them before it ends. Their sizes depend on
the basis alone, so the pool serves them from memory it already holds, and a
block of columns peaks no higher than one build. The issue's "apply allocates
nothing" would need those two to take caller scratch, which is a change to
`src/integrals/`, not to this module; the test holds the sigma to what it can
own: a second `apply` of four columns peaks no higher in the pool than one bare
`nonsymmetric_jk` call, and leaves the workspace where it was.

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

## Tests

`test/gtest/cis_sigma_gpu.cpp`, over the setup steps' basis and
`pipeline.eri`'s integral-direct classes:

- `CisSigmaSpin.IsSymmetric/*`: ⟨b1, A b2⟩ = ⟨b2, A b1⟩ for random b on
  H₂O / cc-pVDZ (d shells, so the spherical bridge) with synthetic C (uniform
  on [−0.5, 0.5)) and ascending synthetic ε, both spins;
- `CisSigmaSpin.BlockEqualsColumns/*`: a block of four against the same four
  columns applied one call each;
- `CisSigma.SecondApplyAllocatesNothing`: a second apply of four columns
  peaks no higher in the device pool than one bare `nonsymmetric_jk` (see
  "The memory"), and leaves the workspace unchanged;
- `CisSigma.Refusals`;
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
| `sigma.cpp` | The checks, the buffers, and the five steps of a column |
| `CMakeLists.txt` | The module library |
