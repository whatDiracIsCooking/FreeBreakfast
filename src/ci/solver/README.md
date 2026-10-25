# Solver

The lowest roots of one spin of the CI Hamiltonian, through `davidson_solve`
over any sigma that is a `LinearOperator`, and S² on a CI vector.

## Module

`free_breakfast.ci.solver`, also re-exported by `free_breakfast.ci`; target
`free_breakfast::ci::solver`. It re-exports `free_breakfast.recipes.davidson`,
whose options and info its own types hold and whose `LinearOperator` a sigma is
written against.

| Name | Role |
|---|---|
| `ci_solve(res, sigma, alpha, beta, diagonal, n_roots, options)` | `Result<CiRoots>`: the `n_roots` lowest roots of spin S = \|N_α − N_β\| / 2. A template over the sigma. |
| `CiSolverOptions` | `spin_penalty` (λ, 1 hartree; 0 keeps every spin), `davidson` (`DavidsonOptions`, residual tolerance 1e-12), `fp32_tolerance` (the single-precision stage's, 1e-6; 0 turns it off) and `olsen_correction` (Olsen's correction in place of the plain one; off) |
| `CiRoots` | `energies`, `spin_squares` (⟨S²⟩), `vectors` (a device buffer in the full layout, root r at [r · dim, (r + 1) · dim)), `davidson` (`DavidsonInfo` of the FP64 stage), `fp32_iterations` (the single-precision stage's, 0 when none ran) |
| `host_spin_square(alpha, beta, c, s2c)` | S² c on the host; `Status` |
| `CiSpinSquare::create(res, alpha, beta)` | Upload S²'s tables; `Result<CiSpinSquare>` |
| `CiSpinSquare::apply(res, k, x, y, layout = full)` | Y = S² X for size(layout) × k device blocks: a `LinearOperator`; `Status` |
| `CiSpinSquare::penalize(res, k, λ, target, x, y, layout = full)` | Y += λ (S² − target) X, the penalty ci_solve adds into sigma's output; `Status` |
| `CiSpinSquare::penalize_diagonal(res, λ, target, diagonal, shifted, layout = full)` | shifted = diagonal + λ (diag(S²) − target), the preconditioner's diagonal, from the full diagonal into either layout; `Status` |

`sigma` applies H to a device block of CI vectors in `host_sigma`'s layout:
the GPU sigma in a calculation, `host_sigma` wrapped as a `LinearOperator` in
the tests. One that also takes a `CiLayout` (`CiSigma`) lets a singlet solve
run in the symmetric layout (below). `diagonal` is diag(H) on the device, as
`CiSigma::diagonal` writes it.

## Design

**Spin control is a penalty, not filtering.** Davidson runs on
H + λ (S² − S(S+1)), with S the lowest spin of the (N_α, N_β) sector,
S = \|N_α − N_β\| / 2. Every state in the sector has spin S' ≥ S, and H commutes
with S², so the penalty leaves the spin-S roots and their energies exactly
where they are and lifts a spin-S' state by λ (S'(S'+1) − S(S+1)) ≥ 2λ (S + 1).
Choosing the spin is choosing the sector: N_α − N_β = 2S. Filtering, solving
for more roots and keeping those of the wanted spin, was rejected because how
many more is not known in advance: N₂'s CAS(6,6) has six triplets below its
second singlet, and every extra root is another sigma column per iteration,
the expensive part. The penalty costs one S² per sigma column instead, which
is far cheaper than sigma (O(dim · N (n − N + 1)) against O(dim · n⁴)).

λ must exceed the span of the wanted roots over the lowest unwanted state, less
its lift; 1 hartree is ample for valence active spaces. When it is not, a root
comes back with the wrong ⟨S²⟩, and `ci_solve` returns `Numerical` rather than
a mixed spectrum (any root more than 1e-6 off S(S+1)). λ = 0 turns the control
off, and the roots are the sector's lowest of any spin.

In a singlet solve over `CiSigma` the symmetric layout (below) already keeps
the Davidson to even S, exactly, so the triplets the penalty used to lift are
not in its space at all. The penalty stays for the rest: quintets and higher
are even too, each lifted by at least 6λ, and a sigma without a layout (the
tests' `host_sigma`), or an M_s = 1 sector, still runs in the full layout and
needs it for everything.

**Energies are ⟨H⟩.** The Ritz value θ is the unit Ritz vector's Rayleigh
quotient of the penalized operator, so θ − λ (⟨S²⟩ − S(S+1)) is its quotient
of H itself, contamination or not.

**S² = S_z (S_z + 1) + N_β − Σ_pq E^β_pq E^α_qp.** S₋S₊ = Σ_pq a†_pβ a_pα a†_qα
a_qβ reorders to E^β_pq (δ_pq − E^α_qp), so both factors are same-spin
excitations and go through the sets' own replacement lists: nothing leaves the
sector, and no string set for (N_α ± 1, N_β ∓ 1) is needed. Each operator is
even in its spin, so it passes the other spin's creators with no sign and
carries only its own list's.

**Olsen's correction is an option, off by default.** With
`olsen_correction` the correction is δ = M⁻¹ (r − ε x),
ε = xᵀ M⁻¹ r / xᵀ M⁻¹ x, in place of the plain M⁻¹ r, over
M = θ − diag(H + penalty). A CI matrix is diagonally dominant, so M ≈ θ − H
and M⁻¹ r = M⁻¹ (H − θ) x is nearly −x, the Ritz vector itself; projected off
V, which holds x, it leaves a small, noisy remainder. Olsen's is the
first-order Jacobi–Davidson correction, with xᵀ δ = 0 by construction.
`davidson_solve` passes X for it. Per column it is two elementwise launches
and two dots: M⁻¹ x into the correction column, xᵀ M⁻¹ x and (M⁻¹ x)ᵀ r
(= xᵀ M⁻¹ r, M being diagonal) into the device workspace in device pointer
mode, then δ from r, x and those two dots, with ε formed in the kernel.
Nothing comes back to the host, so it adds no sync. A zero xᵀ M⁻¹ x falls
back to ε = 0. With the option off, the correction is the same launch on r
that it always was, and the solve is bitwise the one before the option
existed.

Over the diagonal M it does not pay. On `bench/ci_sigma_precision` (below
the GD+k table, same conditions, parent 34b1ca5 → `olsen_correction` on) it
is slower on 19 of the 3080's 28 solves and 11 of the gfx1200's 21 that both
converge, at 0.88–1.07× and 0.91–1.19×; iteration counts move both ways, by
up to ~20%. On the gfx1200 it converges two CAS(12,12) 2.0 Å symmetric
solves whose FP64 stage stalls in the parent (#318): `fp32_tolerance` 1e-4 in
60 + 135 iterations (24.0 s) and 1e-6 in 82 + 109 (23.0 s), and it adds no
new `Divergence`. The energies differ from the parent's by at most 1.1e-12
on the 3080 (3.8e-13 FP64-only) and 1.8e-11 on the gfx1200 (2.6e-12
FP64-only), within the parent's own spread across solve modes, 2.8e-12 and
5.6e-11. #287 replaces M with a P-space operator; it is to re-bench Olsen's
correction there and turn it on by default if it wins.

| | | full, FP64-only | symmetric, FP64-only | two-stage 1e-6 | two-stage 1e-7 |
|---|---|---|---|---|---|
| CAS(10,10), 1.0977 Å | 3080 | 0.81 → 0.80 s (112 → 109) | 0.39 → 0.39 s (108 → 103) | 0.32 → 0.33 s (46+60 → 48+60) | 0.32 → 0.30 s (56+57 → 57+45) |
| | gfx1200 | 2.32 → 2.03 s (112 → 106) | 0.95 → 0.92 s (108 → 103) | 0.88 → 0.91 s (47+60 → 48+61) | 0.89 → 0.90 s (56+53 → 57+52) |
| CAS(12,12), 1.0977 Å | 3080 | 11.6 → 12.7 s (99 → 115) | 5.27 → 5.26 s (91 → 97) | 3.71 → 4.20 s (47+61 → 55+66) | 3.74 → 3.99 s (51+55 → 71+53) |
| | gfx1200 | 31.3 → 26.2 s (119 → 97) | 12.7 → 13.6 s (91 → 101) | Divergence in both | 13.3 → 14.1 s (51+59 → 71+52) |
| CAS(10,10), 2.0 Å | 3080 | 1.21 → 1.33 s (206 → 225) | 0.74 → 0.75 s (210 → 205) | 0.60 → 0.62 s (103+108 → 103+107) | 0.58 → 0.60 s (122+87 → 122+89) |
| | gfx1200 | 3.42 → 3.76 s (206 → 225) | 1.83 → 1.81 s (212 → 205) | 1.73 → 1.76 s (103+108 → 103+108) | 1.71 → 1.71 s (122+88 → 121+85) |
| CAS(12,12), 2.0 Å | 3080 | 21.7 → 22.1 s (195 → 208) | 10.1 → 10.1 s (192 → 189) | 7.18 → 7.50 s (90+103 → 95+108) | 7.02 → 7.12 s (112+97 → 112+95) |
| | gfx1200 | 54.1 → 49.6 s (208 → 184) | 26.6 → 25.4 s (201 → 189) | Divergence → 23.0 s (82+109) | 23.4 → 23.4 s (113+92 → 103+98) |

**The preconditioner is r / (θ − diag(H + penalty)).** diag(S²) is
S_z (S_z + 1) + N_β less the doubly occupied orbitals, so the penalty's
diagonal is free, and including it keeps the lifted determinants from being
over-weighted. The denominator is held off zero at 1e-3. In the symmetric
layout it is the same diagonal on the triangle's determinants: the packed
operator's own diagonal adds H's element between a determinant and its mirror, which is
nonzero only where the two differ by one orbital of each spin, and the
preconditioner needs only an approximation. diag(H) and diag(S²) are both
unchanged by the swap, so the correction stays symmetric too.

**Guesses are the 2 · n_roots determinants of lowest diag(H), spread.** In
the symmetric layout they are triangle elements, each a determinant and its
mirror at once, so a determinant and its mirror, which have one diagonal, no
longer take two of the guesses; and the spread is drawn over the triangle, so
every guess is symmetric by construction. A
determinant has one spatial symmetry, and both H and a diagonal preconditioner
keep to it, so from unit vectors alone a root of a symmetry no guess has never
enters the subspace: N₂'s second and third singlets were lost that way. Each
guess therefore also carries a 1e-3 deterministic spread over every
determinant (a splitmix64 hash of its position, so runs are reproducible), and
the columns are orthonormalized. The subspace holds up to 16 · n_roots
columns (at least 2 · n_roots, at most dim). All of it is on the device: the
lowest determinants by 2 · n_roots argmin passes over diag(H) (each the least
(value, address) above the last, so ties go to the lower address, as a stable
sort's would, without sorting dim values), the spread by a hash kernel, and
the orthonormalization by modified Gram-Schmidt, twice, over the BLAS.

**A singlet solve runs on packed vectors.** When the sigma takes a
`CiLayout`, the two sets are one (N_α = N_β), the penalty is on and n_roots
fits the triangle, `ci_solve` runs the whole Davidson in
`CiLayout::symmetric` (`../sigma/`): the guesses and the penalized diagonal
are built on the triangle, sigma and the penalty apply to packed blocks, ⟨S²⟩
is taken there (the packing is an isometry, so the dots are the full
vectors'), and the converged roots are unpacked with `ci_unpack` into
`CiRoots::vectors` in the full layout, so `CiSigma::rdm1`, `CiRdm::rdm12` and
every caller read them as before. λ = 0 asks for every spin, odd S included,
so it keeps the full layout.

The alternative was full-length vectors with only sigma's work skipped:
sigma would read and write the triangle of each full column, and the scratch
would halve, but not V and A V, which at CAS(14,14) with three roots and the
16 · n_roots subspace are about 9 GB, as much as G itself, and not the
Davidson's own FP64 gemms over them either, a third of the kernel time at
CAS(12,12) (see the table below). Packed vectors halve all of it, and
`recipes/davidson` is unchanged: the √2 on the off-diagonal elements is what
keeps its plain dot product, and so the packed operator's symmetry, right. The
cost is an unpack at the end, n_roots full vectors. A full-length Davidson
would also have had to keep its vectors symmetric against roundoff, the
guesses' spread included; in packed vectors the symmetry is the layout, so
there is nothing to drift and no re-symmetrisation to schedule.

**The residual tolerance is tight, 1e-12 relative.** A vector's error is first
order in its residual and the 1-RDM is read from the vector, so the default
residual of `davidson_solve` (1e-8 relative, ~1e-6 hartree here) would leave
the 1-RDM near 1e-5. 1e-12 relative is ~1e-10 hartree at molecular energies,
which roundoff in sigma still clears.

**FP32 first, then FP64, when the sigma can.** A sigma with an
`apply_fp32` (`CiSigma`'s: float D, G and gemm, ~1e-7 relative, see
`../sigma/`) is solved in two stages. Stage 1 is `davidson_solve` over the
penalized `apply_fp32` to `fp32_tolerance`; stage 2 is `davidson_solve` over
the penalized `apply` to `davidson.residual_tolerance`, from stage 1's n_roots
Ritz vectors in place of the guesses. Stage 2 is why this is safe: its
operator, preconditioner and tolerance are exactly an FP64-only solve's, and
no column of A V mixes the precisions, because it restarts rather than
carrying stage 1's A V over. Stage 1 only chooses where stage 2 starts. Its
vectors are V S for an FP64 orthonormal V and an orthogonal S, so orthonormal
whatever the noise in the A V they came from, and a bad start costs stage 2
iterations, not accuracy. Stage 1 running out of iterations or subspace is
therefore no error; stage 2 goes on from its last Ritz vectors. A sigma
without `apply_fp32` (`host_sigma` in the tests), `fp32_tolerance` = 0, or a
`fp32_tolerance` at or below the FP64 tolerance runs the single FP64 solve as
before. `recipes/davidson` is unchanged.

FP32 noise floors the residual near 1e-7 relative, so stage 1 cannot go much
further, and where it stalls it spends its whole `max_iterations` (N₂
CAS(12,12) at 1e-7: 400 FP32 iterations on the 3080, though not on the
gfx1200). 1e-6, the default, converged in every case measured and was the
fastest setting that did. `bench/ci_sigma_precision` over
`tools/emit_ci_bench_integrals.py`'s N₂ cc-pVDZ active spaces, three singlets,
`max_iterations` 400 (100, the default, does not converge these
FP64-only), each card idle (the 3080 held through `devtools/lib.sh`'s GPU
slots), all in the full layout, before the symmetric one (whose numbers
follow below):

| | | σ per column, FP64 / FP32 | `ci_solve` FP64-only | two-stage, 1e-6 | iterations, FP64-only → FP32 + FP64 |
|---|---|---|---|---|---|
| CAS(10,10), 1.0977 Å | 3080 | 1.77 / 0.26 ms | 0.95 s | 0.74 s (1.28×) | 134 → 71 + 78 |
| | gfx1200 | 2.52 / 0.80 ms | 2.63 s | 2.45 s (1.07×) | 134 → 71 + 88 |
| CAS(12,12), 1.0977 Å | 3080 | 35.6 / 5.33 ms | 15.2 s | 12.5 s (1.22×) | 140 → 78 + 94 |
| | gfx1200 | 38.3 / 11.0 ms | 43.3 s | 32.2 s (1.34×) | 165 → 68 + 72 |
| CAS(10,10), 2.0 Å | 3080 | 1.60 / 0.25 ms | 1.47 s | 1.18 s (1.25×) | 234 → 108 + 135 |
| | gfx1200 | 1.60 / 0.49 ms | 3.93 s | 3.78 s (1.04×) | 234 → 108 + 135 |
| CAS(12,12), 2.0 Å | 3080 | 35.8 / 5.30 ms | 25.7 s | 18.1 s (1.42×) | 228 → 115 + 131 |
| | gfx1200 | 38.4 / 10.9 ms | 60.6 s | 52.5 s (1.16×) | 232 → 94 + 131 |

At every `fp32_tolerance` from 1e-4 to 1e-7, every two-stage energy is within
2e-12 of the FP64-only one (6e-12 on the gfx1200 at CAS(12,12)). The whole solve gains far less than σ for two
reasons. First, stage 2 is long: 44–67% of the FP64-only iteration count,
because 1e-6 is half the decades to 1e-12 and a restart from n_roots vectors
drops the Krylov subspace. Carrying stage 1's whole subspace V over instead
would cost one FP64 apply of up to 16 · n_roots columns, about as many as
the iterations it saves. Second, Davidson's own FP64 work, the V^T A V and
projection gemms over up to 48 columns, is untouched by stage 1: a third of
the 3080's kernel time over the bench's five CAS(12,12) solves (nsys), and on
the gfx1200 at least ~150 ms of an FP64-only iteration's ~260 is outside σ,
more than σ's own (inferred from the timings: the image's rocprofv3 lacks
`libhsa-amd-aqlprofile`). FP64 emulation for
stage 2 does not help on these cards either: cuBLAS 13.0's Ozaki-style
`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`, forced on (`EAGER`) in a standalone
gemm of a whole CAS(12,12) vector's D (853,776 × 78, times 78 × 79) on the
3080, is exact to 3e-16 but takes 186 ms against DGEMM's 57 ms (SGEMM:
1.3 ms). With k = P = 78 there is too little reduction
for the splitting to pay for itself.

Measured with `bench/ci_sigma_precision` on the RTX 3080, idle and held
through `devtools/lib.sh`'s GPU slots, over the same N₂ cc-pVDZ active spaces,
three singlets, `max_iterations` 400. "Full" is `ci_solve` held to the full
layout (a wrapper that hides `CiSigma`'s layout parameter), which is the
solve as it was before the symmetric layout; the workspace is what one
column of `apply` grows a fresh `DeviceResources` to, G and one chunk of D:

| | σ per column FP64 / FP32, full → symmetric | `apply` workspace FP64 / FP32, MiB | `ci_solve` FP64-only, full → symmetric | two-stage at 1e-6, full → symmetric |
|---|---|---|---|---|
| CAS(10,10), 1.0977 Å | 2.09 / 0.28 → 1.03 / 0.15 ms | 53.8 / 26.9 → 27.0 / 13.5 | 0.99 → 0.50 s (131 → 114 it.) | 0.89 → 0.44 s |
| CAS(12,12), 1.0977 Å | 39.4 / 5.77 → 20.3 / 3.35 ms | 579 / 321 → 322 / 193 | 15.1 → 7.90 s (114 → 126) | 11.6 → 6.06 s |
| CAS(10,10), 2.0 Å | 1.90 / 0.25 → 0.88 / 0.18 ms | 53.8 / 26.9 → 27.0 / 13.5 | 1.55 → 0.98 s (225 → 238) | 1.21 → 0.81 s |
| CAS(12,12), 2.0 Å | 41.2 / 6.21 → 21.3 / 3.43 ms | 579 / 321 → 322 / 193 | 26.3 → 13.4 s (211 → 226) | 20.2 → 9.36 s |

Every symmetric-layout energy, at every `fp32_tolerance` from 1e-4 to 1e-7,
is within 9e-13 of the full layout's FP64-only one. σ halves, as the rows do;
the workspace less its fixed 64 MiB (32 in float) of D halves too. The
Davidson's own buffers halve with the vectors: at CAS(12,12) with three roots
V and A V are 625 → 313 MiB, A X, R and the corrections 59 → 29, and
`ci_solve`'s guesses, roots and diagonal 65 → 33, with the 20 MiB of unpacked
roots added at the end. At CAS(12,12) the whole solve gains about what σ
does, 1.9–2.2×, because the Davidson's FP64 gemms halve with it; at
CAS(10,10), where each iteration's fixed costs weigh more, 1.5–2×. Iteration
counts move by about 10% either way: the packed subspace is a different
Krylov sequence, from guesses that are each a determinant and its mirror. In
the symmetric layout the single-precision stage converged at 1e-7 in all four
cases, where the full layout's stalled at CAS(12,12) above; the full layout
was not rerun at 1e-7 here.

**The Davidson collapse keeps X_prev (GD+k, `../../recipes/davidson/`).**
With three roots the 48-column subspace collapses every ~15 iterations, and
each collapse onto the Ritz vectors alone used to drop the search direction.
Keeping the previous Ritz vectors beside them cuts the FP64 iterations by
10–25%. Times are from `bench/ci_sigma_precision`, max_iterations 400, three
singlets, the symmetric layout except the "full" column. Each card was idle
with both of its GPU slots held, and the parent commit was run in the same
hold. Iterations are FP32 + FP64:

| | | full, FP64-only | symmetric, FP64-only | two-stage 1e-6 | two-stage 1e-7 |
|---|---|---|---|---|---|
| CAS(10,10), 1.0977 Å | 3080 | 0.95 → 0.81 s (138 → 112) | 0.47 → 0.40 s (128 → 108) | 0.36 → 0.32 s (54+70 → 46+60) | 0.36 → 0.32 s (59+67 → 56+57) |
| | gfx1200 | 2.43 → 2.09 s (138 → 112) | 1.13 → 0.95 s (128 → 108) | 1.00 → 0.88 s (54+68 → 47+60) | 1.02 → 0.89 s (59+65 → 56+53) |
| CAS(12,12), 1.0977 Å | 3080 | 13.8 → 11.7 s (130 → 99) | 6.03 → 5.32 s (110 → 91) | 4.89 → 3.73 s (48+75 → 47+61) | 4.54 → 3.76 s (58+72 → 51+55) |
| | gfx1200 | 34.2 → 31.5 s (131 → 119) | 15.1 → 12.7 s (109 → 91) | Divergence in both | 14.7 → 13.4 s (57+64 → 51+59) |
| CAS(10,10), 2.0 Å | 3080 | 1.36 → 1.22 s (232 → 206) | 0.82 → 0.74 s (229 → 210) | 0.65 → 0.61 s (110+116 → 103+108) | 0.65 → 0.58 s (125+110 → 122+87) |
| | gfx1200 | 3.86 → 3.42 s (232 → 206) | 2.00 → 1.83 s (229 → 212) | 1.86 → 1.73 s (110+116 → 103+108) | 1.94 → 1.71 s (126+113 → 122+88) |
| CAS(12,12), 2.0 Å | 3080 | 24.2 → 21.9 s (221 → 195) | 11.9 → 10.1 s (230 → 192) | 8.56 → 7.18 s (103+132 → 90+103) | 8.24 → 7.02 s (124+122 → 112+97) |
| | gfx1200 | 60.7 → 54.3 s (233 → 208) | 30.6 → 26.6 s (231 → 201) | Divergence in both | 27.6 → 23.4 s (125+117 → 113+92) |

Every solve that both builds converge is 1.01–1.35× faster, on both cards.
On the gfx1200 the FP64 stage of the CAS(12,12) two-stage solves stalls just
above its 1e-12 tolerance (~1.1e-10 absolute) and ends in `Divergence`. That
happens in 6 of the parent's 14 such solves, and with GD+k in a 7th: 2.0 Å,
symmetric, `fp32_tolerance` 1e-4, which the parent converged in 34 + 206
iterations, 29.9 s (#318). The 3080 converges all 28. The energies differ from
the parent's by at most 1.3e-12 on the 3080 (5.5e-13 FP64-only) and 1.1e-11 on
the gfx1200 (2.5e-12 FP64-only). That is within the parent's own spread across
its solve modes, 2.4e-12 and 5.3e-11.

**Nothing of dim size touches the host.** The penalty and the preconditioner
run on the device, so no block leaves it during the Davidson. On the host
each cost a block's round trip and a sync per call, and a single-threaded S²
per column, every iteration. The setup and the readback are on the device too:
diag(H) comes in as a device array, the guesses and the penalized diagonal are
built from it there, ⟨S²⟩ per root is S² X and a dot there, and the roots
stay there as `CiRoots::vectors`, which `CiSigma::rdm1` and `CiRdm::rdm12`
take as they are. Only the n_roots Ritz values and ⟨S²⟩s are read back.
`CiSpinSquare` gathers S² X one thread per row, like the GPU sigma's second
gather: S² is real symmetric, so row J walks its own α links a†_p a_q and, for
each, finds the β partner a†_q a_p in its β string's n × n slot table (which
holds sign · (target + 1), or 0). `penalize` adds λ (S² − S(S+1)) X into the
sigma's Y in the same pass. The preconditioner is one elementwise launch per
column (two launches and two device-mode dots with Olsen's correction), θ
passed by value, over diag(H + penalty), which `penalize_diagonal`
builds once: one thread per determinant counts its α links a†_p a_p whose β
partner survives, the doubly occupied orbitals.

## Tests

`test/gtest/ci_solver_gpu.cpp`, over `host_sigma` wrapped as a
`LinearOperator`, against `test/gtest/data/ci/direct_spin1.json`. Every case
there has a triplet below its second singlet, so the spin control is exercised:

- the lowest three singlets of every case: energies to 1e-8, ⟨S²⟩ to 1e-8 of 0,
  and the 1-RDM (`ci.rdm`) to 1e-8 on each root that is not one of a degenerate
  level, its trace N_α + N_β;
- the lowest three triplets, from the M_s = 1 sector, against direct_spin1's
  M_s = 0 ones (the spin-summed 1-RDM does not depend on M_s);
- with λ = 0, direct_spin1's lowest three roots, a triplet among them;
- over `CiSigma` itself, every case's singlets and (M_s = 1) triplets, two-
  stage by default and FP64-only at `fp32_tolerance` = 0, each to the same
  tolerances, the first with `fp32_iterations` > 0 and the second with 0, and
  each recording the layout it was applied in: symmetric for the singlets,
  full for the triplets;
- over `CiSigma` with λ = 0, the full layout and a triplet among the lowest
  three;
- every solve above, with `olsen_correction` off and on;
- `MaxIterationsExceeded` when given two iterations;
- the refusals;
- `CiSpinSquare`'s `apply` and `penalize` against `host_spin_square` on
  random blocks of three vectors, each column to 1e-12 of its norm, over seven
  sectors (open and closed shells, an empty spin, a full one, a single orbital,
  N_α < N_β), `penalize_diagonal` against diag(S²) from the strings' popcounts
  (also in place), and its refusals;
- the same in the symmetric layout over six sets for both spins, on packed
  swap-symmetric blocks against `host_spin_square` packed, with
  `penalize_diagonal` exactly the full layout's on the triangle, and refused
  over two sets.

## Files

| File | Role |
|------|------|
| `solver.cppm` | Module interface: `ci_solve` (a template, with its two stages and two layouts), its options and roots, `host_spin_square`, `CiSpinSquare` |
| `solver.cpp` | `host_spin_square`; `CiSpinSquare`'s tables and launches; the preconditioner's launches, and Olsen's dots; the guesses |
| `ci_solver_bridge.h` | The launchers' declarations, shared by `solver.cpp`'s GMF and the `.cu` |
| `ci_solver.cu` | The S² gather and the penalized diagonal (in either layout, over `../sigma/ci_symmetric.h`), the preconditioner's two kernels, the argmin passes and the guess spread |
| `CMakeLists.txt` | The device library `free_breakfast.ci.solver.device` and the module `free_breakfast.ci.solver` |
