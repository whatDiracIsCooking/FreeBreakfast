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
| `ci_solve(res, sigma, alpha, beta, diagonal, n_roots, options, start = {})` | `Result<CiRoots>`: the `n_roots` lowest roots of spin S = \|N_α − N_β\| / 2, from the guesses or, when `start` (n_roots device vectors in the full layout) is not empty, from those. A template over the sigma. |
| `CiSolverOptions` | `spin_penalty` (λ, 1 hartree; 0 keeps every spin), `davidson` (`DavidsonOptions`, residual tolerance 1e-12), `fp32_tolerance` (the single-precision stage's, 1e-6; 0 turns it off), `olsen_correction` (Olsen's correction in place of the plain one; off), `restart_with_previous` (the FP64 stage from [X, X_prev] in place of X; off) and `spin_averaged_diagonal` (precondition with the spin-averaged diag(H), from the sigma's `exchange()`; off) |
| `CiRoots` | `energies`, `spin_squares` (⟨S²⟩), `vectors` (a device buffer in the full layout, root r at [r · dim, (r + 1) · dim)), `davidson` (`DavidsonInfo` of the FP64 stage), `fp32_iterations` (the single-precision stage's, 0 when none ran) |
| `host_spin_square(alpha, beta, c, s2c)` | S² c on the host; `Status` |
| `CiSpinSquare::create(res, alpha, beta)` | Upload S²'s tables; `Result<CiSpinSquare>` |
| `CiSpinSquare::apply(res, k, x, y, layout = full)` | Y = S² X for size(layout) × k device blocks: a `LinearOperator`; `Status` |
| `CiSpinSquare::penalize(res, k, λ, target, x, y, layout = full)` | Y += λ (S² − target) X, the penalty ci_solve adds into sigma's output; `Status` |
| `CiSpinSquare::penalize_diagonal(res, λ, target, diagonal, shifted, layout = full, exchange = nullptr)` | shifted = diagonal + λ (diag(S²) − target), the preconditioner's diagonal, from the full diagonal into either layout; with `exchange` ((ij\|ji), n × n on the device) the diagonal spin-averaged first; `Status` |

`sigma` applies H to a device block of CI vectors in `host_sigma`'s layout:
the GPU sigma in a calculation, `host_sigma` wrapped as a `LinearOperator` in
the tests. One that also takes a `CiLayout` (`CiSigma`) lets a singlet solve
run in the symmetric layout (below). `diagonal` is diag(H) on the device, as
`CiSigma::diagonal` writes it. `spin_averaged_diagonal` also wants the sigma's
`exchange()`, the (ij|ji) of its H on the device (`CiSigma::exchange`), and is
`InvalidConfig` without one.

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
the orthonormalization by modified Gram-Schmidt, twice, over the BLAS, or,
with `davidson.canonical_orthonormalization` (below), by `orthonormalize`
(`../../recipes/orthogonalize/`), with no host round trip per pair. The
columns are independent by construction, so `ci_guesses` returns
`Numerical` if `orthonormalize` ever drops one.

**A start replaces the guesses.** `start`, when not empty, is n_roots CI
vectors in the full layout, as `CiRoots::vectors` holds them: in a CASSCF
macro loop the last iteration's roots, whose H is a small step from this
one's. They take the guesses' place and nothing else changes: on the
symmetric path `ci_pack` (`../sigma/`) packs them onto the triangle, they are
orthonormalized (`orthonormalize`, `Numerical` if one drops: a previous
solve's roots are orthonormal only to its convergence, and `davidson_solve`
takes orthonormal columns), and they start stage 1 (below) when there is one,
stage 2 otherwise. Stage 1 only chooses where stage 2 starts, so seeding it is
seeding both: from converged roots stage 1 stops at its first iteration, its
Ritz vectors a rotation of the start within its span, and stage 2 recovers the
roots in the first iteration of its own. `ci_pack` is `ci_unpack`'s adjoint,
which on vectors symmetric under the swap is its inverse, and maps any other
vector to its symmetric part: a start's odd-S part, if any, cannot enter the
packed space. Only n_roots columns start, against the guesses' 2 · n_roots, so
in a space hardly bigger than that (H₂CO's CAS(4,3), six triangle elements) a
warm start can take an iteration or two more than the guesses, which span
nearly all of it at once; each is two sigma columns there. An empty `start` is
the solve without one, to the bit.

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
iterations, 29.9 s (#318, since fixed: below). The 3080 converges all 28. The energies differ from
the parent's by at most 1.3e-12 on the 3080 (5.5e-13 FP64-only) and 1.1e-11 on
the gfx1200 (2.5e-12 FP64-only). That is within the parent's own spread across
its solve modes, 2.4e-12 and 5.3e-11.

**Canonical orthonormalization is an option, off by default.**
`davidson.canonical_orthonormalization` orthonormalizes the guesses and each
iteration's new corrections as a block, by `orthonormalize` (canonical
orthogonalization over a Gram matrix, one syevd and a sync, then a
Newton–Schulz step or a second canonical pass), in place of Gram–Schmidt's
norm sync per column. Per iteration it is even: `bench/ci_sigma_precision`
(same conditions as above, parent 4e21392 → the option on, the "on" binary
being the pre-option build in which it always was), the FP64-only solves,
which take the same iteration counts to within a few, sum to 51.6 → 51.8 s
on the 3080 and 133.1 → 133.8 s on the gfx1200. The two-stage solves take a
different rounding path, and their FP32 stage, at its noise floor, moves by
several iterations either way. One of them, CAS(12,12) at 1.0977 Å,
symmetric, `fp32_tolerance` 1e-7, stalls its FP32 stage for all 400
iterations on both cards (51 in the parent), which makes the option slower on
balance: 126.2 → 131.1 s over the 3080's 28 solves, 199.4 → 231.7 s over the
21 the gfx1200's parent converges. Without that solve it is 1% slower on
each card. On the gfx1200 it converges one solve whose FP64 stage stalls in
the parent (#318), CAS(12,12) at 2.0 Å, symmetric, 1e-4, in 58 + 149
iterations (25.6 s), and adds no new `Divergence`; the other six stall as
before, so the Gram–Schmidt is not what stalls them. The energies differ from
the parent's by at most 1.4e-12 on the 3080 (3.7e-13 FP64-only) and 1.1e-11
on the gfx1200 (3.1e-12 FP64-only), within the parent's own spread across
solve modes, 2.8e-12 and 5.6e-11. With the option off, both are the
Gram–Schmidt they always were: on the two CAS(10,10) cases, 7 solves each,
every root energy to 15 decimals and every iteration count matched the
parent's on both cards.

| | | full, FP64-only | symmetric, FP64-only | two-stage 1e-6 | two-stage 1e-7 |
|---|---|---|---|---|---|
| CAS(10,10), 1.0977 Å | 3080 | 0.81 → 0.80 s (112 → 112) | 0.39 → 0.39 s (108 → 108) | 0.31 → 0.32 s (46+60 → 45+65) | 0.31 → 0.34 s (56+57 → 60+57) |
|  | gfx1200 | 2.01 → 2.11 s (112 → 112) | 0.95 → 0.96 s (108 → 108) | 0.89 → 0.92 s (47+60 → 45+66) | 0.90 → 0.97 s (56+53 → 60+57) |
| CAS(12,12), 1.0977 Å | 3080 | 11.5 → 11.6 s (99 → 106) | 5.24 → 5.31 s (91 → 91) | 3.69 → 3.97 s (47+61 → 53+57) | 3.75 → 7.58 s (51+55 → 400+55) |
|  | gfx1200 | 31.4 → 31.7 s (119 → 115) | 12.7 → 12.7 s (91 → 91) | Divergence in both | 13.3 → 43.9 s (51+59 → 400+56) |
| CAS(10,10), 2.0 Å | 3080 | 1.22 → 1.20 s (206 → 205) | 0.74 → 0.73 s (210 → 209) | 0.61 → 0.60 s (103+108 → 103+110) | 0.58 → 0.57 s (122+87 → 121+90) |
|  | gfx1200 | 3.43 → 3.45 s (206 → 205) | 1.84 → 1.83 s (212 → 208) | 1.74 → 1.77 s (103+108 → 103+109) | 1.72 → 1.75 s (122+88 → 121+90) |
| CAS(12,12), 2.0 Å | 3080 | 21.7 → 21.8 s (195 → 202) | 10.0 → 10.1 s (192 → 191) | 7.16 → 7.05 s (90+103 → 81+105) | 6.99 → 6.80 s (112+97 → 94+96) |
|  | gfx1200 | 54.2 → 54.9 s (208 → 211) | 26.6 → 26.0 s (201 → 195) | Divergence in both | 23.3 → 23.8 s (113+92 → 106+99) |

**Restarting the FP64 stage from [X, X_prev] is an option, off by default.**
With `restart_with_previous`, stage 1's `davidson_solve` also writes out the
Ritz vectors of the iteration before its last, X_prev (its
`previous_eigenvectors`, into the guesses' second n_roots columns), and
stage 2 starts from `orthonormalize([X, X_prev])` (`../../recipes/orthogonalize/`),
at most 2 · n_roots columns, a mode at or below
`davidson.linear_dependence_floor` (or under the Gram matrix's rounding)
dropped, in place of X alone. The argument for stage 2 above still holds:
X and X_prev are each V S for an FP64-orthonormal V of stage 1's, the start
is orthonormalized in FP64, and stage 2's operator, preconditioner and
tolerance are unchanged; X_prev only widens where it starts. The
orthonormalization's n × 2 n_roots product is the device workspace's,
smaller than the sigma's unblocked scratch; `FciStep` leaves the option off,
so its memory estimate is unchanged.

It does not pay. On `bench/ci_sigma_precision` (same conditions as the
tables above, parent 8e85883 → the option on, each card idle with both
GPU slots held, the two run back to back in one hold) stage 2's iterations
move both ways, by up to ~10 either way, and the extra n_roots FP64 columns
of its first apply, and the earlier collapse they bring, cost about what
the iterations save. Over the 3080's 20 two-stage solves it is 74.3 → 76.1 s
(faster on 5), stage 2 1844 → 1822 iterations; over the 13 the gfx1200's
parent converges, 66.4 → 67.0 s (faster on 7), 1185 → 1166. On the gfx1200
it converges two CAS(12,12) 2.0 Å symmetric solves whose FP64 stage stalls
in the parent (#318), `fp32_tolerance` 1e-4 in 34 + 160 iterations (24.3 s)
and 1e-6 in 95 + 116 (24.6 s), and adds no new `Divergence`. The energies
differ from the parent's by at most 1.8e-12 on the 3080 and 3.5e-11 on the
gfx1200 (the FP64-only solves are untouched, to the bit), within the
parent's own spread across solve modes, 2.8e-12 and 5.6e-11; a 1e-12
criterion sits below that spread on both cards. With the option off, both
stages are the parent's: on the two CAS(10,10) cases, 7 solves each, every
root energy to 15 decimals and every iteration count matched the parent's
on both cards.

| | | full, FP64-only | symmetric, FP64-only | two-stage 1e-6 | two-stage 1e-7 |
|---|---|---|---|---|---|
| CAS(10,10), 1.0977 Å | 3080 | 0.81 → 0.81 s (112 → 112) | 0.39 → 0.40 s (108 → 108) | 0.31 → 0.31 s (46+60 → 46+55) | 0.31 → 0.31 s (56+57 → 56+52) |
|  | gfx1200 | 2.26 → 2.11 s (112 → 112) | 0.95 → 0.95 s (108 → 108) | 0.89 → 0.88 s (47+60 → 47+58) | 0.89 → 0.83 s (56+53 → 56+44) |
| CAS(12,12), 1.0977 Å | 3080 | 11.5 → 11.7 s (99 → 99) | 5.24 → 5.31 s (91 → 91) | 3.67 → 3.91 s (47+61 → 47+56) | 3.70 → 3.96 s (51+55 → 51+56) |
|  | gfx1200 | 31.0 → 31.4 s (119 → 119) | 12.7 → 12.7 s (91 → 91) | Divergence in both | 13.3 → 14.5 s (51+59 → 51+69) |
| CAS(10,10), 2.0 Å | 3080 | 1.20 → 1.22 s (206 → 206) | 0.73 → 0.74 s (210 → 210) | 0.60 → 0.59 s (103+108 → 103+98) | 0.57 → 0.59 s (122+87 → 122+90) |
|  | gfx1200 | 3.44 → 3.42 s (206 → 206) | 1.84 → 1.84 s (212 → 212) | 1.74 → 1.69 s (103+108 → 103+100) | 1.72 → 1.73 s (122+88 → 122+88) |
| CAS(12,12), 2.0 Å | 3080 | 21.6 → 21.9 s (195 → 195) | 10.0 → 10.1 s (192 → 192) | 7.15 → 7.45 s (90+103 → 90+108) | 6.99 → 7.22 s (112+97 → 112+105) |
|  | gfx1200 | 54.0 → 54.3 s (208 → 208) | 26.6 → 26.6 s (201 → 201) | Divergence → 24.6 s (95+116) | 23.3 → 23.7 s (113+92 → 113+94) |

**The FP64 stage's stall on the gfx1200 was the Ritz values, not the
sigma (#318).** On the gfx1200 the FP64 stage of six of the 14 CAS(12,12)
two-stage solves stalled at a residual of 1.1–4e-10, just above or a few
times its 1e-12 tolerance (~1.09e-10 at |E| ≈ 109), and ended in
`Divergence`; the 3080 converged them all. The FP64 sigma is not the cause:
over the converged roots and random vectors, `CiSigma::apply` on the
gfx1200 is within 1.4e-13 of `host_sigma` (~1e-15 of ‖H c‖), bitwise
repeatable, and symmetric to 1e-16. What stalled was θ. At the stall the
Ritz vectors were already converged, ‖A x − ρ x‖ = 1e-11 to 4e-11 with ρ
their Rayleigh quotient, equal to the FP64-only energies, but every θ sat
1.0–1.3e-10 above its ρ, so the residual A x − θ x was (ρ − θ) x. That offset
was there from the FP64 stage's first iteration and never moved. It came from
Vᵀ A V's diagonal: `davidson_solve` formed it by a DGEMM reduced over the
vector length, whose terms vᵢ (A v)ᵢ ≈ θ vᵢ² all share a sign, and rocBLAS's
DGEMM of that shape on the gfx1200 sums them up to ~1e-12 relative high,
where its `dot` and the 3080's cuBLAS DGEMM stay near 1e-15. A collapse
writes diag(θ) back into Vᵀ A V, so the offset never washes out, and the
FP64 stage starts from nearly converged vectors, so its first iteration's
offset is the one that stays. `davidson_solve` now rewrites each new
diagonal element by `dot` (`../../recipes/davidson/`: a dot per new column,
in device pointer mode, no sync). The off-diagonal elements, whose terms
cancel, were already accurate.

The alternatives were the two the issue named. A stagnation-aware
convergence test would have accepted a θ wrong by 1e-10, so a residual that
measures θ's error and not the vector's; the vectors were fine, so the
fix belongs in θ. A residual against a reorthogonalized basis would not have
moved it: V was orthonormal to 1e-14. Two others were tried on the way:
making `linear_dependence_floor` relative to each correction's norm turned
every `Divergence` into `MaxIterationsExceeded` at 400 iterations, the same
stall run longer, so it is not part of this; and `canonical_orthonormalization`
and `olsen_correction`, measured above, each moved one or two of the six by
a different rounding path.

`bench/ci_sigma_precision`, same conditions as the tables above, parent
1626bc5 → the diagonal by `dot`, both run back to back with both of each
card's GPU slots held:

| | | full, FP64-only | symmetric, FP64-only | two-stage 1e-6 | two-stage 1e-7 |
|---|---|---|---|---|---|
| CAS(10,10), 1.0977 Å | 3080 | 0.59 → 0.56 s (75 → 75) | 0.44 → 0.44 s (108 → 108) | 0.36 → 0.36 s (46+62 → 46+62) | 0.36 → 0.34 s (52+59 → 53+52) |
|  | gfx1200 | 1.81 → 1.59 s (81 → 81) | 0.99 → 0.97 s (108 → 108) | 0.92 → 0.87 s (46+62 → 46+58) | 0.92 → 0.85 s (52+57 → 53+49) |
| CAS(12,12), 1.0977 Å | 3080 | 13.5 → 13.7 s (120 → 119) | 6.04 → 6.56 s (106 → 106) | 4.17 → 4.27 s (43+66 → 43+62) | 4.10 → 4.04 s (55+56 → 55+49) |
|  | gfx1200 | 31.6 → 32.3 s (120 → 122) | 14.5 → 14.6 s (106 → 106) | Divergence → 12.2 s (43+59) | Divergence → 12.7 s (55+53) |
| CAS(10,10), 2.0 Å | 3080 | 1.41 → 1.42 s (226 → 227) | 0.84 → 0.85 s (225 → 226) | 0.67 → 0.66 s (120+105 → 120+105) | 0.65 → 0.64 s (138+87 → 138+87) |
|  | gfx1200 | 3.75 → 3.75 s (225 → 225) | 1.97 → 1.97 s (227 → 227) | 1.79 → 1.84 s (120+100 → 120+106) | 1.82 → 1.87 s (137+88 → 138+93) |
| CAS(12,12), 2.0 Å | 3080 | 22.0 → 23.0 s (183 → 195) | 10.6 → 10.5 s (191 → 190) | 7.47 → 7.56 s (85+106 → 85+109) | 7.06 → 7.36 s (97+93 → 97+95) |
|  | gfx1200 | 50.7 → 50.2 s (191 → 191) | 25.3 → 25.4 s (189 → 190) | 22.9 → 22.8 s (85+108 → 85+108) | 22.1 → 22.4 s (98+92 → 97+96) |

The four other gfx1200 `Divergence`s converge too: at 1.0977 Å, the full
layout at 1e-6 in 58 + 64 iterations (28.0 s) and symmetric 1e-5 in 39 + 65
(12.7 s); at 2.0 Å, full 1e-6 in 104 + 109 (48.1 s) and symmetric 1e-5 in
64 + 125 (23.0 s). All 28 solves converge on both cards. Over the 22 the
gfx1200's parent converges, 228.0 → 229.0 s; over the 3080's 28, 135.0 →
138.4 s, mostly one solve (CAS(12,12) at 2.0 Å, full, FP64-only), whose
iteration count moved 183 → 195 on the new rounding path: iteration counts
move by a few either way, and the dots themselves, one length-n
reduction per new column, are nothing beside a sigma. The energies close up:
every solve's worst root is within 6.7e-13 of the card's own FP64-only
solve on the 3080 (2.4e-12 in the parent) and 1.3e-12 on the gfx1200
(9.7e-11), and the FP64-only energies of the two cards agree to every printed
decimal. The parent's gfx1200 spread across solve modes, 5.6e-11 in the
tables above, was this offset, not FP64 sigma noise.


**The spin-averaged diagonal is an option, off by default.** Determinants
of one spatial configuration (the same doubly and singly occupied orbitals)
differ in diag(H) only by the exchange among their open shells, and a spin
eigenstate mixes them, so a per-determinant diagonal breaks spin in the
correction, which the penalty then has to undo. With `spin_averaged_diagonal`
the preconditioner's diag(H) is replaced by its average over each
configuration's determinants in the sector (DETCI's default,
`HD_AVG = EVANGELISTI`). It has a closed form: over the n_o open shells,
n_α of them α, the exchange −Σ_{i<j} δ_{σ_i σ_j} K_ij becomes −p Σ_{i<j} K_ij,
p = [n_α(n_α − 1) + n_β(n_β − 1)] / [n_o(n_o − 1)] the probability that two
open shells have the same spin, and every term with a closed shell is the same
across the configuration. So it is one change to `penalize_diagonal`'s pass,
with no search over partners: the thread already walks its α links a†_p a_p,
which give the α occupations, and its β slot table's (p, p) entries give the
β ones; it adds Σ_{i<j open} (δ_{σ_i σ_j} − p) K_ij from the sigma's (ij|ji).
It applies in both layouts, since a triangle element with three or more open
shells has partners beyond its mirror. Only the preconditioner takes it: the
guesses are still the determinants of lowest per-determinant diag(H). Off,
`penalize_diagonal` gets no `exchange` and the solve is the one before the
option existed.

It does not pay over the diagonal M either. `bench/ci_sigma_precision`, same
conditions as the tables above, three singlets, `max_iterations` 400, the
3080 idle with both of its GPU slots held, the option off → on in the same
binary and the same hold: over the 28 singlet solves 132.3 → 133.4 s (faster
on 11), 4297 → 4222 iterations; over the M_s = 1 triplets of CAS(12,12) at
1.0977 Å (`shift` 1, the full layout), 25.3 → 25.8 s (faster on 1 of 5),
394 → 433. The FP64-only solves take 1–12 more iterations each; the
two-stage solves move both ways, the FP32 stage most. Every energy, on or
off, is within 6.8e-13 of the FP64-only full-layout solve's; on against off
of the same mode, within 6.1e-13 in 32 of the 33 pairs and 1.1e-12 in the
33rd (CAS(10,10) at 2.0 Å, symmetric, 1e-6, whose off solve is itself
6.4e-13 from the FP64-only one). **The gfx1200 numbers are missing: no HIP
device was available for this run.** #287's P-space is to use this diagonal
for its Q block, and to re-bench the option there.

| | full, FP64-only | symmetric, FP64-only | two-stage 1e-6 | two-stage 1e-7 |
|---|---|---|---|---|
| CAS(10,10), 1.0977 Å | 0.76 → 0.73 s (108 → 111) | 0.39 → 0.40 s (101 → 104) | 0.33 → 0.32 s (45+59 → 45+56) | 0.36 → 0.33 s (58+57 → 54+51) |
| CAS(12,12), 1.0977 Å | 12.67 → 13.12 s (111 → 123) | 6.00 → 6.21 s (109 → 116) | 4.18 → 3.85 s (51+67 → 42+55) | 3.75 → 4.03 s (60+51 → 53+61) |
| CAS(10,10), 2.0 Å | 1.39 → 1.40 s (238 → 239) | 0.62 → 0.62 s (152 → 151) | 0.51 → 0.52 s (57+98 → 56+102) | 0.50 → 0.51 s (69+84 → 67+87) |
| CAS(12,12), 2.0 Å | 22.24 → 22.97 s (205 → 214) | 10.32 → 10.95 s (207 → 214) | 7.14 → 7.47 s (112+102 → 97+108) | 7.12 → 6.97 s (128+91 → 116+91) |

The triplets, CAS(12,12) at 1.0977 Å, M_s = 1, full layout, 3080:

| FP64-only | two-stage 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|
| 7.47 → 7.27 s (76 → 76) | 4.83 → 5.03 s (41+38 → 50+38) | 4.52 → 4.64 s (45+34 → 55+32) | 4.27 → 4.44 s (49+29 → 60+29) | 4.19 → 4.39 s (54+28 → 66+27) |

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
- over `CiSigma`, every case's singlets and triplets started from their own
  converged roots, two-stage and FP64-only: at most two iterations per stage
  (one, here), in the same layout as the cold solve, the energies to 1e-12 and
  the 1-RDMs to 1e-10 of the cold solve's, and the golden tolerances; and an
  empty start bit-identical to the cold solve, energies, ⟨S²⟩, iteration
  counts and vectors;
- over `CiSigma` with λ = 0, the full layout and a triplet among the lowest
  three;
- every solve above, with `olsen_correction` off and on, and with
  `davidson.canonical_orthonormalization` on; the two-stage ones also with
  `restart_with_previous`, whose FP64 stage's first apply then takes more
  than the three columns of X it takes without; and with
  `spin_averaged_diagonal` on, through a test-uploaded (ij|ji) for the host
  sigma and `CiSigma::exchange` for `CiSigma` (the warm starts and
  `MaxIterationsExceeded` too);
- `MaxIterationsExceeded` when given two iterations;
- the refusals, a start of the wrong size and `spin_averaged_diagonal` over a
  sigma whose `exchange()` is null among them, and `Numerical` for a start of
  zero vectors;
- `CiSpinSquare`'s `apply` and `penalize` against `host_spin_square` on
  random blocks of three vectors, each column to 1e-12 of its norm, over seven
  sectors (open and closed shells, an empty spin, a full one, a single orbital,
  N_α < N_β), `penalize_diagonal` against diag(S²) from the strings' popcounts
  (also in place), and its refusals;
- the same in the symmetric layout over six sets for both spins, on packed
  swap-symmetric blocks against `host_spin_square` packed, with
  `penalize_diagonal` exactly the full layout's on the triangle, and refused
  over two sets;
- `penalize_diagonal` with `CiSigma::exchange` against the spin-averaged
  diag(H) by brute force (each determinant's `host_diagonal` averaged over its
  configuration's determinants, plus the penalty's diagonal), to 1e-12, over
  every direct_spin1 case in four sectors (N_α − N_β of 0, 2, 4 and −2), and
  for the singlet sector in the symmetric layout too.

## Files

| File | Role |
|------|------|
| `solver.cppm` | Module interface: `ci_solve` (a template, with its two stages and two layouts), its options and roots, `host_spin_square`, `CiSpinSquare` |
| `solver.cpp` | `host_spin_square`; `CiSpinSquare`'s tables and launches; the preconditioner's launches, and Olsen's dots; the guesses |
| `ci_solver_bridge.h` | The launchers' declarations, shared by `solver.cpp`'s GMF and the `.cu` |
| `ci_solver.cu` | The S² gather and the penalized, optionally spin-averaged, diagonal (in either layout, over `../sigma/ci_symmetric.h`), the preconditioner's two kernels, the argmin passes and the guess spread |
| `CMakeLists.txt` | The device library `free_breakfast.ci.solver.device` and the module `free_breakfast.ci.solver` |
