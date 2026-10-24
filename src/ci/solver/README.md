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
| `CiSolverOptions` | `spin_penalty` (λ, 1 hartree; 0 keeps every spin), `davidson` (`DavidsonOptions`, residual tolerance 1e-12) and `fp32_tolerance` (the single-precision stage's, 1e-6; 0 turns it off) |
| `CiRoots` | `energies`, `spin_squares` (⟨S²⟩), `vectors` (a device buffer, root r at [r · dim, (r + 1) · dim)), `davidson` (`DavidsonInfo` of the FP64 stage), `fp32_iterations` (the single-precision stage's, 0 when none ran) |
| `host_spin_square(alpha, beta, c, s2c)` | S² c on the host; `Status` |
| `CiSpinSquare::create(res, alpha, beta)` | Upload S²'s tables; `Result<CiSpinSquare>` |
| `CiSpinSquare::apply(res, k, x, y)` | Y = S² X for size() × k device blocks: a `LinearOperator`; `Status` |
| `CiSpinSquare::penalize(res, k, λ, target, x, y)` | Y += λ (S² − target) X, the penalty ci_solve adds into sigma's output; `Status` |
| `CiSpinSquare::penalize_diagonal(res, λ, target, diagonal, shifted)` | shifted = diagonal + λ (diag(S²) − target), the preconditioner's diagonal; `Status` |

`sigma` applies H to a device block of CI vectors in `host_sigma`'s layout:
the GPU sigma in a calculation, `host_sigma` wrapped as a `LinearOperator` in
the tests. `diagonal` is diag(H) on the device, as `CiSigma::diagonal` writes
it.

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

**Energies are ⟨H⟩.** The Ritz value θ is the unit Ritz vector's Rayleigh
quotient of the penalized operator, so θ − λ (⟨S²⟩ − S(S+1)) is its quotient
of H itself, contamination or not.

**S² = S_z (S_z + 1) + N_β − Σ_pq E^β_pq E^α_qp.** S₋S₊ = Σ_pq a†_pβ a_pα a†_qα
a_qβ reorders to E^β_pq (δ_pq − E^α_qp), so both factors are same-spin
excitations and go through the sets' own replacement lists: nothing leaves the
sector, and no string set for (N_α ± 1, N_β ∓ 1) is needed. Each operator is
even in its spin, so it passes the other spin's creators with no sign and
carries only its own list's.

**The preconditioner is r / (θ − diag(H + penalty)).** diag(S²) is
S_z (S_z + 1) + N_β less the doubly occupied orbitals, so the penalty's
diagonal is free, and including it keeps the lifted determinants from being
over-weighted. The denominator is held off zero at 1e-3.

**Guesses are the 2 · n_roots determinants of lowest diag(H), spread.** A
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
slots):

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
column, θ passed by value, over diag(H + penalty), which `penalize_diagonal`
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
  tolerances, the first with `fp32_iterations` > 0 and the second with 0;
- `MaxIterationsExceeded` when given two iterations;
- the refusals;
- `CiSpinSquare`'s `apply` and `penalize` against `host_spin_square` on
  random blocks of three vectors, each column to 1e-12 of its norm, over seven
  sectors (open and closed shells, an empty spin, a full one, a single orbital,
  N_α < N_β), `penalize_diagonal` against diag(S²) from the strings' popcounts
  (also in place), and its refusals.

## Files

| File | Role |
|------|------|
| `solver.cppm` | Module interface: `ci_solve` (a template, with its two stages), its options and roots, `host_spin_square`, `CiSpinSquare` |
| `solver.cpp` | `host_spin_square`; `CiSpinSquare`'s tables and launches; the preconditioner's launch; the guesses |
| `ci_solver_bridge.h` | The launchers' declarations, shared by `solver.cpp`'s GMF and the `.cu` |
| `ci_solver.cu` | The S² gather, the penalized diagonal, the preconditioner kernel, the argmin passes and the guess spread |
| `CMakeLists.txt` | The device library `free_breakfast.ci.solver.device` and the module `free_breakfast.ci.solver` |
