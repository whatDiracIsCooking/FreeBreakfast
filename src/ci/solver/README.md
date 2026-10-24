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
| `CiSolverOptions` | `spin_penalty` (λ, 1 hartree; 0 keeps every spin) and `davidson` (`DavidsonOptions`, residual tolerance 1e-12) |
| `CiRoots` | `energies`, `spin_squares` (⟨S²⟩), `vectors` (a device buffer, root r at [r · dim, (r + 1) · dim)), `davidson` (`DavidsonInfo`) |
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
| `solver.cppm` | Module interface: `ci_solve` (a template), its options and roots, `host_spin_square`, `CiSpinSquare` |
| `solver.cpp` | `host_spin_square`; `CiSpinSquare`'s tables and launches; the preconditioner's launch; the guesses |
| `ci_solver_bridge.h` | The launchers' declarations, shared by `solver.cpp`'s GMF and the `.cu` |
| `ci_solver.cu` | The S² gather, the penalized diagonal, the preconditioner kernel, the argmin passes and the guess spread |
| `CMakeLists.txt` | The device library `free_breakfast.ci.solver.device` and the module `free_breakfast.ci.solver` |
