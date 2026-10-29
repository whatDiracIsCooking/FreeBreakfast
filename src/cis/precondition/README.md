# Precondition

CIS's Davidson start and preconditioner, over the orbital-energy diagonal
Δε_ia = ε_a − ε_i: the diagonal itself, the unit-vector guess on its lowest
entries, and the correction r / (θ − Δε).

## Module

`free_breakfast.cis.precondition`; target `free_breakfast::cis::precondition`.
The launchers are `free_breakfast::cis::device`'s, in
`cis_precondition_bridge.h`, defined in `cis_precondition.cu` (target
`free_breakfast::cis::precondition::device`).

| Name | Role |
|---|---|
| `cis::diagonal(res, n_occ, n_virt, orbital_energies, delta)` | Δε into the n_ov device vector `delta`, element i · n_virt + a, from the n_occ + n_virt device orbital energies, occupied first; `Status` |
| `cis::guess(res, n_ov, delta, guess_count, vectors)` | The `guess_count` lowest Δε as unit vectors, ascending, ties to the lower index, into the n_ov × guess_count column-major device block `vectors`: orthonormal by construction; `Status` |
| `cis::Preconditioner(res, n_ov, delta, max_columns)` | r / (θ − Δε) over the borrowed device `delta`, for calls of up to `max_columns` columns: a `DavidsonPreconditioner<…, double>` |

```cpp
import free_breakfast.core.resources;
import free_breakfast.recipes;
import free_breakfast.cis.precondition;
namespace fb = free_breakfast;

fb::Buffers::device<double> delta(n_ov, res.device());
FBF_TRY(fb::cis::diagonal(res, n_occ, n_virt, eps.data(), delta.data()));
fb::Buffers::device<double> start(n_ov * guess_count, res.device());
FBF_TRY(fb::cis::guess(res, n_ov, delta.data(), guess_count, start.data()));
const fb::cis::Preconditioner precondition{res, n_ov, delta.data(), n_roots};
const auto info = FBF_TRY(fb::davidson_solve<double>(res, sigma, precondition, n_ov, n_roots,
                                                     max_subspace, start.data(), guess_count,
                                                     values.data(), vectors.data(), options));
```

## Design

**Where it came from.** The issue (#195) asked for a port of the private
`cis/precondition`, without its clustered exact-block correction (#202). That
source is not reachable from this tree, so this was written from the issue's
specification, with `../../recipes/davidson/`'s `DavidsonPreconditioner`
contract and `../../ci/solver/`'s diagonal preconditioner as the pattern.

**Δε, not diag(A).** A's diagonal is Δε_ia plus the two-electron
2 (ia|ia) − (ii|aa) for a singlet (−(ii|aa) for a triplet). Δε needs only the
orbital energies, which the SCF already has, and is the same for both spins;
the exact diagonal would cost an integral pass per element. Δε is the
leading term, and the Davidson converges to A's roots whichever M it is
preconditioned with: M only sets the iteration count.

**The clamp.** A root concentrated on one excitation has θ ≈ its Δε, and θ
can land on one exactly, so |θ − Δε| is held at or above 1e-3 with its sign
kept (a zero denominator taken as +1e-3), as the CI solver's is. Keeping the
sign keeps the correction on the side of the pole it came from; a correction
the floor made large in one element is scaled to unit norm by the Davidson
before it is used, so the floor only bounds the ratio between elements.

**One launch per call, θ in a buffer allocated up front.** The kernel covers
the whole n_ov × k block, one thread per element, and reads θ_c from a device
buffer of `max_columns` doubles allocated by the constructor; the CI solver
launches once per column with θ by value instead. So a call allocates nothing.
`theta` is the caller's pageable host memory, which an upload does not promise
to have read by the time it returns (`docs/style.md` "Host-device copies"), so
the call synchronizes after the launch: one more sync per Davidson iteration,
beside the several `davidson_solve` already takes.

**The guess is a host select.** `cis::guess` reads Δε back and partial-sorts the
indices on the host by (Δε, index). That runs once per solve, over n_ov
doubles, and the explicit tie on the index is what makes the order exact and
the result deterministic. The CI solver's device radix select
(`ci::device::pick_lowest`) earns its keep on CI vectors of millions of
elements; it is private to `ci`, and CIS's n_ov is far below that. The unit
vectors are written by one memset and one 8-byte copy of `kOne` per column,
needing no kernel and no sync: `kOne` is static storage.

**No Olsen correction, no P-space.** CIS's A is diagonally dominant in the same
way a CI matrix is, so the plain M⁻¹ r is nearly the Ritz vector too, but
nothing here needs Olsen's yet; the X `davidson_solve` passes for it is
accepted and not read. The clustered exact-block correction for
near-degenerate Δε is #202, and lands in this module.

## Tests

`test/gtest/cis_precondition_gpu.cpp`:

- the diagonal against the host to 1e-12, and its refusal of an empty
  occupied or virtual space;
- the guess against a host partial sort, with a tie inside the selected set and
  one straddling its last slot, every other element zero, and a second call
  identical; its refusal of a count outside [1, n_ov];
- the correction against r / (θ − Δε), with θ exactly on one diagonal entry
  (finite, +1e-3) and just above another (−1e-3, its sign kept), and in place;
  its refusal of more columns than it was built for;
- `davidson_solve` on a synthetic diagonally dominant A = diag(Δε) + coupling,
  through this preconditioner and guess, against `diagonalize`'s spectrum.

## Files

| File | Role |
|------|------|
| `precondition.cppm` | Module interface: `cis::diagonal`, `cis::guess`, `cis::Preconditioner` |
| `precondition.cpp` | Implementation: argument checks, the launches, the guess's select |
| `cis_precondition_bridge.h` | The launchers' declarations, shared by the `.cpp`'s GMF and the `.cu` |
| `cis_precondition.cu` | The diagonal and correction kernels |
| `CMakeLists.txt` | The device library and the module |
