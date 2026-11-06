# PCG

Block preconditioned conjugate gradients on A X = B for a symmetric positive definite A known only as a [`LinearOperator`](../linear_operator/README.md), with a second `LinearOperator` as the preconditioner, or none. It is the one PCG the response solves are moving onto: `response::cphf`'s lockstep solve, `mcscf::zvector(s)`' per-column and Schur solves, and `RelaxedHessian`'s unpreconditioned inner CI solve (milestone "Response consolidation", #829–#831). No caller uses it yet.

## Module

`free_breakfast.recipes.pcg`, also re-exported by `free_breakfast.recipes`. It re-exports `free_breakfast.recipes.linear_operator`.

## Design

`pcg_solve<T>(res, op, precondition, n, k, b, x, options)` solves the k columns of B at once, each by its own CG recurrence, from X = 0 (R = B) or, with `start_from_x`, from the X passed in (R = B − A X, one product over every column). P = S = M R over every column to start. Each sweep then:

1. **Picks the active columns**: those whose ‖r‖₂ is above their tolerance, and not stopped (below).
2. **Applies A once**, to the active columns' directions gathered into one contiguous block (as `zvectors`' Schur path does), or to P in place when every column is active.
3. **Steps each active column**: p·Ap, then α = r·s / p·Ap, x += α p, r −= α Ap, ‖r‖₂.
4. **Applies M once**, to the residuals of the columns just stepped, gathered the same way, and sets p = s + β p with β = r·s_new / r·s_old.

A column that has stopped is never read or written again: its X stays as it was. That is the recurrence `response::cphf` and `mcscf::zvectors` run today, step for step, so moving them onto it leaves their sweep counts unchanged.

**No preconditioner.** With `precondition` `nullptr`, S is R itself: no copy, and r·s is ‖r‖₂², the norm step 3 already took. That is `RelaxedHessian`'s inner CI CG, which forms rr from its `nrm2`.

**Stopping rules.**

| Rule | Behaviour |
|---|---|
| Tolerance | Per column, absolute, on ‖r_j‖₂: `tolerance` for all, or `tolerances`, one per column. A relative rule passes `rel · ‖b_j‖`, as `RelaxedHessian`'s `response_tolerance` will; a column whose right-hand side is round-off can be switched off with a tolerance at its norm. |
| `max_iterations` | A column still active at the start of sweep `max_iterations + 1` is `MaxIterationsExceeded`. With `stop_silently` the solve stops there instead, returns normally, and flags those columns `converged = false`: `RelaxedHessian`'s CI response stops silently at `response_max_iterations`. |
| Curvature | p·Ap ≤ 0 (or NaN) is `Divergence` when `check_curvature` is on, the default. Off, that column stops there, its X as the previous sweep left it, flagged `converged = false`, and the others carry on: `RelaxedHessian`'s inner CG ends a vector's solve on such a direction (round-off's other spins) rather than failing. |

A NaN residual norm is never at or below its tolerance, so its column stays active until its NaN curvature stops it.

**True residuals.** With `true_residuals` (the default) one more product over every column gives ‖A x_j − b_j‖₂, reported per column: the recurrence's R drifts from it in rounding. `RelaxedHessian`'s inner solve does not take it.

**None of the buffers is the workspace.** R, P and the gathered input and output blocks of an apply are held across `op.apply` and `precondition.apply`, either of which may take `res.device_workspace()`, so they are one pool buffer of the call's own, 4 n k elements, allocated on entry (`docs/style.md` "Scratch memory"). X and B are the caller's.

**Host pointer mode, and a sync per reduction.** Each p·Ap, r·s and ‖r‖ is a host-mode `dot` or `nrm2`, read when it is formed, as the callers' loops do; `op` and `precondition` run in host pointer mode too. Batching those reductions in device mode would save syncs, but would change nothing the callers' sweep counts depend on, and is left for when a caller's profile asks for it.

**Device.** The call makes `res.dev_idx()` current for its duration and restores the previous device.

| Outcome | Return |
|---|---|
| Every column converged, or stopped under `stop_silently` / `check_curvature` off | `PcgInfo` |
| A column active after `max_iterations` sweeps | `MaxIterationsExceeded` |
| p·Ap ≤ 0 with `check_curvature` | `Divergence` |
| A shape or option outside the documented bounds | `InvalidConfig` |
| A BLAS, runtime, `op` or `precondition` failure | that error, passed up |

After an error, X holds the iterate as far as it got.

## Public API

| Symbol | Role |
|---|---|
| `pcg_solve<T>(…)` | `Result<PcgInfo>`: X = A⁻¹ B for k columns. Any `real_fp` `T`; a template instantiated by the caller. `precondition` is a `LinearOperator<T>` lvalue or `nullptr`. |
| `PcgOptions` | `tolerance` (1e-10), `tolerances` (empty), `max_iterations` (100), `start_from_x` (false), `check_curvature` (true), `stop_silently` (false), `true_residuals` (true) |
| `PcgInfo` | `sweeps`; per column `iterations` (products of A), `converged`, and `residuals` (empty without `true_residuals`) |

## Usage

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;
namespace fb = free_breakfast;

fb::DeviceResources res{0};
Hessian op{...};       // any fb::LinearOperator<Hessian, double>
Diagonal jacobi{...};  // M ~ A^-1, also a LinearOperator, or pass nullptr

fb::Buffers::device<double> x(static_cast<size_t>(n) * k, res.device());
fb::PcgOptions options;
options.tolerance = 1e-8;
const auto info = fb::pcg_solve(res, op, jacobi, n, k, b.data(), x.data(), options);
```

## Tests

`test/recipes/pcg_gpu.cpp`, over a dense A held as a one-gemm operator. It covers:

- a dense SPD model, three columns to 1e-10 with no preconditioner, against a host Cholesky solve, with an operator that overwrites the device workspace on every apply, every apply in host pointer mode;
- per-column tolerances: each column stops at its own sweep, each apply covers only the columns still running, and each column matches a solve of it alone (same products, x within 1e-12 relative);
- a Jacobi preconditioner on an ill-conditioned diagonal (six decades) within CG's bound of 24 sweeps, which the identity path exceeds;
- `Divergence` on an indefinite matrix, and with `check_curvature` off the stopped column untouched while the other converges;
- `MaxIterationsExceeded` at one sweep, and under `stop_silently` a normal return with every column flagged and its true residual reported;
- `start_from_x` from a converged X: no sweep, X returned as given;
- shape and option refusals.

## Files

| File | Role |
|------|------|
| `pcg.cppm` | Module interface: `pcg_solve`, its options and info |
| `CMakeLists.txt` | The module `free_breakfast.recipes.pcg` |
