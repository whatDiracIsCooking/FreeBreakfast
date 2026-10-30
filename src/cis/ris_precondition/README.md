# CIS RIS preconditioner

`free_breakfast.cis.ris_precondition`; target
`free_breakfast::cis::ris_precondition`. The RIS-J preconditioner of Zhou &
Parker (2024; gpu4pyscf `tdscf/ris.py`) for `davidson_solve`, beside the
diagonal and clustered ones in `../precondition/`, and its RIS-JK extension
with the RI exchange term.

## Why

The diagonal correction r / (θ − Δε) treats each excitation ia on its own.
Where A couples near-degenerate excitations, which it does most through the
Coulomb term 2 (ia|jb), that coupling is what the diagonal cannot supply and
what the Davidson then spends iterations recovering. RIS supplies it cheaply,
through a resolution of the identity over a *minimal* auxiliary basis
(`../ris_basis/`: one uncontracted shell per l on each atom).

## The approximation

    A ≈ Δε + 2 L Lᵀ,   L_ia,Q = B_Q,ia,   L Lᵀ = (ia|P) M⁻¹ (P|jb)

B = L_M⁻¹ (mn|P) is the fitted factor over the RIS basis, built by the same
chain `pipeline.eri` builds its own with: `metric`, `three_center`,
`density_fit` (over `recipes::cholesky`, never the dense solver's `potrf`),
and then `b_mo` over C's virtual and occupied columns gives L's transpose, the
naux × n_ov matrix whose column i · n_virt + a is excitation ia, as for
`DfSigma`'s B_vo. M⁻¹ = L_M⁻ᵀ L_M⁻¹, so this is gpu4pyscf's J^{-1/2}
form, (ia|P) J^{-1/2} J^{-1/2} (P|jb), spelled in the Cholesky factor. It is **independent of `[eri]`**: the preconditioner fits its own
RIS basis whichever ERI mode the sigma uses, and frees that B once L is built.

## The correction

Per column c of the residual block, with Dt = θ_c − Δε under the diagonal
correction's clamp (|Dt| held at or above 1e-3, its sign kept, by the same
`device::precondition` kernel), Sherman–Morrison–Woodbury gives

    (Dt − 2 L Lᵀ)⁻¹ r = Dt⁻¹ (r + L y),   (½ I − Lᵀ Dt⁻¹ L) y = Lᵀ Dt⁻¹ r

exactly, with no iteration. The capacitance matrix ½ I − Lᵀ Dt⁻¹ L is
naux × naux (tens of functions), symmetric but **indefinite**: θ sits between
the Δε by construction, so Dt⁻¹ has both signs. A Cholesky does not apply, so
it is factored by `lu_solve` (`../../recipes/lu/`, pivoted LU). If U comes out
exactly singular for some θ, that column takes y = 0, which is the diagonal
correction r / Dt, and never a NaN.

On the device, per column: Dt⁻¹ (the kernel over a vector of ones), Lᵀ Dt⁻¹
(a dgmm), the capacitance matrix (a copy of ½ I and one gemm), its right-hand
side (a gemv-shaped gemm) and `lu_solve`. Then over the whole block: r copied
into the correction (unless it is already there), L Y added (one gemm) and
Dt⁻¹ applied in place (the kernel). `lu_solve` takes the device workspace and
synchronizes twice per column, so every buffer the correction holds across it
is the preconditioner's own, allocated by `create`.

## RIS-JK: the exchange term

A CIS singlet's A is Δε + 2 (ia|jb) − (ij|ab), so RIS-J drops the exchange
coupling. `RisOptions::exchange` adds it over the same fit:

    A ≈ Δε + 2 L Lᵀ − K_RI,   (K_RI)_ia,jb = Σ_Q T_ij^Q T_ab^Q

with T_oo and T_vv `b_mo` of the same B over the occupied, and the virtual,
columns on both sides: two more MO transforms, not a second three-centre
build. Applied to a vector, K_RI is `DfSigma`'s exchange over one column: a
batched gemm over a against T_vv, then one gemm against T_oo.

−K_RI is not low-rank in L, so Woodbury no longer inverts the shifted system.
With M = Dt − 2 L Lᵀ, the system (θ − A) z = r is (M + K_RI) z = r, and
left-preconditioned by M it is

    (I + M⁻¹ K_RI) z = M⁻¹ r = z₀,

z₀ the RIS-J answer. Each column runs an **unrestarted GMRES** on it, from
z₀: Arnoldi with classical Gram-Schmidt twice, each M⁻¹ the Woodbury solve
over ½ C⁻¹ (C the capacitance matrix, inverted once per column by `lu_solve`
against ½ I, and zero when C is exactly singular, as RIS-J's y is). GMRES and
not CG or MINRES: M⁻¹ is indefinite, θ sitting among the Δε by construction,
so M⁻¹ K_RI is not symmetric positive definite even though A is symmetric.
K_RI is the small term beside the Coulomb one, so M⁻¹ K_RI has a small
spectral radius and a few steps at a loose tolerance do:
`RisOptions::k_max_inner_iters` (10) caps the Krylov space and
`k_inner_tol` (1e-2) stops it once the residual estimate is that fraction of
‖z₀‖.

**Degrading.** z is written only at the end, so a breakdown before any
step, a zero on R's diagonal, a value that is not finite, or no step allowed
(`k_max_inner_iters = 0`) leaves z exactly z₀, the RIS-J answer. A lucky
breakdown (the next Arnoldi vector's norm at most 1e-14: the space is
invariant) ends the iteration with the exact solution of the space. A cap
reached first keeps GMRES's iterate, whose residual is never above z₀'s.

The vectors are the device's; the Hessenberg column comes back each step
(two Gram-Schmidt passes' coefficients and the new vector's norm), and the
Givens rotations and the back substitution run on the host, over at most
`k_max_inner_iters` entries. The Krylov basis and the rest of the GMRES's
scratch are the workspace's, carved after `lu_solve` returns and grown to
size by `create`; ½ C⁻¹, which `lu_solve` writes, is the preconditioner's own.

## Public API

| Name | Role |
|---|---|
| `cis::RisPreconditioner::create(res, basis, classes, atoms, c, n_occ, delta, max_columns, ris)` | `Result<RisPreconditioner>`: the RIS basis over `atoms` at `ris` (`RisOptions`), fitted against `basis` over its pair `classes`, and L over the spherical RHF orbitals `c`, the first `n_occ` occupied; `delta` is `cis::diagonal`'s, borrowed |
| `RisPreconditioner::operator()(res, theta, ritz_vectors, residual, correction)` | `Status`: the `DavidsonPreconditioner` call, z = (θ − Δε − 2 L Lᵀ)⁻¹ r per column, refined towards (θ − Δε − 2 L Lᵀ + K_RI)⁻¹ r with `ris.exchange`; `correction` may be `residual` |
| `RisPreconditioner::naux()` | The RIS basis's function count |
| `RisPreconditioner::factor()` | Lᵀ on the device, naux × n_ov, for a check that rebuilds the operator |
| `RisPreconditioner::factor_oo()`, `factor_vv()` | T_oo (naux × n_occ²) and T_vv (naux × n_virt²) on the device, empty without exchange, for the same check |

`pipeline.cis`'s `CisStep` builds it when `cis::Options::ris` is set (`[cis]
ris = true`, and `ris_exchange = true` for RIS-JK, with
`ris_k_max_inner_iters` and `ris_k_inner_tol`), in place of the diagonal or
clustered correction; `cis::guess` still supplies the start.

## Tests

`test/gtest/cis_ris_precondition_gpu.cpp`, over H₂O / STO-3G's own RHF: z
satisfies (θ − Δε − 2 L Lᵀ) z = r to 1e-7 relative to r against the operator
rebuilt on the host from Δε and `factor()`, the clamp included, at θ below,
between and above the Δε, and is not the diagonal correction; the same call in
place gives the same bytes; θ exactly on a Δε gives a finite z that still
solves the clamped system; `Sp` and `Spd` carry 6 and 11 functions; and the
refusals. With exchange, against the operator rebuilt from Δε, `factor()`,
`factor_oo()` and `factor_vv()`: at a tight `k_inner_tol` and a Krylov cap
above n_ov (an unrestarted GMRES spans the whole space by then), z solves
(θ − Δε − 2 L Lᵀ + K_RI) z = r to 1e-6 and is not the RIS-J answer; zero
inner iterations reproduces RIS-J to 1e-12; and θ on a Δε still gives a
finite z that solves the clamped system. `test/gtest/cis_step_gpu.cpp` runs
`CisStep` with `ris = true`, and with `ris_exchange = true` beside it, to the
reference's roots, and `test/gtest/lu_gpu.cpp` the LU.

## Files

| File | Role |
|------|------|
| `ris_precondition.cppm` | Module interface: `cis::RisPreconditioner` |
| `ris_precondition.cpp` | `create` (the fit, L, and T_oo and T_vv) and the per-call Woodbury solve and GMRES |
| `CMakeLists.txt` | The module `free_breakfast.cis.ris_precondition` |
