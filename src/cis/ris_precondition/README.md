# CIS RIS-J preconditioner

`free_breakfast.cis.ris_precondition`; target
`free_breakfast::cis::ris_precondition`. The RIS-J preconditioner of Zhou &
Parker (2024; gpu4pyscf `tdscf/ris.py`) for `davidson_solve`, beside the
diagonal and clustered ones in `../precondition/`.

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

## Public API

| Name | Role |
|---|---|
| `cis::RisPreconditioner::create(res, basis, classes, atoms, c, n_occ, delta, max_columns, ris)` | `Result<RisPreconditioner>`: the RIS basis over `atoms` at `ris` (`RisOptions`), fitted against `basis` over its pair `classes`, and L over the spherical RHF orbitals `c`, the first `n_occ` occupied; `delta` is `cis::diagonal`'s, borrowed |
| `RisPreconditioner::operator()(res, theta, ritz_vectors, residual, correction)` | `Status`: the `DavidsonPreconditioner` call, z = (θ − Δε − 2 L Lᵀ)⁻¹ r per column; `correction` may be `residual` |
| `RisPreconditioner::naux()` | The RIS basis's function count |
| `RisPreconditioner::factor()` | Lᵀ on the device, naux × n_ov, for a check that rebuilds the operator |

`pipeline.cis`'s `CisStep` builds it when `cis::Options::ris` is set (`[cis]
ris = true`), in place of the diagonal or clustered correction; `cis::guess`
still supplies the start.

## Tests

`test/gtest/cis_ris_precondition_gpu.cpp`, over H₂O / STO-3G's own RHF: z
satisfies (θ − Δε − 2 L Lᵀ) z = r to 1e-7 relative to r against the operator
rebuilt on the host from Δε and `factor()`, the clamp included, at θ below,
between and above the Δε, and is not the diagonal correction; the same call in
place gives the same bytes; θ exactly on a Δε gives a finite z that still
solves the clamped system; `Sp` and `Spd` carry 6 and 11 functions; and the
refusals. `test/gtest/cis_step_gpu.cpp` runs `CisStep` with `ris = true` to
the reference's roots, and `test/gtest/lu_gpu.cpp` the LU.

## Files

| File | Role |
|------|------|
| `ris_precondition.cppm` | Module interface: `cis::RisPreconditioner` |
| `ris_precondition.cpp` | `create` (the fit and L) and the per-call Woodbury solve |
| `CMakeLists.txt` | The module `free_breakfast.cis.ris_precondition` |
