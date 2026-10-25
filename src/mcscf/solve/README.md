# Solve

The two-step, state-averaged CASSCF loop on the GPU: `casscf_solve`, which
optimizes the orbitals and the CI of `nroots` roots at fixed weights until the
weighted energy is stationary in both.

## Module

`free_breakfast.mcscf.solve`, target `free_breakfast::mcscf::solve`,
re-exported by `free_breakfast.mcscf`. It re-exports `free_breakfast.ci.solver`,
whose `CiSolverOptions` its options hold and whose `CiRoots` its solution does.

```cpp
Result<CasscfSolution> casscf_solve(
    DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
    const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
    std::span<CoulombPairClass> j_classes, std::span<OrderedPairClass> k_classes,
    const SphericalMatrixAO &c, int32_t ncore, int32_t ncas, int32_t nelecas,
    int32_t nroots, std::span<const double> weights = {}, const CasscfOptions &options = {});
```

The arguments through `k_classes` are `casscf_fock`'s; `c` is the start, its
columns core, active, virtual as the rest of the layer takes them. The roots
are the `nroots` lowest of spin |N_α − N_β| / 2, N_β = nelecas / 2: singlets
for an even `nelecas`. `weights` are refused (`InvalidConfig`) if any is
negative, if they do not sum to 1 within 1e-12 or if their count is not
`nroots`; empty means equal.

`CasscfSolution` holds C, the `CiRoots` at C (each root's energy and ⟨S²⟩,
and the CI vectors on the device), the weighted energy, ‖g‖, the weighted γ and
Γ and each root's γ on the host, and one `CasscfIteration` per macro iteration
(the energy, ΔE, ‖g‖, the trust radius, the CI's Davidson iterations, the
micro iterations and the Hessian products of the step, and whether the
iteration was accepted).

## The loop

Each macro iteration, at the current C:

1. **The integrals**: `casscf_integrals`, the active-space Hamiltonian and
   ppaa, papa.
2. **The CI**: `ci_solve` over the GPU sigma (`CiSigma`, created once, its
   integrals set per iteration), from `ci_solve`'s own guesses: **not
   warm-started** (see "Not done"). Its relative residual tolerance is
   min(1e-8, 1e-4 ‖g‖) of the last iteration, held at or above
   `options.ci`'s (1e-12 by default), so the early iterations do not
   over-solve and the last ones are as tight as a lone `ci_solve`.
3. **The RDMs and derivatives**: `CiRdm::rdm12` at the weights, read back,
   and `casscf_fock`: the weighted energy and the gradient g.
4. **Convergence**: |ΔE| < `conv_tol` over the last accepted iteration and
   ‖g‖ < `conv_tol_grad`. The first iteration has no ΔE, so the loop takes at
   least two.
5. **Rejection**: a rise past 1e-11 hartree over the last accepted iteration
   drops these orbitals, halves the step from the accepted ones and sets the
   trust radius to its new length. The step is not re-solved: its κ's
   `skew_eig` is kept, and `skew_expm` takes the half as its t.
6. **The step**: `OrbitalHessian` H at C plus the CI response D (below), and
   up to `max_micro` solves of the quadratic model at these frozen integrals,
   E + g·x + ½ xᵀ(H + D)x. Each is the lowest root of the augmented Hessian
   [[0, gₘᵀ], [gₘ, H + D]], gₘ the model gradient; its eigenvector (v₀, v)
   gives dx = v / v₀, with (H + D − λ) dx = −gₘ. dx is cut to what is left of
   the trust radius, and the model gradient moves to gₘ + (H + D) dx. The
   solves stop once ‖gₘ‖ is a tenth of ‖g‖, the radius is reached, or
   `max_micro` is.

   **The solves share one subspace.** H + D is fixed for the macro
   iteration, and only gₘ changes between its solves, so they keep one
   orthonormal Q (n_pairs × q) and (H + D) Q, grown by every product taken
   and dropped when the macro iteration ends. A solve projects:
   the lowest root (μ, (u₀, u)) of the (q + 1)-square
   [[0, (Qᵀgₘ)ᵀ], [Qᵀgₘ, Qᵀ(H + D)Q]] (`diagonalize`), and its residual
   r = u₀ gₘ + (H + D) Q u − μ Q u (its first row is 0 by the small
   problem's own equation). Only while ‖r‖ is above the tolerance below does
   Q grow, by r / (μ − diag(H + D)) (the denominator held off zero at 1e-8),
   orthogonalized against Q twice and normalized, and one product. Then
   dx = Q u / u₀ and (H + D) dx = (H + D) Q u / u₀, a gemv of products
   already taken. Q holds at most 40 columns (`kAhMaxSubspace`); past that
   it restarts on the current direction Q u / ‖u‖, whose image is a gemv of
   the old one's, so no product is lost. A solve adds at most 50 columns,
   and a correction re-orthogonalized to 1e-12 or less ends it; either way
   its last root is taken, and the energy test guards the step.

   This is what `davidson_solve` from e₀ did, for fewer products. Its subspace was [e₀, (0, Q)]: e₀'s row is the only one a
   correction has after the first, and re-orthogonalizing against e₀
   removes it. But it applied the operator to e₀ itself, an H product of
   zero, once per solve; and each solve after a macro iteration's first
   started over, though gₘ = g + (H + D) x lies almost wholly in the span
   the last one built.
7. **Rotate**: C ← C exp(κ(x)), κ unpacked to the full antisymmetric nmo × nmo
   (`unpack_orbital_pairs`) and exponentiated by `skew_eig` and `skew_expm`.

**The CI response is learned, BFGS-style.** H is the Hessian at fixed CI, and
the CI's relaxation softens the true one by H_oc (H_cc − E)⁻¹ H_co. Without it
the loop is linear: from these starts H₂O CAS(4,4) took 22 macro iterations
and H₂O AVAS 24, each cutting ‖g‖ by a steady 0.4–0.5. That term is not
built (it needs the CI Hamiltonian's derivative along every κ); D learns it
from the steps instead. After each accepted step s = t x, with B s =
t (H + D) x the model's and y = g_new − g_old the gradient's actual change, CI
relaxed at both ends, D gains the BFGS pair y yᵀ / (y·s) − (Bs)(Bs)ᵀ / (s·Bs),
taken only when both denominators are positive (past 1e-8 of their factors'
norms), so B stays positive definite. The last 8 steps are kept, as vectors
and scales, so D v is two gemms and a dgmm on the device. With it every case
here converges in 9 to 13. A symmetric rank-one update (SR1) was tried first
and rejected: it admits negative curvature, and near convergence a noisy pair
gave H + D a direction the augmented Hessian then stepped the full radius
along.

An accepted step's actual ΔE against the model's t g·x + ½ t² xᵀHx doubles the
radius, up to `trust_radius`, when the two are within 25%.

## Defaults, and what each settles

| Option | Default | Settles |
|---|---|---|
| `conv_tol` | 1e-10 | ΔE of the weighted energy at convergence |
| `conv_tol_grad` | 1e-6 | ‖g‖ at convergence. The weighted energy is quadratic in the orbitals' error there, but each root's energy and the natural occupations are first order: ~‖g‖ / (H's least eigenvalue). The tests run at 1e-9 for 1e-8 per-root energies |
| `max_cycle` | 50 | macro iterations, rejected ones included, before `MaxIterationsExceeded` |
| `trust_radius` | 0.5 | the initial and largest ‖x‖ of a step |
| `max_micro` | 4 | augmented-Hessian solves per macro iteration on its frozen integrals |
| `ci` | `CiSolverOptions{}` | the CI's spin penalty and its tightest residual tolerance |
| `tau_j`, `tau_k` | 1e-14 | the screening of every J and K build |

**The cost is the Hessian products.** Each H·κ is two J and two K builds when
there is a core (`hessian/`), and each column added to Q costs one;
`CasscfIteration::hessian_products` counts them. A macro iteration's first
solve pays for every column it needs; a later solve pays only for what its
own residual still lacks, often half or less. The integrals and the
Hessian's own setup are a few J and K per active pair and per core orbital,
once per macro iteration.

**The subspace allocates nothing per iteration.** Q, (H + D) Q beside gₘ, D's
terms and scales, diag(H + D), the residual, the correction, the step pair
(dx, (H + D) dx), and the projected problem are buffers of the solve's, sized
once for `kAhMaxSubspace` columns and 2 × 8 BFGS terms. diag(H + D) is
uploaded once per macro iteration, and the preconditioner is a kernel of this
module's (`ah_precondition.cu`), μ passed by value. Per column added there
are a handful of syncs (the small problem's upload, `diagonalize`, its
readback, the residual's and the correction's norms, the new row's readback),
against the two J and two K of its product. The kernel is this module's, not
a shared one in `recipes/davidson`: `ci_solve`'s preconditioner divides with
a 1e-3 floor over a spin-shifted diagonal, this one multiplies by the
reciprocal with a 1e-8 floor, and a common kernel would carry both as options
for two callers.

**The tolerance follows ‖gₘ‖**: a residual of about 1e-2 ‖gₘ‖ (relative to
max(1, max |diag(H + D)|)), times max(|μ|, the projected problem's largest
|eigenvalue|), so the step is good to about 1% at any ‖g‖. A fixed one would
stop the solve at e₀ once ‖g‖ fell below it, and the step would be zero.

**Why a persistent augmented Hessian, and not Steihaug CG or GLTR** over the
same products: those take the Newton step, or a boundary point of the trust
region when H + D is indefinite, where the augmented Hessian takes a
level-shifted one. Swapping the step's model would move every case's macro
trajectory and the radius's behaviour with it, which the issue's bar (macro
iterations within one) leaves no room to absorb, and the products it could
save are few: past a macro iteration's first solve, the persistent subspace
already prices a later one at its residual's few columns, and most macro
iterations here take one solve. Steihaug was not measured.

## Not done

- **The CI warm start** (#282), issue #251's "warm-started from the previous
  iteration's vectors". `ci_solve` takes no starting vectors, and adding them
  is not the small change it was: its singlet path now runs on the packed
  vectors of `CiLayout::symmetric` (the triangle I_α ≥ I_β), and only
  `ci_unpack` exists, so the last roots, which `CiRoots` holds in the full
  layout, have no way back in. Each macro iteration solves the CI from the
  determinant guesses; the tolerance schedule above still keeps the early
  ones short, and the macro-iteration counts do not depend on it.

- **A coupled orbital–CI step** (PySCF's `mc1step`): the CI is re-solved, not
  stepped, each macro iteration, and its coupling to the orbitals is only
  learned (D), not built.
- **Frozen orbitals**: every core orbital rotates.
- **Root following**: the roots are the lowest `nroots` each iteration.

## Tests

`test/gtest/mcscf_solve_gpu.cpp`, against `test/gtest/data/casscf/casscf.json`'s
converged, polished results, from each case's recorded start, set up as
`mcscf_fock_gpu.cpp` does: H₂O cc-pVDZ CAS(4,4) [2, 3, 5, 6], N₂ cc-pVDZ
CAS(6,6), H₂CO cc-pVDZ CAS(4,3) at [0.5, 0.5] and [0.7, 0.3], and H₂O cc-pVDZ
by AVAS, at `conv_tol_grad` 1e-9:

- each root's energy and the weighted one to 1e-8;
- the state-averaged natural occupations to 1e-6;
- ‖g‖ below `conv_tol_grad`, recomputed from scratch with a cold CI;
- Cᵀ S C = 1 to 1e-10;
- no accepted iteration's ΔE above 1e-11;
- each case's macro iterations printed, and at most 20;
- `MaxIterationsExceeded`, and the refusals.

## Files

| File | Role |
|------|------|
| `solve.cppm` | Module interface: `casscf_solve`, `CasscfOptions`, `CasscfIteration`, `CasscfSolution` |
| `solve.cpp` | The loop, the augmented Hessian's persistent subspace over H + D, the BFGS pairs |
| `ah_precondition.cu` | The preconditioner's kernel, r / (μ − diag(H + D)) for one column |
| `ah_precondition_bridge.h` | Its launcher, shared by the `.cu` and `solve.cpp`'s global module fragment |
| `CMakeLists.txt` | The device library `free_breakfast.mcscf.solve.device` and the module `free_breakfast.mcscf.solve` |
