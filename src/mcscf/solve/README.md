# Solve

The two-step, state-averaged CASSCF loop on the GPU: `mcscf::solve`, which
optimizes the orbitals and the CI of `nroots` roots at fixed weights until the
weighted energy is stationary in both.

## Module

`free_breakfast.mcscf.solve`, target `free_breakfast::mcscf::solve`,
re-exported by `free_breakfast.mcscf`. It re-exports `free_breakfast.ci.solver`,
whose `ci::SolverOptions` its options hold and whose `ci::Roots` its solution does.

```cpp
Result<mcscf::Solution> mcscf::solve(
    DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
    const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
    std::span<CoulombPairClass> j_classes, std::span<OrderedPairClass> k_classes,
    const SphericalMatrixAO &c, int32_t ncore, int32_t ncas, int32_t nelecas,
    int32_t nroots, std::span<const double> weights = {}, const mcscf::Options &options = {},
    const Buffers::device<double> *b = nullptr, int32_t naux = 0);
```

The arguments through `k_classes` are `response::fock`'s; `c` is the start, its
columns core, active, virtual as `../../response/README.md` orders them. The roots
are the `nroots` lowest of spin |N_α − N_β| / 2, N_β = nelecas / 2: singlets
for an even `nelecas`. `weights` are refused (`InvalidConfig`) if any is
negative, if they do not sum to 1 within 1e-12 or if their count is not
`nroots`; empty means equal. `b` and `naux` are B for the builds
`options.density_fitting` fits (see "Density fitting" below); null means none.

`mcscf::Solution` holds C, the `ci::Roots` at C (each root's energy and ⟨S²⟩,
and the CI vectors on the device), the weighted energy, ‖g‖, the weighted γ and
Γ and each root's γ on the host, and one `mcscf::Iteration` per macro iteration
(the energy, ΔE, ‖g‖, the trust radius, the CI's Davidson iterations in each
of its two stages, the micro iterations and the Hessian products of the step, and whether the
iteration was accepted).

## The loop

Each macro iteration, at the current C:

1. **The integrals**: `casscf_integrals`, the active-space Hamiltonian and
   ppaa, papa.
2. **The CI**: `ci::solve` over the GPU sigma (`ci::Sigma`, created once, its
   integrals set per iteration), **warm-started** from the last iteration's
   roots (`ci::solve`'s `start`; the first iteration is from its own guesses).
   The last roots are those of the last orbitals, accepted or rejected: the
   nearest H either way. The root order and count do not change, and the RDMs
   do not depend on a vector's sign. Late in the loop, where C barely moves,
   the single-precision stage then stops at its first iteration and the
   double one starts close: over the five test cases the CI's iterations fall
   from 239 + 213 to 156 + 182 in total (single + double precision) on the
   RTX 3080, the macro-iteration counts unchanged. On H₂CO's CAS(4,3), six
   triangle elements, the early iterations take one to three more, since the
   cold guesses span nearly the whole space at once (see
   `../../ci/solver/`); each is at most two sigma columns of six. Its
   relative residual tolerance is
   min(1e-8, 1e-4 ‖g‖²) of the last iteration, held at or above
   `options.ci`'s (1e-12 by default), so the early iterations do not
   over-solve and the last ones are as tight as a lone `ci::solve`. ‖g‖²,
   not ‖g‖: the steps converge quadratically, so the CI has to keep ahead of
   the next ‖g‖. At 1e-4 ‖g‖ (what the learned response's linear steps took)
   the CI's error in g, up to ~60 times its residual, held the last steps
   back and cost every case one macro iteration (H₂O 8 rather than 7: ‖g‖
   went 4.2e-7 to 3.2e-7), and on H₂CO SA2 it left the final ‖g‖ at 7.6e-10
   inside the loop but 1.5e-9 recomputed with a tight CI.
3. **The RDMs and derivatives**: `ci::Rdm::rdm12` at the weights, read back,
   and `response::fock`: the weighted energy and the gradient g.
4. **Convergence**: |ΔE| < `conv_tol` over the last accepted iteration and
   ‖g‖ < `conv_tol_grad`. The first iteration has no ΔE, so the loop takes at
   least two.
5. **Rejection**: a rise past 1e-11 hartree over the last accepted iteration
   drops these orbitals, halves the step from the accepted ones and sets the
   trust radius to its new length. The step is not re-solved: its κ's
   `skew_eig` is kept, and `skew_expm` takes the half as its t.
6. **The step**: `response::Hessian` H at C plus the CI's response R (below),
   H̃ = H + R, and up to `max_micro` solves of the quadratic model at these
   frozen integrals,
   E + g·x + ½ xᵀH̃x. Each is the lowest root of the augmented Hessian
   [[0, gₘᵀ], [gₘ, H̃]], gₘ the model gradient; its eigenvector (v₀, v)
   gives dx = v / v₀, with (H̃ − λ) dx = −gₘ. dx is cut to what is left of
   the trust radius, and the model gradient moves to gₘ + H̃ dx. The
   solves stop once ‖gₘ‖ is a tenth of ‖g‖, the radius is reached, or
   `max_micro` is. A step x along negative curvature, xᵀH̃x < 0, may then
   be cut to a quarter of the radius (see "The trust region" below).

   **The solves share one subspace.** H̃ is fixed for the macro
   iteration, and only gₘ changes between its solves, so they keep one
   orthonormal Q (n_pairs × q) and H̃ Q, grown by every product taken
   and dropped when the macro iteration ends. A solve projects:
   the lowest root (μ, (u₀, u)) of the (q + 1)-square
   [[0, (Qᵀgₘ)ᵀ], [Qᵀgₘ, QᵀH̃Q + σ]] (`diagonalize`), σ = 1e-8 the level
   shift (see "The level shift" below), and its residual
   r = u₀ gₘ + H̃ Q u − (μ − σ) Q u (its first row is 0 by the small
   problem's own equation). Only while ‖r‖ is above the tolerance below does
   Q grow, by r / (μ − σ − diag(H)) (the denominator held off zero at 1e-8),
   orthogonalized against Q twice and normalized, and one product. Then
   dx = Q u / u₀ and H̃ dx = H̃ Q u / u₀, a gemv of products
   already taken. Q holds at most 40 columns (`kAhMaxSubspace`); past that
   it restarts on the current direction Q u / ‖u‖, whose image is a gemv of
   the old one's, so no product is lost. A solve adds at most 50 columns,
   and a correction re-orthogonalized to 1e-12 or less ends it; either way
   its last root is taken, and the energy test guards the step.

   This is what `davidson_solve` from e₀ did, for fewer products. Its subspace was [e₀, (0, Q)]: e₀'s row is the only one a
   correction has after the first, and re-orthogonalizing against e₀
   removes it. But it applied the operator to e₀ itself, an H product of
   zero, once per solve; and each solve after a macro iteration's first
   started over, though gₘ = g + H̃ x lies almost wholly in the span
   the last one built.
7. **Rotate**: C ← C exp(κ(x)), κ unpacked to the full antisymmetric nmo × nmo
   (`unpack_orbital_pairs`) and exponentiated by `skew_eig` and `skew_expm`.

**The CI response is computed, per product.** H is the Hessian at fixed CI.
With the CI re-solved each macro iteration, the energy the loop minimizes is
E(x) with the CI relaxed, and its Hessian is the Schur complement of the
coupled one over (x, δ), δ_r ⊥ c_r the change of root r:

    H̃ = H − H_oc H_cc⁻¹ H_co,   (H_co x)_r = 2 w_r (1 − c_r c_rᵀ) H′(x) c_r,
    (H_cc)_r = 2 w_r (H₀ − E_r),   H_oc δ = Σ_r 2 w_r ⟨δ_r| H′(·) |c_r⟩,

H′(x) the derivative of the active-space Hamiltonian along κ(x). So
R x = H_oc δ with δ_r = −(H₀ − E_r)⁻¹ (1 − c_r c_rᵀ) H′(x) c_r (w_r cancels),
and a product of H̃ is H's plus, per column:

- **H′(x)** on the host, from F^I, ppaa and papa, read back once per macro
  iteration. With C ← C(1 + κ) and T = ncore + t,
  (tu|vw)′ = X_tuvw + X_utvw + X_vwtu + X_wvtu, X_tuvw = Σ_p κ_pT (pU|vw),
  and h_eff′_tu = Σ_p κ_pT F^I_pU + κ_pU F^I_Tp + Σ_p,i κ_pi (4 (pi|tu) −
  (pt|iu) − (pu|it)), the last F^I's change through the core density, i
  core. **No J or K**: every integral it takes has two active indices, so it
  is in ppaa or papa. E_core′ is left out: it multiplies c_r, which the
  projector removes.
- **b_r = H′ c_r**: a second `ci::Sigma`, its integrals set to (0, h_eff′,
  (tu|vw)′), applied to the roots, one sigma each.
- **δ_r** by conjugate gradients on H₀ − E_r off every root (the projector
  1 − Σ_s c_s c_sᵀ), to 1e-8 of the right-hand side, unpreconditioned: one
  sigma per iteration, 3 (H₂CO) to 20 (H₂O AVAS) of them here. The other
  roots' part is exact, −(c_s · b_r) / (E_s − E_r) c_s: it is what makes H̃
  the Hessian of
  Σ_r w_r E_r for unequal weights, and cancels for equal ones. Off the roots
  H₀ − E_r is positive on the singlets; H′ is spin-free, so b_r and the
  Krylov space stay in the roots' spin.
- **H_oc δ** is the gradient in x of h_eff′·γ_T + ½ (tu|vw)′·Γ_T,
  γ_T = Σ_r w_r (⟨c_r|E|δ_r⟩ + ⟨δ_r|E|c_r⟩): half the difference of
  `ci::Rdm::rdm12` of c + δ and of c − δ at the weights, two RDM calls, and
  the adjoint of the H′ map above on the host, packed by
  `pack_orbital_pairs`.

So H̃ costs what H costs in J and K, and per product adds (1 + the CG's
iterations) sigmas per root, two RDMs and O(nmo ncas⁴) host work. At these
CAS sizes that is 0.7 to 4% of the time of a product's two J and two K
(measured with syncs around each half, over the whole solve: H₂CO 0.035 s of
4.7 s, H₂O 0.07 s of 3.4 s, H₂O AVAS 0.19 s of 4.8 s).

Near the solution the step is Newton's on the relaxed energy, so ‖g‖ falls
quadratically: every case here converges in 6 to 8 macro iterations, where
the learned BFGS correction it replaced (D, the last 8 accepted steps' pairs
y yᵀ / (y·s) − (Bs)(Bs)ᵀ / (s·Bs) of the gradient's actual change y) took 9
to 15. The products per macro iteration barely move, so the J and K builds
halve with the macro iterations, and `mcscf::solve`'s wall time falls 32 to
48%. On the RTX 3080 (CUDA), both of its GPU slots held, the two builds
interleaved, three runs each (median; the macro and product counts were the
same in every run unless shown as a range); σ counts the sigma vectors
applied, single precision (ci::solve's first stage) included:

| case | H + D: macro, Hx, J + K, σ, s | H̃: macro, Hx, J + K, σ, s |
|---|---|---|
| H₂O CAS(4,4) | 13, 105, 908, 172, 16.6 | 7, 51, 464, 713, 8.7 |
| N₂ CAS(6,6) | 13, 78, 1227, 270–295, 34.3 | 8, 48–49, 752–756, 1100–1150, 21.3 |
| H₂CO SA2 | 9 or 10, 43 or 48, 457 or 510, 82 or 91, 17.6 or 19.7 | 6, 31, 310, 308, 12.0 |
| H₂CO SA2 [0.7, 0.3] | 9, 43, 457, 82, 17.6 | 6, 26, 290, 275, 11.3 |
| H₂O AVAS CAS(8,6) | 15, 128, 1485, 291, 28.5 | 8, 71–72, 802–806, 1669, 15.6 |

The sigmas rise 3.4 to 5.7 times, nearly all of them the response's CG, and
cost next to nothing at these sizes; the J and K builds are the time.

Without either, the loop is linear: from these starts H₂O CAS(4,4) took 22
macro iterations and H₂O AVAS 24, each cutting ‖g‖ by a steady 0.4–0.5.

## The trust region

An accepted step's actual ΔE against the model's t g·x + ½ t² xᵀH̃x doubles the
radius, up to `trust_radius`, when the two are within 25%. A rise past 1e-11
halves the step and the radius (step 5 of the loop).

**A step along negative curvature is cut short unless the model has earned
the radius.** When xᵀH̃x < 0 the quadratic model has no minimum along x: the
augmented Hessian's root runs out to the radius, and nothing in the model says
where the surface turns up. Such a step keeps its full length only if the last
accepted step's ΔE agreed with its model (the same 25% test; true before the
first step, false after a rejection) **and** the gradient drives it,
|g·x| ≥ |½ xᵀH̃x| / 10. Otherwise it is cut to a quarter of the radius, the
length two rejections would have halved it to, without spending the two macro
iterations.

Why both tests. On H₂O AVAS, at the macro-iteration-2 orbitals, H̃ has an
eigenvalue near −1.3e-2 along a mode nearly orthogonal to g. Whether the
subspace solve reaches it depends on where it stops, and that is chaotic in
the last digits: the CI's ceiling (`kCiLoosest`) moved by a factor of 3 moves
the CI energies by ~1e-9 and is enough to flip it. When it is found, the
full-radius step along it (g·x ≈ −1e-7, the model's −1.7e-3 all curvature)
raised the energy by 2.2e-3, the half step by 4.2e-5, and the case took 10 macro
iterations, not 8. H₂O CAS(4,4) and N₂ take negative-curvature steps too
(in their first three macro iterations), but gradient-driven ones (|g·x| about |½ xᵀH̃x|) after
a model that agreed to a few percent. At full radius those are the fast way
down: cutting every negative-curvature step took H₂O from 7 to 11 macro
iterations and N₂ from 8 to 11–12. The gradient test alone does not suffice on
AVAS either: after the cut step along the mode, the next is gradient-driven
(|g·x| ≈ |½ xᵀH̃x|) but follows a 36% disagreement, and at full radius it was
rejected (11 macro iterations at 6 of 9 ceilings).

H₂O AVAS's macro iterations with the CI's ceiling moved (Hessian products in
brackets; the RTX 3080, both GPU slots held):

| ceiling | 1e-8 | 2e-8 | 3e-8 | 5e-8 | 1e-7 | 3e-7 | 1e-6 | 3e-6 | 1e-5 |
|---|---|---|---|---|---|---|---|---|---|
| full radius | 8 (68) | 8 (68) | 8 (68) | 10 (81) | 8 (67) | 10 (81) | 10 (82) | 10 (81) | 10 (80) |
| cut | 8 (70) | 8 (68) | 8 (68) | 8 (68) | 8 (82) | 8 (81) | 8 (83) | 8 (81) | 8 (80) |

At the shipped 1e-8 nothing is cut in any case here, and every case's
trajectory is the one without the rule. The quarter was measured, not
derived: on AVAS, a cut to an eighth of the radius crept along the mode and
took 10 macro iterations, and to a half overshot and took 12.

A smaller initial radius (0.25) was measured too: AVAS took 6 macro
iterations at most ceilings but 12 at 1e-5, and H₂O CAS(4,4) went from 7 to 8.
A line search on rejection (a cubic through E and g·x at both ends of the
step) was not tried: it still spends the rejected macro iteration, and AVAS's
bar is 8.

## The level shift

The augmented Hessian's model is H̃ + σ, σ = 1e-8 (`kAhLevelShift`, PySCF's
`ah_level_shift`), so no step divides by a curvature below σ. The gradient,
gₘ's updates and the energy test still take H̃ itself: the shift moves the
step, not what the step is judged by.

**H̃ cannot resolve a curvature below ~1e-10.** Its products carry the CI
response's conjugate gradients (to 1e-8 of their right-hand side) and the
CI's own residual: QᵀH̃Q's two triangles differ by 1e-11 to 1e-10 at every
column. A mode flatter than that is noise, and a step along it is noise
divided by noise.

HF cc-pVDZ DF CAS(10,7) (`PipelineNevpt2.PcRefusalKeepsSc`) has one. The
F 1s is active, its occupation 2 − ~1e-5, and so is its correlating partner,
an active natural orbital of occupation ~1e-5 (a mix of the two weakly
occupied orbitals, 0.016 and 0.018 on the diagonal): rotating it into the
virtuals all but leaves the energy alone. Once ‖g‖ fell to ~5e-9 the
subspace solve, its tolerance then ~1e-12, grew to 25–30 columns and found
that mode at a Ritz value of 5e-12, under the lowest genuine one (6.5e-3);
a gradient component of ~1e-14 along it made a step of 1e-2 to 1e-1. That
left the quadratic region: ‖g‖ rose to 4e-6–6e-4, and the energy, flat to
~1e-10 there, was rejected or accepted on noise. The loop then cycled,
reject, a small Newton step back to ‖g‖ ~3e-9, the same long step, until
`max_cycle`. Which run cycled was chaotic in the last digits, so it showed
as a flake: unshifted, 2 of 12 runs failed on the RTX 3080 and 4 of 19 on
the RX 9060 XT (the passing ones in 16 to 47 macro iterations); shifted, 20
of 20 passed on each card, in 12 to 18 (#463). The shift bounds the step along
the mode at the same point to ~1e-4.

Along a curvature λ the shift scales the step by λ / (λ + σ): nothing at
the 6.5e-3 above, and a mode with λ ≪ σ is left to converge by a factor
σ / (λ + σ) per step, where its gradient, λ times the displacement, is
already ~λ or less. `mcscf_solve_gpu.cpp`'s five cases take the macro
iterations and Hessian products of the tables above with the shift (the RX
9060 XT, the slow tier).

## Defaults, and what each settles

| Option | Default | Settles |
|---|---|---|
| `conv_tol` | 1e-10 | ΔE of the weighted energy at convergence |
| `conv_tol_grad` | 1e-6 | ‖g‖ at convergence. The weighted energy is quadratic in the orbitals' error there, but each root's energy and the natural occupations are first order: ~‖g‖ / (H's least eigenvalue). The tests run at 1e-9 for 1e-8 per-root energies |
| `max_cycle` | 50 | macro iterations, rejected ones included, before `MaxIterationsExceeded` |
| `trust_radius` | 0.5 | the initial and largest ‖x‖ of a step |
| `max_micro` | 4 | augmented-Hessian solves per macro iteration on its frozen integrals |
| `ci` | `ci::SolverOptions{}` | the CI's spin penalty and its tightest residual tolerance |
| `tau_j`, `tau_k` | 1e-14 | the screening of every integral-direct J and K build |
| `density_fitting` | `Direct` | which builds take B (below) |

**The cost is the Hessian products.** Each H·κ is two J and two K builds when
there is a core (`response/hessian/`), plus the CI's response to it (sigmas and RDMs,
no J or K), and each column added to Q costs one;
`mcscf::Iteration::hessian_products` counts them. A macro iteration's first
solve pays for every column it needs; a later solve pays only for what its
own residual still lacks, often half or less. The integrals and the
Hessian's own setup are a few J and K per active pair and per core orbital,
once per macro iteration.

**The subspace allocates nothing per iteration.** Q, H̃ Q beside gₘ,
diag(H), the residual, the correction, the step pair (dx, H̃ dx), and the
projected problem are buffers of the solve's, sized once for
`kAhMaxSubspace` columns, and so are the response's CI blocks (b, δ, c ± δ),
its three CG vectors and its RDMs. diag(H) is copied once per macro
iteration (the response's diagonal is not built: diag(H) preconditions
well enough), and the preconditioner is a kernel of this
module's (`device::ah_precondition.cu`), μ passed by value. Per column added there
are a handful of syncs (the small problem's upload, `diagonalize`, its
readback, the residual's and the correction's norms, the new row's readback),
against the two J and two K of its product. The kernel is this module's, not
a shared one in `recipes/davidson`: `ci::solve`'s preconditioner divides with
a 1e-3 floor over a spin-shifted diagonal, this one multiplies by the
reciprocal with a 1e-8 floor, and a common kernel would carry both as options
for two callers.

**The tolerance follows ‖gₘ‖**: a residual of about 1e-2 ‖gₘ‖ (relative to
max(1, max |diag(H)|)), times max(|μ|, the projected problem's largest
|eigenvalue|), so the step is good to about 1% at any ‖g‖. A fixed one would
stop the solve at e₀ once ‖g‖ fell below it, and the step would be zero.

**Why a persistent augmented Hessian, and not Steihaug CG or GLTR** over the
same products: those take the Newton step, or a boundary point of the trust
region when H̃ is indefinite, where the augmented Hessian takes a
level-shifted one. Swapping the step's model would move every case's macro
trajectory and the radius's behaviour with it, which the issue's bar (macro
iterations within one) leaves no room to absorb, and the products it could
save are few: past a macro iteration's first solve, the persistent subspace
already prices a later one at its residual's few columns, and most macro
iterations here take one solve. Steihaug was not measured.

## Density fitting

`mcscf::DensityFitting` picks which of the loop's builds are fitted over the B
`solve` is given (`DensityFitted::b`, naux × nmo², as the density-fitted
`response::Hessian::create` reads it):

| Mode | Fitted | Converges to |
|---|---|---|
| `Direct` (default) | nothing; B, if given, is unused | the exact CASSCF |
| `ApproxHessian` | the orbital Hessian H alone (PySCF's `approx_hessian`) | the exact CASSCF |
| `Full` | every build: the integrals, the Fock, the energy, the gradient and H | the DF-CASSCF |

The B input is the shape of the density-fitted `casscf_integrals`,
`response::fock` and `Hessian::create` overloads, a `Buffers::device<double>`
and naux, but by pointer, since a wrapper cannot be default-constructed and
none is the common case. A mode that needs B without one is `InvalidConfig`,
as is a B that is not naux nmo² doubles over naux ≥ 1, both before the loop
runs. The `Hessian` keeps B's storage by pointer, and B is the caller's, so it
outlives every macro iteration's operator.

**In `ApproxHessian` only the step's model is fitted.** The integrals, the CI,
the energy and the gradient stay integral-direct, so the loop's stationary
point, and its convergence test, are the exact CASSCF's: the fit moves the
path to it, and nothing else. Each product of H then takes its two J and two K
from B, and so does the diagonal in its create (`../../response/hessian/`);
the CI's response R, which takes no J or K, the integrals' J and K and the
gradient's are what they were. The fitted H is not the exact one's Hessian,
so the Newton step is not exact either; how far that costs macro iterations
depends on the fitting set, which is why the issue's bar was within one
macro iteration of `Direct`.

**The fit costs no macro iterations here, and saves 29 to 58% of the wall
time.** Over aug-cc-pVDZ-acCD-6-6, the set `../../response/hessian/` compares
the fitted H to the exact one over, every case takes `Direct`'s macro
iterations, and its Hessian products to within one. On the RTX 3080 (CUDA,
preset `default`), both of its GPU slots held and no other compute process on
the card before or after, three runs each (median; counts the same in every
run unless shown as a range), at `conv_tol_grad` 1e-9, the solve alone (B is
built once per case, outside it):

| case | `Direct`: macro, Hx, s | `ApproxHessian`: macro, Hx, s | saved |
|---|---|---|---|
| H₂O CAS(4,4) | 7, 50, 8.53 | 7, 50, 4.42 | 48% |
| N₂ CAS(6,6) | 8, 45–46, 21.14 | 8, 45–46, 14.92 | 29% |
| H₂CO SA2 | 6, 31, 11.64 | 6, 31, 4.89 | 58% |
| H₂CO SA2 [0.7, 0.3] | 6, 26–27, 10.95 | 6, 26–27, 4.85 | 56% |
| H₂O AVAS CAS(8,6) | 8, 68–70, 15.44 | 8, 68–70, 10.60 | 31% |

What is left of `ApproxHessian`'s time is the integrals' and the gradient's
integral-direct J and K, once per macro iteration, which the products no longer
dominate. Each case ran `Direct` first in the process, so any warm-up falls on
`Direct`'s side; it is small next to these gaps (H₂O's `Direct` is the 8.7 s of
the table above, measured alone). `Direct` stays the default all the same, as the
issue sets it: `ApproxHessian` needs a B the caller builds and holds
(naux nmo² doubles) and a fitting set chosen, which a default cannot assume.
Choosing it from the config is #386's.

**In `Full` every build is fitted, and the loop converges to the DF-CASSCF.**
Each macro iteration takes the density-fitted `casscf_integrals`
(`../../integrals/active_space/`), `response::fock` (`../../response/fock/`)
and `Hessian::create` (`../../response/hessian/`), all over the one B, so no J
or K in the loop is integral-direct: `u`, the J and K classes and the taus go
unread. The stationary point, and the convergence test with it, are now the
fitted energy's, which is PySCF's `mcscf.CASSCF(mf, ...).density_fit()`: the
tests compare it to `test/gtest/data/casscf/casscf_df.json`, fitted over
cc-pVDZ-JKFIT. The CI's response R takes ppaa and papa as the integrals left
them, fitted too.

**`Full` takes `Direct`'s macro iterations in every case, and under 2% of its
wall time.** Measured as the table above was (RTX 3080, preset `default`,
both GPU slots held and no other compute process on the card before or after,
median of three runs at `conv_tol_grad` 1e-9, the solve alone with B built
outside it), in one session for all three modes; `Full` over cc-pVDZ-JKFIT,
the other two as above:

| case | `Direct`: macro, Hx, s | `ApproxHessian`: macro, Hx, s | `Full`: macro, Hx, s |
|---|---|---|---|
| H₂O CAS(4,4) | 7, 50, 8.80 | 7, 50, 4.55 | 7, 50, 0.15 |
| N₂ CAS(6,6) | 8, 45–46, 21.50 | 8, 45–46, 15.37 | 8, 45–46, 0.22 |
| H₂CO SA2 | 6, 31, 12.04 | 6, 31, 4.98 | 6, 31, 0.10 |
| H₂CO SA2 [0.7, 0.3] | 6, 26–27, 11.29 | 6, 25–26, 4.98 | 6, 25–26, 0.09 |
| H₂O AVAS CAS(8,6) | 8, 70, 15.99 | 8, 70, 10.76 | 8, 68, 0.28 |

What `ApproxHessian` leaves, the integrals' and the gradient's
integral-direct J and K, was nearly all of the time: over B at cc-pVDZ each is
a few gemms, and the loop's cost falls to the CI and its response. It is a
different answer, though, not a faster route to the same one: the fitted
energy sits 2e-5 (water) to 2.6e-4 Eh (N₂) from the exact CASSCF over this
set (`test/gtest/data/casscf/README.md`). So `Direct` stays the default, as
the issue sets it; choosing a mode from the config is #386's.

## Not done

- **A one-step joint (x, c) augmented Hessian** (Werner–Knowles,
  `mc1step_symm`): the CI is re-solved each macro iteration, so only the
  orbital step needs the coupling, and the Schur complement above gives it
  over the orbital pairs alone. A joint Davidson would pay a whole product
  (two J and two K) for every iteration its CI block needs; here the CI's
  share is a CG over sigmas inside each product. Nor is the CI stepped
  within the micro iterations (PySCF's `mc1step` `update_casdm`): with one
  micro solve per macro iteration in every case here, there is nothing for
  it to feed.
- **A preconditioned response CG** for large CAS spaces: unpreconditioned,
  its iterations grow with H₀'s spread over the gap above the roots.
- **Frozen orbitals**: every core orbital rotates.
- **Root following**: the roots are the lowest `nroots` each iteration.

## Tests

`test/gtest/mcscf_solve_gpu.cpp`, against `test/gtest/data/casscf/casscf.json`'s
converged, polished results, from each case's recorded start, set up as
`response_fock_gpu.cpp` does: H₂O cc-pVDZ CAS(4,4) [2, 3, 5, 6], N₂ cc-pVDZ
CAS(6,6), H₂CO cc-pVDZ CAS(4,3) at [0.5, 0.5] and [0.7, 0.3], and H₂O cc-pVDZ
by AVAS, at `conv_tol_grad` 1e-9:

- each root's energy and the weighted one to 1e-8;
- the state-averaged natural occupations to 1e-6;
- ‖g‖ below `conv_tol_grad`, recomputed from scratch with a cold CI;
- Cᵀ S C = 1 to 1e-10;
- no accepted iteration's ΔE above 1e-11;
- each case's macro iterations printed, with the CI's iterations per stage,
  and at most 20;
- `MaxIterationsExceeded`, and the refusals (a density-fitted mode without B,
  and a B of the wrong size among them).

Each case runs twice, `Direct` and then `ApproxHessian` over
aug-cc-pVDZ-acCD-6-6 (built once per case, outside the timed solve): the
same references to the same tolerances, and at most one macro iteration more
than `Direct`. Each run's macro iterations, Hessian products and wall time
are printed.

Each case also runs in `Full` over cc-pVDZ-JKFIT, against
`test/gtest/data/casscf/casscf_df.json`'s DF-CASSCF from the same start, to
the same bars; ‖g‖ is recomputed from scratch over the same B (the fitted
integrals and Fock), and the B's naux must be the reference's. The set is
the one that file was fitted over (its `aux_basis`; the sha256 recorded there
is the tree's `data/aux_basis/cc-pVDZ-JKFIT.json`'s), not the acCD set above:
at 1e-8 two fitting sets would differ by far more than the bar.

## Files

| File | Role |
|------|------|
| `solve.cppm` | Module interface: `mcscf::solve`, `mcscf::Options`, `mcscf::Iteration`, `mcscf::Solution` |
| `solve.cpp` | The loop, the augmented Hessian's persistent subspace over H̃, the CI's response R |
| `device::ah_precondition.cu` | The preconditioner's kernel, r / (μ − diag(H)) for one column |
| `ah_precondition_bridge.h` | Its launcher, shared by the `.cu` and `solve.cpp`'s global module fragment |
| `CMakeLists.txt` | The device library `free_breakfast.mcscf.solve.device` and the module `free_breakfast.mcscf.solve` |
