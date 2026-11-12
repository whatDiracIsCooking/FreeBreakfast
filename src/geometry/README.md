# Geometry

A Cartesian quasi-Newton minimizer: BFGS under a trust radius, with the rigid-body translations and rotations projected out, over a caller-supplied energy and gradient; its counterpart for the minimum-energy crossing of two states (MECI); the intrinsic reaction coordinate (IRC) from a transition state; and the harmonic vibrational analysis of a Cartesian Hessian.

## Module

`free_breakfast.geometry`, namespace `free_breakfast::geometry`.

## Design

**`optimize` is host-side, and knows nothing of chemistry.** The state is a coordinate vector of 3N values and an n × n Hessian. Neither needs the device, so `optimize` touches none and its tests run on a device-less box (the module itself links `recipes` for `harmonic_analysis`). It does not know about `Molecule`, the SCF or the pipeline: `optimize(x0, energy_and_gradient, options)` takes the starting coordinates as a `std::span<const double>` and a `std::function<Result<double>(x, gradient)>` that returns the energy at `x` and writes its gradient. An error from the callback (an SCF that did not converge) ends the optimization with that error, unchanged. An optional fourth argument, `on_iteration`, is called with each `Iteration` as it is recorded (`optimize_meci` takes one too, of its own `Iteration`), so a caller keeps the walk's log when an error loses the `Solution`: the pipeline's optimize step writes its `optimize.json` and `optimize.xyz` from it as the walk goes. The pipeline's optimize step (#497) supplies the callback: it reruns the chain from molecule to gradient at each `x`. Units are the callback's own; the defaults assume bohr and hartree.

**The method is BFGS, not RFO.** Each step:

1. Project the gradient g (below).
2. The quasi-Newton step p = −B⁻¹g, by a Cholesky solve of the n × n Hessian B, then projected again.
3. If |p| is longer than the trust radius, scale p back to it.
4. Evaluate the energy at x + p. A **rise** over the energy at x rejects the step: x, g and B stay, the trust radius becomes |p|/4, and the next step is retried from x. Rejected evaluations count against `max_steps`.
5. Otherwise take the step. Update the trust radius from the ratio ρ of the actual energy change to the quadratic model's g·p + ½ p·B·p (Nocedal and Wright's Algorithm 4.1): ρ < ¼ sets it to |p|/4; ρ > ¾ with p at the radius doubles it, up to `max_trust_radius`; otherwise it stays.
6. The BFGS update of B over s = p and y the change in projected gradient, B ← B + yyᵀ/(y·s) − (Bs)(Bs)ᵀ/(s·Bs), applied only when the curvature y·s is positive past rounding, which keeps B positive definite.

B starts as `initial_hessian` times the identity. A model Hessian can replace that later without changing the loop. RFO was the alternative: it handles an indefinite B, which BFGS's skipped updates avoid, but it needs an eigensolve of the (n+1)-square augmented Hessian per step, and BFGS needs only a Cholesky solve.

**The projection.** With `project_rigid_body` (the default), the coordinates are points in space, x, y, z each. At every geometry the three unit translations and the three infinitesimal rotations about the centroid, ω × (rᵢ − c), are orthonormalized by Gram-Schmidt, and both g and p lose their components along them. A motion with nothing left after the ones before it is dropped: the rotation about a linear arrangement's axis, every rotation of a single point. The centroid is unweighted, because the module has no masses. An energy that is invariant to rigid motions has a gradient with no rigid components anyway, so the projection changes no converged geometry; what it buys is that rounding and the BFGS update never move the centroid or rotate the points. Every step sums to zero over the points, and has zero first-order rotation (Σᵢ (rᵢ − c) × sᵢ = 0) about the geometry it is taken from. Turn it off for a function whose coordinates are not points in space, as the quadratic-well and Rosenbrock tests do; it needs a size divisible by 3 when on.

**Convergence** is Gaussian's four criteria, each settable, with Gaussian's default thresholds:

| Option | Default | Of |
|---|---|---|
| `max_force` | 4.5e-4 | the largest \|component\| of the projected gradient |
| `rms_force` | 3.0e-4 | its root mean square over the n coordinates |
| `max_displacement` | 1.8e-3 | the largest \|component\| of the step that reached the geometry |
| `rms_displacement` | 1.2e-3 | that step's root mean square |

All four must hold at an accepted geometry. The displacement is of the step just taken, not, as in Gaussian, of the step predicted next, so each norm in an `Iteration` is a measured one. The first evaluation has no step, so it converges on the forces alone: a start at a minimum costs one evaluation. Gaussian's extra rule, convergence on forces a hundred times under the threshold whatever the displacement, is not implemented.

The other defaults: an initial trust radius of 0.3 (Gaussian's MaxStep), a maximum of 1.0, `initial_hessian` 0.5, and `max_steps` 100 evaluations after the first. Running out of `max_steps` is `MaxIterationsExceeded`; a non-finite energy or gradient is `Numerical`; a bad input or option is `InvalidConfig`.

**The result.** `Solution` holds the final coordinates `x`, its `energy` and `iterations`: one `Iteration` per evaluation, the first at `x0`, with the energy, the two force norms at that geometry, the two displacement norms of the step that reached it (0 for the first), the trust radius the next step is taken under, whether it was `accepted`, and under constraints each one's distance and multiplier (below). A rejected entry's energy is the higher one that rejected it.

## Distance constraints

`Options::constraints`, a list of `DistanceConstraint {atoms, distance}`, holds interatomic distances fixed while `optimize` minimizes over the rest (#1260; the pipeline's `[optimize] constraints`, `../pipeline/optimize/README.md`). The method is a projection over the BFGS above, with a SHAKE/RATTLE-like correction:

1. **The start** is moved onto the constraints before the first evaluation, by the correction in 4 along the constraint gradients at x0.
2. **The projection** gains the constraint gradients a_k = dd_k/dx (the pair's unit vector at one atom, its negative at the other), Gram-Schmidted after the rigid motions. They are orthogonal to the rigid motions already: a distance does not change under one. Constraint gradients with nothing left after the ones before them (1e-8 of their length) are linearly dependent, three atoms held collinear for one, and are refused, `Numerical`.
3. **The step** is −(PBP)⁺ g, P now the projector off the rigid motions and the a_k, by one Cholesky solve of PBP + (I − P), as `optimize_meci`'s seam step is. Projecting −B⁻¹g afterwards, as the unconstrained walk does, lets B's coupling between the free and the constrained directions turn the free step: an overshoot by the stiff curvature B has not learned, which near convergence the trust radius never recovers from.
4. **The correction.** x + p moves back onto the constraints along the a_k of the geometry the step left, as SHAKE moves along the last step's bonds: x + Σ_j μ_j a_j(x), μ by Newton's method on the k distances, whose Jacobian is a_i(x′)·a_j(x), a k × k solve by Gaussian elimination. It stops at 1e-12 of each distance (relative, absolute under 1), or fails, `Numerical`, after 50 iterations or at a singular Jacobian. Every evaluated geometry, rejected ones too, meets the constraints to that.
5. **The update.** B gets the BFGS update over P s, s the corrected step, and y the change in the Lagrangian's gradient at the new multipliers, g⁺_P − (g − Σ_k λ⁺_k a_k(x)), projected by the old P: SQP's choice, which carries the constraints' curvature λ_k ∂²d_k/∂x² into B. (The change in P-projected gradients, projected by the old P, is the same vector: `optimize_meci`'s seam update is this one.)
6. **The rounding floor.** An energy change under 1000 ulps of the energies, `optimize_meci`'s floor, neither rejects a step nor moves the trust radius. Near a tight convergence a step's energy change f²/2k is under the energy's rounding, and without the floor that noise rejects steps and halves the radius until the walk stalls short of its thresholds (a Lennard-Jones cluster held at one distance stalled at forces of 4e-8 against 1e-10). The unconstrained walk is left as it was.

**Convergence** is the four criteria on the projected gradient and on the corrected step. **The multipliers** λ_k, each `Iteration`'s `multipliers`, are the least-squares fit g = Σ_k λ_k a_k + (the projected gradient), one k × k solve of (A Aᵀ) λ = A g. At a converged geometry λ_k is dE*/dd_k, the slope of the constrained minimum's energy along that distance with the rest relaxed; `distances` holds each constraint's distance at that geometry.

**Refusals**, `InvalidConfig`: a point out of range or named twice in one constraint, a pair named twice (either order), a distance not finite and above 0, constraints over a size not divisible by 3, and any constraints at all in `optimize_meci`.

## Minimum-energy crossings

`optimize_meci(x0, energies_and_gradients, options)` finds the lowest point of the seam where two states are degenerate, by gradient projection (Bearpark, Robb and Schlegel, CPL 223, 269 (1994); Sicilia, Bearpark, Blancafort and Robb, JCTC 4, 257 (2008)). It is host-side and chemistry-free like `optimize`, and takes the same `Options`. The callback is `Result<std::array<double, 2>>(x, g_i, g_j, h_ij)`: it returns the two energies and writes each one's gradient and their coupling vector h_IJ, the shape `mcscf::df_intersection`'s `Intersection {g_i, g_j, h_ij}` comes in (#774 is its caller). The energies may come in either order; the lower one, with its gradient, is taken as I at each geometry. Only h_IJ's direction is read, so its scale and sign (the CI phase) are free.

Each step, at x:

1. **The branching plane.** x = g_J − g_I and y = h_IJ lose their rigid-body components (with `project_rigid_body`), and are orthonormalized, x first. A y with nothing left after x (under 1e-8 of the larger of \|x\| and \|y\|) is dropped: an uncoupled crossing, h ≡ 0, has a seam of dimension 3N − 7, and the plane is x alone. P is the projector off the rigid motions and the plane.
2. **The gap.** A Newton step on E_J − E_I along x̂: −(E_J − E_I)/\|x\| x̂. Near a conical intersection the gap is linear in the distance from the seam, so this lands on it to first order.
3. **The seam.** A BFGS step on the mean energy Ē = (E_I + E_J)/2 in the intersection space: −(PBP)⁺ P ḡ, ḡ = (g_I + g_J)/2, by one Cholesky solve of PBP + (I − P) (B on the intersection space, the identity on the rest). B starts as `initial_hessian` times the identity and gets the BFGS update over s = P p and the change in P ḡ, projected by the old P, when y·s is positive past rounding. The new geometry's P in that change is what carries the plane's turn, so B learns the seam's Lagrangian curvature and not just Ē's.
4. **The trust radius** caps the sum: scaled back to it when longer.

**A carried plane** (#902). A callback with no branching plane at x writes the mean gradient into both g_I and g_J and zeros into h_IJ (`nevpt2::qd_pc_intersection` inside a degenerate CASSCF pair, `../nevpt2/intersection/README.md`). The evaluation then takes x̂, ŷ and ‖x‖ from the last one, x̂ and ŷ re-orthonormalized against this geometry's rigid motions, and only P ḡ is new: P is the same before and after such a step, so B learns Ē's curvature and not a plane's turn. The gap step is kept on the carried x̂. The first evaluation has no plane to carry, and is refused (`Numerical`). The iteration reports `plane_carried`.

**No step is rejected.** The composite step is not the descent of any single function, so an energy rise is no evidence of a bad step. The trust radius instead follows the ratio ρ of the actual change in Ē + (E_J − E_I) to the model's (ḡ·p + ½ p'·B·p' for the mean, with p' = P p, and x·p for the gap): ρ outside [¼, 7/4] sets it to \|p\|/4; ρ within [¾, 5/4] with p at the radius doubles it, up to `max_trust_radius`; otherwise it stays. The test is two-sided because the model's change may be a rise (a gap step up a tilted cone), and it is skipped when that change is under 1000 ulps of the energies. When either end of the step carried its plane, the gap terms leave both changes: around a cone x̂ turns with the direction to the seam, so a carried x̂ does not model the gap, and with it in the ratio a toy cone's trust radius fell to 1e-17. Such a walk's gap is already under `max_gap` there (#902's 4.8e-11).

**Convergence** needs five criteria at once: the gap at or below `max_gap` (1e-5 Eh by default; `optimize` ignores it), and `optimize`'s four, on the projected force −P ḡ and on the step that reached the geometry. The projected force is the one #770's reference reports. The first evaluation converges on the gap and the forces alone.

**The result.** `MECISolution` holds the final `x`, its `lower` and `upper` energies, and `iterations`: one `MECISolution::Iteration` per evaluation, the first at `x0`, with both energies and the gap, the two force norms, the two displacement norms of the step that reached it, the trust radius the next step is taken under, `branching_dimension`, 2, or 1 when y was dropped, and `plane_carried`. The refusals are `optimize`'s: `InvalidConfig` for bad input or options (and `max_gap` not above 0), `Numerical` for a non-finite energy, gradient or coupling, a gradient difference that vanishes after the projection (other than a mean gradient alone), or a first evaluation that is a mean gradient alone, `MaxIterationsExceeded` once `max_steps` runs out, and the callback's own error unchanged.

## The intrinsic reaction coordinate

`irc(start, hessian, direction, energy_and_gradient, options, on_point)` walks the steepest-descent path in mass-weighted Cartesians, q = √m x, from the transition state `start` down one side (#1269; the pipeline's `[irc]`, `../pipeline/optimize/README.md`). It is host-side and chemistry-free like `optimize`, and takes `optimize`'s callback; it reads `start`'s elements only for their masses, `harmonic_analysis`'s (the `:masses` partition), so its mass-weighted coordinates are the ones the Hessian's normal modes are in. `hessian` is the Cartesian Hessian at the TS, `harmonic_analysis`'s layout, and `direction` the Cartesian displacement to leave along (the imaginary mode as `harmonic_analysis` gives it, or its negative); only its direction is read.

**The method is Gonzalez and Schlegel's second-order one** (GS2; J. Chem. Phys. 90, 2154 (1989)), chosen over the local quadratic approximation (LQA) because its points lie on the path to the accuracy of the hypersphere optimization, not of a model integrated along it: an error in one point is not carried forward as a drift, and GS2 is what Gaussian and ORCA run by default. Each step, from point k at q_k with the mass-weighted gradient g_k:

1. **The pivot**, half a step down: q* = q_k + (s/2) u, u = −g_k/\|g_k\|, or at the TS (whose gradient is zero) the unit mass-weighted `direction`.
2. **The hypersphere.** Point k + 1 is the minimum of the energy on the sphere \|q − q*\| = s/2, where the gradient is parallel to q − q*. The sphere passes through q_k, so the arc from q_k is s long.
3. **The constrained Newton step.** From a trial p = q − q* with gradient g, the quadratic model with Hessian H is minimized on the sphere: in H's eigenbasis over the internal coordinates (the complement of the rigid motions at the trial, diagonalized by the `:quasi_newton` partition's host Jacobi), p'_i = (b_i p_i − g_i)/(b_i − λ), λ below every b_i and fixed by \|p'\| = s/2, the trust-region secular equation, solved by bisection to the last bit. When even λ = b_0 leaves \|p'\| short (the hard case, nothing along the lowest eigenvector), the shortfall is made up along it, on the side the trial leaned. The first trial is the model's from q_k itself, so a quadratic surface costs one evaluation per point.
4. **The Hessian update.** H starts as `hessian`, mass-weighted, and gets Bofill's update over every two evaluations in a row: SR1 and Powell's symmetric Broyden mixed by φ = (ξ·s)²/(\|ξ\|²\|s\|²), ξ = y − Hs. BFGS would force H positive definite, and near the TS it is not.
5. **The first sphere.** It passes through the TS, whose gradient is zero, so the TS is a stationary point of E on it, and a minimum: E's curvature on the sphere there is the Hessian's over its tangent space, the positive modes. (Point k on a later sphere is stationary too, but its curvature there is lowered by the Lagrange multiplier \|g_k\|/(s/2).) On a stiff, sharply curved valley the model's step from the mode-displaced point can jump the sphere's diameter back to the TS, and the search converges there (#1285: (x² − 1)² + 200(y − 2x²)² at s = 0.05 stopped `EnergyRose` with the TS alone on both sides). A first sphere that converges behind its pivot is therefore searched again from the start's Hessian, and once a trial is ahead of the pivot, a model step that would take the next one behind it is replaced by a step down the tangential gradient along the sphere's great circle: the Newton step on E's curvature along it, at most a quarter turn, halved until it is ahead. Convergence is still judged on the model's own correction. A first sphere that converges ahead, as every chemistry case so far does, is searched once and is unchanged. Each search has `max_sphere_iterations` evaluations, and the point's `sphere_iterations` counts both.
6. **Convergence** of the sphere: the gradient's part tangent to it, \|g − (g·p̂)p̂\|, at or below `sphere_gradient` and the next correction at or below `sphere_displacement`. A point whose tangential gradient is t lies within t/κ of the sphere's minimum, κ the curvature of E on the sphere there (the least eigenvalue of H − λI over its tangent space); `[irc]`'s reference test derives its bars from that.

The rigid motions (with `project_rigid_body`, the default) are projected off the gradient, the steps, the Hessian's updates and the way out, in mass-weighted coordinates, so the centre of mass stays where it is and no step rotates the molecule.

**The stops.** A point whose Cartesian force (projected, largest component) is at or below `max_force` is a minimum (`Minimum`). Before each step, a mass-weighted \|g\| that halved over the last step, 2\|g_k\| ≤ \|g_(k−1)\|, stops the walk (`MinimumWithinStep`): along the path dE/dσ = −\|g\|, so \|g\| carried on linearly from the last two points reaches 0 within one more step, the path's minimum is within a step of the last point, and the next sphere would hold it. That sphere's own minimum has no continuation of the path (it flips between the sphere's two sides, the iteration never settling), which is why the walk stops rather than steps. The test reads the walk, not the model (#1278). Its first form, the model's minimum along the steepest descent \|g\|/(u·H·u) within s/2, is the minimum along a straight line, and a point off the path by its sphere's tolerance has a part of its gradient across the valley; in a soft valley with a stiff floor that part puts the line's minimum within half a step while the path still runs on along the valley (the Cope PC-NEVPT2 IRC stopped 39 steps short, its \|g\| about 4e-4 against the default `sphere_gradient` of 1e-4). \|g\| sees that part only at second order. It holds while \|g\| is well above `sphere_gradient`, below which a sphere's point is not resolved anyway. A point whose energy rose over the last's, or that lies behind its pivot, is past the minimum and is not kept (`EnergyRose`). Then `max_points` (`MaxPoints`), and a sphere that does not converge in `max_sphere_iterations` evaluations (`SphereNotConverged`, the point not kept). None is an error: the path so far is returned with its `stop`, and `irc_stop_name` spells it for a history.

| Option | Default | Of |
|---|---|---|
| `step_size` | 0.1 | s, amu^½ bohr (Gaussian's default) |
| `max_points` | 50 | the points after the TS |
| `max_force` | 4.5e-4 | the force at which a point is a minimum, hartree/bohr |
| `sphere_gradient` | 1e-4 | the tangential mass-weighted gradient's 2-norm, hartree/(amu^½ bohr) |
| `sphere_displacement` | 1e-3 | the next correction's 2-norm, amu^½ bohr |
| `max_sphere_iterations` | 20 | evaluations per point |

**The result.** `IrcPath` holds the points, the TS first, and the `stop`. Each `IrcPoint` has its arc length k s, energy, coordinates, the projected Cartesian gradient's 2-norm and largest component, the evaluations its sphere took and its converged tangential gradient. `on_point` is called with each as it is kept, so a caller keeps the walk on an error. Bad input (no atoms, an element outside 1 to 118, a Hessian of the wrong size, a non-finite value, a direction that is a rigid motion, an option out of range) is `InvalidConfig`; a non-finite energy or gradient is `Numerical`; the callback's error ends the walk unchanged.

## Harmonic analysis

`harmonic_analysis(res, atoms, hessian)` turns a 3N × 3N Cartesian Hessian (hartree/bohr², element (3A + k, 3B + l) at `hessian[3N(3A + k) + 3B + l]`) into `Vibrations`: the harmonic `frequencies` in cm⁻¹, ascending, an imaginary one reported as minus its magnitude, and the Cartesian normal `modes`, one row of 3N per frequency. It does not care how the Hessian was computed. It is `pyscf.hessian.thermo.harmonic_analysis` with `imaginary_freq=False`, step for step:

1. **Mass-weighting** by the isotope-averaged masses, PySCF's `elements.MASSES` (the standard atomic weights), copied into the `:masses` partition for Z = 1 to 118 (`irc` weights with them too): H_mw = M^-½ H M^-½.
2. **The rigid motions**: `optimize`'s own rigid-body basis (the `:rigid_body` partition), each atom's components scaled by √m, so it spans the translations and rotations in mass-weighted coordinates. Gram-Schmidt drops the rotation about a linear molecule's axis by itself, so a linear molecule loses five, a single atom three and everything else six; there is no separate rotor test.
3. **Projection** by a change of basis, not a P H P: the projector P = I − QQᵀ is diagonalized, and its eigenvectors of eigenvalue 1 (the last 3N − k, ascending) are an orthonormal basis B of the internal motions, PySCF's `bvec`. Bᵀ H_mw B is diagonalized in that basis, so the rigid motions are not reported, rather than reported as near-zero frequencies a caller would have to filter.
4. **Frequencies and modes**: an eigenvalue λ (hartree / bohr² / amu) is the wavenumber sign(λ) √(|λ| / 1822.888486209) × 219474.6313632 (CODATA 2018: the dalton in electron masses and the hartree in cm⁻¹; PySCF's own CODATA 2014 set agrees to 1e-9 relative). The mode is B v divided by √m per atom, PySCF's `norm_mode`: unit length mass-weighted, not in Cartesian. Its sign is arbitrary, and so is the basis of a degenerate set.

Both eigensolves are `recipes::diagonalize` and the products `gemmwich` and `gemm`, on `res`'s stream; the 3N-square matrices are built on the host, where the problem is small. The two physical constants live in `frequencies.cpp` rather than `core/constants`, which no other unit reads them from yet (`docs/style.md`, "Numeric constants"). Bad input (no atoms, a Hessian of the wrong size, a non-finite value, an element outside 1 to 118) is `InvalidConfig`.

## Usage

```cpp
import free_breakfast.geometry;

namespace geo = free_breakfast::geometry;

std::vector<double> x0 = /* 3N coordinates, bohr */;
auto result = geo::optimize(
    x0, [&](std::span<const double> x, std::span<double> gradient)
            -> free_breakfast::Result<double> { /* energy at x; gradient into gradient */ });
if (result) {
  // result->x, result->energy, result->iterations
}

auto meci = geo::optimize_meci(
    x0, [&](std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
            std::span<double> h_ij) -> free_breakfast::Result<std::array<double, 2>> {
      /* the two energies at x; their gradients and coupling into the spans */
    });
if (meci) {
  // meci->x, meci->lower, meci->upper, meci->iterations
}

free_breakfast::DeviceResources res{0};
auto vib = geo::harmonic_analysis(res, molecule.atoms(), hessian); // 3N x 3N, row-major
if (vib) {
  // vib->frequencies (cm^-1), vib->modes (frequencies.size() x 3N)
}
```

`harmonic_analysis_footprint(atoms, syevd_lwork)` is what it takes (#1044's model): the projector, H_mw, the m × m projected Hessian and the two spectra as scratch (m the internal modes), and the device workspace, the largest of syevd's work at 3N and at m and the two n × m products carved from it. `syevd_lwork` is syevd's work by order through at least 3N, as `FootprintDimensions::syevd_lwork` holds it; the rigid motions are counted unweighted, since their number does not depend on the masses. The semi-numerical `[hessian]` step composes it (`pipeline/hessian`).

## Tests

`test/gtest/geometry_host.cpp`: a quadratic well reached exactly in one step when the starting Hessian is the well's, and to 1e-10 by the BFGS updates when it is anisotropic; the extended Rosenbrock function; a four-point Lennard-Jones cluster converging to the regular tetrahedron at E = −6, with its centroid fixed to 1e-12 and every step free of rotation, and again at the default thresholds; an uphill step rejected with the trust radius cut to a quarter and the retry taken from the accepted geometry; a start at a minimum converging in one evaluation; and the refusals, the step cap and a failing callback. Distance constraints (`GeometryOptimizeConstrained.*`, #1260): that cluster with one distance held off its start and off the free minimum, held to 1e-12 at every evaluation with the centroid fixed, the raw gradient less λ times the constraint gradient under the force threshold at the optimum, and λ the constrained minimum's slope dE*/dd by a central difference of two more walks at ±1e-5 (bar 2e-9: truncation h²E*'''/6 = 7e-10, rounding 1e-10); two coupled distances sharing a point; and the refusals, among them three collinear points whose constraint gradients are dependent, `Numerical`.

The same file holds `optimize_meci`'s cases, on analytic two-state models: a 2 × 2 diabatic Hamiltonian over the six distances of four points, so every rigid motion is an exact zero mode, with linear and quadratic couplings and a known crossing (the reference arrangement, up to a rigid motion). The thresholds are explicit (gap 1e-12 Eh, max and rms force 1e-10 Eh/bohr, max and rms displacement 1e-9 bohr), and each model states its least seam curvature (≥ 0.37 Eh/bohr²) and its plane's slopes (≥ 0.138 Eh/bohr), so the converged point is within force/curvature + gap/slope ≈ 3e-10 bohr of the crossing; the bar is 1e-8 bohr after alignment. A peaked cone and one tilted in its plane each reach it from a start off the seam and from one on the seam away from its minimum; an uncoupled crossing (h ≡ 0, its seam 3N − 7) reaches it with y dropped at every evaluation; a starting Hessian 20 times too soft shrinks the trust radius after the first step, and still reaches it. Every run holds the centroid to 1e-12 and every step free of rotation. Plus energies handed upper-first, the refusals, the step cap, a failing callback, a non-finite coupling and a vanishing gradient difference. And a carried plane (#902): on the peaked cone, a callback giving the mean gradient alone under a gap of 1e-9 Eh (under `max_gap` 1e-8, as #902's walk is) reaches the crossing to 1e-7 bohr (force/curvature + max_gap/slope = 7.3e-8; measured 7.6e-10 in 37 evaluations, 11 carried), with `plane_carried` exactly where the gap is under that floor and never at the start; from a start on the seam it is refused.

`irc`'s cases (`GeometryIrc.*`, #1269), on analytic surfaces: a curved valley, (x² − 1)² + 2(y − x²/2)² + z²/2 for one hydrogen, both sides at tight sphere thresholds, every point held to GS2's condition by a recomputation here (on its sphere about the pivot to 1e-12, its tangential gradient under the threshold, the gradient pointing back at the pivot), the energy falling, and each side stopped with \|g\| halved over its last step, so its last point within a step of (±1, ½, 0) (the case the stop is for: without it each side ends with a sphere that holds the minimum near its pivot and does not converge); a soft valley bent on an arc of radius 10 about a stiff floor, 1e-3 u² + 3(r − 10)² + z²/200, at the default sphere thresholds, where the model's line test stopped 10 steps short (point 20 of 30, 1.01 amu^½ bohr out), each side within a step of its minimum (#1278); a stiff, sharply curved valley, (x² − 1)² + k(y − 2x²)² + z²/2 at k = 200 and s = 0.05 (#1285's case), k = 400 and s = 0.05, and k = 200 and s = 0.1, where the first sphere's search returned to the TS and each side stopped with it alone: each side now leaves it, point 1 on its sphere with its tangential gradient converged, and runs to within a step of (±1, 2, 0); a quadratic saddle walked along its axis in one evaluation per point (the model is the surface) to `max_points`; two hydrogens on a skew axis bound by ((r − 2)² − 1)², the centre of mass and the axis held to 1e-12 at every point under the projection and the two sides reaching r = 1 and 3 within a step; a start at a minimum (the sphere's minimum is behind its pivot) and a sphere that cannot converge, each stopping with the start alone; the callback's error and a non-finite energy; and the refusals.

`test/gtest/geometry_frequencies_gpu.cpp`: PySCF RHF/cc-pVDZ Hessians of bent water, CO₂ along z, HCN off every axis and linear water (a saddle point, its degenerate bend imaginary), against `pyscf.hessian.thermo.harmonic_analysis` of the same matrix (`test/gtest/data/frequencies/`, `tools/emit/frequencies_reference.py`): the mode count (3N − 6, or 3N − 5), the frequencies to 1e-3 cm⁻¹, and the modes up to sign as each degenerate set's mass-weighted projector to 1e-7; the negated water Hessian, every frequency imaginary; a spring between two hydrogens against √(k/μ) and its mode along the bond; a single atom with no modes; and the refusals. Last `harmonic_analysis_footprint` over bent water and linear CO₂, held to one call's pool peak by `test/support/pool_footprint.h`'s two bars (sound and tight), and a single atom's, empty.

## Files

| File | Role |
|------|------|
| `geometry.cppm` | Module interface: `Options`, `Iteration`, `Solution`, `optimize`; `MECISolution`, `optimize_meci`; `IrcOptions`, `IrcStop`, `IrcPoint`, `IrcPath`, `irc`, `irc_stop_name`; `Vibrations`, `harmonic_analysis`, `harmonic_analysis_footprint` |
| `geometry.cpp` | `optimize`'s loop |
| `meci.cpp` | `optimize_meci`: the branching plane, the composite step and its trust radius |
| `irc.cpp` | `irc`: GS2's pivot, hypersphere step (the secular equation, the hard case), the first sphere's retry off the TS and Bofill update, and the stops |
| `quasi_newton.cppm` | Internal partition: the projection off an orthonormal basis, the max and rms norms and the Cholesky step the optimizers share, and the host Jacobi eigensolve `irc` diagonalizes its model with |
| `masses.cppm` | Internal partition: the isotope-averaged masses `harmonic_analysis` and `irc` mass-weight with |
| `rigid_body.cppm` | Internal partition: the orthonormal rigid-body basis, optionally mass-weighted, both units project with |
| `frequencies.cpp` | `harmonic_analysis` and `harmonic_analysis_footprint`, and the two unit conversions |
| `CMakeLists.txt` | The module library |
