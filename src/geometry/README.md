# Geometry

A Cartesian quasi-Newton minimizer: BFGS under a trust radius, with the rigid-body translations and rotations projected out, over a caller-supplied energy and gradient; its counterpart for the minimum-energy crossing of two states (MECI); and the harmonic vibrational analysis of a Cartesian Hessian.

## Module

`free_breakfast.geometry`, namespace `free_breakfast::geometry`.

## Design

**`optimize` is host-side, and knows nothing of chemistry.** The state is a coordinate vector of 3N values and an n × n Hessian. Neither needs the device, so `optimize` touches none and its tests run on a device-less box (the module itself links `recipes` for `harmonic_analysis`). It does not know about `Molecule`, the SCF or the pipeline: `optimize(x0, energy_and_gradient, options)` takes the starting coordinates as a `std::span<const double>` and a `std::function<Result<double>(x, gradient)>` that returns the energy at `x` and writes its gradient. An error from the callback (an SCF that did not converge) ends the optimization with that error, unchanged. The pipeline's optimize step (#497) supplies the callback: it reruns the chain from molecule to gradient at each `x`. Units are the callback's own; the defaults assume bohr and hartree.

**The method is BFGS, not RFO.** Each step:

1. Project the gradient g (below).
2. The quasi-Newton step p = −B⁻¹g, by a Cholesky solve of the n × n Hessian B, then projected again.
3. If |p| is longer than the trust radius, scale p back to it.
4. Evaluate the energy at x + p. A **rise** over the energy at x rejects the step: x, g and B stay, the trust radius becomes |p|/4, and the next step is retried from x. Rejected evaluations count against `max_steps`.
5. Otherwise take the step. Update the trust radius from the ratio ρ of the actual energy change to the quadratic model's g·p + ½ p·B·p (Nocedal and Wright's Algorithm 4.1): ρ < ¼ sets it to |p|/4; ρ > ¾ with p at the radius doubles it, up to `max_trust_radius`; otherwise it stays.
6. The BFGS update of B over s = p and y the change in projected gradient, B ← B + yyᵀ/(y·s) − (Bs)(Bs)ᵀ/(s·Bs), applied only when the curvature y·s is positive past rounding, which keeps B positive definite.

B starts as `initial_hessian` times the identity. A model Hessian can replace that later without changing the loop. RFO was the alternative: it handles an indefinite B, which BFGS's skipped updates avoid, but it needs an eigensolve of the (n+1)-square augmented Hessian per step, and the one host eigensolver in the tree, `sad`'s cyclic Jacobi, is private to that module. BFGS needs only a Cholesky solve.

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

**The result.** `Solution` holds the final coordinates `x`, its `energy` and `iterations`: one `Iteration` per evaluation, the first at `x0`, with the energy, the two force norms at that geometry, the two displacement norms of the step that reached it (0 for the first), the trust radius the next step is taken under, and whether it was `accepted`. A rejected entry's energy is the higher one that rejected it.

## Minimum-energy crossings

`optimize_meci(x0, energies_and_gradients, options)` finds the lowest point of the seam where two states are degenerate, by gradient projection (Bearpark, Robb and Schlegel, CPL 223, 269 (1994); Sicilia, Bearpark, Blancafort and Robb, JCTC 4, 257 (2008)). It is host-side and chemistry-free like `optimize`, and takes the same `Options`. The callback is `Result<std::array<double, 2>>(x, g_i, g_j, h_ij)`: it returns the two energies and writes each one's gradient and their coupling vector h_IJ, the shape `mcscf::df_intersection`'s `Intersection {g_i, g_j, h_ij}` comes in (#774 is its caller). The energies may come in either order; the lower one, with its gradient, is taken as I at each geometry. Only h_IJ's direction is read, so its scale and sign (the CI phase) are free.

Each step, at x:

1. **The branching plane.** x = g_J − g_I and y = h_IJ lose their rigid-body components (with `project_rigid_body`), and are orthonormalized, x first. A y with nothing left after x (under 1e-8 of the larger of \|x\| and \|y\|) is dropped: an uncoupled crossing, h ≡ 0, has a seam of dimension 3N − 7, and the plane is x alone. P is the projector off the rigid motions and the plane.
2. **The gap.** A Newton step on E_J − E_I along x̂: −(E_J − E_I)/\|x\| x̂. Near a conical intersection the gap is linear in the distance from the seam, so this lands on it to first order.
3. **The seam.** A BFGS step on the mean energy Ē = (E_I + E_J)/2 in the intersection space: −(PBP)⁺ P ḡ, ḡ = (g_I + g_J)/2, by one Cholesky solve of PBP + (I − P) (B on the intersection space, the identity on the rest). B starts as `initial_hessian` times the identity and gets the BFGS update over s = P p and the change in P ḡ, projected by the old P, when y·s is positive past rounding. The new geometry's P in that change is what carries the plane's turn, so B learns the seam's Lagrangian curvature and not just Ē's.
4. **The trust radius** caps the sum: scaled back to it when longer.

**No step is rejected.** The composite step is not the descent of any single function, so an energy rise is no evidence of a bad step. The trust radius instead follows the ratio ρ of the actual change in Ē + (E_J − E_I) to the model's (ḡ·p + ½ p'·B·p' for the mean, with p' = P p, and x·p for the gap): ρ outside [¼, 7/4] sets it to \|p\|/4; ρ within [¾, 5/4] with p at the radius doubles it, up to `max_trust_radius`; otherwise it stays. The test is two-sided because the model's change may be a rise (a gap step up a tilted cone), and it is skipped when that change is under 1000 ulps of the energies.

**Convergence** needs five criteria at once: the gap at or below `max_gap` (1e-5 Eh by default; `optimize` ignores it), and `optimize`'s four, on the projected force −P ḡ and on the step that reached the geometry. The projected force is the one #770's reference reports. The first evaluation converges on the gap and the forces alone.

**The result.** `MECISolution` holds the final `x`, its `lower` and `upper` energies, and `iterations`: one `MECISolution::Iteration` per evaluation, the first at `x0`, with both energies and the gap, the two force norms, the two displacement norms of the step that reached it, the trust radius the next step is taken under, and `branching_dimension`, 2, or 1 when y was dropped. The refusals are `optimize`'s: `InvalidConfig` for bad input or options (and `max_gap` not above 0), `Numerical` for a non-finite energy, gradient or coupling or a gradient difference that vanishes after the projection, `MaxIterationsExceeded` once `max_steps` runs out, and the callback's own error unchanged.

## Harmonic analysis

`harmonic_analysis(res, atoms, hessian)` turns a 3N × 3N Cartesian Hessian (hartree/bohr², element (3A + k, 3B + l) at `hessian[3N(3A + k) + 3B + l]`) into `Vibrations`: the harmonic `frequencies` in cm⁻¹, ascending, an imaginary one reported as minus its magnitude, and the Cartesian normal `modes`, one row of 3N per frequency. It does not care how the Hessian was computed. It is `pyscf.hessian.thermo.harmonic_analysis` with `imaginary_freq=False`, step for step:

1. **Mass-weighting** by the isotope-averaged masses, PySCF's `elements.MASSES` (the standard atomic weights), copied into `frequencies.cpp` for Z = 1 to 118: H_mw = M^-½ H M^-½.
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

## Tests

`test/gtest/geometry_host.cpp`: a quadratic well reached exactly in one step when the starting Hessian is the well's, and to 1e-10 by the BFGS updates when it is anisotropic; the extended Rosenbrock function; a four-point Lennard-Jones cluster converging to the regular tetrahedron at E = −6, with its centroid fixed to 1e-12 and every step free of rotation, and again at the default thresholds; an uphill step rejected with the trust radius cut to a quarter and the retry taken from the accepted geometry; a start at a minimum converging in one evaluation; and the refusals, the step cap and a failing callback.

The same file holds `optimize_meci`'s cases, on analytic two-state models: a 2 × 2 diabatic Hamiltonian over the six distances of four points, so every rigid motion is an exact zero mode, with linear and quadratic couplings and a known crossing (the reference arrangement, up to a rigid motion). The thresholds are explicit (gap 1e-12 Eh, max and rms force 1e-10 Eh/bohr, max and rms displacement 1e-9 bohr), and each model states its least seam curvature (≥ 0.37 Eh/bohr²) and its plane's slopes (≥ 0.138 Eh/bohr), so the converged point is within force/curvature + gap/slope ≈ 3e-10 bohr of the crossing; the bar is 1e-8 bohr after alignment. A peaked cone and one tilted in its plane each reach it from a start off the seam and from one on the seam away from its minimum; an uncoupled crossing (h ≡ 0, its seam 3N − 7) reaches it with y dropped at every evaluation; a starting Hessian 20 times too soft shrinks the trust radius after the first step, and still reaches it. Every run holds the centroid to 1e-12 and every step free of rotation. Plus energies handed upper-first, the refusals, the step cap, a failing callback, a non-finite coupling and a vanishing gradient difference.

`test/gtest/geometry_frequencies_gpu.cpp`: PySCF RHF/cc-pVDZ Hessians of bent water, CO₂ along z, HCN off every axis and linear water (a saddle point, its degenerate bend imaginary), against `pyscf.hessian.thermo.harmonic_analysis` of the same matrix (`test/gtest/data/frequencies/`, `tools/emit/frequencies_reference.py`): the mode count (3N − 6, or 3N − 5), the frequencies to 1e-3 cm⁻¹, and the modes up to sign as each degenerate set's mass-weighted projector to 1e-7; the negated water Hessian, every frequency imaginary; a spring between two hydrogens against √(k/μ) and its mode along the bond; a single atom with no modes; and the refusals.

## Files

| File | Role |
|------|------|
| `geometry.cppm` | Module interface: `Options`, `Iteration`, `Solution`, `optimize`; `MECISolution`, `optimize_meci`; `Vibrations`, `harmonic_analysis` |
| `geometry.cpp` | `optimize`'s loop |
| `meci.cpp` | `optimize_meci`: the branching plane, the composite step and its trust radius |
| `quasi_newton.cppm` | Internal partition: the projection off an orthonormal basis, the max and rms norms and the Cholesky step both optimizers share |
| `rigid_body.cppm` | Internal partition: the orthonormal rigid-body basis, optionally mass-weighted, both units project with |
| `frequencies.cpp` | `harmonic_analysis`, the mass table and the two unit conversions |
| `CMakeLists.txt` | The module library |
