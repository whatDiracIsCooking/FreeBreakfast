# Geometry

A Cartesian quasi-Newton minimizer: BFGS under a trust radius, with the rigid-body translations and rotations projected out, over a caller-supplied energy and gradient; and the harmonic vibrational analysis of a Cartesian Hessian.

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

free_breakfast::DeviceResources res{0};
auto vib = geo::harmonic_analysis(res, molecule.atoms(), hessian); // 3N x 3N, row-major
if (vib) {
  // vib->frequencies (cm^-1), vib->modes (frequencies.size() x 3N)
}
```

## Tests

`test/gtest/geometry_host.cpp`: a quadratic well reached exactly in one step when the starting Hessian is the well's, and to 1e-10 by the BFGS updates when it is anisotropic; the extended Rosenbrock function; a four-point Lennard-Jones cluster converging to the regular tetrahedron at E = −6, with its centroid fixed to 1e-12 and every step free of rotation, and again at the default thresholds; an uphill step rejected with the trust radius cut to a quarter and the retry taken from the accepted geometry; a start at a minimum converging in one evaluation; and the refusals, the step cap and a failing callback.

`test/gtest/geometry_frequencies_gpu.cpp`: PySCF RHF/cc-pVDZ Hessians of bent water, CO₂ along z, HCN off every axis and linear water (a saddle point, its degenerate bend imaginary), against `pyscf.hessian.thermo.harmonic_analysis` of the same matrix (`test/gtest/data/frequencies/`, `tools/emit/frequencies_reference.py`): the mode count (3N − 6, or 3N − 5), the frequencies to 1e-3 cm⁻¹, and the modes up to sign as each degenerate set's mass-weighted projector to 1e-7; the negated water Hessian, every frequency imaginary; a spring between two hydrogens against √(k/μ) and its mode along the bond; a single atom with no modes; and the refusals.

## Files

| File | Role |
|------|------|
| `geometry.cppm` | Module interface: `Options`, `Iteration`, `Solution`, `optimize`; `Vibrations`, `harmonic_analysis` |
| `geometry.cpp` | The loop, the projection and the Cholesky solve |
| `rigid_body.cppm` | Internal partition: the orthonormal rigid-body basis, optionally mass-weighted, both units project with |
| `frequencies.cpp` | `harmonic_analysis`, the mass table and the two unit conversions |
| `CMakeLists.txt` | The module library |
