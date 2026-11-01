# Root rotation

The term the rotations among the averaged roots add to an **equal-weight**
state-averaged DF-CASSCF Hessian, `casscf_hessian.tex` eq. (rot), and the
rotations themselves, eq. (theta), for every nuclear coordinate at once:

    ΔH_I(x, y) = 2 Σ_{s≠I} ℋ^x_Is ℋ^y_Is / (E_I − E_s),
    θ^x_sr = ℋ^x_rs / (E_r − E_s),
    ℋ^x_rs = ⟨c_r|H̃^x + H′(κ^x)|c_s⟩.

At equal weights E_SA is invariant to the roots' rotations among themselves,
so they are not SA parameters and λ^x does not carry them, but root I's
energy is not invariant: the roots are fixed by ⟨c_r|H|c_s⟩ = 0 for r ≠ s,
whose first derivative is eq. (theta). E_I is stationary in θ_sI with
∂²E_I/∂θ_sI² = 2(E_s − E_I) and total derivative 2ℋ^x_Is along the response,
and eliminating θ gives eq. (rot). **The factor 2 is not optional**: the
product appears once for each order of x and y, and one copy misses by half
the term, which with three equal roots is as large as the Hessian itself.
At unequal weights the rotations are parameters, their response is inside
λ^x, and this term is not added: the call refuses unequal weights.

## Module

`free_breakfast.mcscf.root_rotation`, target
`free_breakfast::mcscf::root_rotation`, re-exported by `free_breakfast.mcscf`.

```cpp
Result<mcscf::RootRotation> mcscf::root_rotation(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    const SphericalMatrixAO &h, const Buffers::device<double> &b,
    const SphericalMatrixAO &c, int32_t ncore, int32_t ncas, int32_t nalpha, int32_t nbeta,
    const double *roots, std::span<const double> weights, std::span<const double> energies,
    const Derivatives &derivatives, const double *kappa, double max_rotation = 10.0);
```

The orbitals, roots and integrals are [`../derivative/`](../derivative/README.md)'s
`df_derivatives` inputs (B is `density_fit`'s, naux nao² long; naux is
read from its length), and `derivatives` is that call's result at the same
orbitals and roots: only its `hamiltonian`, H̃^x_rs, is read, and that is
the same at any weights. `weights` are the average's, all equal (to 1e-10
relative); `energies` each root's, in any common origin. `kappa` is λ^x's
orbital part, the response of the SA parameters:

| Input | Where | Shape, element |
|---|---|---|
| `kappa` | device | (n_pairs, n): (k, x) at k + n_pairs x, k over `response::pack_orbital_pairs`' packed pairs. x_pq = K_pq at p > q, C → C e^K: [`../zvector/`](../zvector/README.md)'s `zvectors` orbital block as it comes, on the SA right-hand sides of `df_derivatives` (`rhs_orbital` packed; `rhs_ci` column r over 2 w_r, zvector's CI convention) |

λ^x's CI part is not taken: it is off every root, and ⟨δc_r|H|c_s⟩ =
E_s ⟨δc_r|c_s⟩ = 0, so it does not enter ℋ.

## Outputs

All on the host, x = 3 A + k (atom A, axis k), n = 3 n_atom, column-major,
**x last** as everywhere in this layer.

| Member | Shape, element | Contents |
|---|---|---|
| `coupling` | (nroots, nroots, n): (r, s, x) at r + nroots (s + nroots x) | ℋ^x_rs = H̃^x_rs + ⟨c_r\|H′(κ^x)\|c_s⟩, the orbital-relaxed coupling (the derivative-coupling numerator), symmetric in r and s, **zero on the diagonal**: ⟨c_r\|H′\|c_r⟩ would need the core energy's change, which the off-diagonal elements do not see, and no consumer needs it |
| `theta` | (nroots, nroots, n): (s, r, x) at s + nroots (r + nroots x) | θ^x_sr = ℋ^x_rs / (E_r − E_s): root r's first-order change along root s, dc_r/dx ⊃ Σ_s θ^x_sr c_s, the fixture's `_theta` vector convention. Antisymmetric, zero diagonal |
| `hessian` | (n, n, nroots): (x, y, I) at x + n (y + n I) | eq. (rot) for root I, symmetric in x and y |

`theta` is what the CI part of a root's own response lacks at equal weights:
root r's full first-order change is λ^x's CI column r plus Σ_s θ^x_sr c_s.
The issues after this one (#620, #621) read `theta` and `hessian` as laid
out here; `coupling` row I is the fixture's `coupling` (nroots, n) of root I.

One root returns zeros (n × n for `hessian`) and does not touch `kappa`.

## The coupling's orbital part

⟨c_r|H′(κ)|c_s⟩ is the active-space Hamiltonian's first derivative along
C → C (1 + κ), between two roots: h_eff′·γ_S + ½ (tu|vw)′·Γ_S, γ_S and Γ_S the
symmetrized transition RDMs ½(⟨c_r|E|c_s⟩ + ⟨c_s|E|c_r⟩), which give it
exactly since H′ is real symmetric. They come by polarization, a quarter of
the difference of `ci::Rdm::rdm12` at c_r + c_s and at c_r − c_s (two calls
per pair). Its gradient in κ is
[`../relaxed_hessian/`](../relaxed_hessian/README.md)'s `response_gradient`,
the same adjoint `RelaxedHessian`'s H_oc uses, over F^I (from `response::fock`
at the averaged RDMs; any RDMs give the same F^I) and ppaa, papa
(`casscf_integrals`); packed, one gemv against every column of `kappa` gives
the orbital part for all x. No core term enters: a root pair's transition
density has no core part, ⟨c_r|c_s⟩ being 0, which is why this is not
`response::fock`'s gradient at the transition RDMs.

## The degeneracy refusal

The term diverges as E_s → E_I: at an intersection the adiabatic energy is
not differentiable and has no Hessian. A pair is refused (Numerical, the
message naming both roots and the gap in hartree) when

- |E_r − E_s| < 1e-8 hartree, whatever the coupling (`RelaxedHessian`'s
  floor for dropping the other roots' mixing, where the quotient is no
  longer a number), or
- max_x |ℋ^x_rs| / |E_r − E_s| = max_x |θ^x_sr| > `max_rotation`, 10 per
  bohr by default.

θ is the rate at which the two roots mix per bohr, so it is the
dimensionless test of "near": in a two-state model the quadratic expansion of
E_I holds for displacements well under 1 / (2|θ|), so past 10 per bohr the
Hessian describes E_I only within 0.05 bohr, less than a typical zero-point
amplitude, and a harmonic analysis over it is not meaningful. The fixture's
cases sit at |θ| ≤ 1.9 per bohr (three roots, gap 0.066 hartree). The
threshold is far above the 1e-8 hartree floor: a coupling of 0.05
hartree/bohr is refused under a gap of 5e-3 hartree.

## Tests

`test/gtest/mcscf_root_rotation_gpu.cpp` (suite `McscfRootRotation`),
against `test/gtest/data/casscf_hessian/`'s equal-weight state averages, the
orbitals and roots **loaded from the fixture** (its README's quirks are
handled as `mcscf_derivative_gpu.cpp` does): two roots and three at equal
weights, each root's `coupling`, θ and `rot`,

- over the fixture's λ^x, to 1e-9 absolute;
- over λ^x solved here by `zvectors` (Schur) from `df_derivatives`' SA
  right-hand sides, the chain a Hessian runs, to 1e-8, with that λ^x's
  orbital part against the fixture's on the way;
- and that the term is not half the reference's (the factor 2).

Measured on the RTX 3080 (CUDA): over the fixture's λ^x, ℋ ≤ 4.6e-14, θ ≤
7e-13 and eq. (rot) ≤ 3.5e-13 (of max |rot| 0.049 for two roots, 0.48 for
three); over the solved λ^x, whose orbital part is the fixture's to 4.9e-9
(two roots) and 8e-10 (three) of 0.41, ℋ ≤ 6.5e-11, θ ≤ 5.4e-10 and eq. (rot)
≤ 7.5e-11. 0.1 s a case over the fixture's λ^x, 0.5–0.8 s with the solve,
the setup included. The gfx1200 was not run.

The refusals: the two-root case's roots given energies 1e-3 hartree apart
(couplings up to 0.055, so |θ| ≈ 55 per bohr) are refused with "1.000e-03
hartree apart" in the message and accepted under `max_rotation` 1e3; a
coincident pair is refused whatever the threshold; unequal weights, an
energy count, a non-positive threshold and a derivative of the wrong root
count are InvalidConfig. One root gives zeros.

## Files

| File | Role |
|------|------|
| `root_rotation.cppm` | Module interface: `RootRotation`, `root_rotation` |
| `root_rotation.cpp` | Its implementation: the transition RDMs, the coupling's orbital part, eqs. (theta) and (rot) and the refusal; host code over the RDMs, `response_gradient` and one gemv per pair, no kernels |
| `CMakeLists.txt` | The `free_breakfast.mcscf.root_rotation` target |
