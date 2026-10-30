# Relaxed Hessian

The orbital Hessian with the CI relaxed, H̃ = H + R, over the packed pairs at
fixed orbitals and roots: `response::Hessian`'s H plus the CI's response R.
`mcscf::solve`'s augmented Hessian steps over it, and the state-averaged
Z-vector of a CASSCF gradient drives the same product.

## Module

`free_breakfast.mcscf.relaxed_hessian`, target
`free_breakfast::mcscf::relaxed_hessian`. **Internal to the MCSCF layer**:
`free_breakfast.mcscf` does not re-export it, and `solve` links it PRIVATE.
Everything is in `namespace free_breakfast::mcscf`.

| Name | What it is |
|------|------------|
| `ResponseIntegrals` | F^I, ppaa and papa on the host, column-major as the device's: what H′(x) contracts |
| `response_integrals` | H′(x): h_eff′ and (tu\|vw)′ along κ(x), row-major as `ci::Sigma::set_integrals` takes them |
| `response_gradient` | Its adjoint: the antisymmetric nmo × nmo whose packed pairs are the gradient in x of h_eff′·γ + (tu\|vw)′·Γ |
| `RelaxedHessian` | H̃ at one set of orbitals and roots: `apply` (y = H̃ x, one column), `respond` (δ and its transition RDMs γ_T, Γ_T for one column), `delta`, `size` |

`RelaxedHessian::create` takes the `response::Hessian`, the sigma whose
integrals are H₀ at the same orbitals, the `ci::Rdm` and the strings both are
over, the `ci::Roots` of H₀, the weights and the `ResponseIntegrals`. It keeps
pointers to the first four of those (and the roots), which must outlive it
and not change under it, and allocates its own second sigma (for H′(x)) and
the CI-sized scratch. A product synchronizes the stream: κ(x) and the
transition RDMs are read back, and H′ and its adjoint are host work.

## What a product is

`apply` is `response::Hessian::apply`, then R x added:

1. κ(x) unpacked (`unpack_orbital_pairs`) and read back, and H′(x) from it
   on the host (`response_integrals`); no J or K, every integral it takes
   has two active indices.
2. b_r = H′ c_r, the second sigma applied to the roots.
3. δ_r = −(H₀ − E_r)⁻¹ (1 − Σ_s c_s c_sᵀ) b_r by conjugate gradients over
   H₀'s sigma, to 1e-8 of the right-hand side (at most 200 sigmas per root),
   and the other roots' part exactly, −(c_s · b_r) / (E_s − E_r) c_s (left
   out for a gap under 1e-8 hartree).
4. γ_T and Γ_T: half the difference of `ci::Rdm::rdm12` of c + δ and of
   c − δ at the weights (Γ_T's halved again, for the ½ of the two-body
   energy).
5. R x = H_oc δ: `response_gradient` of γ_T and Γ_T, uploaded, packed
   (`pack_orbital_pairs`) and added to y.

Steps 1 to 4 are `respond`, so a caller that needs δ itself (the CI part of a
Z-vector's multipliers) or the transition RDMs takes them without H's product.
The derivation, the costs and what H̃ bought the loop are in
[`../solve/README.md`](../solve/README.md), "The CI response is computed, per
product".

## Tests

None of its own: `test/gtest/mcscf_solve_gpu.cpp` drives every product
through `mcscf::solve`.

## Files

| File | Role |
|------|------|
| `relaxed_hessian.cppm` | Module interface: `ResponseIntegrals`, `response_integrals`, `response_gradient`, `RelaxedHessian` |
| `relaxed_hessian.cpp` | H′(x) and its adjoint, the CI response's conjugate gradients and transition RDMs, and the product |
| `CMakeLists.txt` | The module `free_breakfast.mcscf.relaxed_hessian` |
