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
| `RelaxedHessian` | H̃ at one set of orbitals and roots: `apply` (y = H̃ x, one column or a block), `respond` (δ and its transition RDMs γ_T, Γ_T for one column), `couple` (w = −(H₀ − E)⁻¹ P y for a CI right-hand side y, and H_oc w; one column or a block), `apply_coupled` (the unreduced operator over (x, x_c), a block), `ci_diagonal` (diag H₀), `delta`, `size`, `weights` |

`RelaxedHessian::create` takes the `response::Hessian`, the sigma whose
integrals are H₀ at the same orbitals, the `ci::Rdm` and the strings both are
over, the `ci::Roots` of H₀, the weights and the `ResponseIntegrals`. It keeps
pointers to the first four of those (and the roots), which must outlive it
and not change under it, and allocates its own second sigma (for H′(x)) and
the CI-sized scratch. A product synchronizes the stream: κ(x) and the
transition RDMs are read back, and H′ and its adjoint are host work.
Its last argument, `response_tolerance` (1e-8 by default), is where the CI
response's conjugate gradients stop, relative to their right-hand side: every
product is good to about that. A test that compares an exact solve with one
through H̃ tightens it.

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
`couple` is steps 3 to 5 with b = y, a CI block the caller gives (projected
off every root first, so the other roots' part is nothing): w into `delta()`
and H_oc w over the packed pairs, the orbital side of a Z-vector's CI
right-hand side.
## Block calls

`apply`, `couple` and `apply_coupled` over a block of k columns are what
`mcscf::zvectors` (`../zvector/README.md`, "Many right-hand sides") iterates
on. They share the per-column work above: H's product over the block in one
call (`response::Hessian::apply`, which still builds each column's J and K
on its own), then H′(x_j) c and H_oc δ_j per column (host work, a readback
each). What the block changes is the CI side:

- **The block `apply`** runs every column's CI response at once: the
  conjugate gradients of step 3 over all k · nroots vectors in lockstep, each
  vector at its own root's E_r, with one sigma of the whole block per step,
  every vector's dot by one strided-batched gemm and one readback, and the
  updates as dgmm and geam over the block. A vector stops at
  `response_tolerance` of its right-hand side as before (its step lengths
  zero after that). The other roots' part is one gemm of the overlaps.
  Each column agrees with the one-column `apply` to the CI response's
  tolerance, not to the bit (another summation order); `delta()` holds the k
  blocks afterwards.
- **The block `couple`** is the one-column `couple` over the block's lockstep
  solve.
- **`apply_coupled`** solves nothing: y = H x + H_oc x_c and
  y_c,r = P (H′(x) c_r + (H₀ − E_r) x_c,r), the CI rows unweighted (the
  Z-vector's convention), (H₀ − E_r) x_c one sigma of the block. With one
  string set for both spins (N_α = N_β), y_c is then put **exactly** in the
  roots' sector under the swap of the α and β strings, (y_c ± y_cᵀ)/2 per
  vector, the sign the roots' parity (`create` reads it off c₀ · c₀ᵀ: + for
  even S, as `ci::Layout::symmetric` has it). An iteration built from it by
  axpy and a swap-symmetric diagonal stays in that sector to the bit. Without
  it the projection's roundoff seeds the other one, which in the full
  determinant space holds the M_s = 0 triplets: below an excited singlet
  root, they make H₀ − E_r indefinite, and a conjugate gradients that
  reaches them diverges (water's SA-2 did, at p·Ap = −1.3e-13).

Their scratch is a member, grown to the widest block seen: r, p, A p and one
product (dim × k nroots each) and a few k nroots vectors, plus `delta()` and
b at k blocks. Six CI blocks per column, so a block of 3N columns over a
large active space is that much device memory.

The derivation, the costs and what H̃ bought the loop are in
[`../solve/README.md`](../solve/README.md), "The CI response is computed, per
product".

## Tests

None of its own: `test/gtest/mcscf_solve_gpu.cpp` drives every product
through `mcscf::solve`, `test/gtest/mcscf_zvector_gpu.cpp` `couple`
through `mcscf::zvector`'s CI right-hand side, and
`test/gtest/mcscf_zvectors_gpu.cpp` the block calls through
`mcscf::zvectors`.

## Files

| File | Role |
|------|------|
| `relaxed_hessian.cppm` | Module interface: `ResponseIntegrals`, `response_integrals`, `response_gradient`, `RelaxedHessian` |
| `relaxed_hessian.cpp` | H′(x) and its adjoint, the CI response's conjugate gradients (one column, and a block's in lockstep) and transition RDMs, and the products |
| `CMakeLists.txt` | The module `free_breakfast.mcscf.relaxed_hessian` |
