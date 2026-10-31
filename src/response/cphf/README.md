# CPHF

The closed-shell coupled-perturbed Hartree–Fock equations, M U = B, for a
block of right-hand sides at once, on the GPU over the density-fitted J and K.
The response of RHF orbitals to a perturbation: a nuclear displacement for the
analytic Hessian, an electric field for the polarizability.

## Module

`free_breakfast.response.cphf`, target `free_breakfast::response::cphf`,
re-exported by `free_breakfast.response`.

```cpp
struct response::Cphf {
  Buffers::device<double> solution; // U, (n_vir n_occ) x ncol, U_ai at a + n_vir i
  std::vector<double> residuals;    // ||M U_x - B_x|| per column
  int32_t iterations;               // lockstep sweeps
};

Result<response::Cphf> response::cphf(
    DeviceResources &res, const Buffers::device<double> &b, int32_t naux,
    const Buffers::device<double> &c, std::span<const double> energies, int32_t nocc,
    const Buffers::device<double> &rhs, double tolerance = 1e-10,
    int32_t max_iterations = 100);
```

`b` is the fitted factor as `df_coulomb` reads it, `c` the nao × nmo orbitals
(the first `nocc` occupied), `energies` their orbital energies on the host,
`rhs` the right-hand sides in the solution's layout.

## The equations

    M_ai,bj = (ε_a − ε_i) δ_ab δ_ij + 4 (ai|bj) − (ab|ij) − (aj|ib)

is the real singlet RHF orbital Hessian over the virtual–occupied pairs:
symmetric, and positive definite exactly when the reference is a minimum
under real singlet rotations. It is never formed. A product is

    (M U)_ai = (ε_a − ε_i) U_ai + C_aᵀ G(D_U) C_i,
    D_U = 2 (X + Xᵀ),  X = C_v U C_oᵀ,  G = J − K/2,

two gemms to D_U, one `df_coulomb` of it, one two-factor `df_exchange` of
L Rᵀ with L = [C_v U, C_o] and R = [C_o, C_v U] (D_U = 2 L Rᵀ, so
K[D_U]/2 = K[L Rᵀ]), and two gemms back to the ai block: one SCF iteration's
work per column. The columns of a sweep go through it together, as the
block forms of `df_coulomb` and `df_exchange` (`density_fitting/jk/`): every
gemm above is one batched over the block, and J and K are one build of the
whole block's densities.

For a nuclear coordinate x, B^x_ai = −F^(x)_ai + ε_i S^x_ai + [C_vᵀ G(D_S^x) C_o]_ai
with D_S^x = 2 C_o S^x_oo C_oᵀ, and U is PySCF's `solve_mo1` `mo1`'s
virtual–occupied block. For a field, B = −r_ai and α_xy = 4 B_x · U_y.

## The solver

Preconditioned conjugate gradients from U = 0, preconditioner (ε_a − ε_i)⁻¹,
every column in lockstep: each sweep takes one product per column still
above `tolerance` in ||r|| and leaves the stopped ones as they are, so the
sweep count is the slowest column's. Its interface and stopping rules are
`mcscf::zvector`'s: an absolute tolerance on each column's residual, the true
residual from one more product at the end, MaxIterationsExceeded when a
column is still above it after `max_iterations` sweeps, and Divergence when a
direction has p·Mp ≤ 0. That last is not the solver failing: M is not
positive definite, so the RHF reference is unstable and a CPHF over it has no
meaning.

All columns share M, so a sweep's products are one batched build over the
active columns' densities: their P gathered side by side, M P scattered back.
A batch is as wide as half the device's free memory holds at the call, its
own L, R, D_U, J, K and G C_o and the J/K scratch (X and Y,
2 · naux · 2 n_occ · nao doubles a column) together, and the int32_t BLAS
dispatch allows; a sweep wider than that runs in batches of it. The
workspace is left grown to that scratch.

### Batched against one column at a time

The CPHF wall clock of 3 N_atom random right-hand sides, the per-column
loop it replaced against the batched build (`bench/cphf_batched.cpp`,
`free_breakfast_cphf_batched`), the second build solving over the first's
orbitals: decane (C₁₀H₂₂, 32 atoms) in cc-pVDZ / cc-pVDZ-JKFIT, nao 250,
n_occ 41, naux 1206, 96 columns, on the gfx1200 (HIP), the card held.

| Product | Wall clock | Sweeps | max residual |
| --- | --- | --- | --- |
| one column at a time | 190.3 s | 18 | 9.96e-11 |
| batched | 167.4 s | 18 | 9.96e-11 |

1.14× faster; the two U agree to 1.3e-15 (max |U| 0.27). The flops are the
same: the batch buys wider gemms and fewer launches (by the arithmetic about
20 columns a build at half of 16 GB free, so X and Y's batched gemms are
about 20 · 82 columns wide where one column's are 82). The RTX 3080 (CUDA)
was not benchmarked.

## Tests

`test/gtest/response_cphf_gpu.cpp` (suite `ResponseCphf`), against
`test/gtest/data/cphf/` (`tools/emit/cphf_reference.py`) over the
`density_fitting/` h2o_ccpvdz integrals: the nine nuclear columns against
`pyscf.df.hessian.rhf`'s `mo1` over the same fitting set, the dipole columns'
polarizability against PySCF's, one column alone against the lockstep block,
an exhausted budget and the refusals.

## Files

| File | Role |
|------|------|
| `cphf.cppm` | Module interface: `response::Cphf`, `response::cphf` |
| `cphf.cpp` | The product and the lockstep conjugate gradients |
| `CMakeLists.txt` | Build configuration |
