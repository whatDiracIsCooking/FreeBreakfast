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
    int32_t max_iterations = 100, bool batched = false);
```

`b` is the fitted factor as `df_coulomb` reads it, `c` the nao × nmo orbitals
(the first `nocc` occupied), `energies` their orbital energies on the host,
`rhs` the right-hand sides in the solution's layout. `batched` builds a
sweep's products as one block (below); it is off by default.

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
work per column.

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

### `batched`, off by default

All columns share M, so with `batched` a sweep's products are one batched
build over the active columns' densities: their P gathered side by side, the
gemms above batched over the block, J and K by the block forms of
`df_coulomb` and `df_exchange` (`density_fitting/jk/`), and M P scattered
back. A batch is as wide as half the device's free memory holds at the call,
its own L, R, D_U, J, K and G C_o and the J/K scratch (X and Y,
2 · naux · 2 n_occ · nao doubles a column) together, and the int32_t BLAS
dispatch allows; a sweep wider than that runs in batches of it. **The batch
takes half the free VRAM, and the workspace is left grown to its X and Y**
after the call. Off, each column's product is its own `df_coulomb` and
`df_exchange`, as before the option existed, bit for bit.

It is off because it is not faster on both cards. The CPHF wall clock of 3
N_atom random right-hand sides (`bench/cphf_batched.cpp`,
`free_breakfast_cphf_batched`), the second run solving over the first's
orbitals: decane (C₁₀H₂₂, 32 atoms) in cc-pVDZ / cc-pVDZ-JKFIT, nao 250,
n_occ 41, naux 1206, 96 columns, 18 sweeps either way, the card held.

| Card | One column at a time | `batched` | Ratio |
| --- | --- | --- | --- |
| gfx1200 (HIP) | 190.3 s | 167.4 s (best of 2) | 1.14× faster |
| RTX 3080 (CUDA) | 184.04 s | 185.66 s | 0.9% slower |

On the 3080 that is one timed solve each after a warm-up; the warm-ups were
184.08 s and 171.83 s, so the batched path's run-to-run spread (about 14 s)
is larger than the difference, and a process outside the card lock held
922 MiB on it throughout. The two paths' U agree to 1.3e-15 (gfx1200) and
1.8e-15 (3080), max |U| 0.27–0.28. The flops are the same: the batch buys
wider gemms and fewer launches (by the arithmetic about 20 columns a build
on the gfx1200's 16 GB, X and Y's batched gemms about 20 · 82 columns wide
where one column's are 82).

## Tests

`test/gtest/response_cphf_gpu.cpp` (suite `ResponseCphf`), against
`test/gtest/data/cphf/` (`tools/emit/cphf_reference.py`) over the
`density_fitting/` h2o_ccpvdz integrals: the nine nuclear columns against
`pyscf.df.hessian.rhf`'s `mo1` over the same fitting set, the dipole columns'
polarizability against PySCF's, one column alone against the lockstep block,
an exhausted budget and the refusals; and `batched` against the default path
on the nuclear and dipole columns.

## Files

| File | Role |
|------|------|
| `cphf.cppm` | Module interface: `response::Cphf`, `response::cphf` |
| `cphf.cpp` | The product and the lockstep conjugate gradients |
| `CMakeLists.txt` | Build configuration |
