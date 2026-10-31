# `free_breakfast.integrals.eri.density_fitting.gradient`

The nuclear gradients of the two density-fitting integrals, contracted with a
caller's matrix, on the GPU:

    metric_gradient(res, aux, W)                       G[A][k] = Σ_PQ  W_PQ   d(P|Q)/dR_Ak
    three_center_gradient(res, basis, classes, aux, Γ) G[A][k] = Σ_mnP Γ_mnP  d(mn|P)/dR_Ak

by McMurchie–Davidson over the kernels `gmetric<lP, lQ>` and
`geri3<la, lb, lP>` that [FelizJueves](../../../../../deps/FelizJueves)
generates (`deps/FelizJueves/docs/implementation/gradient.md`, "The
density-fitting gradients"); the same derivative integrals uncontracted, over
its `ipmetric<lP, lQ>` and `xeri3<la, lb, lP>`; and over those, the
density-fitted J and K differentiated by every nuclear coordinate, the
two-electron half of an RHF Hessian's skeleton Fock derivatives:

    metric_derivative(res, aux, M1)                         M1[k][P][Q]  = d(P|Q)/dA_k, A on P
    three_center_derivative(res, basis, classes, aux, A, X) X[k][m][n][P] = (∇_k m n|P), P on atom A
    df_jk_derivative(res, basis, classes, aux, B, L, C, nocc, Jx, Kx, g)
                                                            J^(x), K^(x), u^x − M^x d

Also re-exported by
`free_breakfast.integrals.eri.density_fitting` and `free_breakfast.integrals`.
Link `free_breakfast::integrals::eri::density_fitting::gradient`.

```cpp
// W: naux x naux on the device, spherical; gamma: (m, n, P) at P + naux (n + nao m)
auto g_metric = metric_gradient(res, *aux, w);                           // n_atom x 3, host
auto g_eri3 = three_center_gradient(res, *basis, *classes, *aux, gamma); // n_atom x 3, host
// G[A][k] at (*g)[3 A + k]
```

Both matrices are **spherical** and any matrix: W in `metric`'s spherical
layout (column-major), Γ in `three_center`'s, P fastest. No symmetry is
assumed of either. The signs and halves of an energy are the caller's: a DF-J
gradient is `three_center_gradient` at Γ = D ⊗ d plus `metric_gradient` at
W = −½ d dᵀ (FelizJueves's gradient.md has the DF-K weights too).

## The build

1. **Cartesian.** The kernels contract against the Cartesian matrix. Since
   the spherical integrals are Uᵀ X U (U the `spherical_transform_matrix`),
   Σ W_sph X_sph = Σ (U W Uᵀ) X_cart, so W_cart = U_aux W U_auxᵀ (two gemms)
   and Γ_cart = Γ ×₁ U ×₂ U ×₃ U_aux on its three indices (a gemm on P, a
   strided-batched gemm on n, a gemm on m), into the d0 kernel's layout,
   (m, n, P) at (m n_ao + n) n_aux + P.
2. **Staging**, as the energy builds stage it (`metric/README.md`,
   `three_center.cppm`): the aux records; the metric's aux shell pairs P ≤ Q,
   staged swapped where lP > lQ; for (mn|P) mmd's records regrouped by shell
   pair and crossed with every aux shell into cells. The gradient kernels
   read a little more: each pair's or cell's **atoms** (from the bases'
   `atom_shell_offset`, hence `place`), and each bra record's two exponents
   (mmd's `ea`, `eb`), for the Cartesian rule on A and B.
   `gradient_bridge.h` has the layouts.
3. **One launch per class** that has a pair or a cell, one thread each,
   `kFourWarps` to a block, every one atomically adding into one zeroed
   device G[n_atom][3], which is downloaded. The atomics make a run
   reproducible to round-off, not bitwise.

The scratch comes from `res.device_workspace()`; each call synchronizes
`res.stream()` once to download G (the three-center one also once after
reading mmd's records back).

## The kernels' forms

FelizJueves's builders pick each class's form from measured floors: geri3 is
low-stack from la + lb + lP = 5 (49 of 70 classes), gmetric from lP + lQ = 10
(4 of 28). The low-stack form holds no Hermite density, which is what keeps
(ff|i) at about 13.8 KB of `sm_86` stack. The emitter forces neither.

## Limits

- The energy kernels' ladders, read from their bridges: orbital shells
  through f (`kThreeCenterMaxMomentum`), aux shells through i
  (`kThreeCenterAuxMaxMomentum`, `kMetricMaxMomentum`).
  `tools/emit/df_gradient_kernels.py` refuses to write another ladder, and
  `gradient_launch.cu` checks the tables against them at compile time.
- Both bases made by `place` over one molecule; G has one row per atom.
- The Cartesian Γ, n_cart² naux_cart doubles, held in full, and within the
  kernels' 32-bit index; past the device's free memory the call is
  Unsupported before anything is allocated.

## The derivative integrals

`metric_derivative` writes M1[k][P][Q] = d(P|Q)/dA_k, A the centre of P, at
Q + naux (P + naux k), spherical (3 naux² doubles). The metric reads A − B
only, so each slab is antisymmetric and d(P|Q)/dR_Xk = [P on X] M1[k][P][Q] +
[Q on X] M1[k][Q][P]. The pairs are `metric_gradient`'s; `ipmetric` writes
each pair's block and its negated mirror by plain stores into the Cartesian
matrices, each made spherical as U_auxᵀ M1 U_aux.

`three_center_derivative` writes X[k][m][n][P′] = (∇_k m n|P) = −d(mn|P)/dA_k,
A the centre of m (libcint's `int3c2e_ip1`), for the aux functions P on one
atom, at P′ + nblk (n + nao (m + nao k)). The other two centres follow:
d(mn|P)/dR_Xk = −[m on X] X_mn − [n on X] X_nm + [P on X] (X_mn + X_nm), the
last by translational invariance, so no second kernel. The whole
3 nao² naux tensor (9 GB at 500 AOs and 1500 aux functions) is never asked for
at once: the caller walks the aux atoms. The cells are `three_center_gradient`'s
whose aux shell sits on the atom, and `xeri3` (which ignores the cells' atoms)
writes each cell's (m, n) and (n, m) slots by plain stores into the Cartesian
block; that is made spherical by a gemm with the atom's block of U_aux (U_aux
is block-diagonal by shell, and an atom's shells are contiguous), a
strided-batched gemm with U per Cartesian m, and a gemm with U.

## The DF J/K derivative

`df_jk_derivative` takes the fit as `density_fit` leaves it (B = L⁻¹ (P|mn),
L) and the occupied orbitals, D = 2 C_occ C_occᵀ, and writes per coordinate
x = 3 A + k the derivatives at fixed D of J_mn = Σ (mn|ls) D_ls and
K_mn = Σ (ml|ns) D_ls over the fit, both the integrals' and the metric's (the
design's §7, PySCF's `df.hessian.rhf` at auxbasis response 2):

    J^(x) = T^x d + T M⁻¹ (u^x − M^x d)                         d = M⁻¹ u, u = Tᵀ vec D
    K^(x) = 2 Σ_i (T^x_i c_iᵀ + c_i T^x_iᵀ − c_i M^x c_iᵀ)      c_i = T_i M⁻¹, (T_i)_mP = Σ_l (ml|P) C_li

Each matrix is S + Sᵀ (K's twice that) for a half S that only ever needs
X_mn (the derivative on m's centre). jx and kx accumulate S, stored
transposed, while one aux atom B at a time is built
(`three_center_derivative`'s block, and the same last product with m and n's
places swapped):

| piece of S^(x) | from atom B's block | onto |
|---|---|---|
| J: Σ_P X_mn,P d_P | gemv with d's rows on B | x = 3B + k, and Y_k |
| J: −[m on A] Σ_P X_mn,P d_P | Y_k, after the walk | x = 3A + k, columns m on A (axpy) |
| u^x: −2 Σ_{m on A} X_mn,P D_mn and 2 Σ X_mn,P D_mn | gemv over the columns m on A, and over all | g's rows on B |
| K: Σ_lP X_ml,P Z_P,ln, Z_P,ln = Σ_i c_P,is C_li | gemm with Z's rows on B | 3B + k, and V1_k |
| K: Σ_lP X_lm,P Z_P,ln | gemm, the swapped block | 3B + k |
| K: −Σ_{l on A} X_lm,P Z_P,ln | gemm over the swapped block's rows l on A | 3A + k |
| K: −Σ_{P on B, i} c_P,im F_P,in, F = M1[k] c | gemm | 3B + k |
| K: −[m on A] V1_k | after the walk | 3A + k, columns m on A |

and −M^x d goes into g first, from M1's rows and columns on each atom (two
gemvs per coordinate). After the walk, g = u^x − M^x d is copied out when
asked for, solved in place by L (one trsm over every coordinate), each slab is
symmetrized (a geam through one slab of scratch), and J gets T M⁻¹ g =
Bᵀ L⁻¹ g, symmetric already, for every coordinate in one gemm.

The scratch is one request: the staging, the fitted half-transformed pairs c
(naux nocc nao), M1 and its Cartesian copy (3 naux² and 3 naux_cart²), g, and
one aux atom's blocks (the Cartesian 3 nao_cart² nblk_cart, X and its swap
3 nao² nblk each, Z nao² nblk, c's rows and F nocc nao nblk each), sized by the
largest atom. Past the device's free memory the call is Unsupported, naming
the bytes, before anything is staged; the 32-bit indices of the kernels (3
naux_cart², 3 nao_cart² nblk_cart) and of the BLAS calls (naux nao², 3 n_atom
nao²) likewise.

## Tests

`test/gtest/df_gradient_gpu.cpp`, against `test/gtest/data/gradient/`'s
`density_fitting` section: PySCF's `int2c2e_ip1` and `int3c2e_ip1` + `ip2`
contracted with the recipe's W_PQ = 1/(P+Q+1) and Γ_mnP = 1/(m+n+P+1), for
the skew water in cc-pVDZ / cc-pVDZ-JKFIT, cc-pVTZ / cc-pVTZ-JKFIT and
cc-pVTZ / cc-pVTZ-acCD-6-6 (aux to i, so (ff|i) and (i|i)), to 1e-10 of
max|G|, each component's sum over atoms below 1e-10; and the refusals. The
one exception is the acCD-6-6 metric, held to 1e-9: there libcint's
`int2c2e_ip1` is itself off by about 1.7e-11 on the set's tight O s and p
contractions, while this tree's matches a 40-digit reference of the same
contraction to about 1e-13 (#494's PR has the numbers).

`test/gtest/df_jk_derivative_gpu.cpp`, against `test/gtest/data/df_jk_derivative/`
(`tools/emit/df_jk_derivative_reference.py`): J^(x), K^(x), J^(x) − ½ K^(x)
(PySCF's `make_h1` less its core-Hamiltonian derivative) and u^x − M^x d at
PySCF's converged DF-RHF orbitals, for the skew water in cc-pVDZ /
cc-pVDZ-JKFIT, cc-pVTZ / cc-pVTZ-JKFIT and cc-pVTZ / cc-pVTZ-acCD-6-6 (aux to
i), to 1e-10 of the largest element; ½ tr(D J^(x)) and −¼ tr(D K^(x)) against
`gradient::df_rhf`'s Coulomb and exchange terms over the same orbitals;
`metric_derivative` and `three_center_derivative`, put back together by the
centre rules above with the recipe's W and Γ, against `metric_gradient` and
`three_center_gradient`; and the refusals, the memory one included.
