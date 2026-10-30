# `free_breakfast.integrals.eri.density_fitting.gradient`

The nuclear gradients of the two density-fitting integrals, contracted with a
caller's matrix, on the GPU:

    metric_gradient(res, aux, W)                       G[A][k] = Σ_PQ  W_PQ   d(P|Q)/dR_Ak
    three_center_gradient(res, basis, classes, aux, Γ) G[A][k] = Σ_mnP Γ_mnP  d(mn|P)/dR_Ak

by McMurchie–Davidson over the kernels `gmetric<lP, lQ>` and
`geri3<la, lb, lP>` that [FelizJueves](../../../../../deps/FelizJueves)
generates (`deps/FelizJueves/docs/implementation/gradient.md`, "The
density-fitting gradients"). Also re-exported by
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
