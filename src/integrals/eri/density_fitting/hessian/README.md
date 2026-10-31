# `free_breakfast.integrals.eri.density_fitting.hessian`

The second derivatives of the two density-fitting integrals, contracted with
a caller's matrix and never stored, on the GPU:

    metric_hessian(res, aux, W)                       H[A k][B l] = Σ_PQ  W_PQ   d²(P|Q)/dR_Ak dR_Bl
    three_center_hessian(res, basis, classes, aux, Γ) H[A k][B l] = Σ_mnP Γ_mnP  d²(mn|P)/dR_Ak dR_Bl

by McMurchie–Davidson over the kernels `d2metric<lP, lQ>` and
`d2eri3<la, lb, lP>` that [FelizJueves](../../../../../deps/FelizJueves)
generates (`deps/FelizJueves/docs/implementation/hessian.md`, "The DF metric
Hessian" and "The three-center DF Hessian"). Also re-exported by
`free_breakfast.integrals.eri.density_fitting` and `free_breakfast.integrals`.
Link `free_breakfast::integrals::eri::density_fitting::hessian`.

```cpp
// W: naux x naux on the device, spherical; gamma: (m, n, P) at P + naux (n + nao m)
auto h_metric = metric_hessian(res, *aux, w);                           // 3 n_atom x 3 n_atom, host
auto h_eri3 = three_center_hessian(res, *basis, *classes, *aux, gamma); // 3 n_atom x 3 n_atom, host
// H[A][k][B][l] at (*h)[(3 A + k) 3 n_atom + 3 B + l]
```

Both matrices are **spherical** and any matrix, in the gradient's layouts
(`gradient/README.md`): no symmetry is assumed of either. The signs and
halves of an energy are the caller's: a DF-RHF Hessian's skeleton Coulomb
part is `three_center_hessian` at Γ = D ⊗ d plus `metric_hessian` at
W = −½ d dᵀ, and its exchange part the same at `gradient::df_rhf`'s Γᴷ and
Wᴷ.

## The build

The gradient's (`gradient/README.md`), one derivative order up: the matrices
made Cartesian, the same staging with each pair's or cell's atoms and each
bra record's exponents, and one launch per class that has a pair or a cell,
one thread each, `kFourWarps` to a block, every one atomically adding into
one zeroed device H[n_atom][3][n_atom][3], which is downloaded. The kernels
read the gradient kernels' parameter block, with H in place of G, so
`hessian.cpp` stages exactly as `gradient.cpp` does.

**The aux centre is differentiated by the Hermite shift.** Aux shells reach
i, and the Cartesian rule twice on the aux centre would raise its E table to
l = 8. Its single-center E does not depend on where it sits, so d/dC moves
only the Hermite Gaussian: d²/dC_k dC_l [r]⁽⁰⁾ = [r + 1_k + 1_l]⁽⁰⁾, and the
kernels read the Boys tower two indices up (L + 2, 14 at (ff|i)) with no
raised aux table. The metric's two centres are both moved that way. No
centre is inferred from the others by translational invariance, so the sum
rule Σ_A H[A k][j] = 0 is a check of each class, not an identity.

## The kernels' forms

FelizJueves's builders pick each class's form from measured floors
(`kernel_d2eri3.LOWSTACK_MIN_L`, `kernel_d2metric.LOWSTACK_MIN_L`): the
low-stack form holds no Hermite density. The emitter forces neither.

## Limits

The gradient's: orbital shells through f, aux shells through i (the energy
kernels' ladders, read from their bridges; `tools/emit/df_hessian_kernels.py`
refuses another ladder and `hessian_launch.cu` checks the tables at compile
time), both bases made by `place` over one molecule, and the Cartesian Γ held
in full within the kernels' 32-bit index, refused before anything is
allocated past the device's free memory.

## Tests

`test/gtest/df_hessian_gpu.cpp`, against `test/gtest/data/hessian/`: PySCF's
`int2c2e_ipip1` / `ip1ip2` and `int3c2e_ipip1` / `ipvip1` / `ip1ip2`
contracted with the recipe's W_PQ = 1/(P+2Q+1) and Γ_mnP = 1/(m+2n+3P+1),
neither symmetric, for the skew water in cc-pVDZ / cc-pVDZ-JKFIT, cc-pVTZ /
cc-pVTZ-JKFIT and cc-pVTZ / cc-pVTZ-acCD-6-6 (aux to i, so (ff|i) and
(i|i)), to 1e-10 of max|H|; the sum rule of every one of the 28 + 70 classes
on its own (a three-atom system per class, one shell a centre, the recipe kept where only that class reaches it), to 1e-10 of the
class's max|H|; and the refusals. The one exception is the acCD-6-6
metric, held to 1e-9, as the gradient's is: there libcint's `int2c2e_ipip1`
/ `ip1ip2` are themselves off by up to 4e-11 on the set's tight O s and p
contractions, while this tree's matches a 40-digit reference of the same
contraction to about 3e-13 (#593's PR has the numbers). The acCD case takes
about 42 s on the RTX 3080, so it is a slow-tier case.
