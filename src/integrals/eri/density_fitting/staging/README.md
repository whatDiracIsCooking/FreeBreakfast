# `free_breakfast.integrals.eri.density_fitting.staging`

**Internal**, not re-exported: `free_breakfast.integrals.eri.density_fitting`
does not re-export it, and only `gradient/` and `hessian/` link it, PRIVATE.

The launch staging the DF derivative kernels share. The Hessian kernels
(`d2metric`, `d2eri3`) read the gradient kernels' (`gmetric`, `geri3`,
`ipmetric`, `xeri3`) parameter block, with H in place of G, so both stage
through this:

- the aux records and the metric's aux shell pairs, each with its two atoms;
- mmd's bra records regrouped by shell pair, with the two primitives'
  exponents, and the three-center cells, each with its three atoms;
- the refusals of bases the kernels cannot take;
- W and Γ made Cartesian (`gradient/README.md`, "The build", steps 1 and 2).

`staging.cpp` checks that the gradient's and the Hessian's record layouts
(`gradient_bridge.h`, `hessian_bridge.h`) agree, and that the momentum
extents its interface spells out match the energy kernels' limits: the
interface cannot include the bridges, as no module interface may.

It is a module of its own, not a partition, because a partition belongs to
one named module and the gradient and the Hessian are two.
