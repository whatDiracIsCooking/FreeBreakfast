# `free_breakfast.integrals.eri.density_fitting.three_center`

`three_center(res, basis, classes, aux, b)`: the three-center integrals

    (mn|P) = ∫∫ m(r1) n(r1) P(r2) / r12

of an orbital `BasisSet` and an auxiliary one on the GPU, by
McMurchie–Davidson over the kernels `eri3<la, lb, lP>` that
[FelizJueves](../../../../../deps/FelizJueves) generates
(`deps/FelizJueves/docs/implementation/three_center.md`), spherical on all
three indices, into the caller's `b`. Link
`free_breakfast::integrals::eri::density_fitting::three_center`.

```cpp
auto aux = place(resolve_aux_basis("cc-pVTZ-JKFIT")->primitive_normalized(), molecule);
auto classes = primitive_pair_classes(res, basis);
Buffers::device<double> b(aux->nao_sphe() * basis.nao_sphe() * basis.nao_sphe(), res.device());
FBF_TRY(three_center(res, basis, *classes, *aux, b.data()));
FBF_TRY(density_fit(res, nao, naux, metric, b.data()));   // b becomes B in place
```

## Layout

`b` is `(P|mn)`, `naux` x `nao²` column-major: element `(P, n + nao m)` at
`P + naux (n + nao m)`, P fastest (`[m][n][P]`), `nao` and `naux` the two
bases' spherical counts. That is exactly what
[`density_fit`](../fit/README.md) takes as its `b`. It is exactly symmetric in
m and n.

## The build

1. **Bra records**: `mmd`'s `PrimitivePairClass`es (`zeta`, `P`, `P − A`,
   `P − B`, `s`), copied back and regrouped per class by shell pair,
   `(off_a, off_b)` ascending, `mmd`'s descending-ζ order kept within each
   pair: an `eri3` thread walks its pair's records as one run, and `mmd`'s
   order interleaves the pairs of a class. A pair whose bra has the higher
   momentum is stored swapped by `mmd`; the kernel's mirrored store writes
   both blocks either way, so nothing is flagged.
2. **Aux records**: the [metric](../metric/README.md)'s, one per aux
   primitive, `s = c (π/α)^{3/2}`.
3. **Cells**: one per (AO shell pair `i <= j`, aux shell), per class
   `(la, lb, lP)`, the pair outer and the aux shell inner, so neighbouring
   threads write neighbouring aux columns. The bra, aux and cell tables are
   field-major (`kThreeCenterBraFields`, `kThreeCenterAuxFields`,
   `kThreeCenterCellFields` in `three_center_bridge.h`), with U and U_aux,
   in one upload.
4. **One launch per class** that has a cell, all 50 classes through `(ff|g)`
   into one Cartesian tensor `nao_cart² naux_cart` doubles, `[m][n][P]`,
   P fastest, one thread per cell, `kFourWarps` to a block. Every element
   belongs to one cell and is a plain store with its `(m, n)` mirror, so
   there is no zeroing and no atomics.
5. **Spherical**: the aux index as one gemm with `U_auxᵀ` over the
   `naux_cart` x `nao_cart²` view; the n index as one strided-batched gemm
   with `U` over the `nao_cart` blocks of m; the m index as one gemm with `U`
   over the `(naux nao)` x `nao_cart` view.
6. **The average**: the last two gemms round `(m, n)` and `(n, m)` in
   different orders, so `b` is written as half of each element plus half of
   its mirror (`three_center_symmetrize`, out of place), which is exactly
   symmetric: the halving is exact and addition commutes.

The staging, the Cartesian tensor and the transform's intermediate come from
`res.device_workspace()`; the call synchronizes twice (after the records'
download and after the upload) and not after the launches.

**The normalisation** is the metric's: no factor beyond the bases' own
`primitive_normalized()` coefficients, because the spherical transforms
absorb what FelizJueves's libcint-convention `c` carries beyond ours
([`../metric/README.md`](../metric/README.md), "The normalisation"). The
three-center bra `s` is `mmd`'s, the same one J and K read.

## Limits

- Orbital shells through f (`kThreeCenterMaxMomentum`), aux shells through g
  (`kThreeCenterAuxMaxMomentum`); a higher one is `Unsupported`, naming the
  shell.
- **The whole tensor must fit, and is refused up front if not.** The staging,
  the Cartesian tensor and the intermediate (`naux · nao_cart²`) are counted
  before anything is allocated; past `max_bytes` (a test's forced limit, the
  default unlimited) or the device's free memory beyond what the workspace
  already holds, the call is `Unsupported`, naming the bytes. Splitting over
  m would need FelizJueves's mirrored store changed.
- `nao_cart² naux_cart` within `INT32_MAX`, the kernels' 32-bit index, else
  `Unsupported`.
- **The stack.** The unrolled high classes spill to stack: after device link
  on the RTX 3080 (sm_86), `(ff|g)` (`eri3_334`) takes 70,232 B a thread and
  `(df|g)` 32,040 B, at 255 registers. The CUDA runtime grows the context's
  per-thread stack at the first such launch, for every thread the card can
  hold (68 SMs x 1536, so about 7.3 GB for `(ff|g)`), and keeps it. It fit on
  the 10 GB card; on a card with less free, that launch fails with a
  GpuRuntime error rather than aborting. The size refusal above does not
  count this reservation. A low-stack eri3 is FelizJueves's to emit.
- No screening: every cell is launched.
- FelizJueves tested its kernels against libcint at `L <= 4` only; the
  classes above, to `(ff|g)` at L = 10, are first checked here, against
  PySCF.

## Tests

`test/gtest/df_three_center_gpu.cpp`, against `test/gtest/data/density_fitting/`'s
`int3c2e` for FelizJueves's water, each element divided by this tree's three
function norms (PySCF's are unit-normalised): cc-pVDZ / cc-pVDZ-JKFIT (to
`(dd|f)`) and cc-pVTZ / cc-pVTZ-JKFIT (to `(ff|g)`) to `1e-12 · max|(mn|P)|`,
exactly `(m, n)`-symmetric, two runs bit-identical; a g orbital shell and an h
aux shell `Unsupported`; the size refusal at a forced limit one byte short,
and the run at the limit.
