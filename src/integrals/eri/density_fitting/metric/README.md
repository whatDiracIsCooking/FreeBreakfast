# `free_breakfast.integrals.eri.density_fitting.metric`

`metric(res, aux, m)`: the Coulomb metric

    (P|Q) = ∫∫ P(r1) Q(r2) / r12

of an auxiliary (fitting) `BasisSet` on the GPU, by McMurchie–Davidson over
the kernels `metric<lP, lQ>` that [FelizJueves](../../../../../deps/FelizJueves)
generates (`deps/FelizJueves/docs/implementation/metric.md`). It fills both
members of a `MatrixAOPair` made for `aux`: the Cartesian metric, and the
spherical one, which is what `density_fit` takes. Also re-exported by
`free_breakfast.integrals.eri.density_fitting` and `free_breakfast.integrals`.
Link `free_breakfast::integrals::eri::density_fitting::metric`.

```cpp
auto aux = place(resolve_aux_basis("cc-pVTZ-JKFIT")->primitive_normalized(), molecule);
MatrixAOPair m(res, *aux, *upload_spherical_transform_matrix(res, *aux));
FBF_TRY(metric(res, *aux, m));
// m.spherical().data(): naux x naux, column-major, exactly symmetric
```

## The build

1. **Records.** One per aux primitive, every shell's once, in one table that
   every class reads: the centre, the exponent α and the weight
   `s = c (π/α)^{3/2}`, `c` the coefficient as the `BasisSet` holds it.
   Field-major, `kMetricRecordFields` segments (`metric_bridge.h`).
2. **Pairs.** One per aux shell pair P ≤ Q by shell index, in its class
   (lP, lQ) with lP ≤ lQ. A pair whose first shell has the higher momentum is
   staged swapped, as FelizJueves's `metric_classes` does (metric.md, "The
   host reference"): no flag is needed, because the kernel stores every
   element with its mirror. Each pair is its two record ranges and its two
   shells' first Cartesian AOs, `kMetricPairFields` segments per class.
3. **One launch per class** that has a pair, every one into
   `m.cartesian()`, one thread per pair, `kFourWarps` to a block. The kernel's
   stores are plain: every element belongs to exactly one pair, so there is no
   zeroing, no atomics, and a run is bit-reproducible.
4. **The transform**, `m.cartesian_to_spherical(res)`: U^T M U with
   `upload_spherical_transform_matrix`'s U. Its blocks are general in l, and
   the aux g of cc-pVTZ-JKFIT is the first l = 4 they meet.
5. **The average.** U^T M U is two gemms, which round (a, b) and (b, a) in
   different orders: on water in cc-pVDZ-JKFIT the two differ by up to
   3.6e-15 of max|M|. `m.spherical()` is replaced by half of itself plus half
   of its transpose, in place by
   [`recipes.symmetrize`](../../../../recipes/symmetrize/README.md), which is
   exactly symmetric: each pair is summed once, and addition commutes.

The staging comes from `res.device_workspace()`. The
call synchronizes once, after the upload, so the host staging can go.

## The normalisation: no factor reconciles them, because none is needed

FelizJueves's weight is `s = c (π/α)^{3/2}` with no extra N (metric.md, "The
aux record"), where its `c` "already carries libcint's normalisation". Ours is
`ShellStamp::primitive_normalized()`'s. The two `c` differ, and the metric is
right with ours anyway:

- **FelizJueves's `c`** is its oracle's (`deps/FelizJueves/test/oracle/basis.py`):
  libcint's `_libcint_ctr_coeff` times `CINTcommon_fac_sp(l)`, so the file's
  coefficient times `gto_norm(l, α) = N(α, l) √(4π/(2l+1))`, times libcint's
  contraction normaliser, times 1/√(4π) (s), √(3/4π) (p) or 1 (d and up).
- **Ours** is the file's coefficient times
  `N(α, l) = (2α/π)^{3/4} (4α)^{l/2} / √((2l−1)!!)`, the (l, 0, 0) component's
  normaliser, and nothing else.

So for s and p the two are the same up to the contraction normaliser, and from
d up FelizJueves's is ours times `√(4π/(2l+1))` (and that normaliser): its
Cartesian functions are libcint's, ours are this tree's. That factor is not
applied here, because it belongs to the Cartesian *convention*, and each
convention carries its own way to the spherical functions. libcint's
cart-to-spherical matrix absorbs `√((2l+1)/4π)`; ours,
`spherical_transform_matrix`, is the solid-harmonic table scaled by
`√((2l−1)!!)`, which maps exactly this tree's N-normalised Cartesians onto
unit-normalised real solid harmonics (`src/basis/basis_set.cppm`). U^T M U
over our `c` is therefore PySCF's `int2c2e`, as the test measures: to
1.4e-14 of max|M| in both cc-pVDZ-JKFIT and cc-pVTZ-JKFIT, the g rows to
4.3e-14 of their own 0.87.

**The contraction normaliser** is the one real difference, and it vanishes
for every element the tests reach. PySCF renormalises each contracted
function to unit self-overlap; this tree takes the file's coefficients as
given (`Shell`). Every cc-pVDZ-JKFIT and cc-pVTZ-JKFIT shell of H, B–Ne,
Al–Ar and Kr is one primitive with coefficient 1, so the two agree there
exactly. Ga–Br's carry contracted fitting shells, whose functions are then off
unit norm by the file's contraction. That only rescales those fitting
functions, which the fitted (mn|P) M⁻¹ (P|ls) is invariant under, provided
(mn|P) is built over the same `BasisSet`; it shows only in a comparison of M
itself against PySCF.

## Limits

- Aux shells through i (`kMetricMaxMomentum`, the emitted table's extent,
  checked in `metric_launch.cu`); a higher one is InvalidConfig. That limit
  is the **aux** basis's and is deliberately above the orbital
  `kMaxAngularMomentum` (f): a JKFIT or RI set carries shells two l above the
  orbital set it fits, so the two ladders are separate knobs
  (`tools/emit/df_kernels.py`'s `AUX_MAX_L` against its `MAX_L`).
- `aux.nao_cart()²` within INT32_MAX, the kernels' 32-bit index.
- FelizJueves tested its kernels against libcint at lP + lQ ≤ 4 only, and
  the classes above that against its own host tower. All 19 of them are
  first checked here, against PySCF: (d|f) and (f|f) in both carried cases
  and (p|g), (d|g), (f|g) and (g|g) in the cc-pVTZ-JKFIT one, by
  `test/gtest/df_metric_gpu.cpp`; then every class through (i|i), by
  `test/gtest/df_high_aux_gpu.cpp`.
- **No carried fitting set reaches h or i.** Both `data/aux_basis/` sets stop
  at g, so the aux h and i classes are reachable only through a path-read
  basis or a hand-built `BasisSet` — which is what their test uses.

## Tests

`test/gtest/df_metric_gpu.cpp`, against `test/gtest/data/density_fitting/`'s
`int2c2e` for FelizJueves's water: the spherical metric of cc-pVDZ-JKFIT
(aux f) and cc-pVTZ-JKFIT (aux g) to 1e-12 · max|M|, the Cartesian and the
spherical metric exactly symmetric, two runs bit-identical; and the refusals,
whose unsupported shell is k, i being covered.

`test/gtest/df_high_aux_gpu.cpp` runs the rest of the ladder — all 28 classes
through (i|i), at L = 12 — over `test/gtest/data/df_high_aux/`'s hand-built
one-shell-per-l basis on two skew centres, against the same `int2c2e`, to the
same 1e-12 · max|M|. It clears that by two orders: the measured worst is
1.2e-14 of max|M| on the RTX 3080, so the tower does not visibly lose
accuracy up the aux ladder. The case prints the figure.
