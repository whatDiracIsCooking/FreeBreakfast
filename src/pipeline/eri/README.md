# Pipeline: Eri

The pipeline's two-electron integrals, chosen once by `[eri]` and built once
for every reader: integral-direct (J's and K's pair classes) or
density-fitted (the factor B = L⁻¹ (P|mn), M = L Lᵀ, of the orbital basis
over an auxiliary one, with that aux basis and L kept beside it).

## Module

`free_breakfast.pipeline.eri`, target `free_breakfast::pipeline::eri`. It
re-exports `core.wwr` (the product holds `Buffers`), J's and K's
integral-direct modules (for `CoulombPairClass` and `OrderedPairClass`), and
the `basis_set` and `primitive_pair_classes` steps, whose products its `View`
names. `io.basis`, `recipes.matrix_ao` and `integrals.eri.density_fitting`
are private: `run` is defined in the implementation unit.

## The step

```cpp
enum class EriMode : uint8_t { IntegralDirect, DensityFitting };

struct IntegralDirect {
  mutable std::vector<CoulombPairClass> j_classes;  // coulomb's
  mutable std::vector<OrderedPairClass> k_classes;  // exchange's
};

struct DensityFitted {
  Buffers::device<double> b;  // naux x nao^2, [Q][n][m], Q fastest
  int32_t naux;
  BasisSet aux;               // placed over the Molecule
  Buffers::device<double> l;  // naux x naux, M = L L^T, lower triangle
};

struct EriOutput {
  std::variant<IntegralDirect, DensityFitted> eri;
};

struct EriStep {
  static constexpr std::string_view kName = "eri";

  EriMode mode = EriMode::IntegralDirect;
  std::string aux_basis{};  // a compiled-in fitting set's name, or a path

  Result<EriOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput> in,
      DeviceResources &res) const;
};
```

`ScfStep`, `CasscfStep` and the rest read `EriOutput` as an ordinary `View` entry and
branch on the alternative, so one pipeline builds its integrals once and
cannot mix a fitted SCF with a direct CASSCF: one aux basis per pipeline, and
a mix is two runs.

### Integral-direct

`coulomb_pair_classes` and `ordered_pair_classes` over the shared
`PrimitivePairClassesOutput`, at their default thresholds. An `aux_basis` is
`InvalidConfig`.

**The classes are `mutable`.** `coulomb` and `exchange` write each class's
scratch (`std::span<CoulombPairClass>`, not `const`), one call at a time per
set. A product is read through a `const` `View`, and every reader of this one
is a device step, which the planner runs one at a time on the device chain
(`pipeline/README.md`), so the classes are written by one build at a time
whichever step reads them.

### Density-fitted

On `res.stream()`:

1. The aux basis: `aux_basis`, or, left empty, `<BasisSet::name()>-acCD-6-6`
   (below). Then `resolve_aux_basis`, and `place(stamp.primitive_normalized(),
   molecule)`, as the `BasisSet` step places the orbital basis.
2. B's and L's bytes, `naux · (nao² + naux)` doubles, against the device's free memory:
   past it, `Unsupported` naming the bytes, since the allocation would
   otherwise abort (`CreatePolicy`).
3. `metric` into a `MatrixAOPair` made for the aux basis, over its own
   `upload_spherical_transform_matrix`.
4. `three_center` into B, `[m][n][P]` with P fastest.
5. The spherical metric copied into `l`, its own `naux²` buffer, and
   `density_fit` of B over it, in place: `l` is then L, and B `[Q][n][m]`,
   what `df_coulomb` and `df_exchange` read
   (`src/integrals/eri/density_fitting/jk/README.md`). The product keeps B,
   L and the placed aux basis.

Each failure is the call's own, passed through: an unreadable aux file is
`IO`; an unknown name, a malformed set or a missing element is
`InvalidConfig`; an orbital shell above f, an aux shell above i, or
three_center's scratch past the device's memory is `Unsupported`, naming the
shell or the bytes; a metric that is not positive definite is `Numerical`.

**The default aux basis is keyed off `BasisSet::name()`**, not the config's
spelling of the basis, since only `resolve_basis`'s compiled-in branch sets
it (the SAD stamps' rule): a basis read from a path gets no default, even if
its document claims a carried name. A basis with no carried `-acCD-6-6` set
(6-31G, 6-31G*, STO-3G, MINAO, or any read from a path) and no `aux_basis` is
`InvalidConfig`, naming the basis and asking for `aux_basis`. The default is
picked here rather than in `pipeline/config` so that config never loads a
basis.

- **The aux basis and L are held in `DensityFitted`, not a second product.**
  The DF-RHF gradient (`pipeline/gradient`) needs both: its derivative
  integrals are over the placed aux basis, and its fitted density and pairs
  take L⁻ᵀ. Products are keyed by type, so a second `BasisSetOutput` cannot
  exist, and a product of their own would only add a type: they mean
  something only beside the B they were made with, so they ride in the same
  alternative, and an integral-direct pipeline holds neither. L costs naux²
  doubles beside B's naux nao²; the basis is host data.
- **No `SphericalTransformOutput`.** The orbital U is not read: `three_center`
  builds its own U and U_aux, and the metric needs only the aux basis's U,
  which the step uploads itself. A `View` entry the step does not read would
  only order the pipeline for no reason.
- **`Device`-shaped**, because B and the classes live on the device.
- **The transients are freed stream-ordered.** The metric pair is released at
  return, after the copy into `l` that reads it was queued on the same
  stream. `three_center` and `density_fit` each synchronize the stream
  internally (their READMEs), so this step is not the device contract's
  no-wait ideal any more than they are.

## The stack

`three_center`'s launches make the CUDA runtime reserve per-thread stack for
every thread the card can hold, and keep it for the context
(`three_center/README.md`, "The stack"). The worst frame in the family is
`(dd|g)`'s 13,856 B, about **1.4 GB** on a 10 GB RTX 3080, which cc-pVTZ /
cc-pVTZ-JKFIT reaches along with `(ff|g)`: the classes above `(dd|g)` are
emitted in FelizJueves's low-stack form, so `(ff|g)` is only 6,320 B (it was
70,232 B, and the reservation 7.3 GB, before that form existed). The step
adds only B and the metric, a few MB at water's size, but a card with less
free memory than the reservation wants fails that launch with `GpuRuntime`.

## Config

```toml
[eri]
mode = "density_fitting"      # default "integral_direct"
aux_basis = "cc-pvdz-jkfit"   # optional: a compiled-in fitting set, or a BSE JSON path
```

A missing or empty `[eri]` is filled in, integral-direct, for any step that
reads `EriOutput`. `aux_basis` is valid only under `"density_fitting"`. A path
is joined to the input directory only when it is not a compiled-in fitting
set's name (`is_builtin_aux_basis`), which is `BasisSetStep::basis`'s rule
over the other table (`pipeline/config`).

## Tests

`test/gtest/pipeline_eri_gpu.cpp`:

- The step's contract is checked by `static_assert`, in `test/pipeline/steps_host.cpp`.
- The step's B for water in cc-pVDZ / cc-pVDZ-JKFIT and in cc-pVTZ /
  cc-pVTZ-JKFIT (f orbitals, g fitting functions) is compared with the B that
  `density_fit` makes of PySCF's `int2c2e` and `int3c2e`
  (`test/gtest/data/density_fitting/`). That B is over PySCF's
  unit-normalised functions, and this tree's are N_i times those. The fitting
  norms D cancel exactly: the step's metric is D M D, its L is D L, and its B
  is L⁻¹ D⁻¹ D (P|mn) N_m N_n. So each element of the step's B is divided by
  N_m N_n, and the fitted (mn|ls) = BᵀB of the two is compared, to 1e-12 of
  its maximum (1.1e-13 in DZ, 1.9e-13 in TZ). B entry by entry is the wrong
  check: its last rows, the metric's smallest pivots, amplify the integrals'
  ~1e-14 differences by L's conditioning, to 5.5e-12 (DZ) and 9.8e-12 (TZ)
  of max|B|.
- Integral-direct holds as many J and K classes as `coulomb_pair_classes` and
  `ordered_pair_classes` make, and refuses an `aux_basis`.
- Density fitting with no `aux_basis` over cc-pVDZ is the B of
  cc-pVDZ-acCD-6-6; over STO-3G, or over cc-pVDZ read from its file, it is
  `InvalidConfig`.
- An aux basis lacking an element of the molecule is `InvalidConfig`, and an
  unknown name is `IO` that lists the compiled-in names.

## Files

| File | Role |
|------|------|
| `eri.cppm` | Module interface: `EriMode`, `IntegralDirect`, `DensityFitted`, `EriOutput`, `EriStep` |
| `eri.cpp` | `EriStep::run`, the default aux basis, and B's and L's build |
| `CMakeLists.txt` | The module library |
