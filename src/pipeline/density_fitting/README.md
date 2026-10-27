# Pipeline: DensityFitting

The density-fitting factor B = L⁻¹ (P|mn), M = L Lᵀ, of the orbital basis
over an auxiliary one, built once per geometry for the DF J and K builds to
read.

## Module

`free_breakfast.pipeline.density_fitting`, target
`free_breakfast::pipeline::density_fitting`. It re-exports `core.wwr` (the
product holds `Buffers`) and the `basis_set` and `primitive_pair_classes`
steps, whose products its `View` names. `io.basis`, `recipes.matrix_ao` and
`integrals.eri.density_fitting` are private: `run` is defined in the
implementation unit.

## The step

```cpp
struct DensityFittingOutput {
  Buffers::device<double> b;  // naux x nao^2, [Q][n][m], Q fastest
  int32_t naux;
};

struct DensityFittingStep {
  static constexpr std::string_view kName = "density_fitting";

  std::string aux_basis{};  // a compiled-in fitting set's name, or a path

  Result<DensityFittingOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput> in,
      DeviceResources &res) const;
};
```

`run`, on `res.stream()`:

1. `resolve_aux_basis(aux_basis)`, then `place(stamp.primitive_normalized(),
   molecule)`, as the `BasisSet` step places the orbital basis.
2. B's bytes, `naux · nao²` doubles, against the device's free memory:
   past it, `Unsupported` naming the bytes, since the allocation would
   otherwise abort (`CreatePolicy`).
3. `metric` into a `MatrixAOPair` made for the aux basis, over its own
   `upload_spherical_transform_matrix`.
4. `three_center` into B, `[m][n][P]` with P fastest.
5. `density_fit` of B over the spherical metric, in place: B is then
   `[Q][n][m]`, what `df_coulomb` and `df_exchange` read
   (`src/integrals/eri/density_fitting/jk/README.md`).

Each failure is the call's own, passed through: an unreadable aux file is
`IO`; an unknown name, a malformed set or a missing element is
`InvalidConfig`; an orbital shell above f, an aux shell above g, or
three_center's scratch past the device's memory is `Unsupported`, naming the
shell or the bytes; a metric that is not positive definite is `Numerical`.

- **The aux basis is the step's own, not a second product.** Products are
  keyed by type, so a second `BasisSetOutput` cannot exist. A product of its
  own would only add a type, because nothing but this step reads the aux basis.
  It goes out of scope with the metric when `run` returns.
- **No `SphericalTransformOutput`.** The orbital U is not read: `three_center`
  builds its own U and U_aux, and the metric needs only the aux basis's U,
  which the step uploads itself. A `View` entry the step does not read would
  only order the pipeline for no reason.
- **`ScfStep` calls `run` itself** under `[scf] eri = "density_fitting"`,
  over the products its own `View` holds, and keeps the B as its own
  (`scf/README.md`): this step's product is for a reader that wants B
  without an SCF, and the SCF does not read it.
- **`Device`-shaped**, because B lives on the device.
- **The transients are freed stream-ordered.** The metric, which becomes L, is
  released at return, after the trsm that reads it was queued on the same
  stream. `three_center` and `density_fit` each synchronize the stream
  internally (their READMEs), so this step is not the device contract's
  no-wait ideal any more than they are.

## The stack

`three_center`'s `(ff|g)` class, reached by cc-pVTZ / cc-pVTZ-JKFIT, makes
the CUDA runtime reserve about 7.3 GB of per-thread stack on a 10 GB RTX 3080,
and keep it for the context (`three_center/README.md`, "The stack"). The step
adds only B and the metric, a few MB at water's size, but a card that has
less free memory fails that launch with `GpuRuntime`.

## Config

```toml
[density_fitting]
aux_basis = "cc-pvdz-jkfit"   # required: a compiled-in fitting set, or a BSE JSON path
```

A path is joined to the input directory only when it is not a compiled-in
fitting set's name (`is_builtin_aux_basis`), which is `BasisSetStep::basis`'s
rule over the other table (`pipeline/config`).

## Tests

`test/gtest/pipeline_density_fitting_gpu.cpp`:

- The step's contract is checked by `static_assert`.
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
- An aux basis lacking an element of the molecule is `InvalidConfig`, and an
  unknown name is `IO` that lists the compiled-in names.

## Files

| File | Role |
|------|------|
| `density_fitting.cppm` | Module interface: `DensityFittingOutput`, `DensityFittingStep` |
| `density_fitting.cpp` | `DensityFittingStep::run` |
| `CMakeLists.txt` | The module library |
