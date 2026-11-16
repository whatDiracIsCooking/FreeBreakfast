# Pipeline: Xyz

The root step of a calculation: read an XYZ geometry's atoms, on the host,
into a `GeometryOutput`.

## Module

`free_breakfast.pipeline.xyz`, target `free_breakfast::pipeline::xyz`. It
re-exports `free_breakfast.atom` and `free_breakfast.molecule`: the
`GeometryOutput` holds `Atom`s, and `OptimizeOutput`, declared here, a
`Molecule`.

## The step

```cpp
struct GeometryOutput {
  std::vector<Atom> atoms;  // bohr, in file order
};

struct XyzStep {
  static constexpr std::string_view kName = "xyz";
  static constexpr std::string_view kTable = "molecule";
  static constexpr std::array<std::string_view, 1> kKeys{"xyz"};

  std::filesystem::path xyz{};

  Result<GeometryOutput> run(View<> in) const;  // read_xyz

  // After an [optimize]: the final.xyz it wrote, its atoms the optimum's.
  static Result<GeometryOutput> load(const std::filesystem::path &from,
                                     View<OptimizeOutput> in);
};

struct OptimizeOutput {  // OptimizeStep's product (pipeline/optimize)
  Molecule molecule;
  double energy;
};
```

A thin wrapper over `io.xyz`'s `read_xyz`, and nothing more: a missing or
malformed file is `ErrorKind::IO`, exactly as there.

- **A root step**, so it reads nothing (`View<>`). Its one option, the file,
  is its input.
- **It reads `[molecule]`.** `kName` is the step's identity (its log lines,
  the registry); `kTable` is the table its option is read from, which it
  shares with `MoleculeStep`, the table's head. So `[molecule] xyz = ...` is
  read as it always was, and `MoleculeStep` reads the rest of the table. A
  `[xyz]` table is no table: a written table requests its head only, and this
  step heads none.
- **The path is already resolved.** Which directory a relative path in the
  TOML is relative to is the front end's decision (`pipeline.config`: the
  input directory). It is made once, before any step exists, so the step
  holds a path it can open as given. There is no shared "input directory"
  product: that would be configuration routed through the graph, the first
  field of a context object.
- **`Host`-shaped**: the atoms are host data. The file is read beside the
  device chain rather than on it, and the nuclei's copy to the device is
  `MoleculeStep`'s, over this step's product.

## The load, and `OptimizeOutput`

A config with `[optimize]` plans the geometry **loaded** from the `final.xyz`
that step writes (`../optimize/README.md`, `../config/README.md`), so every
step reading `MoleculeOutput` runs at the optimum. The load's `View` reads
`OptimizeOutput`, which orders it after the optimization. A file whose atoms
are not the optimized molecule's, element by element, is `InvalidConfig`; an
unreadable one is `IO`, as `read_xyz` reports it. The front end plans this
load itself; `[molecule] from` is refused. The charge and multiplicity are
`[molecule]`'s, read by `MoleculeStep` as without an `[optimize]`.

`OptimizeOutput` is `OptimizeStep`'s product, but it is declared here,
because the load names it and `pipeline.optimize` imports every step that
imports this module. It holds only a `Molecule` and a `double`.

## Tests

The step's contract, its load's among it, is checked by `static_assert` in
`test/pipeline/steps_host.cpp`. Its run, against a direct `read_xyz` of the
same file and with a missing file `IO`, is `test/pipeline/xyz_host.cpp`'s, on
the host. The load needs an `OptimizeOutput`, whose `Molecule` is on the
device, so it is `test/gtest/pipeline_molecule_gpu.cpp`'s: the optimized
atoms read back, other atoms `InvalidConfig`, a missing file `IO`. It runs
after a real optimization in `test/gtest/pipeline_optimize_gpu.cpp`.

## Files

| File | Role |
|------|------|
| `xyz.cppm` | Module interface: `XyzStep`, `GeometryOutput`, `OptimizeOutput` |
| `CMakeLists.txt` | The module library |
