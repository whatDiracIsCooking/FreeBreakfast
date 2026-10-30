# Pipeline: Molecule

The root step of a calculation: read an XYZ geometry into a `MoleculeOutput`.

## Module

`free_breakfast.pipeline.molecule`, target `free_breakfast::pipeline::molecule`.
It re-exports `free_breakfast.molecule`, so an importer can name the
`Molecule` the step's output holds without a second import.

## The step

```cpp
struct MoleculeOutput {
  Molecule molecule;
};

struct MoleculeStep {
  static constexpr std::string_view kName = "molecule";

  std::filesystem::path xyz{};
  int32_t charge{0};
  int32_t multiplicity{1};

  Result<MoleculeOutput> run(View<> in, DeviceResources &res) const;  // read_xyz

  // After an [optimize]: the final.xyz it wrote, at its charge and multiplicity.
  static Result<MoleculeOutput> load(const std::filesystem::path &from,
                                     View<OptimizeOutput> in, DeviceResources &res);
};

struct OptimizeOutput {  // OptimizeStep's product (pipeline/optimize)
  Molecule molecule;
  double energy;
};
```

A thin wrapper over `io.xyz`'s `read_xyz`, and nothing more: a missing or
malformed file is `ErrorKind::IO`, exactly as there.

- **A root step**, so it reads nothing (`View<>`). Its options are its inputs:
  the file, and the charge and multiplicity XYZ does not carry.
- **The path is already resolved.** Which directory a relative path in the
  TOML is relative to is the front end's decision (`pipeline.config`: the config
  file's own directory). It is made once, before any
  step exists, so the step holds a path it can open as given. There is no
  shared "input directory" product: that would be configuration routed through
  the graph, the first field of a context object.
- **`Device`-shaped**, because a `Molecule` uploads its nuclei at
  construction.

## The load, and `OptimizeOutput`

A config with `[optimize]` plans the molecule **loaded** from the `final.xyz`
that step writes (`../optimize/README.md`, `../config/README.md`), so every
step reading `MoleculeOutput` runs at the optimum. The load's `View` reads
`OptimizeOutput`, which orders it after the optimization, and supplies the
charge and multiplicity the file does not carry. A file whose atoms are not
the optimized molecule's, element by element, is `InvalidConfig`; an
unreadable one is `IO`, as `read_xyz` reports it. The front end plans this
load itself; `[molecule] from` is refused.

`OptimizeOutput` is `OptimizeStep`'s product, but it is declared here, because
the load names it and `pipeline.optimize` imports every step that imports this
module. It holds only `Molecule` and a `double`, so it needs nothing more than
this module already imports.

## The device contract it does not keep

A device `run` should only enqueue on `res.stream()`, never synchronize the
host, and leave device selection to the pipeline (`step/README.md`). The
`Molecule` constructor breaks all three. It opens its own `ScopedDeviceIndex`,
copies the nuclei on a new `DeviceStream` of its own, and synchronizes that
stream before returning, because every class's stream reads the nuclei and none
is ordered after the stream the nuclei were copied on.

That costs one small blocking copy per geometry. The step holds its device's
chain while the copy runs, so the chain stalls for one copy of a few doubles per
atom. The pipeline waits on `res.stream()` after every device step anyway, so
this is one wait more, not lost overlap; it stays until events replace those
waits.

It also reads the file on the device chain. Splitting the read into a host step
would need a host-only XYZ parser in `io.xyz`, which returns a device-backed
`Molecule` today. Geometry optimization did not need one: its chain builds each
trial `Molecule` from atoms in memory, with a root step of its own.

## Tests

`test/gtest/pipeline_molecule_gpu.cpp`. The step's contract, its load's among it, is checked by
`static_assert`, in `test/pipeline/steps_host.cpp`; the load itself runs in
`test/gtest/pipeline_optimize_gpu.cpp`. The step's `Molecule` is checked against a direct `read_xyz`
of the same file, and a missing file is checked to come back as `IO`.

## Files

| File | Role |
|------|------|
| `molecule.cppm` | Module interface: `MoleculeStep`, `OptimizeOutput` |
| `CMakeLists.txt` | The module library |
