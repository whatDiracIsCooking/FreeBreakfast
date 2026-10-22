# Pipeline: Molecule

The root step of a calculation: read an XYZ geometry into a `Molecule`.

## Module

`free_breakfast.pipeline.molecule`, target `free_breakfast::pipeline::molecule`.
It re-exports `free_breakfast.molecule`, so an importer can name the step's
output without a second import.

## The step

```cpp
struct MoleculeStep {
  static constexpr std::string_view kName = "molecule";

  std::filesystem::path xyz{};
  int32_t charge{0};
  int32_t multiplicity{1};

  Result<Molecule> run(View<> in, DeviceResources &res) const;  // read_xyz
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

## The device contract it does not keep

A device `run` should only enqueue on `res.stream()`, never synchronize the
host, and leave device selection to the pipeline (`step/README.md`). The
`Molecule` constructor breaks all three. It opens its own `ScopedDeviceIndex`,
copies the nuclei on a new `DeviceStream` of its own, and synchronizes that
stream before returning, because every class's stream reads the nuclei and none
is ordered after the stream the nuclei were copied on.

That costs one small blocking copy per geometry. The step holds its device's
chain while the copy runs, so the chain stalls for one copy of a few doubles per
atom. It is kept rather than worked around until the pipeline (stage 3) has an
event to hand out instead.

It also reads the file on the device chain. Splitting the read into a host step
would need a host-only XYZ parser in `io.xyz`, which returns a device-backed
`Molecule` today. That is worth doing when geom-opt makes the geometry an input
rather than a file, and not before.

## Tests

`test/gtest/pipeline_molecule_gpu.cpp`. The step's contract is checked by
`static_assert`. The step's `Molecule` is checked against a direct `read_xyz`
of the same file, and a missing file is checked to come back as `IO`.

## Files

| File | Role |
|------|------|
| `molecule.cppm` | Module interface: `MoleculeStep` |
| `CMakeLists.txt` | The module library |
