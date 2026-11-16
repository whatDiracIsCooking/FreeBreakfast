# Pipeline: Molecule

The step that puts a calculation's geometry on the device: the `GeometryOutput`
`XyzStep` read (`../xyz/README.md`), at a charge and a multiplicity, into a
`MoleculeOutput`.

## Module

`free_breakfast.pipeline.molecule`, target `free_breakfast::pipeline::molecule`.
It re-exports `free_breakfast.molecule` and `free_breakfast.pipeline.xyz`, so
an importer can name the `Molecule` the step's output holds, and the
`GeometryOutput` it reads, without a second import.

## The step

```cpp
struct MoleculeOutput {
  Molecule molecule;
};

struct MoleculeStep {
  static constexpr std::string_view kName = "molecule";
  static constexpr std::array<std::string_view, 2> kKeys{"charge", "multiplicity"};

  int32_t charge{0};
  int32_t multiplicity{1};

  Result<MoleculeOutput> run(View<GeometryOutput> in, DeviceResources &res) const;
};
```

- **The head of `[molecule]`**, a table it shares with `XyzStep`: the table's
  `xyz` key is that step's, and `charge` and `multiplicity` are this one's
  (`../config/README.md`, "Tables"). A written `[molecule]` requests this
  step, and the planner fills `XyzStep` in from the same table.
- **It reads the geometry, not a file.** The file is read on the host by
  `XyzStep`, ahead of this step and off the device chain; this step only
  builds the `Molecule`, adding the charge and multiplicity XYZ does not carry.
- **`Device`-shaped**, because a `Molecule` uploads its nuclei at
  construction.
- **No load.** After an `[optimize]` it is `XyzStep` that is loaded from
  `final.xyz`, and this step runs over that geometry at `[molecule]`'s own
  charge and multiplicity, which the optimization's chain read too.
  `[molecule] from` is refused.

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

## Tests

`test/gtest/pipeline_molecule_gpu.cpp`: the step's `Molecule` holds the
geometry's atoms at its charge and multiplicity. The step's contract is
checked by `static_assert`, in `test/pipeline/steps_host.cpp`.

## Files

| File | Role |
|------|------|
| `molecule.cppm` | Module interface: `MoleculeStep`, `MoleculeOutput` |
| `CMakeLists.txt` | The module library |
