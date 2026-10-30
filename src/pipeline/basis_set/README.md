# Pipeline: BasisSet

Resolve a basis by name or path and place it over the `MoleculeOutput`'s molecule.

## Module

`free_breakfast.pipeline.basis_set`, target `free_breakfast::pipeline::basis_set`.
It re-exports `free_breakfast.basis` and the Molecule step, so an importer can
name everything in `run`'s signature without a second import.

## The step

```cpp
struct BasisSetOutput {
  BasisSet basis;
};

struct BasisSetStep {
  static constexpr std::string_view kName = "basis_set";

  std::string basis{};

  Result<BasisSetOutput> run(View<MoleculeOutput> in) const;  // resolve_basis, normalize, place
};
```

A thin wrapper over `io.basis`'s `resolve_basis` and `basis`'s `place`, with
the same error kinds:

- A name that is neither a compiled-in set nor an openable file is `IO`, and
  the message lists the compiled-in names.
- A syntactically invalid file is `IO`.
- A malformed basis is `InvalidConfig`.
- So is a basis that lacks an element the molecule uses, because a basis that
  cannot cover the molecule is a configuration error, not a smaller basis.

**The primitives are normalized here.** BSE coefficients are for normalized
primitives, and the integral engines (`integrals.mmd`'s records) take
coefficients as given. So the step places `BasisStamp::primitive_normalized`,
which is what a production run's `BasisSet` is.

**`Host`-shaped.** `place` reads only `Molecule::atoms()`, never the device
nuclei, so the step takes no `DeviceResources&`. The pipeline runs it as free
host work, beside the device chain.

**`basis` is a name or a path**, and `resolve_basis` prefers the name: a
compiled-in set (`sto-3g`, see `data/basis/README.md`) is parsed out of the
binary, anything else is read from disk. That is why the field is a
`std::string` and not a `std::filesystem::path` — the two are told apart by
`io.basis`, not by type.

**A path arrives resolved**, as for `MoleculeStep`: the front end decides what
a relative path in the TOML is relative to. A *name* must reach the step
untouched, so `pipeline/config` asks `is_builtin_basis` before it anchors
anything to the config file's directory.

## Not split into a stamp step, yet

`resolve_basis` returns a geometry-free `BasisStamp`, and `place` lays it over one
geometry. The split exists so one parse can serve many geometries. A pipeline
that runs once per geometry, like geom-opt's inner flow, would want a
`BasisStamp` step outside the loop and a placing step inside it. With one
geometry per run there is nothing to share, so this step does both. Split it
when geom-opt arrives.

## Tests

`test/gtest/pipeline_basis_set_gpu.cpp`:

- The step's contract is checked by `static_assert`, in `test/pipeline/steps_host.cpp`.
- The step's basis is checked, shell by shell, against a direct `read_basis`
  then `primitive_normalized` and `place` of the same file.
- A missing file is checked to be `IO`, and an uncovered element to be
  `InvalidConfig`.

The step itself never touches the device, but its input `Molecule` does, so the
tests need a GPU.

## Files

| File | Role |
|------|------|
| `basis_set.cppm` | Module interface: `BasisSetStep` |
| `CMakeLists.txt` | The module library |
