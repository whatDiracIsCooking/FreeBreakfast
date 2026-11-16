# io/xyz

The geometry reader: a standard XYZ document (count line, comment line, then
`symbol x y z` per atom) parsed into its atoms, a `std::vector<Atom>`. XYZ
coordinates are conventionally in Angstrom while an `Atom`'s position is in
bohr, so every coordinate is multiplied by `bohr_per_angstrom` (from
`free_breakfast.core.constants`) on read — the one physical assumption baked
in here.

**Host-only.** `parse_xyz` / `read_xyz` take no `DeviceResources` and return
no `Molecule`, whose nuclei live on the device: a caller that wants one builds
`Molecule(res, atoms, charge, multiplicity)` itself, with the charge and
multiplicity XYZ does not carry. So the pipeline reads the file in a host
step (`pipeline/xyz`), and the plan's shape counts its atoms without a
device. There is no device-returning twin.

## Errors

Every failure is `ErrorKind::IO`: a missing, non-numeric or zero count line (a
geometry has at least one atom), a malformed atom line, an unknown element
symbol, or a count that disagrees with the number of atom lines. That departs from the layer's usual split (format
modules report `IO`, domain modules `InvalidConfig`) because XYZ has no separate
format module to report the syntax — this one module is both, and every one of
those failures means "this file is not a usable XYZ document", the `IO`
recovery action. Blank trailing lines and `\r\n` endings are tolerated.

Element symbols match case-insensitively (`CL`, `cl` and `Cl` are chlorine).

## Usage

```cpp
import free_breakfast.io.xyz;

auto atoms = free_breakfast::read_xyz("water.xyz");    // Result<std::vector<Atom>>, bohr
auto more = free_breakfast::parse_xyz(text);
Molecule cation(res, *atoms, /*charge=*/1, /*multiplicity=*/2);  // on the device
```

## Files

| File | Role |
|------|------|
| `xyz.cppm` | Primary interface `free_breakfast.io.xyz`; `export import`s the partition |
| `xyz_reader.cppm` | `:reader` — `parse_xyz(text)` / `read_xyz(path)` |
| `xyz_reader.cpp` | Implementation (the element-symbol table stays private here) |
| `CMakeLists.txt` | The `free_breakfast.io.xyz` module target |

## Tests

`test/io/xyz_host.cpp`, host-only.
