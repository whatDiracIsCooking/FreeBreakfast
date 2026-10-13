# io/xyz

The geometry reader: a standard XYZ document (count line, comment line, then
`symbol x y z` per atom) parsed into a `Molecule`. XYZ coordinates are
conventionally in Angstrom while `Molecule` stores bohr, so every coordinate is
multiplied by `bohr_per_angstrom` (from `free_breakfast.core.constants`) on
read — the one physical assumption baked in here.

XYZ carries no charge or multiplicity, so `parse_xyz` / `read_xyz` take both as
arguments (defaulting to a neutral singlet).

## Errors

Every failure is `ErrorKind::IO`: a missing or non-numeric count line, a
malformed atom line, an unknown element symbol, or a count that disagrees with
the number of atom lines. That departs from the layer's usual split (format
modules report `IO`, domain modules `InvalidConfig`) because XYZ has no separate
format module to report the syntax — this one module is both, and every one of
those failures means "this file is not a usable XYZ document", the `IO`
recovery action. Blank trailing lines and `\r\n` endings are tolerated.

Element symbols match case-insensitively (`CL`, `cl` and `Cl` are chlorine).

## Usage

```cpp
import free_breakfast.io.xyz;

auto water = free_breakfast::read_xyz("water.xyz");          // neutral singlet
auto cation = free_breakfast::parse_xyz(text, /*charge=*/1, /*multiplicity=*/2);
```

## Files

| File | Role |
|------|------|
| `xyz.cppm` | Primary interface `free_breakfast.io.xyz`; `export import`s the partition |
| `xyz_reader.cppm` | `:reader` — `parse_xyz(text)` / `read_xyz(path)` |
| `xyz_reader.cpp` | Implementation (the element-symbol table stays private here) |
| `CMakeLists.txt` | The `free_breakfast.io.xyz` module target |
