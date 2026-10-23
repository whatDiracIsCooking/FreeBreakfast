# IO: SAD stamps

The SAD atomic densities in `data/sad/`, compiled into the binary and looked
up by basis-set name and atomic number.

## Module

`free_breakfast.io.sad`, target `free_breakfast::io::sad`.

## Public API

| Symbol | Role |
|---|---|
| `builtin_sad_density(basis, z)` | `std::optional<std::vector<double>>`: element `z`'s baked density in the compiled-in set `basis`, n x n column-major, or nullopt |

Its signature is `sad`'s `SadStampLookup`. The Guess step passes it to
`sad_density`. `sad` takes it as a parameter rather than importing this
module because `tools/sad`, the generator of `data/sad/`, links `sad`. If
`sad` linked this module, the generator would be linked against its own output.

## Design

**Compiled in, as the basis sets are.** `CMakeLists.txt` globs
`data/sad/*/*.npy` at configure time. It reads each file as hex and writes its
bytes, every one a `\x` escape, into a `sv` string literal in
`builtin_sad.cpp`, which it generates from `builtin_sad.cpp.in`. The JSON of
`src/io/basis` can go in a raw string literal, but a `.npy` is binary. The
`sv` suffix makes the literal's own length the view's, so the NUL bytes in
every `.npy` header are kept. At runtime the lookup hands those bytes to
`parse_npy` (`free_breakfast.io.npy`). A parse takes microseconds; the SCF it
replaces takes milliseconds to minutes.

**Looked up by the canonical name, compared exactly.** The key is the
directory name, which the configure checks is the BSE `name` of a
`data/basis` document. The name it is compared with is `BasisSet::name()`,
which only `resolve_basis`'s compiled-in branch sets, always to that same
canonical spelling. So no spelling is folded here; `cc-pvdz` finds nothing.

**Bad bytes abort.** These bytes are this build's own, so one that does not
parse as a square matrix means a broken build. Falling back to the SCF would
only hide that. The tests parse every entry.

**An empty `data/sad/` is allowed.** The table is then empty, and every
element is computed. It is slower but still right, and it is the state the
generator first runs from.

## Tests

`test/gtest/io_sad_host.cpp`:

- Every element of every compiled-in set is baked, at the size its shells
  need.
- Nothing else is baked: not a non-canonical spelling, not an absent element.
- Each baked stamp equals the SCF recomputed exactly as the generator ran it,
  bitwise. The fast tier checks H, C and O of every set; the slow tier checks
  every element through argon, and through neon for cc-pVTZ.

`test/gtest/pipeline_guess_gpu.cpp` checks that the Guess step on a
compiled-in set takes these stamps.

## Files

| File | Role |
|------|------|
| `sad.cppm` | Module interface: `builtin_sad_density` |
| `builtin_sad.cpp.in` | Template for the generated implementation unit: the stamp table and the lookup |
| `CMakeLists.txt` | Generates `builtin_sad.cpp` from `data/sad/`, and the module library |
