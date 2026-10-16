# io/basis

The Basis Set Exchange (BSE) JSON reader: a domain module over `io/json` that
turns a BSE document into a geometry-free `BasisStamp` (from
`free_breakfast.basis`), and optionally places it onto a `Molecule`.

* `parse_basis(text)` / `read_basis(path)` — JSON → `BasisStamp`, keyed by
  atomic number. Elements the molecule does not use may be present.
* `parse_and_assign_basis(text, molecule)` — parse, then `place`.
* `parse_and_assign_aux_basis(text, molecule)` — the same for a
  density-fitting set; a coverage failure's message names the auxiliary role.

Each `electron_shells` block maps to shells by pairing momentum *i* with
coefficient column *i* (an sp / spd block), or — with one momentum and several
columns — one shell per column (a general contraction).

**Only spherical basis sets are accepted**, because every shell in
`free_breakfast.basis` is spherical. `function_type` decides whether a block is
admitted:
`gto_spherical` is; `gto_cartesian` is rejected (`InvalidConfig`) at any l;
bare `gto` or no tag is spherical for l ≤ 1 (where the two coincide) and
rejected as ambiguous from d up; `sto` and anything else are rejected. The
Cartesian functions the integral kernels evaluate map onto this layout through
`spherical_transform_matrix` in `free_breakfast.basis`.

Classification follows the I/O layer's split: malformed JSON or an unreadable
file is `ErrorKind::IO`; a missing or wrong-typed field, a non-numeric
exponent string, or an element the molecule needs but the basis lacks is
`InvalidConfig`.

## Files

| File | Role |
|------|------|
| `basis.cppm` | Primary interface `free_breakfast.io.basis`; `export import`s `:reader` |
| `basis_reader.cppm` | `:reader` — the four entry points above |
| `basis_reader.cpp` | Implementation |
| `CMakeLists.txt` | The `free_breakfast.io.basis` module target |
