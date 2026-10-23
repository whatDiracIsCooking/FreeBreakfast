# io/basis

The Basis Set Exchange (BSE) JSON reader: a domain module over `io/json` that
turns a BSE document into a geometry-free `BasisStamp` (from
`free_breakfast.basis`), and optionally places it onto a `Molecule`.

* `parse_basis(text)` / `read_basis(path)` — JSON → `BasisStamp`, keyed by
  atomic number. Elements the molecule does not use may be present.
* `parse_and_assign_basis(text, molecule)` — parse, then `place`.
* `parse_and_assign_aux_basis(text, molecule)` — the same for a
  density-fitting set; a coverage failure's message names the auxiliary role.
* `resolve_basis(spec)` — a **compiled-in set's name** (`sto-3g`) or a path to
  a BSE JSON file → `BasisStamp`. What a front end calls.
* `is_builtin_basis(spec)` — whether `spec` is a name rather than a path.

## The compiled-in sets

`data/basis/*.json` is wrapped verbatim into a generated module implementation
unit by this directory's `CMakeLists.txt`, so `resolve_basis("sto-3g")` parses
the bytes out of the binary. `builtin_basis.cpp.in` is that generator's
template and carries the reasoning; `data/basis/README.md` says which sets,
which elements, and why two of them stop short of krypton.

Resolution prefers the name, and spelling is folded — lowercased, with `-`,
`_` and spaces dropped — so `STO-3G`, `sto-3g` and `sto3g` are one name. The
characters that distinguish sets (`*`, `+`, parentheses) survive, so no alias
table is needed for the sets carried today. **A path is the escape hatch**: a
basis this build does not carry is read straight from disk with no rebuild,
which is why the name/path pair is one function rather than a mode switch. A
file whose own name collides with a compiled-in one is reached as `./sto-3g`.

**Only a compiled-in set comes back named.** `resolve_basis`'s compiled-in
branch sets `BasisStamp::name` to the set's canonical BSE spelling (`STO-3G`,
however it was asked for); `read_basis` and the path branch leave it empty,
whatever the document's own `name` says. The name is how the SAD guess finds
its baked atomic densities (`src/io/sad`), so it must only ever sit on shells
known to be that set's.

`is_builtin_basis` exists because the discrimination has to happen *before* a
relative path is anchored to a config file's directory — joining a name to
that directory would turn `sto-3g` into a file nobody wrote. Only
`free_breakfast.pipeline.config` needs it; everything else calls
`resolve_basis`.

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
| `basis_reader.cppm` | `:reader` — the entry points above |
| `basis_reader.cpp` | Implementation of the parse / place entry points |
| `builtin_basis.cpp.in` | Template for the generated unit: the compiled-in data, name folding, `resolve_basis` |
| `CMakeLists.txt` | The module target, and the `data/basis/` → `builtin_basis.cpp` generation |
