# io/basis

The Basis Set Exchange (BSE) JSON reader: a domain module over `io/json` that
turns a BSE document into a geometry-free `BasisStamp` (from
`free_breakfast.basis`), and optionally places it onto a `Molecule`.

* `parse_basis(text)` / `read_basis(path)` — JSON → `BasisStamp`, keyed by
  atomic number. Elements the molecule does not use may be present.
* `parse_and_assign_basis(text, molecule)` — parse, then `place`.
* `resolve_basis(spec)` — a **compiled-in set's name** (`sto-3g`) or a path to
  a BSE JSON file → `BasisStamp`. What a front end calls.
* `is_builtin_basis(spec)` — whether `spec` is a name rather than a path.
* `resolve_aux_basis(spec)` — `resolve_basis` over the compiled-in
  **density-fitting** sets (`cc-pVDZ-JKFIT`, `cc-pVTZ-JKFIT`) instead. The
  caller places it over the orbital basis's molecule with
  `place(stamp.primitive_normalized(), molecule)`; an element the set lacks is
  `InvalidConfig` from `place`.
* `is_builtin_aux_basis(spec)` — `is_builtin_basis` over the fitting sets.

## The compiled-in sets

`data/basis/*.json` is wrapped verbatim into a generated module implementation
unit by this directory's `CMakeLists.txt`, so `resolve_basis("sto-3g")` parses
the bytes out of the binary. `builtin_basis.cpp.in` is that generator's
template and carries the reasoning; `data/basis/README.md` says which sets,
which elements, and why two of them stop short of krypton.

`data/aux_basis/*.json`, the fitting sets, go into **a table of their own** in
the same generated unit, reached only through `resolve_aux_basis`. They are
not more orbital sets: `tools/sad` and `io_sad_host` treat every orbital set
as one that needs `data/sad/` stamps, and a SAD density in a fitting basis
means nothing. So `is_builtin_basis` and `resolve_basis` never see a fitting
set's name, and `resolve_aux_basis` never sees an orbital one; either, asked
for the other kind, reads it as a path. `data/aux_basis/README.md` says where
the sets come from.

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
that directory would turn `sto-3g` into a file nobody wrote.
`is_builtin_aux_basis` asks the same of the fitting table, for
`[density_fitting] aux_basis`. Only `free_breakfast.pipeline.config` needs
either; everything else calls `resolve_basis` or `resolve_aux_basis`.

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
| `builtin_basis.cpp.in` | Template for the generated unit: the compiled-in data, name folding, `resolve_basis`, `resolve_aux_basis` |
| `CMakeLists.txt` | The module target, and the `data/basis/` + `data/aux_basis/` → `builtin_basis.cpp` generation |
