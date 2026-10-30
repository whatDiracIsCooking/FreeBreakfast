# io/molden

The [Molden format](https://www.theochem.ru.nl/molden/molden_format.html):
a molecule's atoms, its basis and a set of orbitals, as text that Molden, Jmol,
IboView, Multiwfn and PySCF's `pyscf.tools.molden.load` read. Write-only.

`molden_to_string(atoms, basis, c, energies, occupations)` and `write_molden(path, ...)`
take the atoms as a span (`Molecule::atoms()`) rather than a `Molecule`, which
uploads its nuclei on construction, so the writer stays host-only.

## What the writer converts

The orbitals come in this tree's spherical AO layout. Three things differ from
what a Molden reader expects:

* **Component order.** `BasisSet` orders a shell's spherical components
  `m = -l..+l` (PySCF/libcint). Molden wants `m = 0, +1, -1, +2, -2, ...`. The
  writer permutes each shell; p is `(x, y, z)` on both sides, so s and p are
  unchanged. There are no sign flips: the real solid harmonics in
  `src/basis/harmonics.cppm` match PySCF's, and PySCF's own Molden writer only
  permutes.
* **Coefficients in `[GTO]`.** The pipeline's shells have the primitive
  normalization `N(alpha, l)` folded into their coefficients
  (`BasisStamp::primitive_normalized`). `[GTO]` carries coefficients over
  normalized primitives, so the writer divides `N` back out. The result is the
  basis-set file's own coefficients.
* **AO normalization.** Nothing in this tree renormalizes a contraction, so an
  AO's norm is whatever its basis-set coefficients give: close to 1, not 1.
  Every Molden reader renormalizes the contractions it reads, so the writer
  multiplies each row of `C` by its AO's norm,
  `sqrt(sum_ij c_i c_j (2 sqrt(a_i a_j) / (a_i + a_j))^(l + 3/2))`. Without it
  each orbital would be off by a per-shell factor near 1, invisible in a plot.

The file always declares `[5D7F]` and `[9G]`. A shell above g is
`ErrorKind::Unsupported`: the format has no flag for it. Every orbital is
written `Spin= Alpha`, `Sym= A`.

## Tests

`test/io/molden_host.cpp`.

## Files

| File | Role |
|------|------|
| `molden.cppm` | Primary interface; `export import`s `:writer` |
| `molden_writer.cppm` | `:writer` — `molden_to_string`, `write_molden` |
| `molden_writer.cpp` | The implementation: the reordering and the two normalizations |
| `CMakeLists.txt` | The module library |
