# Basis

Gaussian basis-set description: contracted shells, a basis set with AO indexing, the Cartesian-to-spherical transform, and the geometry-free stamp a basis-set file describes.

## Module

`free_breakfast.basis`

## Design

This module carries only the *description* of a Gaussian basis — the data an integral engine consumes — and evaluates no integrals itself. **No integral engine exists in this tree yet**; where the docs say "the GPU kernels", they mean the consumer this layout is designed for. Everything is in atomic units: shell centers in bohr (a `Vec<double, 3>`), primitive exponents in inverse bohr².

**Spherical basis, evaluated as Cartesian.** Every shell means the `2L+1` real solid harmonics of its angular momentum — there is no Cartesian-basis mode, and the BSE reader (`io/basis`) rejects Cartesian input. A shell of angular momentum L is nonetheless evaluated over the Cartesian Gaussians `x^a y^b z^c` with `a+b+c = L`, giving `(L+1)(L+2)/2` functions — 1 (s), 3 (p), 6 (d) — in the canonical lexicographic-on-(a,b,c)-with-a-highest order: `(x, y, z)` for p, `(xx, xy, xz, yy, yz, zz)` for d. `cartesian_powers(l)` enumerates those `(a, b, c)` triples in that order, for arbitrary L. `kMaxAngularMomentum` is 2 (the intended GPU-kernel L limit; nothing here enforces it); `function_count_cart(l)` returns the Cartesian count.

So a `BasisSet` has two parallel layouts: the Cartesian one the kernels evaluate (`BasisSet::aos_cart`, `ao_offset_cart`, `nao_cart`), `(l+1)(l+2)/2` per shell, and the spherical one the basis means (`BasisSet::nao_sphe`, `ao_offset_sphe`), `2l+1` per shell. `spherical_transform_matrix(basis)` bridges the two: a block-diagonal matrix `U` (Cartesian rows × spherical columns, column-major) holding each shell's real solid harmonic embedding (from `:harmonics`, scaled by `sqrt((2l-1)!!)` to reconcile this tree's Cartesian normalization with a unit spherical AO; an identity for s and p), so `U^T M U` converts any Cartesian-basis matrix `M` — an overlap, a nuclear-attraction matrix, … — into the same quantity over the spherical layout. The spherical columns follow PySCF/libcint (`m = -l..+l`); `test/gtest/basis_host.cpp` pins the d block against libcint's closed-form coefficients and checks `U^T S U = I` for a normalized spherical shell.

- **`Shell`** — one contracted Gaussian shell: an angular momentum L, a list of `Primitive{exponent, coefficient}`, and a `Vec<double, 3>` center. The constructor `check_or_abort`s (aborts rather than returns an error — the one production caller, the BSE reader in `io/basis`, already screens these) if L is negative, there are no primitives, or the zipping constructor's exponent and coefficient counts differ. There is no upper L cap. `function_count_cart()` is the Cartesian count for its L; `function_count_sphe()` is `2l+1`.
- **`AO`** — one Cartesian atomic orbital: an owning `shell_index` and its powers `(a, b, c)`. It stores no center or exponents (its shell resolves those).
- **`BasisSet`** — shells held in the order supplied, with AOs laid out by concatenating each shell's Cartesian functions in that same order. `ao_offset_cart(i)` returns shell i's first AO index (`check_or_abort`s on a bad index); `nao_cart()` is the total. `aos_cart()` is the same layout as a flat `AO` list. `nao_sphe()` / `ao_offset_sphe(i)` are the parallel spherical-layout accessors. A basis made by `place` also records which atom owns which shells — `natoms()`, `atom_shell_offset(a)`, `atom_shell_count(a)` — so per-atom work (the SAD guess in `src/sad`) reads the grouping instead of re-deriving it from shell centers. Built from a bare shell list, it carries none (`natoms() == 0`); the two-argument constructor `check_or_abort`s on offsets that do not partition the shells.
- **`ShellStamp`** — a `Shell` with the center removed: what a basis-set file specifies per element rather than per atom. Same shape contract as `Shell`, and `at(center)` makes a centered `Shell`. `primitive_normalized()` folds `N(α,l) = (2α/π)^{3/4}(4α)^{l/2}/√((2l−1)!!)` into the coefficients — center-independent, so computed once on the stamp rather than once per geometry.
- **`BasisStamp`** — the geometry-free basis *definition*: a `Z → shells` table of `ShellStamp`. **`place(stamp, molecule)`** stamps it onto a molecule's atoms — for each atom, in molecule order, the element's shell stamps at that atom's center — returning `Result<BasisSet>` with the per-atom shell grouping recorded (`ErrorKind::InvalidConfig` if an atom's element is absent from the stamp). Build one stamp, `place` it at each geometry.

The center-independent pieces live in plain headers the module includes in its global module fragment and re-exports, so CUDA / HIP code can `#include` them through `free_breakfast::basis::headers`: `kMaxAngularMomentum` in `angular_momentum.h`, and `function_count_cart`, `function_count_sphe` and `Primitive` in `basis.h`. Six partitions divide the rest: `:shell` (`Shell`), `:ao` (`cartesian_powers` and `AO`), `:harmonics` (the closed-form Schlegel–Frisch table `solid_harmonic_block`), `:basis_set` (`BasisSet` and `spherical_transform_matrix`), `:shell_stamp` (`ShellStamp`), and `:basis_stamp` (`BasisStamp` and `place`). `basis.cppm` aggregates all six and re-exports `free_breakfast.molecule`, so one import gives `Vec` and `Molecule` too.

Parsing a Basis Set Exchange JSON file into a `BasisStamp` is `free_breakfast.io.basis`'s job, not this module's.

## Usage

```cpp
import free_breakfast.basis;

using free_breakfast::Shell;
using free_breakfast::BasisSet;

const free_breakfast::Vec<double, 3> origin{0.0, 0.0, 0.0};

// A single-primitive s shell and a single-primitive p shell at the origin.
BasisSet basis({
    Shell(0, {{3.42, 0.15}}, origin),   // s: 1 AO
    Shell(1, {{0.62, 1.00}}, origin),   // p: 3 AOs
});

auto n     = basis.nao_cart();          // 4
auto off_p = basis.ao_offset_cart(1);   // 1  (p shell owns AOs [1, 4))
```

## Tests

`test/gtest/basis_host.cpp`.

## Files

| File | Role |
|------|------|
| `angular_momentum.h` | Center-independent, shared with device `#include`rs: `kMaxAngularMomentum` |
| `basis.h` | Center-independent, shared with device `#include`rs: `function_count_cart`, `function_count_sphe`, `Primitive` |
| `basis.cppm` | Primary module interface: aggregates the partitions and re-exports `free_breakfast.molecule` |
| `shell.cppm` | `:shell` — re-exports the header names, then defines `Shell` |
| `ao.cppm` | `:ao` — the canonical `cartesian_powers(int)` order and the `AO` descriptor |
| `harmonics.cppm` | `:harmonics` — the closed-form real-solid-harmonic table `solid_harmonic_block(l)` |
| `basis_set.cppm` | `:basis_set` — `BasisSet` (its Cartesian and spherical AO layouts) and `spherical_transform_matrix` |
| `shell_stamp.cppm` | `:shell_stamp` — the geometry-free `ShellStamp` |
| `basis_stamp.cppm` | `:basis_stamp` — the per-element `BasisStamp` plus `place(stamp, molecule) -> Result<BasisSet>` |
| `CMakeLists.txt` | The `free_breakfast.basis.headers` interface library and the module library |
