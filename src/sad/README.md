# SAD

The Superposition-of-Atomic-Densities initial guess: per-atom aufbau occupations laid onto the diagonal of the molecular density matrix, in the spherical AO layout.

## Module

`free_breakfast.sad`

## Design

Pure host data with no GPU link, so it builds and tests on a device-less box. Its inputs are a `BasisSet` made by `place` and the `Molecule` it was placed over; its output is a `std::vector<double>`. The SCF step copies it to the device anyway, and a `wwr` host buffer would pull the GPU runtime into a module that has no other use for it.

**The model.** No atomic SCF is run. Each atom's Z electrons (the neutral atom's count) fill subshells in the Madelung order, `kAufbauOrder` (`1s 2s 2p 3s 3p 4s 3d 4p …` through `7p`, enough for Z ≤ 118). The plain aufbau order is used; the d/f anomalies (Cr, Cu, …) are not encoded, because a spherically averaged guess does not need them. The k-th subshell of momentum l goes on the atom's k-th shell of momentum l, and its electrons are spread evenly over that shell's `2l+1` spherical functions. Off-diagonal entries are zero.

**Why spherical.** Every shell in this tree is spherical (see `src/basis/README.md`). The real solid harmonics of one shell are orthogonal and share a norm, so an equal weight per m is a true spherical average. The Cartesian components are not like that: in this tree's normalization a d shell's `xy` has self-overlap 1/3 against `xx`'s 1, and the six Cartesians hide an s-like `x²+y²+z²`. A consumer that wants a Cartesian-layout density takes `U D Uᵀ` with `U = spherical_transform_matrix(basis)`. Density transforms the opposite way to the `Uᵀ M U` used for Fock and overlap matrices.

**What the trace means.** `trace(D)` is the placed electron count exactly: Σ_A Z_A when the basis has a shell for every occupied subshell, less any subshell it cannot seat. That is the orthonormal-AO picture. Shells of one atom overlap (1s with 2s), so the physical count `trace(D S)` is close to, but not exactly, the same number.

**Neutral atoms.** The guess places Σ_A Z_A electrons whatever the molecule's net charge. Rescaling or re-placing charge is the SCF driver's call.

**Which shells belong to which atom.** The basis says so: `place` records each atom's shell range in the `BasisSet` (`natoms`, `atom_shell_offset`, `atom_shell_count`), and this module reads it. A basis built from a bare shell list carries no grouping (`natoms() == 0`), and one placed over a different molecule has a different atom count; both are `ErrorKind::InvalidConfig`.

## Public API

| Symbol | Role |
|---|---|
| `SubshellFill` | One subshell as `{l, capacity}`, `capacity = 2(2l+1)`. |
| `kAufbauOrder` | The Madelung filling order, 19 subshells, covering Z ≤ 118. |
| `subshell_occupation(z)` | Electrons per subshell for a neutral atom, aligned to `kAufbauOrder`, summing to `z`. Aborts outside [0, 118]. |
| `atomic_diagonal(z, shells)` | One atom's diagonal occupations, one per spherical AO of `shells`. |
| `sad_diagonal(basis, molecule)` | `Result<std::vector<double>>`, length `nao_sphe()`; its sum is the placed electron count. |
| `sad_density(basis, molecule)` | `Result<std::vector<double>>`, the n×n column-major guess, n = `nao_sphe()`. |

## Usage

```cpp
import free_breakfast.basis;
import free_breakfast.sad;

auto basis = free_breakfast::place(stamp, molecule);  // records the atom grouping
if (basis) {
  auto density = free_breakfast::sad_density(*basis, molecule);
  // *density is nao_sphe x nao_sphe, column-major; trace == placed electrons.
}
```

## Tests

`test/gtest/sad_host.cpp`.

## Files

| File | Role |
|------|------|
| `sad.cppm` | Module interface: the occupation model, `sad_diagonal` and `sad_density` |
| `sad.cpp` | Implementation |
| `CMakeLists.txt` | Build configuration |
