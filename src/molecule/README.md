# Molecule

Molecular geometry (`Molecule`), the nuclear repulsion energy and its gradient.

## Module

`free_breakfast.molecule`

## Design

Everything is in atomic units: positions in bohr, energies in hartree, and a nucleus identified by its element (its atomic number Z). In Hartree atomic units the Coulomb prefactor `1/(4·π·ε₀)` is exactly 1, so the nuclear repulsion is a bare `Σ Z_A Z_B / R_AB` with no physical constants.

- **Points** are `wwr::double3` — the backend's own `double3` from WarpWraps' `wwr.vector_types`, so a host-built position and a kernel-built one share a layout. `distance(a, b)` is the Euclidean distance between two of them.
- **`Molecule`** — a `std::vector<Atom>` plus a net `charge` (elementary-charge units, positive for a cation) and a spin `multiplicity` (2S+1, default singlet). Geometry is fixed at construction. `total_nuclear_charge()` sums Z; `electron_count()` is `(Σ Z_A) − charge`.
- An `Atom`'s `position` is already a `double3` (`wwr::double3` names the same type); the free function `nuclear_charge(atom)` reads Z out of it.
- **`nuclear_repulsion_energy(molecule)`** — `Σ_{A<B} Z_A Z_B / R_AB`, each unordered pair counted once; zero for fewer than two atoms.
- **`nuclear_repulsion_gradient(molecule)`** — `dE_nn/dR`, flat `(natoms, 3)` row-major. Each pair contributes equal and opposite pulls, so the per-axis sum is zero by construction.

The module re-exports `free_breakfast.atom` and `wwr.vector_types` (`export import`), so importers get `Atom`, `Element` and `wwr::double3` without separate imports.

## Usage

```cpp
import free_breakfast.molecule;

using free_breakfast::Atom;
using free_breakfast::Element;
using free_breakfast::Molecule;

// H2 at 1.4 bohr separation (Atom and Element come through the re-export).
Molecule h2({Atom{{0.0, 0.0, 0.0}, Element::H},
             Atom{{0.0, 0.0, 1.4}, Element::H}});

auto n_elec = h2.electron_count();                            // 2
auto e_nn   = free_breakfast::nuclear_repulsion_energy(h2);   // 1/1.4 hartree
```

## Tests

`test/gtest/molecule_host.cpp`, including the gradient against a central finite difference of the energy.

## Files

| File | Role |
|------|------|
| `molecule.cppm` | Module interface: `distance`, `Molecule`, `nuclear_charge`, `nuclear_repulsion_energy`, `nuclear_repulsion_gradient` |
| `CMakeLists.txt` | Build configuration |
