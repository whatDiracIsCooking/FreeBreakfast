# Molecule

Molecular geometry (`Molecule`), its nuclei on the device, the nuclear repulsion energy, its gradient and its Hessian.

## Module

`free_breakfast.molecule`

## Design

Everything is in atomic units: positions in bohr, energies in hartree, and a nucleus identified by its element (its atomic number Z). In Hartree atomic units the Coulomb prefactor `1/(4·π·ε₀)` is exactly 1, so the nuclear repulsion is a bare `Σ Z_A Z_B / R_AB` with no physical constants.

- **Points** are `Vec<double, 3>` (from `free_breakfast.core.types`) — the backend's own `double3`, so a host-built position and a kernel-built one share a layout. `distance(a, b)` is the Euclidean distance between two of them.
- **`Molecule`** — a `std::vector<Atom>` plus a net `charge` (elementary-charge units, positive for a cation) and a spin `multiplicity` (2S+1, default singlet), and the nuclei on the device. Geometry is fixed at construction. `total_nuclear_charge()` sums Z; `electron_count()` is `(Σ Z_A) − charge`.
- An `Atom`'s `position` is already a `Vec<double, 3>`; the free function `nuclear_charge(atom)` reads Z out of it.
- **`nuclear_repulsion_energy(molecule)`** — `Σ_{A<B} Z_A Z_B / R_AB`, each unordered pair counted once; zero for fewer than two atoms.
- **`nuclear_repulsion_gradient(molecule)`** — `dE_nn/dR`, flat `(natoms, 3)` row-major. Each pair contributes equal and opposite pulls, so the per-axis sum is zero by construction.
- **`nuclear_repulsion_hessian(molecule)`** — `d²E_nn/dR dR`, flat `(3 natoms, 3 natoms)` row-major, row `3A + k` and column `3B + l`. Each pair adds `T` to its two diagonal blocks and `−T` to its two off-diagonal ones, so it is symmetric and each block row sums to zero by construction.

The module re-exports `free_breakfast.atom` and `free_breakfast.core.types` (`export import`), so importers get `Atom`, `Element` and `Vec` without separate imports.

**The nuclei on the device.** `Molecule(res, atoms, charge, multiplicity)` copies Z and the positions, in atom order, to `res`'s device once: one `MirroredBuffer` per field (`z`, `c`) on a new `DeviceStream` of `res.device()`, written straight into the pinned side and synchronized once, after the copies, so every class's stream can read them with no ordering of its own. A geometry is fixed for a `Molecule`'s lifetime, so one upload serves every build that reads it. A `Molecule` is movable, not copyable, and building one needs a GPU.

A molecule with no atoms aborts, as does a failed allocation (`CreatePolicy`) or copy: the copies only move data just created, so their failure means a broken context, not a bad input. Input that can be empty — an XYZ file — is refused as an `Error` by its reader before a `Molecule` is built.

**The view is a header.** The `MirroredBuffer`s are only nameable in a module, so `view()` hands the launchers a `NucleiView` (`n_nuc`, `z`, `c` as plain `const` pointers), defined once in `nuclei_view.h`. The launchers `#include` it through `free_breakfast::molecule::headers`; this module includes it in its global module fragment and re-exports the name, so the two sides share one entity (the `atom` pattern). A view is valid while its `Molecule` is alive. `device_stream()` is the nuclei's stream, and its `dev_idx()` the device they are on.

## Usage

```cpp
import free_breakfast.core.resources;
import free_breakfast.molecule;

using free_breakfast::Atom;
using free_breakfast::Element;
using free_breakfast::Molecule;

free_breakfast::DeviceResources res{0};
// H2 at 1.4 bohr separation (Atom and Element come through the re-export).
Molecule h2(res, {Atom{{0.0, 0.0, 0.0}, Element::H},
                  Atom{{0.0, 0.0, 1.4}, Element::H}});

auto n_elec = h2.electron_count();                            // 2
auto e_nn   = free_breakfast::nuclear_repulsion_energy(h2);   // 1/1.4 hartree
auto nuclei = h2.view();                                       // for a launcher
```

## Tests

`test/gtest/molecule_gpu.cpp`, including the gradient against a central finite difference of the energy and the nuclei read back from the device; `test/gtest/molecule_host.cpp` for `distance` and `nuclear_charge`, which need no `Molecule`. `nuclear_repulsion_hessian` is tested against PySCF's `hess_nuc` in `test/gtest/one_electron_hessian_gpu.cpp`, beside the one-electron Hessians it is summed with.

## Files

| File | Role |
|------|------|
| `molecule.cppm` | Module interface: `distance`, `Molecule`, `nuclear_charge`, `nuclear_repulsion_energy`, `nuclear_repulsion_gradient`, `nuclear_repulsion_hessian`, and `NucleiView` re-exported |
| `nuclei_view.h` | `NucleiView`'s single definition, for the module and the launchers |
| `CMakeLists.txt` | The interface library `free_breakfast.molecule.headers` and the module |
