# Core Hamiltonian

The Cartesian core Hamiltonian H = T + V on the GPU, formed in place from the
`kinetic` and `potential_en` builds.

## Module

`free_breakfast.integrals.one_electron.core_hamiltonian`, also re-exported by
`free_breakfast.integrals.one_electron` and `free_breakfast.integrals`.

## Design

`core_hamiltonian` runs `check_one_electron_build` and `check_nuclei`
(`upload`) first, so a refused input launches nothing. Then `kinetic` zeroes H
and writes T, and `add_nuclear_attraction` adds V on top. Both kernels only add,
so T + V is formed without a T or V matrix of its own and without a separate
addition.

No kernels of its own, so no bridge header, launcher or device library: the
module and its implementation unit are the whole directory.

## Public API

| Symbol | Role |
|---|---|
| `core_hamiltonian(res, classes, nuclei, h)` | `Status`: the Cartesian H = T + V into `h` |

## Usage

```cpp
import free_breakfast.integrals.one_electron.upload;
import free_breakfast.integrals.one_electron.core_hamiltonian;
import free_breakfast.linalg.matrix_ao;
namespace fb = free_breakfast;

const auto device = fb::upload_primitive_pairs(res, *classes); // once per basis
const auto nuclei = fb::upload_nuclei(res, molecule);          // once per geometry
fb::CartesianMatrixAO h(res, basis);
fb::Status status = fb::core_hamiltonian(res, *device, *nuclei, h);
```

## Tests

`test/gtest/core_hamiltonian_gpu.cpp`: H₂O in STO-3G and cc-pVDZ against
PySCF's `hcore` at 1e-12 · max|H|, plus the zeroing and the class-range
refusal.

## Files

| File | Role |
|------|------|
| `core_hamiltonian.cppm` | Module interface: `core_hamiltonian` |
| `core_hamiltonian.cpp` | Implementation |
| `CMakeLists.txt` | The module `free_breakfast.integrals.one_electron.core_hamiltonian` |
