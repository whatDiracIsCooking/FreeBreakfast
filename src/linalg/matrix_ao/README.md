# MatrixAO

Square AO-basis matrices on the GPU, tagged at compile time by which layout of a `BasisSet` they are over, and a pair that holds one of each and converts between them through a shared transform.

## Module

`free_breakfast.linalg.matrix_ao`, also re-exported by `free_breakfast.linalg`.

## Design

Every shell in this tree means its `2l+1` real solid harmonics, but the integral kernels evaluate its `(l+1)(l+2)/2` Cartesian Gaussians (see `src/basis/README.md`). So any AO-basis matrix has two forms: Cartesian, `basis.nao_cart()` square, and spherical, `basis.nao_sphe()` square. The two coincide when no shell is above p.

- **`BasisLayout`** is `{Cartesian, Spherical}`, and `ao_count<L>(basis)` gives the order a layout has.
- **`MatrixAO<L>`** owns one `n × n` double matrix on the device, column-major with packed leading dimension `n`, so `data()` goes straight into a BLAS call. It is zero-initialised at construction and move-only (`NonCopyable`). The layout is the template parameter, so passing a Cartesian matrix where a spherical one is wanted is a compile error. It records its order and device, and nothing about what it holds. `CartesianMatrixAO` and `SphericalMatrixAO` are the two aliases.
- **`upload_spherical_transform_matrix(res, basis)`** builds U = `spherical_transform_matrix(basis)` on the device once per basis, as a `std::shared_ptr<const Buffers::device<double>>`: column-major `nao_cart × nao_sphe`. It is `nullptr` when U is the identity (no shell above p, or an empty basis), and then nothing is uploaded. (The host `spherical_transform_matrix` differs: it always builds U, an explicit identity in that case.) U is never written after the upload, so every pair over a basis shares one.
- **`MatrixAOPair`** holds a `CartesianMatrixAO` and a `SphericalMatrixAO` for one basis, co-owns that basis's U, and owns its own `nao_cart × nao_sphe` scratch. `cartesian()` and `spherical()` reach the members. Writing one member does not update or invalidate the other; you bring them into step with:

  | Product | Computes | gemmwich |
  | --- | --- | --- |
  | `cartesian_to_spherical(res)` | `spherical = Uᵀ cartesian U` | `OpX = N`, `m = nao_sphe`, `n = nao_cart` |
  | `spherical_to_cartesian(res)` | `cartesian = U spherical Uᵀ` | `OpX = T`, `m = nao_cart`, `n = nao_sphe` |

  Both read the same stored U. When U is the identity (a null U), both are a device-to-device copy, and the pair allocates no scratch.

**Which product suits which matrix.** U is tall, so only one direction is exact for a given kind of matrix, and the caller picks it:

- A matrix of **integrals** over AO pairs (overlap, core Hamiltonian, Fock) is produced Cartesian and wanted spherical: `cartesian_to_spherical`. The spherical AOs are fixed combinations of the Cartesian ones, so their integrals are the same combinations.
- A matrix **contracted against** AO pairs (a density) is produced spherical and wanted Cartesian by the kernels: `spherical_to_cartesian`. It gives the Cartesian matrix whose contraction with any Cartesian integral matrix equals the spherical one's contraction with that matrix's `cartesian_to_spherical`: `tr(U D Uᵀ F) = tr(D Uᵀ F U)`.

The other two directions (`cartesian_to_spherical` on a density, `spherical_to_cartesian` on a Fock) compute something, but not the same quantity in the other layout: a d shell's `r²`-like Cartesian component has no spherical counterpart, and nothing recovers it. The types do not encode which kind a matrix is; this rule lives here and in the doc comments.

**The two are not inverses.** Once a shell is above p, neither round trip returns the matrix it started from. Cartesian → spherical → Cartesian drops the `r²` components. Spherical → Cartesian → spherical gives `UᵀU M UᵀU`, and `UᵀU ≠ I`: U's columns are orthonormal under the Cartesian overlap (`Uᵀ S_cart U = I`), not under the plain dot product. A d shell's `xy` column, for one, is `√3` in the `xy` row, so `(UᵀU)` has a 3 there. What the pair preserves is the trace pairing above. `test/gtest/basis_host.cpp` pins `UᵀU ≠ I`.

**Scratch and streams.** Each pair has its own scratch and U is read-only, so different pairs may run on different streams. Calls on one pair must be ordered on one stream.

**Checks.** The constructor aborts if U does not fit the basis: null for a basis with a shell above p, non-null for one without, or the wrong element count. A U from another basis with the same orders is not detected. `cartesian_to_spherical` and `spherical_to_cartesian` return `InvalidConfig`, before enqueueing anything, if `res` is not on the pair's device.

**Queueing.** Both products are queued on `res.stream()` with no host synchronization, and make `res.dev_idx()` current for the call. `upload_spherical_transform_matrix` synchronizes once, after uploading U from a host staging buffer.

## Public API

| Symbol | Role |
|---|---|
| `BasisLayout` | `Cartesian` or `Spherical` |
| `ao_count<L>(basis)` | `nao_cart()` or `nao_sphe()` |
| `MatrixAO<L>(res, basis)` | One zero `n × n` device matrix in layout `L`; `n()`, `num_elements()`, `dev_idx()`, `data()` |
| `CartesianMatrixAO`, `SphericalMatrixAO` | The two `MatrixAO` aliases |
| `upload_spherical_transform_matrix(res, basis)` | `Result<std::shared_ptr<const Buffers::device<double>>>`: the shared device U, `nullptr` for the identity |
| `MatrixAOPair(res, basis, u)` | One of each layout; `cartesian()`, `spherical()`, `dev_idx()` |
| `MatrixAOPair::cartesian_to_spherical(res)` | `Status`: `spherical = Uᵀ cartesian U` |
| `MatrixAOPair::spherical_to_cartesian(res)` | `Status`: `cartesian = U spherical Uᵀ` |

## Usage

```cpp
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.linalg;
namespace fb = free_breakfast;

fb::DeviceResources res{0};
auto u = fb::upload_spherical_transform_matrix(res, basis);  // once per basis

fb::MatrixAOPair fock(res, basis, *u);
// ... the kernels write fock.cartesian() ...
fb::Status s = fock.cartesian_to_spherical(res);

fb::MatrixAOPair density(res, basis, *u);
// ... the SCF step writes density.spherical() ...
s = density.spherical_to_cartesian(res);
```

## Tests

`test/gtest/matrix_ao_gpu.cpp` covers the layout orders and zero initialisation, U's column-major upload and its absence below d, `cartesian_to_spherical` and `spherical_to_cartesian` against host sandwiches built from `spherical_transform_matrix` on an s+p+d basis, the trace pairing between two pairs sharing one U, the copy path below d, an empty basis, and the constructor's abort on a U that does not fit.

## Files

| File | Role |
|------|------|
| `matrix_ao.cppm` | Module interface: `BasisLayout`, `MatrixAO`, `upload_spherical_transform_matrix`, `MatrixAOPair` |
| `matrix_ao.cpp` | The upload of U, the pair's check on it, `cartesian_to_spherical` and `spherical_to_cartesian` |
| `CMakeLists.txt` | The module target `free_breakfast.linalg.matrix_ao` |
