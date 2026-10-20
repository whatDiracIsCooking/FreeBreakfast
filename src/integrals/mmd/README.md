# MMD

McMurchie–Davidson primitive-pair records: a `BasisSet` turned into the
per-class arrays the generated one-electron kernels read.

## Module

`free_breakfast.integrals.mmd`, also re-exported by `free_breakfast.integrals`.

## Design

For primitives α on A and β on B, the Gaussian product theorem gives one
Gaussian of exponent ζ = α + β centred at P = (αA + βB)/ζ. A record holds what
a kernel needs of that pair and cannot form itself:

| Field | Value |
|---|---|
| `zeta` | ζ = α + β |
| `p_x`, `p_y`, `p_z` | P |
| `pa_x`, `pa_y`, `pa_z` | P − A |
| `pb_x`, `pb_y`, `pb_z` | P − B |
| `s` | c_a c_b (π/ζ)^{3/2} |
| `eb` | β, the ket exponent (the kinetic kernel's) |
| `off_a`, `off_b` | the two shells' first Cartesian AO |

The kernels build the Hermite expansion E from these, including its
exp(−αβ/ζ |A−B|²) prefactor (as exp(ζ X_PA X_PB) per axis), so the host forms
neither. `eb` is read by the kinetic kernels only and `p_*` by the
nuclear-attraction kernels only, so one record set serves S, T and V.

**Every field is one value per record.** A vector is three flat arrays, one per
axis, named as the generated kernels name their parameters, so with one thread
per record a warp's lanes read consecutive doubles on every load. The kernels
take them as plain positional pointers.

**The fields are named once.** `PairFields<Real, Offset>` (`pair_fields.h`) is
the record's fourteen fields as members, over whatever holds each one, and
every holder of a class's records is an instance of it:

| Instance | Holds | Where |
|---|---|---|
| `PairFields<std::span<double>, std::span<int32_t>>` | the storage this module writes into (`PrimitivePairRecords`) | here |
| `PairFields<MirroredBuffer<double>, MirroredBuffer<int32_t>>` | the device owner's buffers (`PairFieldBuffers`) | `one_electron.upload` |
| `PairFields<const double *, const int32_t *>` | the kernels' pointers (base of `PrimitivePairClassView`) | `device_views.h` |

`apply_pair_fields(f, fields)` calls `f` with every field at once, in
declaration order, and `map_pair_fields(f, fields)` builds the `PairFields` of
`f(field)`: that is how one instance becomes another without naming a field.
`apply_pair_fields` is the only other place the names are listed, and it
static_asserts their count (`kPairFields`), so a field added to the struct but
not to it fails to compile rather than going silently unvisited.

The header is plain C++ so the launchers can `#include` it (through
`device_views.h`); this module includes it in its global module fragment and
re-exports the name, as `atom` does.

**The caller owns the storage.** `primitive_pair_classes` counts each class's
records; the caller sizes a `PrimitivePairRecords` (one span per field) per
class from that, and `write_primitive_pairs` fills them. That is what lets the upload hand
it pinned memory and copy it to the device as written, while this module links
no GPU. Storage that does not fit the basis (the wrong class list, or a span of
the wrong length) aborts before anything is written: the writer indexes into it.

**Classes are `la <= lb`.** A shell pair whose bra has the higher momentum is
stored swapped (B as the bra), so a basis whose p shells precede its s shells,
as H₂O's do, still lands in `(s|p)`.

**Diagonal pairs** (a shell against itself) have `off_a == off_b`, which is how
the kernels tell them apart: they write the lower triangle of the block once and
mirror it.

**Coefficients are taken as given** (see `Shell`): normalisation is the basis's
business (`BasisStamp::primitive_normalized`). Nothing is screened.

Host only; no GPU is linked, so it builds and is tested on a device-less box.

## Public API

| Symbol | Role |
|---|---|
| `PairFields<Real, Offset>`, `kPairFields` | The record's fields, named once (`pair_fields.h`, re-exported) |
| `apply_pair_fields(f, fields)`, `map_pair_fields(f, fields)` | Walk every field: call `f` with all of them, or map each into a new `PairFields` |
| `PrimitivePairClass` | One class: `la`, `lb` and its record count `n_rec` |
| `PrimitivePairRecords` | One class's caller-owned storage: `PairFields` of spans |
| `primitive_pair_classes(basis)` | `std::vector<PrimitivePairClass>`: every reached class, ascending, none empty. `BasisSet` already bounds `nao_cart()` to `INT32_MAX`; a class past `INT32_MAX` records aborts |
| `write_primitive_pairs(basis, classes, records)` | Fills each class's storage; aborts if it does not fit the basis |

## Tests

`test/gtest/mmd_host.cpp`.

## Files

| File | Role |
|------|------|
| `pair_fields.h` | `PairFields`, the single definition, for `#include` and `import` alike |
| `mmd.cppm` | Module interface: `PairFields` re-exported, its walkers, `PrimitivePairClass`, `PrimitivePairRecords`, `primitive_pair_classes`, `write_primitive_pairs` |
| `mmd.cpp` | The shell-pair sweep (one pass to count each class, one to write its records by index into the caller's storage), the swap, the per-record arithmetic |
| `CMakeLists.txt` | The interface library `free_breakfast.integrals.mmd.headers` and the module library |
