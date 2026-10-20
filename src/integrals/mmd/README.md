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

**Two blocks per class, laid out once.** A class's twelve real fields are one
block of `kPairReals · n_rec` doubles, in `PairReal` order, and its two AO
offsets one block of `kPairOffsets · n_rec` `int32_t`, in `PairOffset` order;
`field_offset(f, n_rec)` is where field `f` starts. Those enums are the only
place the layout is written down: the host writer, the device owner
(`one_electron.upload`) and the tests all address fields through them.

**The caller owns the storage.** `primitive_pair_classes` counts each class's
records; the caller sizes a `PrimitivePairRecords` (two spans) per class from
that, and `write_primitive_pairs` fills them. That is what lets the upload hand
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
| `PairReal`, `kPairReals`, `PairOffset`, `kPairOffsets`, `field_offset(f, n_rec)` | The record layout |
| `PrimitivePairClass` | One class: `la`, `lb` and its record count `n_rec` |
| `PrimitivePairRecords` | One class's caller-owned storage: `reals` and `offsets` spans |
| `primitive_pair_classes(basis)` | `std::vector<PrimitivePairClass>`: every reached class, ascending, none empty. `BasisSet` already bounds `nao_cart()` to `INT32_MAX`; a class past `INT32_MAX` records aborts |
| `write_primitive_pairs(basis, classes, records)` | Fills each class's storage; aborts if it does not fit the basis |

## Tests

`test/gtest/mmd_host.cpp`.

## Files

| File | Role |
|------|------|
| `mmd.cppm` | Module interface: the layout, `PrimitivePairClass`, `PrimitivePairRecords`, `primitive_pair_classes`, `write_primitive_pairs` |
| `mmd.cpp` | The shell-pair sweep (one pass to count each class, one to write its records by index into the caller's storage), the swap, the per-record arithmetic |
| `CMakeLists.txt` | The module library |
