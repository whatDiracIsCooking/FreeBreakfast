# MMD

McMurchie–Davidson primitive-pair records: a `BasisSet` turned into the
per-class arrays the generated one-electron kernels read, on the device.

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

**A class owns its records.** `PrimitivePairClass` is `la`, `lb`, `n_rec` and
one `n_rec`-long `MirroredBuffer` per field, all on the class's own
`DeviceStream`. `primitive_pair_classes(res, basis)` counts each class's
records, allocates it on a new stream of `res.device()`, writes the records
straight into the pinned side, and queues every field's copy to the device on
that stream. It never synchronizes: the pinned side is kept for the class's
lifetime (a `MirroredBuffer`'s destructor waits out its copies), and a build
launches the class on the same stream, after the copy. The pinned side is also
where the tests read what was written.

**The launchers' view is a header.** `PrimitivePairClass` holds wwr buffers,
which only a module can name, so `view()` hands the launchers a
`PrimitivePairClassView`: the same fields as plain `const` pointers, defined
once in `primitive_pair_class_view.h`. The launchers `#include` it (through
`one_electron.upload`'s `device_views.h`); this module includes it in its
global module fragment and re-exports the name, so the two sides share one
entity (the `atom` pattern).

**Classes are `la <= lb`.** A shell pair whose bra has the higher momentum is
stored swapped (B as the bra), so a basis whose p shells precede its s shells,
as H₂O's do, still lands in `(s|p)`.

**Diagonal pairs** (a shell against itself) have `off_a == off_b`, which is how
the kernels tell them apart: they write the lower triangle of the block once and
mirror it.

**Coefficients are taken as given** (see `Shell`): normalisation is the basis's
business (`BasisStamp::primitive_normalized`). Nothing is screened.

## Public API

| Symbol | Role |
|---|---|
| `PrimitivePairClass` | One class `(la, lb)`: `n_rec`, one `MirroredBuffer` per field, `device_stream()`; `view()` gives a `PrimitivePairClassView` |
| `PrimitivePairClassView` | The class's device fields by pointer, for the launchers (`primitive_pair_class_view.h`, re-exported) |
| `primitive_pair_classes(res, basis)` | `Result<std::vector<PrimitivePairClass>>`: every reached class, ascending, none empty, each on a new `DeviceStream`. `BasisSet` already bounds `nao_cart()` to `INT32_MAX`; a class past `INT32_MAX` records aborts |

## Tests

`test/gtest/mmd_gpu.cpp`.

## Files

| File | Role |
|------|------|
| `primitive_pair_class_view.h` | `PrimitivePairClassView`, the single definition, for `#include` and `import` alike |
| `mmd.cppm` | Module interface: `PrimitivePairClass`, `primitive_pair_classes`, the view re-exported |
| `mmd.cpp` | The shell-pair sweep (one pass to count and size each class, one to write its records by index), the swap, the per-record arithmetic, the copies |
| `CMakeLists.txt` | The interface library `free_breakfast.integrals.mmd.headers` and the module library |
