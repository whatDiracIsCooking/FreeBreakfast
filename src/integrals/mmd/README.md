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
| `xp`, `yp`, `zp` | P, one array per axis |
| `xpa` … `zpa`, `xpb` … `zpb` | P − A, P − B, one array per axis |
| `s` | c_a c_b (π/ζ)^{3/2} |
| `eb` | β, the ket exponent (the kinetic kernel's) |
| `off_a`, `off_b` | the two shells' first Cartesian AO |

The kernels build the Hermite expansion E from these, including its
exp(−αβ/ζ |A−B|²) prefactor (as exp(ζ X_PA X_PB) per axis), so the host forms
neither. `eb` is read by the kinetic kernels only and P by the
nuclear-attraction kernels only, so one record set serves S, T and V.

**Structure of arrays.** Every field is one flat array with an entry per
record, and a vector is split per axis rather than interleaved `x y z`. The
kernels run one thread per record, so a warp's lanes then read consecutive
doubles of one array on every load. On the device the arrays travel as one
FelizJueves `PrimPairSoA` (`upload`).

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
| `PrimitivePairClass` | One class `(la, lb)`'s records as parallel arrays; `size()` is the record count |
| `primitive_pair_classes(basis)` | `std::vector<PrimitivePairClass>`: every reached class, ascending, none empty. Infallible: `BasisSet` already bounds `nao_cart()` to `INT32_MAX` |

## Tests

`test/gtest/mmd_host.cpp`.

## Files

| File | Role |
|------|------|
| `mmd.cppm` | Module interface: `PrimitivePairClass`, `primitive_pair_classes` |
| `mmd.cpp` | The shell-pair sweep, the swap, the per-record arithmetic |
| `CMakeLists.txt` | The module library |
