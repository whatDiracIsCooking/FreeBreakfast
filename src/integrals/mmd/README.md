# MMD

McMurchie–Davidson primitive-pair records: a `BasisSet` turned into the
per-class arrays the generated one-electron kernels read, on the device, with
each record's Schwarz bounds, undressed and dressed, which the J and K screens
derive from.

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
| `ea` | α, the bra exponent |
| `eb` | β, the ket exponent (the kinetic kernel's) |
| `off_a`, `off_b` | the two shells' first Cartesian AO |
| `rank_a`, `rank_b` | each shell's index among the basis's shells of its momentum, in basis order |
| `bound` | I = pair_bound(ζ, la + lb), the record's undressed Schwarz bound: J's screen's `Ip` as a bra and `Iq` as a ket. Formed on the device (below). K reads its own, dressed bound instead ([`exchange/pairs`](../exchange/pairs/README.md)) |
| `bound_ab`, `bound_ba` | the record's DRESSED Schwarz bound, FelizJueves's `ao.k.record_bounds` (the max over the first shell's Cartesians), as stored and swapped. Formed on the device (below) |

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
that stream, then the order and the bounds below. It never synchronizes: the
pinned side is kept for the class's lifetime (a `MirroredBuffer`'s destructor
waits out its copies), and a build launches the class on the same stream,
after them.

**The pinned side is staging, one way.** The records flow host to device and
nothing is copied back: the order and the bounds exist on the device side
only, the pinned side keeps the records in the order the host wrote them, and
the bounds' pinned sides are never written. Every reader — S, T and V, J's
prune (`coulomb_pair_classes`), K's gather (`ordered_pair_classes`) — reads
the device side; a test that wants the records on the host copies them back.
The pinned side stays allocated by design: under AIMD the basis composition is
fixed, and the ζ order depends only on the basis, so a later step can rewrite
`p_*`, `pa_*` and `pb_*` at the same record indices and re-upload without
allocating pinned memory again. Rewriting it in place must first wait (an
event or a synchronization) for the previous upload from it; that entry point
does not exist yet.

**Records are in descending ζ order, per class.** On the class's stream, after
the upload: key `-ζ`, a stable merge sort carrying each record's index
([`src/integrals/sort`](../sort/README.md)), then every field permuted to that
order in place (`mmd_launch.cu`). Equal ζ keep
the order the host wrote (shell pair, then bra primitive). Within a class the
record bound rises with ζ, so this is descending `Ip`: a bra's ket loop runs until `Dmax · Ip · Iq < τ`, so the
longest loops launch first and a build's tail is short. No kernel depends on
the order (each accumulates), and it is fixed per basis, so it costs one sort
per class, once. It also spreads a shell pair's records across warps, so
`jab`'s and the one-electron scatters' atomic adds collide less.

**The bound is a field, formed once per basis.** `I = pair_bound(ζ, la + lb)`,
the pair-factor-free Schwarz bound of FelizJueves's `Target.PAIR_BOUND`, depends
on ζ and the class's `la + lb` only, so it belongs to the record, not to a
build: on the class's stream, after the order, the generated
`pair_bound<la, lb>` (`kernels/pair_bound/`, one thread per record, a Horner
fold of its closed form in √ζ) writes it from the ordered ζ. The host never
writes it. Because it rises
with ζ, it is non-increasing in record order. A class above
`kMaxAngularMomentum` has no kernel, so its bound stays zero; every build
refuses such a class, and `primitive_pair_classes` itself still accepts it. The J build reads it (bras in
record order, kets through `sort_bounds`' sorted copy). The K build does not:
its screen is proved against the DRESSED bound (contraction coefficients and
pair factor in), which depends on the pair's orientation, so the ordered
classes carry it ([`exchange/pairs`](../exchange/pairs/README.md)).
`tools/emit_one_electron_kernels.py` writes the kernels (its small `mmd`
section); `kernels/.clang-format` turns formatting off so they stay as emitted.

**So is the dressed bound, both ways.** `ao.k.record_bounds`, `max_{i∈a}
Σ_{k∈b} Σ_p |s E^{ik}_p| I_p(ζ)`, carries the coefficients and the pair factor,
and takes its max over the FIRST shell, so it depends on the orientation. On the
class's stream, after the order, the generated `record_bound<la, lb>`
(`kernels/record_bound/`, written by `tools/emit_exchange_kernels.py`, one
thread per record) writes `bound_ab` from the record as stored, and
`record_bound<lb, la>` writes `bound_ba` from a swapped view (`la`/`lb` and
`P − A`/`P − B` exchanged). The launcher, `ordered_pair_bound`
(`mmd_bridge.h`), is called by mmd alone; the bounds' readers sit above it and
read the fields: `coulomb_pair_classes` prunes by `bound_ab`, and
`exchange/pairs` gathers `bound_ab` and `bound_ba` into its ordered classes.

**The launchers' view is a header.** `PrimitivePairClass` holds wwr buffers,
which only a module can name, so `view()` hands the launchers a
`PrimitivePairClassView`: the same fields as plain `const` pointers, defined
once in `primitive_pair_class_view.h`. The launchers `#include` it (through
their bridge headers); this module includes it in its
global module fragment and re-exports the name, so the two sides share one
entity (the `atom` pattern).

**Classes are `la <= lb`.** A shell pair whose bra has the higher momentum is
stored swapped (B as the bra), so a basis whose p shells precede its s shells,
as H₂O's do, still lands in `(s|p)`.

**Diagonal pairs** (a shell against itself) have `off_a == off_b`, which is how
the kernels tell them apart: they write the lower triangle of the block once and
mirror it.

**Coefficients are taken as given** (see `Shell`): normalisation is the basis's
business (`BasisStamp::primitive_normalized`). Nothing is screened: S, T and
V read every record. J prunes a copy of them
(`coulomb_pair_classes`, [`../coulomb/`](../coulomb/README.md)).

## Public API

| Symbol | Role |
|---|---|
| `PrimitivePairClass` | One class `(la, lb)`: `n_rec`, one `MirroredBuffer` per field (the three bounds among them), `device_stream()`; `view()` gives a `PrimitivePairClassView` |
| `PrimitivePairClassView` | The class's device fields by pointer, for the launchers (`primitive_pair_class_view.h`, re-exported) |
| `primitive_pair_classes(res, basis)` | `Result<std::vector<PrimitivePairClass>>`: every reached class, ascending, none empty, each on a new `DeviceStream`. `BasisSet` already bounds `nao_cart()` to `INT32_MAX`; a class past `INT32_MAX` records aborts |

## Tests

`test/gtest/mmd_gpu.cpp`, reading the device side copied back: each record's
fields against the Gaussian product theorem, the swap, the streams, and the
pinned side as staging (the device side holds its values, reordered; the
pinned side keeps the host's order); the order, on a shell whose exponents
ascend (so the sort has to move them) and on a class past one sort tile (2816 records), each record matched by its
fields to the one the host wrote; and the bound, for every class of
H₂O/cc-pVDZ, within 16 eps of a host evaluation of its closed form (summed term
by term, not by the kernel's Horner fold) and non-increasing within each class;
`ea`, `eb` and the ranks against the basis; and `bound_ab` and `bound_ba`, for
every class of H₂O/cc-pVDZ, bit for bit against the bound
`ordered_pair_classes` forms for the same record as stored and swapped.

## Files

| File | Role |
|------|------|
| `primitive_pair_class_view.h` | `PrimitivePairClassView`, the single definition, for `#include` and `import` alike |
| `mmd.cppm` | Module interface: `PrimitivePairClass`, `primitive_pair_classes`, the view re-exported |
| `mmd.cpp` | The shell-pair sweep (one pass to count and size each class, one to write its records by index), the swap, the ranks, the per-record arithmetic, the copies, the order and the bounds |
| `mmd_bridge.h` | The order's and the bounds' launchers, declared once for `mmd.cpp`'s GMF and the `.cu` |
| `mmd_launch.cu` | The key and gather kernels, and the launchers: `mmd_order` (key and sort), `mmd_permute` (one field, in place), `mmd_bound` (the generated `pair_bound<la, lb>`), `ordered_pair_bound` (the generated `record_bound<la, lb>`), both tables checked against `kMaxAngularMomentum` |
| `kernels/pair_bound/` | The generated `pair_bound<la, lb>` units and their template header (`tools/emit_one_electron_kernels.py`) |
| `kernels/record_bound/` | The generated `record_bound<la, lb>` units, all nine ordered classes through d, and their template header (`tools/emit_exchange_kernels.py`) |
| `CMakeLists.txt` | The interface library `free_breakfast.integrals.mmd.headers`, the device library and the module library |
