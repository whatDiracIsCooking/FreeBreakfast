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
| `bound` | I = pair_bound(ζ, la + lb), the record's undressed Schwarz bound: J's screen's `Ip` as a bra and `Iq` as a ket. Formed on the device (below). K reads its own, dressed bound instead ([`exchange/pairs`](../eri/integral_direct/exchange/pairs/README.md)) |
| `bound_ab`, `bound_ba` | the record's DRESSED Schwarz bound, FelizJueves's `ao.k.record_bounds` (the max over the first shell's Cartesians), as stored and swapped. Formed on the device (below) |

The kernels build the Hermite expansion E from these, including its
exp(−αβ/ζ |A−B|²) prefactor (as exp(ζ X_PA X_PB) per axis), so this module
forms neither. `eb` is read by the kinetic kernels only and `p_*` by the
nuclear-attraction kernels only, so one record set serves S, T and V.

**Every field is one value per record.** A vector is three flat arrays, one per
axis, named as the generated kernels name their parameters, so with one thread
per record a warp's lanes read consecutive doubles on every load. The kernels
take them as plain positional pointers.

**A class owns its records, on the device only.** `PrimitivePairClass` is
`la`, `lb`, `n_rec`, its own `DeviceStream` (`handle`) and one `n_rec`-long
device buffer per field, all on that stream. Every reader — S, T and V, J's
prune (`coulomb_pair_classes`), K's gather (`ordered_pair_classes`) — reads
them there; a test that wants them on the host copies them back
(`test/gtest/mmd_records.h`).

**The records are formed on the device; the host never writes one.** A basis
of N primitives has O(N²) records but only O(shells) shells and O(shells²)
shell pairs, so that is all the host forms and uploads:

- the **shell table**, once: each shell's primitives (exponents and
  coefficients, flattened, with each shell's first index), centre, first
  Cartesian AO and rank;
- per class, its **shell pairs** in sweep order (`i <= j`, swapped into
  `la <= lb`), each with its first record: the prefix of `K_a · K_b`.

A record is then named by its index in that enumeration (shell pair, then bra
primitive, then ket primitive). One thread per record finds its shell pair by
a binary search of the prefix and its two primitives by a division
(`mmd_launch.cu`'s `record_primitives`). The tables are staged in
`MirroredBuffer`s that live for the call only, so `primitive_pair_classes`
waits for every class's stream before it returns.

**Records are in descending ζ order, per class, and written once, in it.** On
the class's stream, after its shell pairs' upload and a fork from the shell
table's stream: `mmd_order` writes each enumerated record's key `-ζ` (from its
two exponents alone) and its index, and sorts them stably
([`src/recipes/sort`](../../recipes/sort/README.md)); then `mmd_records` writes record
`k` from enumerated record `perm[k]`, every field at once. Nothing is permuted
after. Equal ζ keep the enumeration order (shell pair, then bra primitive).
Within a class the
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
record order, kets through their class's sorted `ket_` copy). The K build does not:
its screen is proved against the DRESSED bound (contraction coefficients and
pair factor in), which depends on the pair's orientation, so the ordered
classes carry it ([`exchange/pairs`](../eri/integral_direct/exchange/pairs/README.md)).
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

**Classes are `la <= lb`** within one basis. A shell pair whose bra has the
higher momentum is stored swapped (B as the bra), so a basis whose p shells precede its s shells,
as H₂O's do, still lands in `(s|p)`.

**Between two bases nothing is swapped.** `mixed_primitive_pair_classes`
sweeps every ordered pair (i of `rows`, j of `cols`) into class (la of i, lb of
j). Its shell table is the two bases one after the other, each shell keeping
its own basis's AO offset and rank, so a record is formed by the same kernels
from the same kind of table. Its offsets index two different matrices' rows and
columns, so none is diagonal, and pair_bound, which depends on `la + lb` alone,
takes `(lb, la)`'s kernel when `la > lb`.

**Diagonal pairs** (a shell against itself) have `off_a == off_b`, which is how
the kernels tell them apart: they write the lower triangle of the block once and
mirror it. Such a pair's record `(α_i, α_j)` and its mirror `(α_j, α_i)` carry
the same P bit for bit. K's gather relies on it: a diagonal record has no
swapped copy of its own, its mirror stands in.

**P's numerator is one explicit FMA, its terms in exponent order.**
`mmd_records` forms `(αA + βB)/ζ` as `fma(e_hi, c_hi, e_lo·c_lo)/ζ`, the
larger-exponent term fused. The terms are ordered before either is formed, so a
record and its mirror run the same operations on the same operands: that is
what gives the diagonal mirror above. It is explicit rather than left to
contraction, so nvcc and HIP clang fuse the same product.

The FMA rounds once where a host's plain expression rounds twice, so P can sit
an ulp away from the host's, and the dressed bound magnifies that: its pair
factor `exp(ζ X_PA X_PB)` on a distant pair, and its Hermite polynomials where
two shells share an atom far from the origin (X_PA ≈ 0, but only to an ulp of
A). Against FelizJueves's unfused goldens the H₁₀ chain moves by up to ~5000 eps,
H₂O/cc-pVDZ by ~38. Those relative errors grow with the coordinates, but only on
bounds the same factors have already shrunk, so the absolute error stays tiny:
at most 2.4 × 10⁻¹⁵ (on a bound near 1). The 5000-eps record is 8 × 10⁻²³ on a
bound of 7 × 10⁻¹¹. So `ExchangeRecordBound` passes a record within 16 eps (the
kernel's own bar) or within 10⁻¹³ absolutely
([`exchange/pairs`](../eri/integral_direct/exchange/pairs/README.md)).

**Coefficients are taken as given** (see `Shell`): normalisation is the basis's
business (`BasisStamp::primitive_normalized`). Nothing is screened: S, T and
V read every record. J prunes a copy of them
(`coulomb_pair_classes`, [`../coulomb/`](../eri/integral_direct/coulomb/README.md)).

## Public API

| Symbol | Role |
|---|---|
| `PrimitivePairClass` | One class `(la, lb)`: `n_rec`, `handle`, one device buffer per field (the three bounds among them), `device_stream()`; `view()` gives a `PrimitivePairClassView` |
| `PrimitivePairClassView` | The class's device fields by pointer, for the launchers (`primitive_pair_class_view.h`, re-exported) |
| `mixed_primitive_pair_classes(res, rows, cols)` | The same, between two bases: every ORDERED shell pair (i of `rows`, j of `cols`), never swapped, so `la > lb` classes appear; `off_a` indexes `rows`' AOs and `off_b` `cols'`. What `overlap_mixed` reads, and nothing else: the symmetric builds would mirror its values |
| `primitive_pair_classes(res, basis)` | `Result<std::vector<PrimitivePairClass>>`: every reached class, ascending, none empty, each on a new `DeviceStream`, finished when it returns. `BasisSet` already bounds `nao_cart()` to `INT32_MAX`; a class past `INT32_MAX` records, or a basis past `INT32_MAX` primitives, aborts |

## Tests

`test/gtest/mmd_gpu.cpp`, reading the records copied back
(`test/gtest/mmd_records.h`, shared with the J, K and pipeline tests): each
record's fields against the Gaussian product theorem, the swap, the streams,
and a diagonal d shell's mirrored records bitwise mirrored, off the axes; the
order, on a shell whose exponents ascend (so the sort has to move them) and on
a class past one sort tile (2816 records), each record matched by its fields
to a host enumeration, stably sorted; and the bound, for every class of
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
| `mmd.cpp` | The shell-pair sweep (each class's shell pairs and their record prefix), the swap, the ranks, the shell and shell-pair tables staged and uploaded, the launches, and the wait |
| `mmd_bridge.h` | `PrimitivePairSource` and `PrimitivePairRecords`, and the records', the order's and the bounds' launchers, declared once for `mmd.cpp`'s GMF and the `.cu` |
| `mmd_launch.cu` | The key and record kernels, and the launchers: `mmd_order` (key and sort), `mmd_records` (every field but the bounds, in the sorted order), `mmd_bound` (the generated `pair_bound<la, lb>`), `ordered_pair_bound` (the generated `record_bound<la, lb>`), both tables checked against `kMaxAngularMomentum` |
| `kernels/pair_bound/` | The generated `pair_bound<la, lb>` units and their template header (`tools/emit_one_electron_kernels.py`) |
| `kernels/record_bound/` | The generated `record_bound<la, lb>` units, all nine ordered classes through d, and their template header (`tools/emit_exchange_kernels.py`) |
| `CMakeLists.txt` | The interface library `free_breakfast.integrals.mmd.headers`, the device library and the module library |
