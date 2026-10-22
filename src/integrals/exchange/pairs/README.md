# Exchange pairs

The ordered primitive-pair classes K's bras and kets are drawn from: `mmd`'s
`la <= lb` records in both orientations, one class per ordered `(la, lb)`,
grouped by first shell, each record with K's **dressed** Schwarz bound and
each group sorted by it, built once per basis.

## Module

`free_breakfast.integrals.exchange.pairs`.

## Why here, not in `mmd`

K is the only consumer. The one-electron matrices and J are symmetric inside a
pair and want `mmd`'s one class per unordered pair of momenta; K is not, so it
wants both orientations of every pair, grouped by the shell that is its output
index, and a record bound that depends on the orientation. Putting that in
`mmd` would make every build pay for records only K reads, and widen `mmd`'s
surface for one caller. If another build ever wants
ordered records, this moves into `mmd` as a partition.

## Design

K_ab = Σ_cd D_cd (ac|bd), by FelizJueves's `kernel_kblock`: its bra `(a, c)` and
ket `(b, d)` are **ordered** shell pairs, and the first shell of each is an
output index (`a` the row, `b` the column). So:

- **Each off-diagonal `mmd` record gives two.** Record `(a, b)` of `mmd`'s
  `(la, lb)` goes to the ordered `(la, lb)` as stored, and to `(lb, la)` with
  `pa_*` and `pb_*`, and `off_a` and `off_b`, swapped. ζ, P and `s` are
  orientation-free and copied as they are. For `la == lb` both land in the
  one class.
- **A diagonal record** (`off_a == off_b`, a shell against itself) gives one.
- So a class holds one record per primitive pair of every ordered shell pair
  of its momenta: all nine classes through d, (p, s) as well as (s, p).

The fields keep `mmd`'s names, with "a" the record's **first** shell and "b"
its second. As the generated `kb<la, lc, lb, ld>` (`../kernels/kb/`) names its
parameters, a bra class passes `p_*`, `zeta`, `pa_*`, `pb_*`, `s` and
`record_bound` as `p_*`, `zeta_ac`, `pa_*`, `pc_*`, `s_ac` and `Ip`; a ket class
the same fields as `q_*`, `eta`, `qb_*`, `qd_*`, `s_bd` and `Iq`. `mmd`'s `eb`
is not carried (kb does not read it), and neither is its undressed `bound`
(below). Two fields are new:

| Field | Value |
|---|---|
| `record_bound` | K's dressed Schwarz bound, FelizJueves's `ao.k.record_bounds`: `I = max_{i in a} Σ_{k in b} Σ_p \|s E^{a_i b_k}_p\| I_p(ζ)`. Formed on the device (below) |
| `rank_b` | the second shell's rank: its index among the basis's shells of momentum `lb`, in basis order |

kb's `Dcd` row is `dc[bra] + dd[ket]`, and the exchange helpers' gather
(`../helpers/exchange_helpers_bridge.h`) lays `Dcd` out so that is
`rank_c * n_shells(ld) + rank_d`: a bra's `dc` is its `rank_b` times the number
of `ld` shells, a ket's `dd` its `rank_b`. The first shell's rank is not
stored: a K block is per first shell (below), and that shell is the group's.

**The bound is dressed, and per orientation.** kb's screen and its
guarantee (FelizJueves #484/#485: every skipped term under `τ`) are proved
against `ao.k.record_bounds`, which carries the contraction coefficients and
the pair factor `exp(-αβ/ζ |AB|²)` (through `s` and the dressed `E`), and takes
its max over the FIRST shell's Cartesians, summing over the second's. So it is
neither `mmd`'s `bound` (`Σ_p I_p(ζ)`, which has neither factor and can fall
short of it: up to 1.2× on H₂O/cc-pVDZ, an s shell with itself) nor the same
for `(a, b)` and `(b, a)`. It decays with `|AB|`: on the H₁₀ chain 54 of 2,500
records are exactly `0.0`, the pair factor underflowed. It is formed once per
basis, on each class's stream after its upload, by the generated
`record_bound<la, lb>` (`kernels/record_bound/`, FelizJueves's
`Target.RECORD_BOUND`, one thread per record in 32-thread blocks, written by
`tools/emit_exchange_kernels.py`; `kernels/.clang-format` keeps them as
emitted). A class above d has no kernel and keeps zeros; the K build refuses
it.

**Grouped by first shell, each group sorted by the bound.** Each class's
records are grouped by their first shell, the groups in basis order. Then, on
the device, each group is sorted by `record_bound`, non-increasing, equal
bounds in gather order: two stable merge sorts
([`src/integrals/sort`](../../sort/README.md)), every record by `-I`, then the
result by its group's index, which leaves each group in the first sort's
order; every field is permuted to it in place and the pinned side refreshed.
That is the order both of kb's screen `break`s (`Ip · Iq · F < τ` down a
block's kets, and `Ip · Iq[ket0] · F < τ` down its bras) need: **an unsorted
group is a wrong K, not a slow one.** The groups do not move, so neither do
`group_start`, the host's group table or any record's first shell. The sort
is once per basis, never per build. (Before the dressed bound, `mmd`'s ζ order
was enough: its bound rises with ζ. The dressed one does not follow ζ.) A K
block takes one bra group and one ket group whole, since all its bras share
the `a` shell whose row the block is, and its kets the `b` shell whose column.
The group table is on the host, one `OrderedPairGroup {shell, ao_offset,
start, count}` per group: `shell` the first shell's index in
`BasisSet::shells()`, `ao_offset` its first Cartesian AO, the records `[start, start + count)`. Its bounds are also on the device, as
`group_start` (`groups.size() + 1` entries, the last `n_rec`), from which the
K build makes its block tables there.

**The gather is on the host, the bound and the sort on the device, once per
basis.** `ordered_pair_classes` synchronizes each `mmd` class's stream (its
pinned side holds the zeta-ordered records only after), walks its records in
order, appends each orientation to its ordered class's group for its first
shell, then writes each ordered class into its pinned side and queues the
copies to the device on a new `DeviceStream`, then the bound, the sort and the
refresh above. The group table and the ranks are host data anyway, and the
gather is a few array copies. Every class's stream is synchronized before the
return, so the pinned side is final (the K build reads its shells' AO offsets
from the group table, not from it). A record whose AO offset starts no shell of the basis (the `mmd`
classes are another basis's) is an `InvalidConfig` error.

## Public API

| Symbol | Role |
|---|---|
| `OrderedPairClass` | One ordered class `(la, lb)`: `n_rec`, one `MirroredBuffer` per field (`record_bound` and `rank_b` among them), `groups` and their bounds `group_start`, `device_stream()`; `view()` gives an `OrderedPairClassView` |
| `OrderedPairGroup` | One group: `shell`, `ao_offset`, `start`, `count` |
| `OrderedPairClassView` | The class's device fields by pointer, for the launchers (`ordered_pair_class_view.h`, re-exported) |
| `ordered_pair_classes(res, basis, classes)` | `Result<std::vector<OrderedPairClass>>` from `primitive_pair_classes(res, basis)`: every reached ordered class, ascending `(la, lb)`, none empty, each on a new `DeviceStream` |

## Tests

`test/gtest/exchange_pairs_gpu.cpp`, on H₂O in STO-3G and cc-pVDZ and the
H₁₀ chain: each class's record count against a host count of ordered
primitive pairs; each group contiguous, one first shell of momentum `la`, the
groups in basis order, each group's count right and its `record_bound`
non-increasing and non-negative, and `group_start` its bounds; each record's
`rank_b`; the `(la, lb)` and `(lb, la)` records the same multiset of
(ζ, P, P − A, P − B, `s`, offsets) with the two orientations swapped; the
device side matching the pinned side; and another basis's classes refused.
On H₂O/cc-pVDZ and the chain, every record's `record_bound` against
FelizJueves's `ao.k.record_bounds` of the same ordered shell pair
(`test/gtest/data/record_bound/`, written by
`tools/emit_record_bound_golden.py`), matched by its two shells and ζ, within
16 eps (FelizJueves's bar for the kernel), a host zero exactly zero: worst
2.95 eps on H₂O/cc-pVDZ (1,024 records) and 2.97 eps on the chain (2,500
records, 54 zeros) on the RX 9060 XT; and every record's P − (P − A) and
P − (P − B) its two shells' centres, so the sort carried P, both
displacements and both offsets with the bound.

## Files

| File | Role |
|------|------|
| `ordered_pair_class_view.h` | `OrderedPairClassView`, the single definition, for `#include` and `import` alike |
| `pairs.cppm` | Module interface: `OrderedPairClass`, `OrderedPairGroup`, `ordered_pair_classes`, the view re-exported |
| `pairs.cpp` | The host gather (the sync, the orientations, the groups, the ranks, the copies), then the bound, the sort, the permutation and the refresh on the device |
| `pairs_bridge.h` | The bound's, the order's and the permutation's launchers, declared once for `pairs.cpp`'s GMF and the `.cu`, and the bound's for `../../coulomb/coulomb.cpp`'s GMF |
| `pairs_launch.cu` | The key and gather kernels, and the launchers: `ordered_pair_bound` (the generated `record_bound<la, lb>`, its table checked against `kMaxAngularMomentum`; also over `mmd`'s own records, as stored, for J's pruning, `coulomb_pair_classes`), `ordered_pair_order` (the two sorts), `ordered_pair_permute` (one field, in place) |
| `kernels/record_bound/` | The generated `record_bound<la, lb>` units, all nine ordered classes through d, and their template header (`tools/emit_exchange_kernels.py`) |
| `CMakeLists.txt` | The interface library `free_breakfast.integrals.exchange.pairs.headers`, the device library and the module library |
