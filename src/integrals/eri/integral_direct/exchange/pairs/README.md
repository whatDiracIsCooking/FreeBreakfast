# Exchange pairs

The ordered primitive-pair classes K's bras and kets are drawn from: `mmd`'s
`la <= lb` records in both orientations, one class per ordered `(la, lb)`,
grouped by first shell, each record with K's **dressed** Schwarz bound and
each group sorted by it, built once per basis on the device, and held there
only; with them, what the K build needs per basis (its block tables and `Dcd`
rows) and the scratch it rewrites per build.

## Module

`free_breakfast.integrals.eri.integral_direct.exchange.pairs`.

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
  of its momenta: all sixteen classes through f, (p, s) as well as (s, p).

The fields keep `mmd`'s names, with "a" the record's **first** shell and "b"
its second. As the generated `kb<la, lc, lb, ld>` (`../kernels/kb/`) names its
parameters, a bra class passes `p_*`, `zeta`, `pa_*`, `pb_*`, `s` and
`record_bound` as `p_*`, `zeta_ac`, `pa_*`, `pc_*`, `s_ac` and `Ip`; a ket class
the same fields as `q_*`, `eta`, `qb_*`, `qd_*`, `s_bd` and `Iq`. `mmd`'s `ea`
and `eb` are not carried (kb reads neither), and neither is its undressed
`bound` (below). Two fields depend on the orientation:

| Field | Value |
|---|---|
| `record_bound` | K's dressed Schwarz bound, FelizJueves's `ao.k.record_bounds`: `I = max_{i in a} Σ_{k in b} Σ_p \|s E^{a_i b_k}_p\| I_p(ζ)`: `mmd`'s `bound_ab` as stored, `bound_ba` swapped |
| `rank_b` | the second shell's rank: its index among the basis's shells of momentum `lb`, in basis order: `mmd`'s `rank_b` as stored, `rank_a` swapped |

kb's `Dcd` row is `dc[bra] + dd[ket]`, and the exchange helpers' gather
(`../helpers/exchange_helpers_bridge.h`) lays `Dcd` out so that is
`rank_c * n_shells(ld) + rank_d`: a bra's `dc` is its `rank_b` times the number
of `ld` shells, a ket's `dd` its `rank_b`. The class stores the `dc` rows, one
per `ld` (below). The first shell's rank is not stored: a K block is per first
shell (below), and that shell is the group's.

**The bound is dressed, and per orientation.** kb's screen and its
guarantee (FelizJueves #484/#485: every skipped term under `τ`) are proved
against `ao.k.record_bounds`, which carries the contraction coefficients and
the pair factor `exp(-αβ/ζ |AB|²)` (through `s` and the dressed `E`), and takes
its max over the FIRST shell's Cartesians, summing over the second's. So it is
neither `mmd`'s `bound` (`Σ_p I_p(ζ)`, which has neither factor and can fall
short of it: up to 1.2× on H₂O/cc-pVDZ, an s shell with itself) nor the same
for `(a, b)` and `(b, a)`. It decays with `|AB|`: on the H₁₀ chain 54 of 2,500
records are exactly `0.0`, the pair factor underflowed. `mmd` forms it, both
orientations of every record, once per basis (`bound_ab` and `bound_ba`, by
the generated `record_bound<la, lb>` in `../../../../mmd/kernels/record_bound/`), so
this module only picks the orientation's. A class above f has no kernel and
has zeros; the K build refuses it.

**Grouped by first shell, each group sorted by the bound.** Each class's
records are grouped by their first shell, the groups in basis order: group
`g` is the shell of rank `g` among the basis's shells of momentum `la`, of
`nprim(a)` times the primitives of every `lb` shell, so the group table is
the basis's alone and is made on the host. Each group is sorted by
`record_bound`, non-increasing, equal bounds in `mmd`'s record order. That is
the order both of kb's screen `break`s (`Ip · Iq · F < τ` down a block's kets,
and `Ip · Iq[ket0] · F < τ` down its bras) need: **an unsorted group is a
wrong K, not a slow one.** The sort is once per basis, never per build.
(Before the dressed bound, `mmd`'s ζ order was enough: its bound rises with
ζ. The dressed one does not follow ζ.) A K block takes one bra group and one
ket group whole, since all its bras share the `a` shell whose row the block
is, and its kets the `b` shell whose column. The groups' bounds are on the
device, as `group_start` (`n_groups + 1` entries, the last `n_rec`): group `g`
is records `[group_start[g], group_start[g + 1])`. The host keeps only their
number, `n_groups`; which shell a group is follows from the basis (rank `g`
among the shells of `la`).

**The gather is on the device, once per basis, straight from `mmd`'s
class.** Each ordered class reads one `mmd` class, the one of its momenta in
`la <= lb` order, by **slot**: for `la != lb` slot `r` is record `r`, as
stored or swapped; for `la == lb` slot `2r` is it as stored and `2r + 1`
swapped, a slot a diagonal record does not have. On a new `DeviceStream`,
`ordered_pair_classes` queues:

1. **The order** (`ordered_pair_order`): two stable merge sorts
   ([`src/recipes/sort`](../../../../../recipes/sort/README.md)), every slot by `-I`, then
   the result by its first shell's rank (`mmd`'s `rank_a` as stored, `rank_b`
   swapped), a diagonal record's swapped slot past every group. Slot order is
   `mmd`'s record order, so equal bounds stay in it. Then `group_start`, by
   bisection over the sorted ranks.
2. **The gather** (`ordered_pair_gather`): record `k` is the `k`-th slot in
   that order, every field written from `mmd`'s, swapped where the slot is.

3. **The K build's per-basis inputs**, once every class is gathered: each
   class's `dc` rows, one per `ld` the basis reaches (`ordered_pair_bra_rows`),
   and its block table as a bra against every class of the set as a ket, in
   the set's order (the exchange helpers' `exchange_blocks`): one block per
   `(a, b)` of a bra group and a ket group with `a <= b`, the blocks the K
   sweep computes (`../README.md`, "The bra-ket swap").

So the records are gathered once, already in order; nothing is permuted after.
`group_start` is formed there too rather than uploaded: an upload would need a
host source that outlives the asynchronous copy, which is a pinned side and a
synchronization again. The host sizes everything from the basis alone:
`n_shells` (shells per momentum, through f at least, K's limit), `n_ao` (its Cartesian
AO count) and, per bra, `n_blk` (each table's block count, one per class of
the set, so its size is the set's class count). A table of more than
`INT32_MAX / 6` blocks is an `InvalidConfig` error.

**The E tables, for a basis with an f shell.** Such a basis launches the
rolled `kbr` (`../README.md`), and every class is a side of some `kbr` launch,
so after its gather each class gets `e_bra` and `e_ket`, its records' undressed
E table as a bra and the same with odd-`|t|` entries negated as a ket
(`exchange_e_table`, `n_rec · exchange_e_table_width(la, lb)` doubles each).
A basis through d gets none (empty buffers, null in the view), and neither
does a class above f, which the build refuses.

**The K build's scratch is the class's** (as J's is `CoulombPairClass`'s):
`dcd`, `geom` and `factor`, a class `(l, l)`'s `rank_offset`, and the set's
own `centre`, `shell_start`, `decay` and `g_max` on its first class only, are
allocated here and left unwritten, and every `exchange` call rewrites them
(`../README.md`, "Scratch"). So an SCF loop
allocates none of them per build, and the K build allows one call at a time
per set of classes.

**No host synchronization, and no host copy.** Each class's stream is forked
from its `mmd` class's (an event, `ForkStream::fork_from`), so it reads
`mmd`'s final records. A bra's tables read every class's `group_start` and
`off_a`, so once every class is gathered, each class's stream is joined into
`res.stream()` and forked back from it for its tables, then joined and forked
once more, so a class's frees follow every table that read it. Each `mmd`
class's stream then waits on `res.stream()`: freeing an `mmd` class queues
its frees on its own stream, so they wait for every read of it, and the
caller may drop the `mmd` classes as soon as the call returns. The records
are device only (`StreamBuffers::device`); a test copies a field back
itself. Since
nothing reads `mmd`'s records on the host, the classes are checked against
the basis by their momenta and record counts: a class missing, repeated or of
another count (the `mmd` classes are another basis's) is an `InvalidConfig`
error.

## Public API

| Symbol | Role |
|---|---|
| `OrderedPairClass` | One ordered class `(la, lb)`: `n_rec`, `n_groups`, the basis's `n_ao` and `n_shells`, its tables' `n_blk` (one per class of the set), its `work_stream` (also `device_stream()`, as `mmd`'s classes name theirs), one device-only `StreamBuffers::device` per field (`record_bound` and `rank_b` among them), the groups' bounds `group_start`, the K build's `dc` rows and `blocks` tables (once per basis) and its scratch `dcd`, `geom`, `factor` and `rank_offset`, and on the set's first class the set's `centre`, `shell_start`, `decay` and `g_max` (per build); `view()` gives an `OrderedPairClassView` |
| `OrderedPairClassView` | The class's device fields by pointer, for the launchers (`ordered_pair_class_view.h`, re-exported) |
| `ordered_pair_classes(res, basis, classes)` | `Result<std::vector<OrderedPairClass>>` from `primitive_pair_classes(res, basis)`: every reached ordered class, ascending `(la, lb)`, none empty, each on a new `DeviceStream`, a set the K build takes whole |

## Tests

`test/gtest/exchange_pairs_gpu.cpp`, on H₂O in STO-3G and cc-pVDZ and the
H₁₀ chain: each class's record count against a host count of ordered
primitive pairs; each group contiguous, one first shell of momentum `la`, the
groups in basis order, each group's count right and its `record_bound`
non-increasing and non-negative, and `group_start` its bounds; each record's
`rank_b`; each class's `n_ao`, `n_shells`, `dc` rows, and `a <= b` block table
against every ket class (`n_blk` and every entry) against a host enumeration
from the groups; the `(la, lb)` and `(lb, la)` records the same multiset of
(ζ, P, P − A, P − B, `s`, offsets) with the two orientations swapped; the
classes bit for bit against a host gather of `mmd`'s records (each group
stably sorted by the bound, the order the earlier host gather made); the view
naming the device fields; and another basis's classes refused. Every check
reads a field copied back explicitly.
On H₂O/cc-pVDZ and the chain, every record's `record_bound` against
FelizJueves's `ao.k.record_bounds` of the same ordered shell pair
(`test/gtest/data/record_bound/`, written by
`tools/emit/record_bound_golden.py`), matched by its two shells and ζ, a host
zero exactly zero. Each record passes within 16 eps (FelizJueves's bar for
the kernel) or within 10⁻¹³ absolutely. `mmd` forms P's numerator with an FMA
and the golden with two roundings, so P can sit an ulp of the coordinates
away. The bound magnifies that relatively: through the pair factor on distant
pairs, and through the Hermite polynomials where two shells share an atom far
from the origin. That happens only on bounds those factors have already
shrunk. Measured on the RTX 3080 and the gfx1200 alike:

- H₂O/cc-pVDZ (1,024 records): worst 37.6 eps, 1.8 × 10⁻¹⁵ absolute.
- The chain (2,500 records, 54 zeros): worst 5,041 eps, which is 8 × 10⁻²³ on
  a bound of 7 × 10⁻¹¹; worst 2.4 × 10⁻¹⁵ absolute.

10⁻¹³ leaves room for molecules further from the origin. It sits below any
screen threshold, and still fails an O(bound) kernel error on every bound above
it. Before the FMA the worst was 2.95 and 2.97 eps (RX 9060 XT). Every record's
P − (P − A) and
P − (P − B) its two shells' centres, so the sort carried P, both
displacements and both offsets with the bound.

## Files

| File | Role |
|------|------|
| `ordered_pair_class_view.h` | `OrderedPairClassView`, the single definition, for `#include` and `import` alike |
| `pairs.cppm` | Module interface: `OrderedPairClass`, `ordered_pair_classes`, the view re-exported |
| `pairs.cpp` | The check against the basis, the sizes from it, then per class the fork from `mmd`'s stream, the order and the gather on the device; after a join, every bra's `dc` rows and tables; and the joins back |
| `pairs_bridge.h` | The order's, the gather's and the `dc` rows' launchers, and the slots the first two share, declared once for `pairs.cpp`'s GMF and the `.cu` |
| `pairs_launch.cu` | The key, group-start, gather and `dc`-row kernels, and the launchers: `ordered_pair_order` (the two sorts and `group_start`), `ordered_pair_gather` (every field, from `mmd`'s class), `ordered_pair_bra_rows` |
| `CMakeLists.txt` | The interface library `free_breakfast.integrals.eri.integral_direct.exchange.pairs.headers`, the device library and the module library |
