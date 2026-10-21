# Exchange pairs

The ordered primitive-pair classes K's bras and kets are drawn from: `mmd`'s
`la <= lb` records in both orientations, one class per ordered `(la, lb)`,
grouped by first shell, built once per basis.

## Module

`free_breakfast.integrals.exchange.pairs`.

## Why here, not in `mmd`

K is the only consumer. The one-electron matrices and J are symmetric inside a
pair and want `mmd`'s one class per unordered pair of momenta; K is not, so it
wants both orientations of every pair, grouped by the shell that is its output
index. Putting that in `mmd` would make every build pay for records only K
reads, and widen `mmd`'s surface for one caller. If another build ever wants
ordered records, this moves into `mmd` as a partition.

## Design

K_ab = Σ_cd D_cd (ac|bd), by FelizJueves's `kernel_k`: its bra `(a, c)` and
ket `(b, d)` are **ordered** shell pairs, and the first shell of each is an
output index (`a` the row, `b` the column). So:

- **Each off-diagonal `mmd` record gives two.** Record `(a, b)` of `mmd`'s
  `(la, lb)` goes to the ordered `(la, lb)` as stored, and to `(lb, la)` with
  `pa_*` and `pb_*`, and `off_a` and `off_b`, swapped. ζ, P, `s` and `bound`
  are orientation-free and copied as they are. For `la == lb` both land in the
  one class.
- **A diagonal record** (`off_a == off_b`, a shell against itself) gives one.
- So a class holds one record per primitive pair of every ordered shell pair
  of its momenta: all nine classes through d, (p, s) as well as (s, p).

The fields keep `mmd`'s names, with "a" the record's **first** shell and "b"
its second. As the generated `k<la, lc, lb, ld>` (`../kernels/k/`) names its
parameters, a bra class passes `p_*`, `zeta`, `pa_*`, `pb_*`, `s` and `bound`
as `p_*`, `zeta_ac`, `pa_*`, `pc_*`, `s_ac` and `Ip`; a ket class the same
fields as `q_*`, `eta`, `qb_*`, `qd_*`, `s_bd` and `Iq`. `mmd`'s `eb` is not carried (kernel_k does not read it). One
field is new:

| Field | Value |
|---|---|
| `rank_b` | the second shell's rank: its index among the basis's shells of momentum `lb`, in basis order |

kernel_k's `Dcd` row is `dc[bra] + dd[ket]`, and the exchange helpers' gather
(`../helpers/exchange_helpers_bridge.h`) lays `Dcd` out so that is
`rank_c * n_shells(ld) + rank_d`: a bra's `dc` is its `rank_b` times the number
of `ld` shells, a ket's `dd` its `rank_b`. The first shell's rank is not
stored: a launch is per first shell (below), and that shell is the group's.

**Grouped by first shell.** Each class's records are stably grouped by their
first shell, the groups in basis order, each group's records in `mmd`'s record
order: descending ζ. `bound` rises with ζ (and depends on nothing else within
a class), so each group is non-increasing in `bound`, which is the order
kernel_k's screen `break` (`Ip · Iq · maxD < τ`, kets sorted descending by
`Iq`) needs, with no sort per build. A K launch takes one ket group whole,
since all its kets share the `b` shell whose column the block is. The group
table is on the host, one `OrderedPairGroup {shell, start, count}` per group:
`shell` the first shell's index in `BasisSet::shells()`, the records
`[start, start + count)`, `count` the launch's `n_ket`.

**The gather is on the host, once per basis.** `ordered_pair_classes`
synchronizes each `mmd` class's stream (its pinned side holds the
zeta-ordered records only after), walks its records in order, appends each
orientation to its ordered class's group for its first shell, then writes
each ordered class into its pinned side and queues the copies to the device on
a new `DeviceStream`. The group table and the ranks are host data anyway, the
whole step is a few array copies, and doing it on the device would want a
device library, a rank table upload and a second sort for one step per basis.
A record whose AO offset starts no shell of the basis (the `mmd` classes are
another basis's) is an `InvalidConfig` error.

## Public API

| Symbol | Role |
|---|---|
| `OrderedPairClass` | One ordered class `(la, lb)`: `n_rec`, one `MirroredBuffer` per field (`rank_b` among them), `groups`, `device_stream()`; `view()` gives an `OrderedPairClassView` |
| `OrderedPairGroup` | One group: `shell`, `start`, `count` |
| `OrderedPairClassView` | The class's device fields by pointer, for the launchers (`ordered_pair_class_view.h`, re-exported) |
| `ordered_pair_classes(res, basis, classes)` | `Result<std::vector<OrderedPairClass>>` from `primitive_pair_classes(res, basis)`: every reached ordered class, ascending `(la, lb)`, none empty, each on a new `DeviceStream` |

## Tests

`test/gtest/exchange_pairs_gpu.cpp`, on H₂O in STO-3G and cc-pVDZ: each
class's record count against a host count of ordered primitive pairs; each
group contiguous, one first shell of momentum `la`, the groups in basis order,
each group's count right and its `bound` non-increasing; each record's
`rank_b`; the `(la, lb)` and `(lb, la)` records the same multiset of
(ζ, P, P − A, P − B, `s`, offsets) with the two orientations swapped; the
device side matching the pinned side; and another basis's classes refused.

## Files

| File | Role |
|------|------|
| `ordered_pair_class_view.h` | `OrderedPairClassView`, the single definition, for `#include` and `import` alike |
| `pairs.cppm` | Module interface: `OrderedPairClass`, `OrderedPairGroup`, `ordered_pair_classes`, the view re-exported |
| `pairs.cpp` | The host gather: the sync, the orientations, the groups, the ranks, the copies |
| `CMakeLists.txt` | The interface library `free_breakfast.integrals.exchange.pairs.headers` and the module library |
