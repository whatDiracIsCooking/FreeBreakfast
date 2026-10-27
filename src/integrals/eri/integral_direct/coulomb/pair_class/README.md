# Pair class

J's own primitive-pair classes, `CoulombPairClass`, and the prune that makes
them from `mmd`'s, `coulomb_pair_classes`. Below the J build's three steps,
which each launch over these classes.

## Module

`free_breakfast.integrals.eri.integral_direct.coulomb.pair_class`, re-exported
by `free_breakfast.integrals.eri.integral_direct.coulomb` and so by
`free_breakfast.integrals`.

## Design

`coulomb_pair_classes(res, classes, tau = 1e-13)` makes J's classes from
`mmd`'s, without the records J cannot need: arXiv:2407.21445 §3.2's
primitive-pair pruning, for two-electron records only. `mmd`'s own records are
untouched, because S, T and V read them and their correctness must not hang on
a two-electron bound.

**J's class** (`CoulombPairClass`) is device-only, on a stream of its own, and
carries only the records J reads: `zeta`, P, `P − A`, `P − B`, `s`, the
undressed `bound` and the two offsets, plus a build's scratch ([`../README.md`](../README.md), "The scratch is the class's"). Its
`view()` is a `PrimitivePairClassView` with the rest (`ea`, `eb`, the dressed
bounds, the ranks) null, so `dq`, `e_hermite`, `jab` and the ket sort read it
as they read `mmd`'s; `ket_view()` is the same over the sorted copies, as
`e_hermite` reads a ket.

**It runs on the device**, queued on `res.stream()`, which first waits on each
of `mmd`'s class streams by event (never a host synchronization):

1. `max B` and `max K` over every record of every class, a block reduction
   per class folded by `atomicMax` on the bits of a non-negative double.
2. Per class, the keep flag (below) and a stable scan of it in one block,
   giving each kept record its place and the class its kept count.
3. One readback of the counts (the only host synchronization, of
   `res.stream()`), to size J's classes; an emptied class is left out.
4. Per kept class, on its own new stream forked from `res.stream()`: its
   records and scratch allocated there, and a scatter of each field into
   place. The scan is stable, so the kept records keep
   `mmd`'s descending ζ, and J's bras their load-balancing order, with no
   sort. The scatters are joined back into `res.stream()`, and each of
   `mmd`'s streams then waits on `res.stream()`, so a later free of `mmd`'s
   records is ordered after every read of them.

**Why J needs it.** `mmd`'s `bound`, J's `Ip`, is `Σ_p I_p(ζ)`: it has no pair
factor `exp(-αβ/ζ |AB|²)` and no contraction coefficients, so it does not fall
with `|AB|`. The pair factor reaches J only through `s E^ab_p`, in `D_q` (a
ket) and in `jab` (a bra). So a far ket's `Dmax` is tiny and the sorted kets
stop early, but a far **bra** is never screened: its warp runs every ket a
near bra would, and `jab` then multiplies the result by `s E^ab_p ≈ 0`. On a
chain the far pairs are almost all of them.

**The bound, and the density it assumes.** Each record's dressed bound
`B = max_{i∈a} Σ_{k∈b} Σ_p |s E^{ik}_p| I_p(ζ)` is FelizJueves's
`ao.k.record_bounds` in `mmd`'s orientation: `mmd`'s `bound_ab`, by the same
generated `record_bound<la, lb>` K reads. With `|[p|q]| <= I_p I_q`, the term of a bra record `(a, b)` and
a ket record `(c, d)` in any element of J is at most `B_ab · Dc · K_cd`, where
`K = w · ncart(la) · B` bounds the ket's sum over every Cartesian pair (`w`
as above) and **`Dc = 2` caps every `|D_cd|`**. A record goes when it is under
`tau` both ways, `B_r Dc max K < tau` and `K_r Dc max B < tau`. The cap is an
assumption, not a theorem, and `coulomb` does not check it: a closed-shell
density has `|D_cd| <= sqrt(D_cc D_dd)` and `D_cc <= 2 (S⁻¹)_cc`, which is 2
for a normalized AO that overlaps nothing and grows with the overlap, without
bound toward linear dependence. The H₁₀ chain's RHF density has `max|D|` 0.33;
the H₂O goldens' Hilbert densities, 1.

**The threshold is `1e-13`, not J's `1e-10`.** Each pruned term is under it,
but an element sums many: on the H₁₀ chain (the measurement's density, below)
pruning at `1e-10` moves J by 6.6e-12 · max|J|, past the libcint bar; at
`1e-11`, 3.5e-13; at `1e-12`, 6.7e-14; at `1e-13`, 7.9e-15. The pair factor
falls so fast that the record count barely moves with it: on H₂₀₀, 3.8% kept
at `1e-10`, 4.4% at `1e-13`, and J's time the same to 15%.

**Measured** on the RX 9060 XT (gfx1200, `--preset hip`, Release), median of
11 runs after one warm-up, J at its default `tau = 1e-10`. The chain is the
H₁₀ golden's H₂ unit (1.4 bohr bond, 5.4 bohr period, cc-pVDZ, 10 Cartesian
AOs) repeated; its density is the H₁₀ RHF density's blocks by unit
separation, the most central pair of H₁₀ at each separation up to 4 units,
past that the 4-unit block times `exp(-0.397 · 5.4 · (d − 4))` (the H₁₀ fit,
`../../exchange/README.md`), zero under `1e-30`:

| Chain | AOs | Records kept | J, every record | J, pruned | Change | J moved |
|---|---|---|---|---|---|---|
| H₁₀ | 50 | 842 of 1,305 (65%) | 2.42 ms | 2.01 ms | −17% | 8e-15 |
| H₂₀ | 100 | 1,947 of 5,110 (38%) | 9.18 ms | 4.61 ms | −50% | 8e-15 |
| H₅₀ | 250 | 5,262 of 31,525 (17%) | 123 ms | 21.5 ms | −83% | 8e-15 |
| H₁₀₀ | 500 | 10,787 of 125,550 (8.6%) | 828 ms | 74.2 ms | −91% | 9e-15 |
| H₂₀₀ | 1,000 | 21,837 of 501,100 (4.4%) | 6,107 ms | 268 ms | −96% | 9e-15 |

"J moved" is `max|J_pruned − J| / max|J|`. The pruning itself is once per
basis: 8–17 ms over these, measured when it ran on the host (it re-ran the
dressed bound and pruned there); it now runs on the device and keeps the same
records. The records kept grow linearly with the chain, every
record quadratically, and so does the memory they hold. The timing program was
a throwaway, not kept in `bench/`.

## Public API

| Symbol | Role |
|---|---|
| `CoulombPairClass` | J's class of records: `mmd`'s fields J reads and a build's scratch, device-only, on its own `DeviceStream`; `view()` gives a `PrimitivePairClassView` with the fields it does not carry null, `ket_view()` the same over the last build's sorted ket copies |
| `coulomb_pair_classes(res, classes, tau = 1e-13)` | `Result<std::vector<CoulombPairClass>>`: J's classes from `classes` (`primitive_pair_classes` of one basis), without every record whose J terms, as bra and as ket, are all under `tau` **for any density with `\|D_cd\| <= 2`** (above), each class in `mmd`'s record order on a new `DeviceStream`, an emptied class left out. `tau = 0` keeps every record. InvalidConfig for a negative or NaN `tau` or a class above f (`kCoulombMaxMomentum`) |

## Tests

`test/gtest/coulomb_gpu.cpp`. The pruned records (`coulomb_pair_classes` at its default): J from them against
J's goldens (`data/coulomb/`) at `1e-12 · max|J|`, at `tau = 0` on H₂O in STO-3G and
cc-pVDZ and on the H₁₀ chain (`data/coulomb/h10_chain_ccpvdz.json`, its RHF
density), and at the default screen on the two H₂O (the chain's screened J
misses the bar with every record too); the record counts pinned, 135 of 135
kept in STO-3G, 572 of 593 in cc-pVDZ and 842 of 1,305 on the chain; how far
pruning moved J recorded (2.0e-14 on the chain, its worst error 2.0e-14); on
all three, at `1e-13`, `1e-10` and `0`, the device prune's records exactly the
host prune's (the criterion above, over `mmd`'s `bound_ab` copied back), every field
J reads, read back from the device, bitwise, in `mmd`'s order; `tau = 0`
keeping every record and a `tau` above every bound none; and the refusals.

## Files

| File | Role |
|------|------|
| `pair_class.cppm` | Module interface: `CoulombPairClass`, `coulomb_pair_classes` |
| `pair_class.cpp` | The prune: the event waits, the maxima and scans, the count readback, J's classes and their scatters |
| `pair_class_bridge.h` | The prune's launchers, declared once for the module and the device TU |
| `pair_class_launch.cu` | The launchers, and the maxima, keep-scan and scatter kernels |
| `CMakeLists.txt` | The device library and the module |
