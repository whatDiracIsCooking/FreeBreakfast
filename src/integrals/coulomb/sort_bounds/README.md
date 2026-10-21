# Sort bounds

Each ket class of the J build sorted descending by its screen bound
`Iq · Dmax`, which is the order `e_hermite`'s screen needs.

## Module

`free_breakfast.integrals.coulomb.sort_bounds`, target
`free_breakfast::integrals::coulomb::sort_bounds`. Only
[`coulomb`](../README.md) uses it, so `free_breakfast.integrals` does not
re-export it.

## Why the kets are sorted, and by what

`e_hermite`'s screen is a `break`, not a `continue`: a lane stops at its first
ket with `Dmax · Ip · Iq < tau`. That skips only kets under `tau` when the kets
arrive in descending `Iq · Dmax`. In any other order a `tau` above 0 gives a
wrong answer, not a slow one.

The key is the product, not `Iq` alone. `Iq` (the class's `bound` field, `mmd`'s `pair_bound`) depends only on
`zeta` and `la + lb`. The pair's Gaussian prefactor `exp(-μ R_AB²)` enters
through `dq`'s `E^{00}_0`, and the contraction and the density come in with it,
so all three are in `Dmax`. `Dmax` changes with the density, so the sort is
redone in every build, after `dq`.

Lane `y` of a warp visits kets `y, y + W, …`, a subsequence of the sorted
order, so it is descending too, and a per-lane `break` is right.

Only kets are sorted here. Bra order has no effect on the result, so the bra
side, `jab` and its `row` input read the class in record order, which
[`mmd`](../../mmd/README.md) already makes descending ζ, so descending `Ip`,
once per basis.

## Design

`SortedKets(device, cls, n_hermite)` allocates, from `device`'s pool, the sort's
scratch and a sorted copy of each array `e_hermite` reads of a ket.
`sort(stream, bound, dmax, dq)` queues three steps on the class's stream:

| Step | Kernel | Does |
|---|---|---|
| key | `sort_bounds_key_kernel`, one thread per record | `key[k] = -(Iq · Dmax)`, `perm[k] = k` |
| sort | `merge_sort_passes` ([`src/integrals/sort`](../../sort/README.md)) | `key` ascending in IEEE-754 total order, `perm` carried, stably |
| gather | `sort_bounds_gather_kernel`, one thread per `D_q` element | place `k` of each sorted copy from record `perm[k]`: `zeta`, P, `Iq`, `Dmax`, and column `k` of each q-major `D_q` row |

- **Descending for free.** The sort only ascends. The bound is never
  negative, so negating it gives the descending order. A bound of exactly 0
  becomes `-0.0`, which total order puts after every negative key, so last.
  A NaN key would need a NaN density.
- **Ties keep record order.** The sort is stable. That doesn't matter for
  correctness, but it makes the order reproducible.
- **One launch per class at these sizes.** Up to 2816 records the sort is a
  single block sort in 1 to 8 warps. Above that it runs merge passes, which
  use the `_alt` scratch.
- **`e_hermite`'s launcher is unchanged.** `view()` returns a
  `PrimitivePairClassView` with `la`, `lb`, `n_rec`, and `zeta` and P aimed at
  the sorted copies, which is all `e_hermite` reads of a ket. Its other
  pointers are null.

The merge sort is [`src/integrals/sort`](../../sort/README.md)'s, shared with
`mmd`, which orders each class's records with it.

## Public API

| Symbol | Role |
|---|---|
| `SortedKets(device, cls, n_hermite)` | Scratch and sorted copies for `cls`, whose `D_q` (q-major, `n_hermite × n_rec`) has `n_hermite` Hermite functions. Allocates and launches nothing. Movable. `cls` must outlive it |
| `SortedKets::sort(stream, bound, dmax, dq)` | `wwrError_t`: queues the key, the sort and the gather on `stream`. Launch errors only |
| `SortedKets::view()` | The class as `e_hermite`'s ket: `zeta` and P sorted, the other pointers null |
| `SortedKets::bound()`, `dmax()`, `dq()` | `Iq`, `Dmax` and `D_q` (q-major), sorted |

## Tests

`test/gtest/sort_bounds_gpu.cpp` uses one s shell of 60 primitives against
itself, 3600 records, which is past the 2816-key tile, so a merge pass runs.
It makes up bounds, `Dmax` and `D_q`, with repeated products and zero `Dmax`
among them. Every sorted array must equal its record order permuted by a host
`std::stable_sort` of the same keys under `std::strong_order`, bit for bit. The
products must not increase, and a zero `Dmax` must come last. Through `J`, the
screen is [`coulomb`](../README.md)'s tests.

## Files

| File | Role |
|------|------|
| `sort_bounds.cppm` / `.cpp` | The module: `SortedKets`, its scratch, and the launcher's arguments |
| `sort_bounds_bridge.h` | `sort_bounds_kets`, declared once for the module's GMF and the `.cu` |
| `sort_bounds_launch.cu` | The key and gather kernels and `sort_bounds_kets` |
| `CMakeLists.txt` | The device library and the module |
