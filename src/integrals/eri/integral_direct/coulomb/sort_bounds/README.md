# Sort bounds

Each ket class of the J build sorted descending by its screen bound
`Iq · Dmax`, which is the order `e_hermite`'s screen needs.

## Target

A device library with no module face,
`free_breakfast::integrals::eri::integral_direct::coulomb::sort_bounds::device`,
driven through `sort_bounds_bridge.h`. Only [`coulomb`](../README.md) uses it:
the buffers it sorts into are `CoulombPairClass`'s `ket_` fields, so the class
owns them, and coulomb's implementation unit includes the bridge in its global
module fragment.

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
[`mmd`](../../../../mmd/README.md) already makes descending ζ, so descending `Ip`,
once per basis.

## Design

`sort_bounds_kets(stream, cls, n_hermite, bound, dmax, dq, …)` queues three
steps on the class's stream, into scratch and sorted copies the caller owns:

| Step | Kernel | Does |
|---|---|---|
| key | `sort_bounds_key_kernel`, one thread per record | `key[k] = -(Iq · Dmax)`, `perm[k] = k` |
| sort | `merge_sort_passes` ([`src/recipes/sort`](../../../../../recipes/sort/README.md)) | `key` ascending in IEEE-754 total order, `perm` carried, stably |
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
- **`e_hermite`'s launcher is unchanged.** `CoulombPairClass::ket_view()`
  returns a `PrimitivePairClassView` with `la`, `lb`, `n_rec`, and `zeta`, P
  and the bound aimed at the sorted copies, which is all `e_hermite` reads of
  a ket. Its other pointers are null.

The merge sort is [`src/recipes/sort`](../../../../../recipes/sort/README.md)'s, shared with
`mmd`, which orders each class's records with it.

## Public API

| Symbol | Role |
|---|---|
| `device::sort_bounds_kets(stream, cls, n_hermite, bound, dmax, dq, key, key_alt, perm, perm_alt, zeta_sorted, p_x_sorted, p_y_sorted, p_z_sorted, bound_sorted, dmax_sorted, dq_sorted)` | `wwrError_t`: queues the key, the sort and the gather on `stream`. Reads `cls`'s `la`, `lb`, `n_rec`, `zeta` and P; `dq` and `dq_sorted` are q-major, `n_hermite × n_rec`, the rest `n_rec` long. Launch errors only |

## Tests

`test/gtest/sort_bounds_gpu.cpp` drives the launcher through its bridge, as
`exchange_helpers_gpu.cpp` drives its own, on a made-up class of 3600
records, which is past the 2816-key tile, so a merge pass runs. It makes up
`zeta`, P, bounds, `Dmax` and `D_q`, with repeated products and zero `Dmax`
among them. Every sorted array must equal its record order permuted by a host
`std::stable_sort` of the same keys under `std::strong_order`, bit for bit. The
products must not increase, and a zero `Dmax` must come last. Through `J`, the
screen is [`coulomb`](../README.md)'s tests.

## Files

| File | Role |
|------|------|
| `sort_bounds_bridge.h` | `sort_bounds_kets`, declared once for coulomb's GMF, the test and the `.cu` |
| `sort_bounds_launch.cu` | The key and gather kernels and `sort_bounds_kets` |
| `CMakeLists.txt` | The device library |
