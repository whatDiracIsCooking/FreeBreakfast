# Non-symmetric J and K

The AO response of a **non-symmetric** Cartesian density `D`, a CIS
transition density say, as one call: `G = J[D] - K[D]/2` (the singlet's) and
`K[D]` (the triplet's), integral-direct, from [`coulomb`](../coulomb/README.md)
and [`exchange`](../exchange/README.md).

## Module

`free_breakfast.integrals.eri.integral_direct.nonsymmetric_jk`, also
re-exported by `free_breakfast.integrals`. It launches nothing of its own: two
`exchange` sweeps, one `coulomb`, and BLAS, so there is no device library.

```cpp
auto mmd = primitive_pair_classes(res, basis);                 // once per basis
auto j_classes = coulomb_pair_classes(res, *mmd);              // once per basis
auto k_classes = ordered_pair_classes(res, basis, *mmd);       // once per basis
FBF_TRY(nonsymmetric_jk(res, *j_classes, *k_classes, d, g, k)); // per density
```

`pipeline.eri`'s `IntegralDirect` holds both sets of classes
(`j_classes`, `k_classes`), so a step reading `EriOutput` passes those.

## Design

`D = S + A`, `S = (D + Dᵀ)/2` and `A = (D - Dᵀ)/2`, each one `geam`. Then

- `J[D] = J[S]`, since `(mn|ls)` is symmetric in `l` and `s`, so `J[A] = 0`;
  and `coulomb` reads its density as symmetric (its `dq` visits each unique
  shell pair once, weighted twice), so `S` is what it must be given anyway.
- `K[D] = K[S] + K[A]`: `exchange` of `S` as `DensitySymmetry::Symmetric` and
  of `A` as `DensitySymmetry::Antisymmetric`, each computing its `a <= b`
  blocks and mirroring the rest with its sign. That is the same `kb` work as
  one sweep over every block, and each part fits its own decay, so it screens
  at least as tightly ([`exchange/`](../exchange/README.md), "A general `D` is
  two sweeps").

In order, all on `res.stream()`:

1. `g = A` (`geam`), `k = K[A]` (`exchange`, `Antisymmetric`), and `w = k`.
2. `g = S` (`geam`), `k = K[S]` (`exchange`, `Symmetric`), and `w += k`, so
   `w = K[D]`.
3. `k = J[S]` (`coulomb` of `g`, which still holds `S`), then
   `g = k - w/2 = J[D] - K[D]/2` (`geam`) and `k = w` (a copy).

**The scratch.** Each build zeroes its own output and reads its density from a
matrix of its own type, `CartesianMatrixAO`, so `g` and `k` take turns as the
density and the output, and `K[A]`, then `K[D]`, waits in `w` across the
builds. `w` is one `nao_cart²` block of `res.device_workspace()`, which grows
only, so after the first call nothing is allocated; neither `coulomb` nor
`exchange` takes the workspace, so holding it across them is allowed
(`docs/style.md`, "Scratch memory"). The builds' own scratch is their classes',
allocated once per basis, which is why calls run **one at a time per set of
classes**: each call writes them. Beside the three builds, the call costs four
`geam`s and two copies of `nao_cart²` doubles.

**The screen.** `tau_j` and `tau_k` default to `coulomb`'s and `exchange`'s
own, `1e-10` and `1e-13`, and the call does nothing to them. K's default holds
on a non-symmetric `D`; J's is coarse for a **dense** one. Its bound is per
skipped term of a bra's `J_p`, and a random `D`, every element of order one,
is the worst case for it, far from the decaying densities its default was
chosen on. On H₂O/cc-pVDZ against libcint (the test's random `D`, RX 9060 XT,
J's classes at their default prune, which moves nothing here):

| `tau_j` | worst `\|G - (vj - vk/2)\|` / `max\|G\|` |
|---|---|
| 1e-10 (default) | 9.0e-12 |
| 3e-11 | 9.3e-12 |
| 1e-11 | 2.2e-12 |
| 3e-12 | 7.0e-13 |
| 1e-12 | 3.4e-13 |
| 0 | 3.4e-13 |

At `tau_j = 0` what is left is K's default screen (K alone at `1e-13`:
1.7e-13 of `max|K|`). So `1e-12` of `max|G|` wants `tau_j <= 3e-12` on such a
density; the test screens J there. The default is unchanged, as `coulomb`'s
"The screen at f" leaves it: the caller (the CIS sigma, #198) picks the screen
its densities need, and a converged transition density is far sparser than a
uniform random one.

**Not checked:** that `|S_cd| <= 2`, the cap J's classes were pruned under
(`coulomb_pair_classes`), as `coulomb` does not check it either. A CIS
transition density of normalised amplitudes is far inside it.

## Public API

| Symbol | Role |
|---|---|
| `nonsymmetric_jk(res, j_classes, k_classes, density, g, k, tau_j = 1e-10, tau_k = 1e-13)` | `Status`: `g = J[D] - K[D]/2` and `k = K[D]` of the Cartesian `density`, from J's and K's classes of its basis, `coulomb` screened at `tau_j` and `exchange` at `tau_k` (their own defaults). Writes the classes' scratch, so one call at a time per set. InvalidConfig, before anything is queued, for a negative or NaN `tau_j` or `tau_k`, a matrix on another device or of another order, or `g` and `k` one matrix or either the density; else `coulomb`'s and `exchange`'s own refusals |

## Tests

`test/gtest/nonsymmetric_jk_gpu.cpp`: `G` and `K` of a random non-symmetric
`D` (uniform on `[-1, 1)`, a fixed seed) on H₂O in STO-3G and cc-pVDZ, against
libcint's `vj - vk/2` and `vk` of the same `D` (PySCF `get_jk(hermi=0)`,
through `test/gtest/data/exchange/*_nonsymmetric.json`, written by
`tools/emit/nonsymmetric_jk_golden.py`), within `1e-12` of each reference's
largest element, unscreened (J's classes pruned at `0`, both `tau`s `0`) and
screened (J's default prune, K at its default `1e-13`, J at `3e-12`: "The
screen" above); `K[Dᵀ] = K[D]ᵀ` and `G[Dᵀ] = G[D]ᵀ` on the device's own
results, to the same bar (not bitwise: an `a == b` block of `K[S]` is computed
whole, and J sums in its atomics' order); and the refusals, each leaving `G`
and `K` as they were.

## Files

| File | Role |
|------|------|
| `nonsymmetric_jk.cppm` | Module interface: `nonsymmetric_jk` |
| `nonsymmetric_jk.cpp` | The checks, the two splits, the three builds and the combination |
| `CMakeLists.txt` | The module library |
