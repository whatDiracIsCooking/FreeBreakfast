# rust/df_metric — the DF metric (P|Q) on the host, in Rust

The density-fitting Coulomb metric `M[P][Q] = (P|Q)` of a molecule over a
fitting set, computed on the host: plain Rust on rayon's threads, no GPU, no
kernel library, no C ABI and no CMake target. It is the matrix
`rust/integrals/eri/density_fitting/metric` builds on a card, from an
independent implementation.

## How

`metric(&aux)` takes an `fb_basis::BasisSet` and returns the **lower triangle**
of M over the spherical functions, packed: `n (n + 1) / 2` doubles for
`n = nao_sphe`, `(P, Q)` with `Q <= P` at `lower_index(P, Q) = P (P + 1) / 2 + Q`.
M is symmetric, so the upper triangle is never computed and never stored:
only shell pairs `A >= B` are visited, and of a shell with itself only the
lower triangle of its block is kept.

One McMurchie-Davidson integral with a single function per electron, for two
primitive Cartesian Gaussians, is

```text
(a|b) = 2 pi^(5/2) / (a b sqrt(a + b))
        * (-1)^lb * sum_{tuv} sum_{t'u'v'} E^a_tuv E^b_t'u'v' R_{t+t',u+u',v+v'}
```

`metric` never forms it. A column of a shell's Cartesian-to-spherical block
is a harmonic polynomial `H` of degree `l`, and Hobson's theorem gives
`H(r) exp(-a r²) = H(∇) exp(-a r²) / (2a)^l`: a spherical Gaussian is the
Hermite Gaussians of the top order `t + u + v = l` alone, under `H`'s own
coefficients. So the Hermite expansion `E` drops out, the exponents enter
only through a scalar weight, and a contracted shell pair sums `R_tuv` of
the one order `la + lb` over its primitive pairs. An element of the
spherical block is then a sum over the two harmonics' nonzero Cartesian
terms of a lookup into that sum: the harmonic blocks are sparse, so no
Cartesian-indexed matrix is formed and no dense transform runs.

**The far field.** Once `x = alpha |A - B|²` is large, `exp(-x)` is nothing
beside the Boys values, and `R_tuv` becomes `sqrt(pi / alpha) / 2` times the
`tuv` derivative of `1 / |A - B|`. The exponents are then left in a scalar
alone, and that scalar factors over the two shells. So for a pair of centers
the block of each `(la, lb)` is built once, from one table of those
derivatives, and every far pair of shells on them is that block times the
product of two per-shell strengths, `sum c / (a^(3/2) (2a)^l)`: no Boys
function, no recurrence and no harmonic sum of its own. A pair of shells is
far when `alpha |A - B|² >= 40 + 4 (la + lb)` for its smallest `alpha`, where
what is dropped is under 1e-17 of what is kept; the test is per pair of
shells, since a tight pair is far at a distance a diffuse one is not.

`metric_via_cartesian` is the textbook route, kept to check and time
against: every Cartesian `(a|b)` through `E`, then a dense sandwich, with no
far field.

| Piece | What |
|---|---|
| `boys(x, f)` | `F_n(x)` for every order asked, `M` the highest: the C++ tree's `eval_boys` (`src/integrals/shared/`), ported. Below `x = M + 8`: `F_M` from its Kummer series as a fixed-degree Horner polynomial, the rest by the downward recurrence. From there on: `F_0 = sqrt(pi/x) erf(sqrt(x)) / 2` and the upward recurrence |
| `boys_series.rs` | Those polynomials' coefficients, copied digit for digit from `eval_boys_M.h` by `gen_boys_series.py`; re-run it if the headers are regenerated |
| the Hermite Coulomb table | `R^n_tuv` at `alpha = a b / (a + b)` over `A - B`, from the Boys values |
| `metric` | Shell pairs `A >= B`, each block directly in the solid harmonics or from the far field, written into the packed triangle. One rayon task per center (a run of shells on one atom), whose rows are one contiguous slice of the triangle, so tasks write disjoint slices; each thread keeps one scratch, between calls too; `RAYON_NUM_THREADS` sets the count |
| `far_pairs(&aux)` | How many shell pairs the far field takes, and how many there are |
| `lower_index(p, q)` | Where `(P, Q)`, `Q <= P`, sits in it |
| `metric_via_cartesian` | The same through the Cartesian integrals |
| `hermite_expansion(l, a)` | `E^i_t` of `x^i exp(-a x²)` on one center, for the Cartesian route alone |

What comes from the rebuild's device-free crates, by path: the XYZ reader
(`fb-io-xyz`), the compiled-in fitting sets (`fb-io-basis`), the shells, the
spherical layout and the Cartesian-to-spherical block (`fb-basis`), and
`.npy` (`fb-io-npy`). So the conventions are the tree's: primitives
normalized, contractions never renormalized, columns `m = -l ..= l`.

The Boys tables reach order 20, so shells through `l = 10`. The carried sets
reach i. `erf` is the `libm` crate's, since `std` has none.

**The build needs the FMA target feature.** The hot loops are `f64::mul_add`
(the Boys series' Horner, as the C++ header's `fma`; the recurrence; the two
sums). Without the feature that is a call into a software fma, about 50%
slower than a plain multiply and add; with it, one instruction.
`.cargo/config.toml` sets `-C target-feature=+fma` for rustc and rustdoc, and
`src/lib.rs` is a `compile_error!` on x86-64 without it, so the two cannot
drift apart. Cargo reads that file only when run from this directory, and a
`RUSTFLAGS` in the environment replaces it.

## Build and run

The toolchain is the image's `/opt/rust`. `Cargo.lock` is not tracked, as in
the sibling directories.

```bash
cd experimental/rust/df_metric
cargo test --release
D=../../../rust/integrals/eri/density_fitting/metric/tests/data
cargo run --release -- $D/h2o.xyz cc-pvdz-jkfit target/h2o.npy \
    $D/h2o_cc-pvdz-jkfit.npy
        # df-metric-host XYZ AUX [OUT.npy [REFERENCE.npy]]
```

`AUX` is a compiled-in name or a BSE JSON path. `OUT.npy` is the packed
triangle, one dimension. `REFERENCE.npy` is a full `n × n` matrix, whose lower
triangle is compared; past `1e-12 max |M_ref|` the binary exits 1. It times
both routes (each again until a second has passed, 20 times at most) and
prints their largest difference.

Nothing in ctest or either gate runs cargo here. This directory is checked by
hand only.

## Checked (2026-11-20)

`cargo test --release`: 8 unit and integration tests and 1 doctest pass;
`cargo clippy --all-targets` and `cargo fmt --check` are clean. The two
integration tests compare both routes' lower triangles with PySCF's
`int2c2e`, using the device metric crate's fixtures and references (its
`tests/data/README.md` says what they pin), at that crate's bar,
`1e-12 max |M_ref|`.

| System | Fitting set | n_aux | l | max \|M - M_ref\| / max \|M_ref\| |
|---|---|---|---|---|
| water | cc-pVDZ-JKFIT | 116 | f | 1.3e-14 |
| iron beside water | def2-universal-JKFIT (contracted) | 377 | i | 1.6e-14 |
| gly10, 73 atoms | cc-pVDZ-JKFIT | 3606 | f | 2.4e-14 |

The gly10 row is by hand, through the binary, with 79% of its shell pairs
from the far field: its reference is that crate's
`reference.py` run on `gly10.xyz` (PySCF 2.14.0), not a tracked file. The
glycine chains are not in the tree either, except `gly5.xyz` and `gly10.xyz`
under `rust/pipeline/scf/tests/data/`.

The unit tests check `boys` against Simpson quadrature on both branches for
every highest order 0 to 20, on either side of each order's switch, the low Hermite coefficients, and two s primitives against
the closed form by both routes, and the far field against the full integral
just past its bound, where what it drops is largest: contracted shells of
every pair of momenta through i, to 1e-13 of each block's largest element.
A third integration test requires the
triangle to carry the same bits on one thread as on eight.

## Measured

The lower triangle alone, release build, on a shared 12-core machine (Ryzen 9
5900X). Below a second per run the time is the best of several; above it, one
run.

### Threads

The direct route, cc-pVDZ-JKFIT, by `RAYON_NUM_THREADS`:

| System | n_aux | Triangle | Far pairs | 1 thread | 2 | 4 | 8 | 8 against 1 |
|---|---|---|---|---|---|---|---|---|
| gly10, 73 atoms | 3606 | 0.048 GiB | 79.1% | 55 ms | 29 ms | 17 ms | 9.2 ms | 6.0x |
| gly30, 213 atoms | 10586 | 0.42 GiB | 92.5% | 0.35 s | 0.19 s | 0.097 s | 0.052 s | 6.7x |
| gly100, 703 atoms | 35016 | 4.6 GiB | 97.7% | 3.31 s | 1.74 s | 0.89 s | 0.47 s | 7.1x |

More than eight threads were not measured. The Cartesian route scales as
well, and has no far field: gly100 takes it 28.4 s on one thread and 3.7 s on
eight.

### One thread

| System | Fitting set | n_aux | Far pairs | Directly spherical | Via Cartesian | Ratio |
|---|---|---|---|---|---|---|
| water | cc-pVDZ-JKFIT | 116 | 0% | 0.14 ms | 0.33 ms | 2.4x |
| iron beside water | def2-universal-JKFIT, through i | 377 | 4.8% | 1.6 ms | 5.2 ms | 3.3x |
| gly10, 73 atoms | cc-pVDZ-JKFIT | 3606 | 79.1% | 0.055 s | 0.30 s | 5.4x |
| gly30, 213 atoms | cc-pVDZ-JKFIT | 10586 | 92.5% | 0.35 s | 2.6 s | 7.5x |
| gly30, 213 atoms | cc-pVTZ-JKFIT, through g | 12319 | 92.4% | 0.52 s | 3.7 s | 7.1x |
| gly100, 703 atoms | cc-pVDZ-JKFIT | 35016 | 97.7% | 3.31 s | 28.4 s | 8.6x |

On the chains the two routes agree to 4.5e-16 of `max |M|`, far field
included.

- **Time is still quadratic in n_aux, but the far field's constant is
  small**: gly100 is 5.4 ns per element of the triangle on one thread, where
  a near pair costs about 17. Nothing is screened; a far element is a
  multiply and a store.
- **A compact molecule gains nothing from the far field.** Water has no far
  pair and the iron fixture 4.8%; a glycine chain is long and thin, so these
  fractions flatter it against a globular molecule of the same size.
- **Packing halves the memory, not the time.** The full matrix was already
  built from the pairs `A >= B` and mirrored, so the triangle saves the
  mirror writes and half the storage: gly100 is 4.6 GiB packed against
  9.1 GiB full.

### What each step bought

The direct route on gly10 and on the iron fixture, one thread, each step on
top of the one before. The Cartesian route shares the Boys function and the
table, and nothing else.

| Step | gly10 | iron beside water |
|---|---|---|
| Through Cartesian integrals (the first version) | 0.32 s | 5.4 ms |
| Directly spherical, dense sandwich | 0.23 s | 3.0 ms |
| The tree's Boys function, ported | 0.23 s | 2.9 ms |
| Scratch buffers allocated once, not five `Vec`s per shell pair | 0.165 s | 2.6 ms |
| Sparse harmonic terms in place of the sandwich, an s\|s path with no table, a table sized to the pair, the recurrence's case chosen outside its inner loop | 0.123 s | 1.6 ms |
| No `erf` call from `x = 36`, where it is 1 | 0.118 s | 1.6 ms |
| `mul_add` in the hot loops, with the FMA feature | 0.113 s | 1.47 ms |
| Rayon, on one thread | 0.115 s | not measured |
| Per-shell weights computed once, and the far field | 0.061 s | 2.1 ms |
| One scratch per thread, not per rayon job | 0.055 s | 1.6 ms |

- **The Boys port bought nothing here**, though it is the better function:
  most pairs of a chain are far apart and already took the cheap asymptotic
  branch, and the series was never the cost.
- **Allocation was the largest single cost removed** before the far field,
  as 285 ns a shell pair had suggested, and it came back once: rayon's
  `for_each_init` builds its state per job, not per thread, and a scratch
  holds two tables of 230 KiB at i. That cost the iron fixture 0.5 ms of
  2.1 until the scratch became a thread-local.
- **The per-shell weights and the far field were not timed apart.**

### Where the near pairs' time went

gly10 before the far field, 118 ms on one thread, taken apart by switching
pieces off (not a profiler; `perf` is not in the image):

| Piece | Time | Share |
|---|---|---|
| The harmonic transform | 24 ms | 20% |
| The `R^n_tuv` recurrence | 29 ms | 25% |
| The Boys function (`exp`, `sqrt`, the recurrences) | 21 ms | 18% |
| The rest: the pair's weights, the top-order sum, the copy into the triangle, the loop | 44 ms | 37% |

The far field removes the first three for a far pair. It has not been taken
apart again since.
