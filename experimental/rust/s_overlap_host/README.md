# rust/s_overlap_host — the AO overlap S on the host, in Rust

The spherical AO overlap `S[μ][ν] = <μ|ν>` of a molecule in a basis,
computed on the host: plain Rust on rayon's threads, no GPU, no kernel
library, no C ABI and no CMake target. It is the matrix `../s_overlap` gets
from free_breakfast's pipeline steps on a card, computed independently, and
built the way `../df_metric` builds the DF metric.

## How

`overlap(&basis)` takes an `fb_basis::BasisSet` and returns the **lower
triangle** of S over the spherical functions, packed: `n (n + 1) / 2`
doubles for `n = nao_sphe`, `(μ, ν)` with `ν <= μ` at
`lower_index(μ, ν) = μ (μ + 1) / 2 + ν`. Only shell pairs `A >= B` are
visited, one rayon task per center, whose rows are one contiguous slice of
the triangle.

**Directly spherical.** Two primitive s Gaussians overlap as
`(π/p)^(3/2) exp(-μ R²)`, `p = a + b`, `μ = ab/p`, `R = A - B`. A column of
a shell's harmonic block is a harmonic polynomial `H` of degree `l`, and
Hobson's theorem gives `H(r - A) exp(-a |r - A|²) = H(∇_A) exp(-a |r - A|²) / (2a)^l`.
Both derivatives pass outside the integral, and `∇_B = -∇_R`, so

```text
<a|b> = (-1)^lb / ((2a)^la (2b)^lb) * H_a(∇_R) H_b(∇_R) (π/p)^(3/2) exp(-μ R²)
```

This is the metric's argument with the Coulomb kernel swapped for a
Gaussian, and the Gaussian is easier. Its derivatives factor over the axes,
`∂^t_X exp(-μX²) = h_t(X) exp(-μX²)` with `h_(t+1) = -2μ (X h_t + t h_(t-1))`.
So there is no Boys function and no `R^n_tuv` table, only three short 1-D
recurrences per primitive pair. A contracted pair sums `h_t h_u h_v` of
the one order `la + lb` over its primitive pairs. An element of the block
is then a sum over the two harmonics' nonzero Cartesian terms of a lookup
into that sum.

**Negligible pairs.** S falls as `exp(-μ R²)`, so a pair of shells with
`μ_min |A - B|² >= 40 + 4 (la + lb)` is left zero. That is the metric's
far-field bound used as a cutoff, and the unit test checks that a full
block just past it is under 1e-15 for every pair of momenta through i. The
metric has a far field; S has nothing beyond its cutoff.

`overlap_via_cartesian` is the textbook route, kept to check and time
against: Obara-Saika per axis for every Cartesian `<a|b>`, then a dense
sandwich `Uaᵀ <a|b> Ub`, with no pair screened.

The conventions are the tree's, because the basis comes from the
rebuild's device-free crates (`fb-io-xyz`, `fb-io-basis`, `fb-basis`,
`fb-io-npy`): primitives normalized, contractions never renormalized, and
columns `m = -l ..= l`. So `diag(S)` is off 1 by the published
coefficients' rounding: 7e-11 in STO-3G, and up to 1.3e-6 in the cc sets
(`../s_overlap/README.md` "What the reference pins, and why").

The hot loops are `f64::mul_add`, as in `../df_metric`. `.cargo/config.toml`
sets `-C target-feature=+fma`, and `src/lib.rs` is a `compile_error!` on
x86-64 without it. Cargo reads that file only when run from this directory,
and a `RUSTFLAGS` in the environment replaces it.

## Build and run

The toolchain is the image's `/opt/rust`. `Cargo.lock` is not tracked, as
in the sibling directories.

```bash
cd experimental/rust/s_overlap_host
cargo test --release
cargo run --release -- ../s_overlap/data/co.xyz cc-pvtz target/co.npy \
    ../s_overlap/data/co_cc-pvtz.npy
        # s-overlap-host XYZ BASIS [OUT.npy [REFERENCE.npy]]
```

`BASIS` is a compiled-in name or a BSE JSON path. `OUT.npy` is the packed
triangle, one dimension. `REFERENCE.npy` is a full `n × n` matrix whose
lower triangle is compared; past `1e-12 max |S_ref|` the binary exits 1.
The binary times both routes (each run repeats until a second has passed,
20 times at most) and prints their largest difference.

Nothing in ctest or either gate runs cargo here. This directory is checked
by hand only.

## Checked (2026-11-21)

`cargo test --release`: 8 unit and integration tests and 1 doctest pass;
`cargo clippy --all-targets` and `cargo fmt --check` are clean.

The integration tests compare both routes' lower triangles with PySCF on
`../s_overlap/data/`'s fixtures and references, at that crate's bar,
`1e-12 max |S_ref|`. The table is the binary's output.

| Molecule | Basis | nao | l | max \|S - S_ref\| / max \|S_ref\| | max \|diag(S) - 1\| |
|---|---|---|---|---|---|
| H2O | STO-3G | 7 | p | 4.4e-16 | 7.0e-11 |
| H2O | cc-pVDZ | 24 | d | 4.4e-16 | 1.0e-6 |
| CO | cc-pVTZ | 60 | f | 8.9e-16 | 1.3e-6 |

The unit tests cover:
- two s primitives against the closed form;
- a single primitive of every momentum through i, unit-normalized in every
  component, by both routes;
- the two routes against each other on contracted shells of every pair of
  momenta through i, to 1e-14;
- the cutoff, as above.

A further integration test requires the triangle to carry the same bits on
one thread as on eight.

## Measured

Release build, Ryzen 9 5900X (12 cores, 24 threads, shared), best of up to
20 runs. gly10 is `rust/pipeline/scf/tests/data/gly10.xyz` (73 atoms); on it
the routes agree to 2e-14.

| System | Basis | nao | Negligible pairs | Spherical, 1 thread | 8 | 24 | Via Cartesian, 24 | Cartesian / spherical |
|---|---|---|---|---|---|---|---|---|
| H2O | cc-pVDZ | 24 | 0% | | | 33 µs | 59 µs | 1.8x |
| CO | cc-pVTZ | 60 | 0% | | | 97 µs | 190 µs | 2.0x |
| gly10 | cc-pVDZ | 734 | 51.7% | | | 1.1 ms | 16 ms | 14.8x |
| gly10 | cc-pVTZ | 1678 | 48.7% | 39 ms | 5.8 ms | 3.8 ms | 52 ms | 13.5x |

- **The ratio grows with the molecule** because the Cartesian route
  screens nothing. Half of gly10's pairs are negligible.
- **The small cases measure overhead, not integrals.** At 24 AOs the time
  is mostly rayon's dispatch to 24 threads.
- Not measured: the time on one thread, or by thread count, of anything
  smaller than gly10.
