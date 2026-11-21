# rust/t_kinetic_host — the AO kinetic energy T on the host, in Rust

The spherical AO kinetic-energy matrix `T[μ][ν] = <μ| -∇²/2 |ν>` of a
molecule in a basis, computed on the host: plain Rust on rayon's threads,
no GPU, no kernel library, no C ABI and no CMake target. It is the matrix
the rebuild's `rust/integrals/one_electron/kinetic` computes on a card,
computed independently, and built the way `../s_overlap_host` builds S.

## How

`kinetic(&basis)` takes an `fb_basis::BasisSet` and returns the **lower
triangle** of T over the spherical functions, packed: `n (n + 1) / 2`
doubles for `n = nao_sphe`, `(μ, ν)` with `ν <= μ` at
`lower_index(μ, ν) = μ (μ + 1) / 2 + ν`. Only shell pairs `A >= B` are
visited, one rayon task per center, whose rows are one contiguous slice of
the triangle.

**T is the Laplacian of S.** The ket is a function of `r - B`, so its
Laplacian in `r` is its Laplacian in `B`, and with `R = A - B` that is the
Laplacian in `R`, which passes outside the integral. For two primitive s
Gaussians, `p = a + b`, `μ = ab/p`:

```text
T(R) = -∇²_R S(R) / 2 = μ (3 - 2μR²) (π/p)^(3/2) exp(-μ R²)
```

**Directly spherical.** `../s_overlap_host` forms S's harmonic block by
Hobson's theorem, `H(r - A) exp(-a |r - A|²) = H(∇_A) exp(-a |r - A|²) / (2a)^l`,
and those derivatives commute with the Laplacian:

```text
<a|T|b> = (-1)^lb / ((2a)^la (2b)^lb) * H_a(∇_R) H_b(∇_R) (-∇²_R / 2) S(R)
```

A Gaussian's derivatives factor over the axes,
`∂^t_X exp(-μX²) = h_t(X) exp(-μX²)` with `h_(t+1) = -2μ (X h_t + t h_(t-1))`.
So a contracted pair sums, for each `t + u + v = la + lb`,
`h_(t+2) h_u h_v + h_t h_(u+2) h_v + h_t h_u h_(v+2)` over its primitive
pairs. That is S's sum with the same three 1-D recurrences run two orders
further. The contraction with the two harmonics is S's, unchanged. There is
no Boys function, no Hermite table and no Cartesian integral.

**Negligible pairs.** A primitive pair's block is `μ` times a polynomial of
degree `la + lb + 2` in `sqrt(μ) R` times `exp(-μ R²)`. So a primitive pair
with `μ |A - B|² >= 40 + 4 (la + lb + 2)` is skipped, and a pair of shells
past it for its `μ_min` is left zero: S's cutoff, two orders up. Past the
bound the term falls with `μ`, so the smallest `μ` gives the largest term.
A unit test checks that a full block just past the bound is under 1e-15
for every pair of momenta through i, at diffuse exponents (`μ ≈ 0.3`) and
at tight ones (`μ ≈ 310`).

The primitive test is the one that pays, as in `../s_overlap_host`.
Between atoms, the tight primitives of a shell pair that is kept are
mostly past the bound. On gly10 in cc-pVTZ skipping them halves the time
(55 ms to 28 ms on one thread) and changes no element by more than
rounding.

**S and T in one pass.** S is the order-`la + lb` sum of the same `h_t`
under the same weight, without the half. So `overlap_and_kinetic(&basis)`
returns both lower triangles, `(S, T)`, from one exponential, one weight
and one set of recurrences per primitive pair: one more accumulation in
the inner sum and one more harmonic contraction per shell pair. On gly10
in cc-pVTZ that costs 1.26-1.31x T alone, against about 1.75x for
`../s_overlap_host` and `kinetic` run one after the other. The block is
one generic function over `const OVERLAP: bool`, so `kinetic` pays
nothing for S. Its T carries `kinetic`'s bits. S is screened by T's
bound, which is past S's own, so it drops nothing `../s_overlap_host`
keeps.

`kinetic_via_cartesian` is the textbook route, kept to check and time
against: Obara-Saika overlaps per axis to `lb + 2`, the 1-D kinetic
`t_ij = b (2j + 1) s_ij - 2b² s_i(j+2) - j (j - 1) s_i(j-2) / 2`, the
Cartesian `t_x s_y s_z + s_x t_y s_z + s_x s_y t_z`, then a dense sandwich
`Uaᵀ <a|T|b> Ub`, with no pair screened.

The conventions are the tree's, because the basis comes from the
rebuild's device-free crates (`fb-io-xyz`, `fb-io-basis`, `fb-basis`,
`fb-io-npy`): primitives normalized, contractions never renormalized, and
columns `m = -l ..= l`.

The hot loops are `f64::mul_add`, as in `../df_metric`. `.cargo/config.toml`
sets `-C target-feature=+fma`, and `src/lib.rs` is a `compile_error!` on
x86-64 without it. Cargo reads that file only when run from this directory,
and a `RUSTFLAGS` in the environment replaces it.

## Build and run

The toolchain is the image's `/opt/rust`. `Cargo.lock` is not tracked, as
in the sibling directories.

```bash
cd experimental/rust/t_kinetic_host
cargo test --release
D=../../../rust/integrals/one_electron/kinetic/tests/data
cargo run --release -- $D/co.xyz cc-pvtz target/co.npy $D/co_cc-pvtz.npy
        # t-kinetic-host XYZ BASIS [OUT.npy [REFERENCE.npy]]
```

`BASIS` is a compiled-in name or a BSE JSON path. `OUT.npy` is the packed
triangle, one dimension. `REFERENCE.npy` is a full `n × n` matrix whose
lower triangle is compared; past `1e-12 max |T_ref|` the binary exits 1.
The binary times both routes and S and T together (each run repeats until
a second has passed, 20 times at most), and prints the routes' largest
difference.

`../s_overlap_host` is a dev-dependency, for the unit test that checks the
joint S against it. Nothing else uses it.

Nothing in ctest or either gate runs cargo here. This directory is checked
by hand only.

## Checked (2026-11-21)

`cargo test --release`: 10 unit and integration tests and 1 doctest pass;
`cargo clippy --all-targets` and `cargo fmt --check` are clean.

The integration tests compare both routes' lower triangles with PySCF's
`int1e_kin` on the GPU kinetic crate's fixtures and references
(`rust/integrals/one_electron/kinetic/tests/data/`, contractions scaled to
the tree's as `D T D`), at that crate's bar, `1e-12 max |T_ref|`. They
compare `overlap_and_kinetic`'s S with PySCF's `int1e_ovlp` from
`../s_overlap/data/` (the same XYZ files byte for byte) at the same bar,
and require its T to carry `kinetic`'s bits.

| Molecule | Basis | nao | l | max \|T - T_ref\| / max \|T_ref\| | joint S: max \|S - S_ref\| / max \|S_ref\| | max \|T\| |
|---|---|---|---|---|---|---|
| H2O | STO-3G | 7 | p | 3.7e-16 | 4.4e-16 | 29.0 |
| H2O | cc-pVDZ | 24 | d | 2.4e-16 | 4.4e-16 | 29.2 |
| CO | cc-pVTZ | 60 | f | 2.4e-16 | 1.0e-15 | 29.2 |

The unit tests cover:
- two s primitives against the closed form, on one center, apart, and
  past the node of `3 - 2μR²`;
- a single primitive of every momentum through i, by both routes, against
  `<T> = a (l + 3/2)` in every component, with no coupling between
  components;
- the two routes against each other on contracted shells of every pair of
  momenta through i, to `1e-14` of the block's largest element;
- the same on a shell pair that is kept but whose tight primitive pairs
  are skipped (`μ R²` about 1800 for them, 0.6 for the diffuse);
- the cutoff, as above;
- `overlap_and_kinetic` on contracted shells of every pair of momenta
  through i, near and where its tight primitive pairs are skipped: T
  bit for bit `kinetic`'s, S `../s_overlap_host`'s to 1e-14.

A further integration test requires the triangles, T alone and S and T
together, to carry the same bits on one thread as on eight.

## Measured

Release build, Ryzen 9 5900X (12 cores, 24 threads, shared), best of up to
20 runs. gly10 is `rust/pipeline/scf/tests/data/gly10.xyz` (73 atoms); on it
the routes agree to 1.7e-13 (6e-15 of `max |T|`).

| System | Basis | nao | Negligible pairs | Spherical, 1 thread | 8 | 24 | Via Cartesian, 24 | Cartesian / spherical |
|---|---|---|---|---|---|---|---|---|
| H2O | STO-3G | 7 | 0% | | | 13 µs | 35 µs | 2.7x |
| H2O | cc-pVDZ | 24 | 0% | | | 37 µs | 103 µs | 2.8x |
| CO | cc-pVTZ | 60 | 0% | | | 122 µs | 306 µs | 2.5x |
| gly10 | cc-pVDZ | 734 | 47.8% | | | 0.80 ms | 30 ms | 37.6x |
| gly10 | cc-pVTZ | 1678 | 45.6% | 28 ms | 4.4 ms | 3.0 ms | 99 ms | 33.5x |

- **T costs about 1.4x S** on gly10 in cc-pVTZ (28 ms against
  `../s_overlap_host`'s 20 ms on one thread): two more orders of the
  recurrences, and the three-term Laplacian in the inner sum.
- **The ratio grows with the molecule** because the Cartesian route
  screens nothing, and its gap over S's Cartesian route is wider: it
  carries `lb + 2` overlaps per axis and three products per element.
  Before primitive pairs were skipped, gly10 in cc-pVTZ took 55 ms,
  8.2 ms and 5.1 ms on 1, 8 and 24 threads.

S and T together, against T alone and against S (`../s_overlap_host`)
and T run one after the other, measured in one sitting:

| System | Basis | Threads | T alone | S alone | S then T | Together | Together / T alone | S then T / together |
|---|---|---|---|---|---|---|---|---|
| gly10 | cc-pVDZ | 24 | 0.79 ms | 0.65 ms | 1.43 ms | 1.00 ms | 1.27x | 1.43x |
| gly10 | cc-pVTZ | 1 | 28 ms | 21 ms | 49 ms | 36 ms | 1.26x | 1.38x |
| gly10 | cc-pVTZ | 8 | 4.3 ms | 3.3 ms | 7.5 ms | 5.6 ms | 1.31x | 1.34x |
| gly10 | cc-pVTZ | 24 | 2.9 ms | 2.4 ms | 5.3 ms | 3.7 ms | 1.26x | 1.45x |

- **Allocate S zeroed, not as a clone of T.** The first version cloned
  T's zeroed triangle for S, which wrote all 11 MB on one thread before
  any task started. On 24 threads that took S and T together to 5.3 ms,
  no faster than running them one after the other. `vec![0.0; n]` takes
  zeroed pages from the allocator, and each task touches its own first.
- The water and CO rows are rayon's dispatch, as above, and say nothing
  about the joint pass.
- **The small cases measure overhead, not integrals.** At 24 AOs the time
  is mostly rayon's dispatch to 24 threads.
- Not measured: the time on one thread, or by thread count, of anything
  smaller than gly10.
