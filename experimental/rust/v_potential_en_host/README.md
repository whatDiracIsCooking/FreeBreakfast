# rust/v_potential_en_host — the nuclear attraction V on the host, in Rust

The spherical AO nuclear-attraction matrix
`V[μ][ν] = -sum_C Z_C <μ| 1/|r - C| |ν>` of a molecule in a basis, point
nuclei, computed on the host: plain Rust on rayon's threads, no GPU, no
kernel library, no C ABI and no CMake target. It is the matrix
`rust/integrals/one_electron/potential_en` builds on a card, computed
independently, and built the way `../s_overlap_host` builds S and
`../df_metric` builds the DF metric.

## How

`potential_en(&basis, &charges)` takes an `fb_basis::BasisSet` and the
nuclei as `PointCharge`s (`point_charges(&molecule)` makes them from a
molecule's atoms). It returns the **lower triangle** of V over the spherical
functions, packed: `n (n + 1) / 2` doubles for `n = nao_sphe`, `(μ, ν)` with
`ν <= μ` at `lower_index(μ, ν) = μ (μ + 1) / 2 + ν`. Only shell pairs
`A >= B` are visited, one rayon task per shell, whose rows are one
contiguous slice of the triangle.

**McMurchie-Davidson.** Two primitive Cartesian Gaussians multiply into
Hermite Gaussians on `P = (a A + b B) / p`, and against one point charge

```text
<a| 1/r_C |b> = 2 pi / p * exp(-mu |A - B|²)
                * sum_tuv E^(ij)_t E^(kl)_u E^(mn)_v R_tuv(p, P - C)
```

with `E` the two-center Hermite expansion per axis and `R_tuv` the Hermite
Coulomb integrals from the Boys function `F_n(p |P - C|²)`. The Boys
function is `../df_metric`'s `boys` (the C++ tree's `eval_boys`, ported),
taken as a path dependency rather than copied.

**Every nucleus in one table.** `E` depends on the primitive pair and not on
the nucleus. So the nuclei are summed into the Hermite table first,
`W_tuv = -sum_C Z_C R_tuv(p, P - C)`, and the Cartesian contraction with `E`
runs once per primitive pair, not once per nucleus. The nuclei then cost
one Boys evaluation and one `R` recurrence each. A contracted pair's
Cartesian block is summed over its primitive pairs, then turned spherical
once, through the harmonic block's nonzeros alone.

Unlike S and the metric, V is not formed directly in the solid harmonics.
The Hobson trick those use needs the integral to depend on `A - B` alone,
but here it also depends on `P - C`. Splitting `∇_A` and `∇_B` into `∇_P`
and `∇_R` brings back the same binomial bookkeeping that `E` already does.

**Negligible pairs.** The product of two Gaussians carries
`exp(-mu |A - B|²)`. The Coulomb integral of what is left is at most `2 pi / p`
per unit charge, wherever the nuclei are. So a pair is left zero when

```text
mu |A - B|²  >=  40 + 4 (la + lb) + ln max(1, sum |Z| * 2 pi / p * |c_a c_b|)
```

This is the overlap's cutoff, raised by the most the Coulomb factor and the
coefficients can add. It is tested per shell pair, at the smallest exponents and
largest coefficients, and again per primitive pair inside a pair that is
kept. The unit test checks that a full block just past it is under 1e-15
for every pair of momenta through i, with nuclei on both shells and between
them.

`potential_en_via_cartesian` is the textbook route, kept to check and time
against: per primitive pair and nucleus, Obara-Saika's vertical recurrence
`[e|0]^(m)`, summed over nuclei and primitive pairs, then the horizontal
recurrence once per contracted pair (it holds no exponent) and a dense
sandwich `Uaᵀ V_ab Ub`, with no pair screened. It shares only the Boys
function with the main route.

The conventions are the tree's, because the basis and the molecule come
from the rebuild's device-free crates (`fb-io-xyz`, `fb-io-basis`,
`fb-basis`, `fb-molecule`, `fb-io-npy`). Primitives are normalized,
contractions are never renormalized, and columns run `m = -l ..= l`. Positions
are in bohr through the tree's CODATA 2018 constant. The Boys tables reach order
20, so shells go through `l = 10`; `potential_en` panics past that.

The hot loops are `f64::mul_add`, as in the siblings. `.cargo/config.toml`
sets `-C target-feature=+fma`, and `src/lib.rs` is a `compile_error!` on
x86-64 without it. Cargo reads that file only when run from this directory,
and a `RUSTFLAGS` in the environment replaces it.

## Build and run

The toolchain is the image's `/opt/rust`. `Cargo.lock` is not tracked, as
in the sibling directories.

```bash
cd experimental/rust/v_potential_en_host
cargo test --release
D=../../../rust/integrals/one_electron/potential_en/tests/data
cargo run --release -- $D/co.xyz cc-pvtz target/co.npy $D/co_cc-pvtz.npy
        # v-potential-en-host XYZ BASIS [OUT.npy [REFERENCE.npy]]
```

`BASIS` is a compiled-in name or a BSE JSON path. The nuclei are the XYZ
file's atoms. `OUT.npy` is the packed triangle, one dimension.
`REFERENCE.npy` is a full `n × n` matrix whose lower triangle is compared;
past `1e-12 max |V_ref|` the binary exits 1. The binary times both routes
(each run repeats until a second has passed, 20 times at most) and prints
their largest difference.

Nothing in ctest or either gate runs cargo here. This directory is checked
by hand only.

## Checked (2026-11-21)

`cargo test --release`: 8 unit and integration tests and 1 doctest pass;
`cargo clippy --all-targets` and `cargo fmt --check` are clean.

The integration tests compare both routes' lower triangles with PySCF's
`int1e_nuc` on the device crate's fixtures and references
(`rust/integrals/one_electron/potential_en/tests/data/`; its `README.md`
says what they pin), at that crate's bar, `1e-12 max |V_ref|`. The table is
the test output.

| Molecule | Basis | nao | l | max \|V_ref\| | max \|V - V_ref\| / max \|V_ref\|, both routes |
|---|---|---|---|---|---|
| H2O | STO-3G | 7 | p | 61.7 | 8.1e-16, 7.8e-16 |
| H2O | cc-pVDZ | 24 | d | 62.2 | 1.4e-15 |
| CO | cc-pVTZ | 60 | f | 64.0 | 5.7e-15 |

The unit tests cover:
- two s primitives and one nucleus against the closed form, by both routes;
- a nucleus 400 bohr from a normalized primitive of every momentum through
  i, which must see a unit charge (`-Z / R` on the diagonal, 0 off it, to
  1e-7), by both routes;
- the two routes against each other on contracted shells of every pair of
  momenta through i, off every axis, with three nuclei, to 1e-13 of the
  block's largest element;
- the cutoff, as above.

A further integration test requires the triangle to carry the same bits on
one thread as on eight.

## Measured

Release build, Ryzen 9 5900X (12 cores, 24 threads, shared), best of up to
20 runs. gly10 is `rust/pipeline/scf/tests/data/gly10.xyz` (73 atoms). On it
the routes agree to 6e-15 (cc-pVDZ) and 1.8e-14 (cc-pVTZ) of `max |V|`; no
PySCF reference was run for it.

| System | Basis | nao | Negligible pairs | One table, 1 thread | 8 | 24 | Via Cartesian, 1 | 24 | Cartesian / one table, 1 thread |
|---|---|---|---|---|---|---|---|---|---|
| H2O | cc-pVDZ | 24 | 0% | 0.24 ms | 59 µs | 65 µs | 0.26 ms | 76 µs | 1.1x |
| CO | cc-pVTZ | 60 | 0% | 1.3 ms | 0.22 ms | 0.22 ms | 1.5 ms | 0.24 ms | 1.2x |
| gly10 | cc-pVDZ | 734 | 48.5% | 0.92 s | 0.12 s | 64 ms | 4.9 s | 0.40 s | 5.3x |
| gly10 | cc-pVTZ | 1678 | 45.9% | 4.2 s | 0.54 s | 0.30 s | 23.8 s | 1.8 s | 5.7x |

- **V costs far more than S.** gly10 in cc-pVTZ takes 4.2 s on one thread
  against S's 39 ms (`../s_overlap_host`). Every kept primitive pair pays a
  Boys evaluation and an `R` recurrence for each of the 73 nuclei, so the
  work is pairs × nuclei, cubic in the molecule. Folding the nuclei into
  one table removes the contraction from that product, not the Boys work.
- **On a small molecule the two routes are close.** With two or three
  nuclei the per-nucleus work dominates both, and OS's recurrence costs
  about what `R`'s does. The gap on gly10 is the screening (the textbook
  route screens nothing) plus the per-nucleus contraction it does and
  the main route does not.
- **One task per shell, not per center.** The first version split by
  center, as the siblings do, and lost to the Cartesian route at 24
  threads on H2O and CO (0.17 ms against 72 µs; 0.90 ms against 0.29 ms)
  because two or three centers are two or three tasks. Splitting by
  shell left gly10 unchanged.
- Not measured: anything larger than gly10.

### Where the time goes (2026-11-21)

Experiments only: the crate's code is unchanged. They were made with
temporary switches and counters in `src/lib.rs`, since reverted. gly10,
one thread, best of 3 to 5 runs.

**The time split, by switching pieces off.** Each row replaces one piece
with a trivial stand-in; the time it saves is that piece's cost. This is not
a profiler, and the pieces overlap a little.

| Piece | cc-pVDZ (0.94 s) | cc-pVTZ (4.06 s) |
|---|---|---|
| The Boys function | 0.47 s, 50% | 1.56 s, 38% |
| The `R` recurrence | 0.21 s, 22% | 0.99 s, 24% |
| The sum over nuclei into `W` | 0.19 s, 20% | 0.91 s, 22% |
| All three together: the per-nucleus work | 0.82 s, 87% | 3.56 s, 88% |
| The contraction with `E`, the `E` expansion, the spherical transform | within noise | within noise |
| What is left: the loops, the screening, `exp(-mu R²)` | 0.12 s | 0.45 s |

So the per-nucleus work is the whole cost. The contraction that folding the
nuclei into one table moved out of the nucleus loop no longer shows.

**The work, counted.** Over the kept shell pairs:

| | CO, cc-pVTZ | gly10, cc-pVDZ | gly10, cc-pVTZ |
|---|---|---|---|
| Primitive pairs kept, of those in kept shell pairs | 7027 of 7684 | 324,912 of 839,189 | 1,110,558 of 2,784,105 |
| (primitive pair, nucleus) evaluations | 14,054 | 23.7 M | 81.1 M |
| ... with `x = p |P - C|² >= 40 + 4L`, far | 32.6% | 93.1% | 91.9% |
| ... with `x >= L + 8`, `boys`' large-`x` branch | 41.9% | 97.4% | 97.0% |
| Distinct (atom A, atom B, `a`, `b`) among the kept primitive pairs | 10.4% | 19.9% | 11.6% |
| `R`-table work, per kept pair against once per distinct pair at its highest `L` | 5.4x | 3.4x | 4.9x |

- **Most primitive pairs are repeats.** A carried cc set is generally
  contracted, stored as segments: an atom's s shells (and p shells) repeat
  the same exponents with different coefficients. `W_tuv` depends on the
  two exponents and the two centers, not on the momenta or the
  coefficients, and a table built to the highest `L` holds every lower
  one. So building each distinct pair's table once would divide the
  per-nucleus work by the last row, 3.4x to 4.9x on gly10. This is
  counted, not timed: the shared tables have to be stored and looked up.
- **Almost every nucleus is far.** On gly10, 92 to 93% of evaluations
  are past `df_metric`'s far-field bound, where `exp(-x)` is under 1e-17 of
  what is kept.

**The far seed, timed.** For a far evaluation the Boys function can be
replaced by the closed form, `R^n_000 = sqrt(pi / p) / 2 * (-1)^n
(2n-1)!! / r^(2n+1)`: no `exp`, no `erf`, no series. That alone took gly10
from 0.94 s to 0.57 s in cc-pVDZ (1.64x) and from 4.34 s to 3.02 s in
cc-pVTZ (1.44x). The errors against PySCF on the three fixtures and
against the Cartesian route on gly10 matched the version without it in
every printed digit; the matrices themselves were not compared bit for bit. It was not kept, because experiments were all this was. The
`R` recurrence and the sum into `W` still run for a far nucleus, so this
is the smallest piece of the multipole idea, not the idea.
