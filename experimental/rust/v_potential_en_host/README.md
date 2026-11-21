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
`A >= B` are visited, one rayon task per pair of atoms. Each task writes
its blocks to a buffer of its own, and one serial pass copies them into the
triangle.

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

**Every primitive pair once.** `W` depends on the two exponents and the two
centers, not on the momenta or the coefficients, and a table built to order
`L` holds every lower one. The carried cc sets are generally contracted and
stored as segments, so an atom's shells repeat the same exponents: on
gly10 only 12–20% of kept primitive pairs are distinct. A pair of atoms
therefore collects its distinct primitive pairs first, each to the highest
`la + lb` any kept shell pair asks of it. It builds each one's `W` once,
and every shell pair's block reads from them. This is why a task is a pair
of atoms: the shells that share a primitive pair are those of the same
two atoms.

**Far nuclei skip the Boys function.** Once `x = p |P - C|²` is past
`../df_metric`'s far-field bound, `40 + 4L`, `exp(-x)` is under 1e-17 of
the Boys values. The seeds of the `R` recurrence then have a closed form,
`R^n_000 = sqrt(pi / p) / 2 * (-1)^n (2n-1)!! / r^(2n+1)`: no `exp`, no
`erf`, no series. On gly10 that covers 92% of (primitive pair, nucleus)
evaluations. The recurrence and the sum into `W` still run for them.

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

| System | Basis | nao | Negligible pairs | Main route, 1 thread | 8 | 24 | Via Cartesian, 1 | 24 | Cartesian / main, 1 thread |
|---|---|---|---|---|---|---|---|---|---|
| H2O | cc-pVDZ | 24 | 0% | 0.12 ms | 66 µs | 79 µs | 0.29 ms | 76 µs | 2.3x |
| CO | cc-pVTZ | 60 | 0% | 0.58 ms | 0.28 ms | 0.27 ms | 1.5 ms | 0.25 ms | 2.6x |
| gly10 | cc-pVDZ | 734 | 48.5% | 0.18 s | 25 ms | 16 ms | 5.1 s | 0.39 s | 29x |
| gly10 | cc-pVTZ | 1678 | 45.9% | 0.60 s | 79 ms | 52 ms | 25 s | 1.8 s | 42x |

- **V still costs more than S**, but no longer 100 times more: gly10 in
  cc-pVTZ takes 0.60 s on one thread against S's 39 ms
  (`../s_overlap_host`). The work is still distinct primitive pairs ×
  nuclei, cubic in the molecule.
- **A small molecule has too few tasks.** One task per pair of atoms is
  three tasks for CO and six for water. They are 2.3x to 2.6x faster than
  the Cartesian route on one thread, but at 24 threads they are slightly
  slower than it (79 µs against 76; 0.27 ms against 0.25), and slower
  than the version with one task per shell was (65 µs; 0.22 ms). Tens of
  microseconds; not chased.
- Not measured: anything larger than gly10.

### What each step bought

gly10, one thread, each step on top of the one before.

| Step | cc-pVDZ | cc-pVTZ |
|---|---|---|
| Every nucleus in one table, one task per shell (the first version) | 0.94 s | 4.06 to 4.34 s |
| The far closed form in place of the Boys function (timed alone, as an experiment) | 0.57 s | 3.02 s |
| With every primitive pair once, one task per pair of atoms | 0.18 s | 0.60 s |

The first row's spread is two runs on a shared machine. The two
optimizations were timed together only after the experiment; sharing
was not timed on its own.

### Where the time went, before those steps (2026-11-21)

Measured on the first version with temporary switches and counters in
`src/lib.rs`, since reverted. gly10, one thread, best of 3 to 5 runs.

**The time split, by switching pieces off.** Each row replaces one piece
with a trivial stand-in; the time it saves is that piece's cost. This is
not a profiler, and the pieces overlap a little.

| Piece | cc-pVDZ (0.94 s) | cc-pVTZ (4.06 s) |
|---|---|---|
| The Boys function | 0.47 s, 50% | 1.56 s, 38% |
| The `R` recurrence | 0.21 s, 22% | 0.99 s, 24% |
| The sum over nuclei into `W` | 0.19 s, 20% | 0.91 s, 22% |
| All three together: the per-nucleus work | 0.82 s, 87% | 3.56 s, 88% |
| The contraction with `E`, the `E` expansion, the spherical transform | within noise | within noise |
| What is left: the loops, the screening, `exp(-mu R²)` | 0.12 s | 0.45 s |

**The work, counted.** Over the kept shell pairs:

| | CO, cc-pVTZ | gly10, cc-pVDZ | gly10, cc-pVTZ |
|---|---|---|---|
| Primitive pairs kept, of those in kept shell pairs | 7027 of 7684 | 324,912 of 839,189 | 1,110,558 of 2,784,105 |
| (primitive pair, nucleus) evaluations | 14,054 | 23.7 M | 81.1 M |
| ... with `x = p |P - C|² >= 40 + 4L`, far | 32.6% | 93.1% | 91.9% |
| ... with `x >= L + 8`, `boys`' large-`x` branch | 41.9% | 97.4% | 97.0% |
| Distinct (atom A, atom B, `a`, `b`) among the kept primitive pairs | 10.4% | 19.9% | 11.6% |
| `R`-table work, per kept pair against once per distinct pair at its highest `L` | 5.4x | 3.4x | 4.9x |

Those two findings are the two steps above. The far closed form was
first timed alone: it left the printed errors against PySCF and the
Cartesian route unchanged in every digit, as they still are with
sharing.

**What is left.** With most Boys calls gone and the tables shared, the
remaining per-nucleus work is the `R` recurrence and the sum into `W`,
mostly for far nuclei. The next step would be a multipole expansion of
those far nuclei per pair of atoms, which removes them from the per-pair
loop altogether. It has not been tried, and the split above has not been
measured again since the two steps.
