# rust/sad_factor — the SAD guess as a factor, on the host

The SAD initial guess as `L`, `n_ao × rank`, with `D_SAD = 2 L Lᵀ`: what a
J/K build that takes the occupied orbitals C of `D = 2 C Cᵀ` can be given
in C's place on the one pass that has no C. Plain Rust over the rebuild's
device-free crates (`fb-sad`, `fb-io-sad`, `fb-basis`, `fb-io-*`): no GPU,
no kernel library, no CMake target. This is the host half only; nothing
here builds J or K.

## Why

`v0.0.4`'s SCF builds K from C's occupied columns on every pass but the
first of a fresh run, whose D is the guess and has no factor
(`../v0.0.4/v0.0.4.html`, "K from the occupied orbitals"). At gly10 in
cc-pVDZ on one GPU that pass is about 12 s of a 58.3 s SCF, against about
2.6 s for a pass from C. K from a factor costs in proportion to the
factor's width.

## How

D_SAD is each atom's stamp `D_A` on its diagonal block and zero between
atoms. A stamp is an atomic SCF's density, `Σ_i n_i c_i c_iᵀ` over its
occupied orbitals with `n_i >= 0`: positive semidefinite, of rank the
atom's occupied count. So

```text
D_A = V Λ Vᵀ   →   L_A = V[:, λ > tol] · sqrt(λ / 2)
```

and L is each atom's `L_A` on that atom's rows, in columns of its own,
atom after atom in the basis's order. The molecule's `n_ao × n_ao` matrix
is never decomposed, or formed.

- **One decomposition per distinct (Z, shells)**, as `fb_sad` runs one SCF
  per distinct (Z, shells): `fb_sad::distinct_stamps` gives the stamps and
  where each goes, `fb_sad::stamp_density` each stamp's block, baked
  (`data/sad/`) or computed. `data/sad/` is read as it is.
- **Cyclic Jacobi**, `fb_sad`'s solver for its l blocks, copied because
  that crate keeps it private. A stamp is an atom's AOs on a side.
- **The cut** is `RANK_TOLERANCE = 1e-10` of the stamp's largest
  eigenvalue. The nonzero eigenvalues of `C n Cᵀ` are bounded below by
  `n_min / λ_max(S)`, so the gap is wide: at gly10 the smallest kept is
  0.49 and the largest dropped 5e-18.
- **The columns are not the atomic orbitals.** Those are orthonormal in
  the atom's overlap, the eigenvectors in the plain dot product. The two
  span the same space; only `L Lᵀ` is read.

## What it gives

`sad_factor(&basis, atoms, baked)` takes `fb_sad::sad_density`'s arguments
and returns L column-major with its width and each element's factor, rank
and gap. `stamp_factor(density, n)` is one stamp's. `density_of` forms
`2 L Lᵀ`, for checks.

**The width is the atoms' occupied counts summed, not the molecule's.**
The spherical average spreads a subshell over every m, so C, N and O each
bring 5 columns (1s, 2s, three 2p) and H brings 1, in any basis.

| | n_ao | occupied | L's width | of occupied | time | max \|2 L Lᵀ − D_SAD\| |
|---|---|---|---|---|---|---|
| gly10, cc-pVDZ | 734 | 155 | 237 | 1.53× | 0.40 ms | 1.6e-15 |
| gly10, cc-pVTZ | 1678 | 155 | 237 | 1.53× | 0.43 ms | 4.0e-15 |
| gly20, cc-pVDZ | 1444 | 305 | 467 | 1.53× | 0.87 ms | 1.6e-15 |
| gly20, cc-pVTZ | 3298 | 305 | 467 | 1.53× | 1.4 ms | 4.0e-15 |
| gly40, cc-pVDZ | 2864 | 605 | 927 | 1.53× | 1.5 ms | 1.6e-15 |
| gly40, cc-pVTZ | 6538 | 605 | 927 | 1.53× | 1.7 ms | 4.0e-15 |

gly20 and gly40 are the binary's runs on geometries the tree does not
carry; the tests stop at gly10.

The time is `sad_factor` whole, with baked stamps, on one thread. From the
width alone, the guess's pass at gly10 would be about 4 s where it is 12 s
now. That is an estimate from the pass's cost per column, not a
measurement: no card has run it.

## Run

```bash
cargo test --release
cargo run --release -- ../../../rust/steps/tests/data/gly10.xyz cc-pvdz [L.npy]
```

`sad-factor XYZ BASIS [OUT.npy]` prints each element's rank and gap, L's
width beside the occupied count, and `max |2 L Lᵀ − D_SAD|` against
`fb_sad::sad_density`; writes L as an `n_ao × rank` `.npy`; and exits 1
past 1e-12. The tests hold the same bar on gly5 (STO-3G, cc-pVDZ) and
gly10 (cc-pVDZ), and each stamp's rank against the Madelung fill's count.

## Not done

- The device half: the guess publishing L beside D₀, and the SCF's first
  pass taking `Density::Occupied(L, rank)`.
- Whether `v0.0.4`'s pass scratch holds a factor wider than `n_occ`
  (`jk_layout`): not read.
- A stamp that is not positive semidefinite is not refused: its negative
  eigenvalues are dropped with the null space and show in
  `largest_dropped`.
