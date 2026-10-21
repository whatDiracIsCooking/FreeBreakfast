# `atomic_scf/`

The spherically averaged atomic SCF a production SAD guess is made of, on the
host: one atom's restricted Hartree–Fock with fractional occupations over its
own one-center integrals, and the block-diagonal molecular guess assembled from
it.

## Module

`free_breakfast.experimental.atomic_scf`

## How it relates to `src/sad`

`free_breakfast.sad` lays each atom's aufbau occupations straight onto the
diagonal, one basis function standing in for one orbital. This module keeps
those occupations — it calls `subshell_occupation` and walks `kAufbauOrder` —
and seats them on real orbitals instead: the k-th subshell of momentum l goes
on the k-th lowest orbital of the atom's l block, its electrons spread evenly
over the 2l+1 m. The result is a dense atomic block, with the 1s/2s coupling
the diagonal guess leaves out, whose trace against S is Z exactly.

## The model

A density spread evenly over m is spherical, so its Fock matrix is
block-diagonal by (l, m) and the same for every m. The SCF therefore
diagonalizes one small radial block per l — the m-average of F over that
momentum's shells — which is PySCF's `AtomSphAverageRHF` exactly, and is
tested against it. The energy is the restricted closed-shell expression over
the spin-summed density, `E = ½ tr D (H + F)`, `F = H + J − K/2`, so a
fractionally occupied orbital keeps part of its own self-interaction: hydrogen
in STO-3G is −0.2729, not −0.4666.

Core-Hamiltonian start, Pulay DIIS on the stacked per-l commutators
`F P S − S P F`, canonical orthogonalization within each l block. The tolerance
bounds both the largest commutator element and the energy change.

**Configurations.** The plain Madelung fill, as in `src/sad`. PySCF's
`NRSRHF_CONFIGURATION` agrees with it per l through Z = 20 and for Ti, V, Cr
and Zn, and differs from Sc onward for most of the open-d and open-f elements
(Fe: Madelung 4s² 3d⁶, PySCF's table 4s⁰ 3d⁸ in its spherical average). The
reference data sticks to elements where the two agree.

## One-center integrals

Every integral has all four centers at the nucleus, so in McMurchie–Davidson
terms every distance is zero:

- the Hermite expansion of `x^n exp(−p x²)` about its own center obeys
  `E^{n+1}_t = E^n_{t−1} / 2p + (t+1) E^n_{t+1}`;
- the Boys function is only ever needed at 0, `F_n(0) = 1/(2n+1)`, and the
  Hermite Coulomb integrals collapse to
  `R_{tuv}(α) = (t−1)!! (u−1)!! (v−1)!! (−2α)^{N/2} / (N+1)` for even t, u, v
  (zero otherwise), `N = t+u+v`.

So S, T, V and `(ij|kl)` are closed form, with no quadrature or table. They are
computed over the Cartesian components, all shell quartets with no
permutational symmetry, then carried to the spherical layout with
`spherical_transform_matrix` — the tree's normalization convention, so the
oxygen block of `test/gtest`'s libcint golden S and T is matched to 1e-12. The
whole n⁴ tensor is held; past 96 Cartesian AOs (≈680 MB) it is refused as
`Unsupported`.

## Public API

| Symbol | Role |
|---|---|
| `AtomicIntegrals` | One atom's S, T, V and `(ij\|kl)` over its spherical layout. |
| `atomic_integrals(z, shells)` | Those integrals for shells on one center. |
| `AtomicScf` | Energy, iterations, the spin-summed density and the per-l orbital energies. |
| `atomic_scf(z, shells, tolerance, max_iterations)` | The atomic SCF. |
| `atomic_scf_density(basis, atoms)` | The molecular guess: one SCF per distinct (Z, shells), each on its atom's block. |

`atomic_scf_density` takes a `std::span<const Atom>` rather than a `Molecule`,
which would need a device for its nuclei; the whole module is host-only.

## Tests

`test/experimental/atomic_scf_host.cpp`, in the host binary
`free_breakfast_tests_experimental_host`:

- the integrals against closed forms for one s primitive, their 8-fold and
  translational symmetry, and the oxygen block of
  `test/gtest/data/one_electron/h2o_{sto3g,ccpvdz}.json`;
- the SCF against PySCF 2.14's `AtomSphAverageRHF` for H–Ar and Zn in STO-3G
  and H, He, C, O, Ne in cc-pVDZ (energy to 1e-8, orbital energies to 1e-6),
  the trace and symmetry of the density, and idempotency for closed-shell Ne;
- the block-diagonal assembly and the refusals.

The reference data is `test/experimental/data/atomic_scf.json`, written by

```sh
uv run --group tools experimental/atomic_scf/emit_reference.py
```

## Not done

- Taskflow: the assembly runs the per-element SCFs serially.
- Charged atoms, ECPs, and the configurations where PySCF's table departs from
  Madelung.
- Permutational symmetry and screening in the two-electron integrals.

## Files

| File | Role |
|------|------|
| `atomic_scf.cppm` | Module interface |
| `atomic_scf.cpp` | The integrals, the SCF and the assembly |
| `emit_reference.py` | Writes the PySCF reference data the tests read |
| `CMakeLists.txt` | Build configuration |
