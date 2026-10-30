# SAD

The Superposition-of-Atomic-Densities initial guess: each atom's spherically averaged atomic SCF density on its block of the molecular density matrix, in the spherical AO layout.

## Module

`free_breakfast.sad`

## Design

Pure host data with no GPU link, so it builds and tests on a device-less box. Its inputs are a `BasisSet` made by `place` and the atoms it was placed over, as a `std::span<const Atom>` (a `Molecule`'s `atoms()`) rather than the `Molecule`, whose nuclei would need a device. Its output is a `std::vector<double>`. The SCF step copies it to the device anyway, and a `wwr` host buffer would pull the GPU runtime into a module that has no other use for it.

**The model.** One restricted Hartree–Fock per atom, with fractional occupations, over the atom's own shells. Its Z electrons (the neutral atom's count) fill subshells in the Madelung order (`1s 2s 2p 3s 3p 4s 3d 4p …` through `7p`, enough for Z ≤ 118). The plain aufbau order is used; the d/f anomalies (Cr, Cu, …) are not encoded, because a spherically averaged guess does not need them. The k-th subshell of momentum l is seated on the k-th lowest orbital of the atom's l block, its electrons spread evenly over the `2l+1` m.

A density spread evenly over m is spherical, so its Fock matrix is block-diagonal by (l, m) and the same for every m. The SCF therefore diagonalizes one small radial block per l (the m-average of F over that momentum's shells), which is PySCF's `AtomSphAverageRHF` exactly, and is tested against it. The energy is the restricted closed-shell expression over the spin-summed density, `E = ½ tr D (H + F)`, `F = H + J − K/2`, so a fractionally occupied orbital keeps part of its own self-interaction: hydrogen in STO-3G is −0.2729, not −0.4666.

Core-Hamiltonian start, Pulay DIIS on the stacked per-l commutators `F P S − S P F`, canonical orthogonalization within each l block. The tolerance (1e-9) bounds both the largest commutator element and the energy change.

One SCF runs per distinct (Z, shells) pair, so a molecule pays for its elements, not its atoms. Its density is an `AtomicDensityStamp`: one-center, so geometry-free, the same block for every atom with that Z and those shells. `place` writes it onto each such atom's diagonal block (`docs/style.md`, "Stamps"); between atoms the guess is zero.

**Baked stamps.** Both forms take an optional `SadStampLookup`, a plain function pointer from (basis name, Z) to an atom's block. For a basis that carries a name (`BasisSet::name`, set only for a compiled-in set), each distinct stamp asks it first, and an element it has skips its SCF. The baked block is checked against the shells by size only (`ErrorKind::InvalidConfig` if it differs); the name is what says the shells are the ones it was computed over. This module never knows where the blocks come from: the Guess step passes `free_breakfast.io.sad`'s `builtin_sad_density`, which reads the stamps `data/sad/` compiles in. Those stamps are written by `tools/sad`, which links this module and passes no lookup, so it can only ever compute them. The lookup is called from the task form's worker threads, so it must be safe to call concurrently.

**Two forms, one result.** `sad_density(basis, atoms)` runs the SCFs one after another and returns the guess. `sad_density(flow, basis, atoms, out)` adds one task to a caller's `tf::Taskflow` (or an enclosing task's `tf::Subflow`) and returns it, so the guess can be ordered against the rest of a graph. When it runs, it finds the distinct stamps, spawns a subflow with one task per stamp that runs its SCF and then `place`s it on each of its atoms, and writes the guess or the error into `out`. The two share everything but that loop. Each SCF is independent and each atom's block is its own, so the stamp tasks need no lock and the result is bitwise the serial one; the error is too, the first failing atom's, though every other SCF still runs.

**Lifetimes in the task form.** Nothing is read at the call: `basis`, the atoms and `out` are captured by reference and used when the task runs, so they must outlive the run, and a predecessor may still be filling `basis` in. Inside, the stamps, their statuses and the density are locals of the task's frame, sized once before any stamp task exists (each holds a reference into them). The task joins its subflow explicitly, because Taskflow's implicit join happens only after the task's callable returns, when those locals are already gone.

**Configurations.** The plain Madelung fill. PySCF's `NRSRHF_CONFIGURATION` agrees with it per l through Z = 20 and for Ti, V, Cr and Zn, and differs from Sc onward for most of the open-d and open-f elements (Fe: Madelung 4s² 3d⁶, PySCF's table 4s⁰ 3d⁸ in its spherical average). The reference data sticks to elements where the two agree.

**What the trace means.** Each atomic block holds Z electrons against the atom's overlap, so `trace(D S)` is Σ_A Z_A exactly. `trace(D)` is not: shells of one atom overlap.

**Neutral atoms.** The guess places Σ_A Z_A electrons whatever the molecule's net charge. Rescaling or re-placing charge is the SCF driver's call.

**Why spherical.** Every shell in this tree is spherical (see `src/basis/README.md`). A consumer that wants a Cartesian-layout density takes `U D Uᵀ` with `U = spherical_transform_matrix(basis)`. Density transforms the opposite way to the `Uᵀ M U` used for Fock and overlap matrices.

**Which shells belong to which atom.** The basis says so: `place` records each atom's shell range in the `BasisSet` (`natoms`, `atom_shell_offset`, `atom_shell_count`), and this module reads it. A basis built from a bare shell list carries no grouping (`natoms() == 0`), and one placed over different atoms has a different atom count; both are `ErrorKind::InvalidConfig`. So is an atom whose basis has fewer linearly independent orbitals of some momentum than it has occupied subshells of it: an SCF over the electrons that fit would be of a different ion.

## One-center integrals

Every integral of the atomic SCF has all four centers at the nucleus, so in McMurchie–Davidson terms every distance is zero:

- the Hermite expansion of `x^n exp(−p x²)` about its own center obeys `E^{n+1}_t = E^n_{t−1} / 2p + (t+1) E^n_{t+1}`;
- the Boys function is only ever needed at 0, `F_n(0) = 1/(2n+1)`, and the Hermite Coulomb integrals collapse to `R_{tuv}(α) = (t−1)!! (u−1)!! (v−1)!! (−2α)^{N/2} / (N+1)` for even t, u, v (zero otherwise), `N = t+u+v`.

So S, T, V and `(ij|kl)` are closed form, with no quadrature or table. They are computed over the Cartesian components, then carried to the spherical layout with `spherical_transform_matrix`, the tree's normalization convention. The whole n⁴ tensor is held; an atom past 96 Cartesian AOs (≈680 MB) is refused as `ErrorKind::Unsupported`.

`(ij|kl)` uses its 8-fold permutational symmetry: only the canonical shell quartets are evaluated (`a ≥ b`, `c ≥ d`, pair `ab ≥` pair `cd`), each contracted over its primitives and then written to all eight permutations of its indices, so the held tensor is exactly symmetric and its evaluation costs about an eighth of the full loop. The spherical transform and the Fock build still run over the whole tensor.

## Public API

| Symbol | Role |
|---|---|
| `sad_density(basis, atoms, baked = nullptr)` | `Result<std::vector<double>>`, the n×n column-major guess, n = `nao_sphe()`. |
| `sad_density(flow, basis, atoms, out, baked = nullptr)` | `tf::Task`: the same guess as one task of `flow`, its SCFs in parallel, written into `out`. |
| `SadStampLookup` | `std::optional<std::vector<double>> (*)(std::string_view basis, int32_t z)`: where `baked` finds an element's block instead of running its SCF. |

Nothing else is exported, except that `free_breakfast.third_party.taskflow` is re-exported for the task form's `tf::FlowBuilder` and `tf::Task`. The aufbau fill and the atomic SCF are internal partitions, imported only by `sad.cpp`, so none of their names are visible to an importer.

## Usage

```cpp
import free_breakfast.basis;
import free_breakfast.sad;

auto basis = free_breakfast::place(stamp, molecule);  // records the atom grouping
if (basis) {
  auto density = free_breakfast::sad_density(*basis, molecule.atoms());
  // *density is nao_sphe x nao_sphe, column-major; trace(D S) == Sum_A Z_A.
}

// The task form, on the node's executor; basis, atoms and out outlive the run.
free_breakfast::tf::Taskflow flow;
free_breakfast::Result<std::vector<double>> guess;
free_breakfast::tf::Task sad = free_breakfast::sad_density(flow, *basis, molecule.atoms(), guess);
node.executor().run(flow).wait();
```

## Tests

- `test/gtest/sad_host.cpp`: each atom's block against PySCF 2.14's `AtomSphAverageRHF` density for H–Ar and Zn in STO-3G, H, He, C, O, Ne in cc-pVDZ, and C and O in cc-pVTZ (f shells; to 1e-7), the block-diagonal assembly, seating by momentum rather than shell position, a ghost atom, and the refusals. Over formaldehyde in cc-pVDZ, the task form against the serial one bitwise, with a predecessor building the basis, and reporting the first failing atom's error. A lookup standing in for an element's SCF in both forms, ignored for an unnamed basis, and refused at the wrong size.
- `test/io/sad_host.cpp`: the compiled-in stamps against this module's SCF, bitwise.
- `test/gtest/sad_gpu.cpp`: the guess over a basis `place`d on a `Molecule`.

The reference data is `test/gtest/data/sad/atomic_scf.json`, written by `tools/emit/sad_reference.py`.

## Not done

- Parallelism inside one atom's SCF (its ERI, transform and Fock loops): the task form runs the atoms in parallel, so the heaviest element sets the time.
- Charged atoms, ECPs, and the configurations where PySCF's table departs from Madelung.
- Screening in the two-electron integrals, and permutational symmetry past their evaluation (the spherical transform, the Fock build).

## Files

Each class and each concern is its own internal partition (`module free_breakfast.sad:name;`), declared and defined in one `.cppm` (`docs/style.md`, "Module-private code"). None is an interface partition, so none has to be re-exported by `sad.cppm`, and `sad.cppm` imports none of them.

| File | Partition | Role |
|------|-----------|------|
| `sad.cppm` | primary | Module interface: `sad_density` |
| `sad.cpp` | implementation | `AtomicDensityStamp`, its `place`, and both forms of `sad_density`: one stamp per distinct (Z, shells), baked or computed, placed on each of its atoms' blocks, serially or as one task per stamp |
| `aufbau.cppm` | `:aufbau` | `kAufbauOrder` and `subshell_occupation`, the Madelung fill |
| `atomic_scf/scf.cppm` | `:scf` | `AtomicScf` and `atomic_scf`: the Fock build, the energy, the per-l commutators, the iteration |
| `atomic_scf/integrals.cppm` | `:integrals` | `AtomicIntegrals` and `atomic_integrals`: checks the shells, calls the two families below, carries them to the spherical layout |
| `atomic_scf/hermite.cppm` | `:hermite` | The same-center Hermite expansion and the zero-distance Hermite Coulomb integrals both families build on |
| `atomic_scf/one_electron.cppm` | `:one_electron` | S, T and V over the Cartesian layout |
| `atomic_scf/eri.cppm` | `:eri` | `(ab\|cd)` over the Cartesian layout |
| `atomic_scf/gemm.cppm` | `:gemm` | `gemm_tn` and `gemm_nn`, the l blocks' matrix products |
| `atomic_scf/eigen.cppm` | `:eigen` | `SymmetricEigen` and `jacobi_eigen`, cyclic Jacobi |
| `atomic_scf/diis.cppm` | `:diis` | `AtomicDiis`, Pulay DIIS over flat vectors |
| `atomic_scf/angular_block.cppm` | `:angular_block` | `AngularBlock` (its m-average and diagonalization), `angular_blocks` (orthogonalizer and aufbau seating per l) and `angular_density` (the scatter back to the full layout) |
| `CMakeLists.txt` | | Build configuration |
