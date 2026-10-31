# Hessian (one-electron)

The one-electron nuclear-Hessian contractions on the GPU, over the `d2s_`,
`d2t_` and `d2v_{la}{lb}` kernels FelizJueves generates: for the overlap S,
the kinetic energy T and the nuclear attraction V,

    H[A, k, B, l] = Σ_mn M_mn d²X_mn / dR_{A,k} dR_{B,l}

over every pair of atoms A, B and axes k, l, for a spherical AO matrix M. The
second-derivative integrals themselves are never formed. The signs and halves
of an energy are the caller's, as in `gradient/`: an RHF Hessian takes M = D
for T and V, and M = −W for S. The nuclear repulsion's own Hessian is
`molecule`'s `nuclear_repulsion_hessian`.

## Module

`free_breakfast.integrals.one_electron.hessian`, also re-exported by
`free_breakfast.integrals.one_electron` and `free_breakfast.integrals`.

## Design

`gradient/`'s, one derivative up. Each function checks its inputs, carries M
onto the Cartesian layout (U M Uᵀ), builds each Cartesian AO's atom from the
basis's per-atom shell map, and launches one generated kernel per `mmd`
class, one thread per record, each class on its own stream (`launch_each`),
into one zeroed `[n_atom, 3, n_atom, 3]` device Hessian, which it downloads.
It waits for `res.stream()` to do so: the Hessian is returned on the host,
flat and row-major as `(3 natoms, 3 natoms)`, row `3A + k` and column
`3B + l`, as `nuclear_repulsion_hessian` is. Before each class's kernel a
small gather (`hessian_launch.cu`, the same as `gradient_launch.cu`'s: each
device library keeps its own) writes the class's record atoms. A basis with no
per-atom map is `InvalidConfig`; for V the molecule must be the one the basis
was placed on (its atom count is checked).

**The weight** is `gradient/`'s: `M[a_r, b_c] + M[b_c, a_r]` off a shell
pair's diagonal and `M[a_r, a_c]` on it, so any M, symmetric or not, in
either layout.

**What each record adds** (FelizJueves's `hessian/kernel_h1e.py` and
`docs/implementation/hessian.md`). Each component `d/dX_k d/dY_l`, X and Y
the pair's two centres, is the McMurchie–Davidson derivative rule applied per
axis twice, from E tables raised two indices (T: four on the ket) above the
class: the AA and BB blocks (`k ≤ l`, mirrored) and all nine of AB, written
into both (A, B) and (B, A), 36 atomics per record. For V the operator moves
with its nucleus C, `d/dC_k [t]^(0) = −[t + 1_k]^(0)`, so per nucleus the
record also adds the centre–nucleus blocks (X, C) and (C, X) and the
nucleus–nucleus block (C, C) from the Hermite integrals up to order L + 2,
45 more atomics. The Hessian is symmetric to rounding, not bitwise.

**The diagonal blocks come from translational invariance.** The kernels
differentiate every centre and, for V, every nucleus directly, and write every
block. But a pair with both shells on atom A adds AA + AB + BA + BB into
H[A, A], which cancels exactly in exact arithmetic (a one-centre integral does
not move with its centre) while each term grows with the exponents: a tight
core primitive leaves its rounding there, 1e-11 to 3e-11 of max|H| in water's
cc-pVDZ T, changing from run to run with the order of the atomics. The
off-diagonal blocks (A ≠ B) carry no one-centre term, so after the download
each diagonal block is replaced by

    H[A, k, A, l] = −Σ_{B ≠ A} H[B, k, A, l]

which brought that T to 5e-12 of max|H| against PySCF, stable from run to run
(PySCF's own diagonal block is that far from its own invariance). The sum over
the row atom is then zero by construction; the sum over the column atom,
Σ_B H[A, k, B, l], is the check, since it vanishes only if the computed (A, B)
and (B, A) blocks are each other's transposes. The kernels still form the
diagonal blocks the host discards; kernels that skip them (the AA and BB
components of every record, and V's nucleus–nucleus block) are FelizJueves's
to emit, a follow-up.

**The kernels.** `kernels/{d2s,d2t,d2v}_{00,01,02,03,11,12,13,22,23,33}.cu`
(every `la <= lb <= 3`), FelizJueves's `--hess-one-e-l 3` ladder, generated
by `tools/emit/one_electron_hessian_kernels.py` and committed as emitted, each
one explicit specialization of its family's kernel template
(`overlap_hessian`, `kinetic_hessian`, `potential_hessian`) in the one device
library; `hessian_launch.cu` includes only `kernels/<family>.h` and launches
through each `<family>_kernels` table, bounded by its extent. `d2v` is
low-stack from `la + lb = 4` ((pf), (dd), (df), (ff)), FelizJueves's own
choice (`kernel_h1e.h1e_lowstack`), unrolled below; its units include the
shared `eval_boys.h`. The largest unit is `d2t_33.cu`, about 456 KB.

**Limits.** f per shell (`launch`'s `kOneElectronMaxL`), `n_ao² ≤ INT32_MAX`
over Cartesian AOs, and every input on the call's device; each is
`InvalidConfig` before anything is launched. The classes must be
`primitive_pair_classes` of the basis given: their AO offsets are not checked
against it. No screening, and V is one thread per record with every nucleus
serial inside it, as `gradient/`'s is.

## Public API

| Symbol | Role |
|---|---|
| `overlap_hessian(res, basis, classes, m)` | `Result<std::vector<double>>`: Σ M d²S/dR dR, `(3 natoms, 3 natoms)` row-major |
| `kinetic_hessian(res, basis, classes, m)` | The same for T |
| `potential_hessian(res, basis, classes, molecule, m)` | The same for V, its operator terms on every nucleus included |

## Usage

```cpp
import free_breakfast.integrals;
import free_breakfast.molecule;
import free_breakfast.recipes.matrix_ao;
namespace fb = free_breakfast;

const auto basis = fb::place(stamp.primitive_normalized(), molecule);
const auto classes = fb::primitive_pair_classes(res, *basis);   // once per basis
fb::SphericalMatrixAO d(res, *basis);                             // the density
const auto h_t = fb::kinetic_hessian(res, *basis, *classes, d);
const auto h_v = fb::potential_hessian(res, *basis, *classes, molecule, d);
const std::vector<double> h_nn = fb::nuclear_repulsion_hessian(molecule);
```

## Tests

`test/gtest/one_electron_hessian_gpu.cpp`: the skew water in cc-pVDZ and
cc-pVTZ (every class through (ff)) against libcint's `int1e_ipipovlp` /
`ipovlpip`, `int1e_ipipkin` / `ipkinip` and `int1e_ipipnuc` / `ipnucip` plus
the operator's `int1e_ipiprinv` / `iprinvip` at each nucleus, assembled as
PySCF's `hessian.rhf.partial_hess_elec` assembles them and contracted with
`test/gtest/data/hessian/`'s recipe matrix, at 1e-11 of max|H|; each one's
translational sum rule over the column atom, Σ_B H[·, B, l] below 1e-11 of
max|H|;
`nuclear_repulsion_hessian` against PySCF's `hess_nuc` at 1e-13 of its
max, and bitwise symmetric; plus the refusals of a matrix of the wrong order,
a basis with no per-atom map and a molecule of another atom count. The
reference script, `tools/emit/one_electron_hessian_reference.py`, holds each
reference Hessian to a fourth-order central difference of the matching
libcint gradient.

## Files

| File | Role |
|------|------|
| `hessian.cppm` | Module interface: `overlap_hessian`, `kinetic_hessian`, `potential_hessian` |
| `hessian.cpp` | Implementation: the checks, the Cartesian M, the AO atoms, the launches, the diagonal blocks |
| `hessian_launch.cu` | The launchers over the generated kernels' tables, and the record-atom gather |
| `hessian_bridge.h` | The launchers' declarations, shared by the `.cu` and the implementation unit's GMF |
| `kernels/` | The generated kernels and their three headers, and the `.clang-format` that leaves them unformatted |
| `CMakeLists.txt` | The device library `free_breakfast.integrals.one_electron.hessian.device` and the module |
