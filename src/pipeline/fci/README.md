# Pipeline: Fci

Full CI over the SCF's canonical orbitals: the lowest singlets of every
electron in every orbital, and each root's 1-RDM. It is CASCI with no core and
every orbital active, and it makes `CasciStep`'s calls, so it adds no
numerics: what it adds is a refusal, up front, of a determinant space the
device cannot hold, and Davidson options a space of millions of determinants
needs.

## Module

`free_breakfast.pipeline.fci`, target `free_breakfast::pipeline::fci`. It
re-exports `free_breakfast.pipeline.scf` (which re-exports every product in
`run`'s signature) and `free_breakfast.recipes.davidson` (`DavidsonInfo`, in
the product). `run` is defined in the implementation unit, so
`integrals.active_space`, J's and K's classes and the `ci` modules are PRIVATE
links.

## The step

```cpp
struct Fci {
  int32_t nmo, n_electrons;
  std::vector<double> energies;      // per root, total (hartree)
  std::vector<double> spin_squares;  // <S^2> per root
  std::vector<double> rdm1;          // root r at [r nmo^2, (r + 1) nmo^2), row-major
  DavidsonInfo davidson;
};

struct FciStep {
  static constexpr std::string_view kName = "fci";

  int32_t nroots = 1;

  Result<Fci> run(View<Molecule, BasisSet, SphericalTransform, PrimitivePairClasses,
                       CoreHamiltonian, Scf> in,
                  DeviceResources &res) const;
};
```

`run`, all on `res.stream()`:

1. **The estimate.** The device bytes the solve holds at its peak, inside
   `ci_solve`'s Davidson, against `wwrMemGetInfo`'s free memory; past it, the
   step returns `Allocation` before anything is allocated. Without this, the
   first allocation to fail would abort the process (`CreatePolicy`), not
   return an error.
2. **The solve**, `CasciStep::run`'s calls with `ncas` the Scf's orbital
   count and `nelecas` every electron (`casci/README.md`): with no core and
   every orbital active, `CasciStep::orbital_order` would be the identity,
   so the Scf's C goes in as it is.
   `active_space_hamiltonian` builds h and (pq|rs) over every orbital, E_core
   the nuclear repulsion alone; `CiSigma` and `ci_solve` find the roots, and
   `CiSigma::rdm1` each root's 1-RDM. The product keeps the energies, ⟨S²⟩, 1-RDMs and
   Davidson record; the CI vectors are dropped.

A singlet solve runs in the symmetric layout (`../../ci/solver/`), on packed
vectors. With s = C(n, N/2) strings, dim = s² determinants, t = s (s + 1) / 2
packed elements, g = min(t, 2 nroots) guesses, m = max(2 nroots,
min(t, 16 nroots)) subspace columns (`ci_solve`'s choices) and
P = n (n + 1) / 2, the estimate counts, in doubles:

| Holder | Size |
|---|---|
| `ci_solve` | g t guesses, nroots t vectors, its t penalized diagonal, and the step's dim diagonal; after the Davidson, nroots dim unpacked roots, counted where they outweigh the Davidson's |
| `davidson_solve` | 2 m t for V and A V, 3 nroots t for A X, R and the corrections, and the m-sized projections |
| `CiSigma::apply`'s scratch | t (P + 1) for G, one vector's (`apply` goes a vector at a time), plus one ≤ 64 MiB chunk of D |
| `CiSigma`'s tables | two string sets (8 bytes a string, 12 a replacement), the P × (P + 1) contraction |
| The integrals | (pq\|rs) n⁴ and h n² |

For H₂O 6-31G (13 orbitals, 1,656,369 determinants, 828,828 packed, 3 roots)
G is 0.61 GB of a 1.45 GB estimate, the Davidson subspace most of the rest;
in the full layout they were 1.2 GB of 2.8 GB. The J and K classes and
`active_space_hamiltonian`'s own scratch are left out; they scale with the
basis, not the determinants. The comparison is against free memory as the
runtime reports it, so memory the pool has cached counts as used: the refusal
errs toward refusing.

What it settled:

- **Not a `CasciStep` run, and not `ci_solve`'s defaults.** Over H₂O
  6-31G's 1.66M determinants, `ci_solve`'s 1e-12 relative residual is past
  what its Davidson reaches: at 100 iterations it stood at 2.3e-9 absolute
  (MaxIterationsExceeded), and with more it stalled at 1.2e-10, 125 iterations
  in (Divergence). At 1e-11 it converged in 112. `CasciStep` takes no solver
  options, so the step makes its calls itself, with `ci_solve` at a 1e-11
  relative residual and 400 iterations. The energies' error is second order in
  the residual, ~1e-19 hartree; the 1-RDMs' first order, ~1e-9.
- **Singlets only**, as `CasciStep`: the Scf is closed-shell, and the spin
  penalty is `ci_solve`'s default.
- **No range check of its own** beyond `nroots >= 1`, which the estimate
  needs; `nroots` past the space is `ci_solve`'s InvalidConfig. With no core
  and every orbital active, none of `orbital_order`'s range checks can fail.
- **The estimate is not exported.** Nothing outside the step needs it, and
  the oversized case is tested through the step.

Errors: InvalidConfig for `nroots` below 1; Allocation for a space past the
device's free memory, naming the electrons, orbitals, roots, the estimate and
the free memory; GpuRuntime for a failed memory query or copy; then the module
calls' own: `ci_solve`'s InvalidConfig for `nroots` past the space,
MaxIterationsExceeded, Divergence, or Numerical for a root of the wrong spin.

## Tests

`test/gtest/pipeline_fci_gpu.cpp`, against PySCF's `fci.FCI` over its own RHF
in this tree's basis files (`test/gtest/data/fci/`,
`tools/emit_fci_step_reference.py`), through the real registry from a config
naming `[fci]`, its Scf at the defaults: H₂O STO-3G (7 orbitals, 441
determinants), N₂ STO-3G (10 orbitals, 14,400) and, under `slow_test.h`, H₂O
6-31G (13 orbitals, 1,656,369). For the lowest three singlets:

- the energy and ⟨S²⟩ to 1e-8;
- the natural occupations to 1e-7, for each root of no degenerate level;
- each 1-RDM symmetric with trace N.

An FCI energy and its natural occupations are invariant under any rotation of
the orbitals, so the two SCFs' convergence does not enter. Also H₂O cc-pVDZ
(1.8e9 determinants) refused as Allocation when the step runs, its Scf built.
The `[fci]` parsing is in `pipeline_config_host.cpp`.

## Files

| File | Role |
|------|------|
| `fci.cppm` | Module interface: `Fci`, `FciStep` |
| `fci.cpp` | The device-memory estimate, and `run`: the refusal, the Hamiltonian, the CI solve, the 1-RDMs |
| `CMakeLists.txt` | The module library |
