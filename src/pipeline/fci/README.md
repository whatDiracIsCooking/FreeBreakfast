# Pipeline: Fci

Full CI over the SCF's canonical orbitals: the lowest roots of the
molecule's spin of every electron in every orbital, and each root's 1-RDM. It is CASCI with no core and
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
struct FciOutput {
  int32_t nmo, n_electrons;
  std::vector<double> energies;      // per root, total (hartree)
  std::vector<double> spin_squares;  // <S^2> per root
  std::vector<double> rdm1;          // root r at [r nmo^2, (r + 1) nmo^2), row-major
  DavidsonInfo davidson;
};

struct FciStep {
  static constexpr std::string_view kName = "fci";

  int32_t nroots = 1;

  Result<FciOutput> run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput,
                       CoreHamiltonianOutput, ScfOutput> in,
                  DeviceResources &res) const;
};
```

`run`, all on `res.stream()`:

1. **The refusal.** The step's own footprint (`footprint(dims)`, below), from
   dimensions it builds out of its inputs (the basis, mmd's record counts
   from `PrimitivePairClassesOutput`, syevd's work through the CI's orders),
   against `DeviceResources::memory_budget()` less one pool chunk (32 MiB)
   of fragmentation headroom, the pipeline's, with what the device
   workspace has still to grow by. Past that, at the least sigma
   scratch, the step returns `Allocation` before anything is allocated;
   without this, the first allocation to fail would abort the process
   (`CreatePolicy`), not return an error. A space of more determinants than
   an `int32_t` counts is refused there too.
2. **The solve**, `CasciStep::run`'s calls with `ncas` the ScfOutput's orbital
   count and `nelecas` every electron (`casci/README.md`): with no core and
   every orbital active, `CasciStep::orbital_order` would be the identity,
   so the ScfOutput's C goes in as it is.
   `active_space_hamiltonian` builds h and (pq|rs) over every orbital, E_core
   the nuclear repulsion alone, and it and the J and K classes are freed once
   h and (pq|rs) are on the host, before the CI allocates; `ci::Sigma` and
   `ci::solve` find the roots, and `ci::Sigma::rdm1` each root's 1-RDM. The
   product keeps the energies, ⟨S²⟩, 1-RDMs and Davidson record; the CI
   vectors are dropped.

**The footprint** is `then(drop(J, K, the Hamiltonian), drop(the CI))`, each
piece its library's own: `coulomb_pair_classes_footprint`,
`ordered_pair_classes_footprint`, `active_space_hamiltonian_footprint`, and
`pipeline/active_space`'s `active_space_ci_footprint` (the string set or
sets, `ci::sigma_footprint`, the diagonal, `ci::solve_footprint`, the
1-RDMs). Nothing is persistent: the product is on the host.

**The sigma's scratch is the one adaptive part.** `ci::Sigma::apply`'s G, one
vector's (`apply` goes a vector at a time) and a ≤ 64 MiB chunk of D, is the
Sigma's own pool buffer (`../../ci/sigma/README.md`). The footprint counts it
at the least the sigma runs with, one string's block of G
(`ci::sigma_apply_bytes` at limit 0), as `core/footprint` has an adaptive
routine's scratch: pre-flight then refuses only a plan that cannot run even
blocked. At run time the step finds the largest `ci::Sigma::set_scratch_limit`
its footprint fits the budget at, by bisection over the footprint itself, and
passes it on; when G for the whole space fits, it sets none, and the sigma is
the unblocked single pass, the same result to the bit. When it does not, G
goes a block of strings at a time, at twice the gathers and gemm. The
footprint's other terms, the CI vectors, the Davidson subspace and the n⁴
integrals among them, are exact (`test/gtest/pipeline_step_footprint_gpu.cpp`
holds a capped run to the footprint byte for byte on the 3080).

What it settled:

- **Not a `CasciStep` run, and not `ci::solve`'s defaults.** Over H₂O
  6-31G's 1.66M determinants, `ci::solve`'s 1e-12 relative residual is past
  what its Davidson reaches: at 100 iterations it stood at 2.3e-9 absolute
  (MaxIterationsExceeded), and with more it stalled at 1.2e-10, 125 iterations
  in (Divergence). At 1e-11 it converged in 112. `CasciStep` takes no solver
  options, so the step makes its calls itself, with `ci::solve` at a 1e-11
  relative residual and 400 iterations. The energies' error is second order in
  the residual, ~1e-19 hartree; the 1-RDMs' first order, ~1e-9.
- **The molecule's spin over closed-shell orbitals**, as `CasciStep`: an
  open-shell ScfOutput (`n_alpha != n_beta`) is refused `Unsupported` first,
  and the sector is N_α − N_β = 2S from `Molecule::multiplicity()`, at
  `ci::solve`'s default spin penalty. The electron count is the ScfOutput's
  `n_alpha + n_beta`; a multiplicity that does not fit it (2S above N, of the
  other parity, or N_α above the orbitals) is `InvalidConfig`, before the
  refusal.
- **No range check of its own** beyond `nroots >= 1`, which the footprint
  needs; `nroots` past the space is `ci::solve`'s InvalidConfig. With no core
  and every orbital active, none of `orbital_order`'s range checks can fail.

Errors: InvalidConfig for `nroots` below 1 or a multiplicity that does not
fit the electrons; Allocation for a space past the
device's memory budget, naming the electrons, orbitals, roots, the footprint
at the least scratch and the budget, or of more determinants than an
`int32_t` counts; GpuRuntime for a failed memory query or copy; then the module
calls' own: `ci::solve`'s InvalidConfig for `nroots` past the space,
MaxIterationsExceeded, Divergence, or Numerical for a root of the wrong spin.

## Tests

`test/gtest/pipeline_fci_gpu.cpp`, against PySCF's `fci.FCI` over its own RHF
in this tree's basis files (`test/gtest/data/fci/`,
`tools/emit/fci_step_reference.py`), through the real registry from a config
naming `[fci]`, its Scf step at the defaults: H₂O STO-3G (7 orbitals, 441
determinants), N₂ STO-3G (10 orbitals, 14,400) and, under `slow_test.h`, H₂O
6-31G (13 orbitals, 1,656,369), and H₂O STO-3G's triplet (245 determinants,
the chain run as the singlet, then the step by hand with the molecule at
multiplicity 3). For the lowest three roots of the spin:

- the energy and ⟨S²⟩ to 1e-8;
- the natural occupations to 1e-7, for each root of no degenerate level;
- each 1-RDM symmetric with trace N.

An FCI energy and its natural occupations are invariant under any rotation of
the orbitals, so the two SCFs' convergence does not enter. Also H₂O cc-pVDZ
(1.8e9 determinants) refused as Allocation when the step runs, its ScfOutput built,
and multiplicities 2, 11 and 13 for H₂O STO-3G refused as InvalidConfig.
The `[fci]` parsing is in `test/pipeline/config_host.cpp`. The footprint, in
`test/gtest/pipeline_step_footprint_gpu.cpp`: N₂ STO-3G under a pool cap of
its footprint and the pool's rounding, which leaves the sigma the least
scratch, held to the footprint; and unbounded, held to it plus G for the
whole space less the least.

## Files

| File | Role |
|------|------|
| `fci.cppm` | Module interface: `FciOutput`, `FciStep` and its footprint |
| `fci.cpp` | The footprint, and `run`: the refusal and the sigma's limit, the Hamiltonian, the CI solve, the 1-RDMs |
| `CMakeLists.txt` | The module library |
