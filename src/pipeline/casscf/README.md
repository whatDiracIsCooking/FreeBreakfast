# Pipeline: Casscf

State-averaged CASSCF from the SCF's orbitals: the start is an active space
named by index or picked by AVAS over MINAO, as the Casci step picks its own;
`mcscf.solve`'s `mcscf::solve` optimizes the orbitals and the CI of the lowest
singlets at fixed weights; the core and the virtuals are then canonicalized,
and the state-averaged natural orbitals optionally written as Molden.

## Module

`free_breakfast.pipeline.casscf`, target `free_breakfast::pipeline::casscf`.
It re-exports `free_breakfast.pipeline.scf` (which re-exports every product in
`run`'s signature), `free_breakfast.pipeline.active_space` (the selection,
`CasscfStep`'s base), and `free_breakfast.mcscf.solve` (`mcscf::Iteration`, in
the product, and `mcscf::Options`, whose defaults the step's are). `run` is
defined in the implementation unit, so J's and K's classes,
`integrals.active_space`, `mcscf.fock`, `diagonalize` and `io.molden` are
PRIVATE links.

The selection -- `ncas`, `nelecas`, `active`, `avas`, `avas_threshold` -- is
`pipeline/active_space`'s ([`active_space/README.md`](../active_space/README.md)).
The loop, its defaults and what each settles are
[`mcscf/solve/README.md`](../../mcscf/solve/README.md)'s. This step adds the
canonicalization and the product.

## The step

```cpp
struct CasscfOutput {
  int32_t ncore, ncas, nelecas;
  SphericalMatrixAO c;                     // AO x MO: core, active, virtual
  std::vector<double> mo_energies;         // diag of the SA Fock over c
  std::vector<double> energies;            // per root, total (hartree)
  std::vector<double> spin_squares;        // <S^2> per root
  std::vector<double> weights;             // nroots, summing to 1
  double energy;                           // sum_r w_r E_r
  double gradient_norm;                    // ||g|| at c
  std::vector<double> vectors;             // root r at [r dim, (r + 1) dim)
  std::vector<double> rdm1;                // root r at [r ncas^2, (r + 1) ncas^2)
  std::vector<mcscf::Iteration> iterations; // one per macro iteration
};

struct CasscfStep : ActiveSpaceSelection {  // ncas, nelecas, active, avas, avas_threshold
  static constexpr std::string_view kName = "casscf";

  int32_t nroots = 1;
  std::vector<double> weights{};       // empty: equal
  double conv_tol = 1e-10;             // mcscf::Options{}'s
  double conv_tol_grad = 1e-6;
  int32_t max_cycle = 50;
  std::filesystem::path molden{};      // the state-averaged natural orbitals, if set

  Result<CasscfOutput> run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput,
                                PrimitivePairClassesOutput, CoreHamiltonianOutput, ScfOutput> in,
                           DeviceResources &res) const;
};
```

`run`, all on `res.stream()`:

1. **The start**: `ActiveSpaceSelection::select`'s C R, as the Casci step's.
2. **The optimization**: `mcscf::solve` from it, over J and K classes built
   from `PrimitivePairClassesOutput` per run, at the step's `nroots`,
   `weights`, `conv_tol`, `conv_tol_grad` and `max_cycle`, every other option
   at `mcscf::Options`' default. Each macro iteration's weighted energy and
   ‖g‖ is logged at info once it returns.
3. **The state-averaged Fock** F = F^I + F^A at the solution, PySCF's
   `mc.get_fock`: `mcscf::fock` over one more `casscf_integrals`, since the
   solution keeps neither. One macro iteration's integrals, no Hessian.
4. **Canonicalization**, PySCF's default (`canonicalization = True`,
   `natorb = False`): the core and the virtual columns each rotated to F's
   eigenvectors within their space, by ascending eigenvalue, by
   `diagonalize`; C ← C R in one gemm. The active orbitals are left as
   optimized, so the CI vectors and 1-RDMs stay over them, and their
   `mo_energies` are F's diagonal there.
5. **The Molden file**, if `molden` is set: the canonical C with its active
   columns rotated to the weighted γ's eigenvectors by descending occupation;
   the core at occupation 2, the virtuals at 0. Each energy is the orbital's
   expectation in F: the canonical eigenvalue for the core and the
   virtuals.

What it settled:

- **Singlets only**, for the Casci step's reason: the SCF is closed-shell.
- **The weights, tolerances and `max_cycle` are refused when the config is
  read**, since none needs the molecule (`config/README.md`); `mcscf::solve`
  checks them again for a step built by hand. The active space's ranges are
  the selection's, when the step runs.
- **`weights` in the product is never empty**: equal ones are filled in.
- **The product carries `mo_energies`**: the canonical orbital energies are
  what canonicalization is for, and recomputing F from C costs a set of
  integrals.

Errors: `ActiveSpaceSelection::select`'s; `mcscf::solve`'s (InvalidConfig for
`nroots` past the determinant count, MaxIterationsExceeded); the module
calls' own (`casscf_integrals`', `mcscf::fock`'s, `diagonalize`'s,
`write_molden`'s); GpuRuntime for a failed copy or gemm.

## Tests

`test/gtest/pipeline_casscf_gpu.cpp`, through the real registry from a config
naming `[casscf]`, its Scf step at the defaults, against PySCF's converged,
polished `mcscf.CASSCF` in `test/gtest/data/casscf/casscf.json`
(`tools/emit_casscf_reference.py`): H₂O cc-pVDZ CAS(4,4) [2, 3, 5, 6], N₂
cc-pVDZ CAS(6,6), H₂CO cc-pVDZ CAS(4,3) at [0.5, 0.5] and [0.7, 0.3], and H₂O
cc-pVDZ by AVAS, each at `conv_tol_grad` 1e-9 (see `mcscf/solve/README.md`
for why):

- each root's energy and the weighted one to 1e-8, ⟨S²⟩ to 1e-8;
- the state-averaged natural occupations to 1e-6;
- the canonical core and virtual orbital energies to 1e-6;
- each CI vector of unit norm, and the weights as written;
- the Molden file: every orbital, at 2, the natural occupations, then 0.

Also `max_cycle` running out failing the step. The `[casscf]` parsing and its
refusals are in `pipeline_config_host.cpp`, and an end-to-end run with
`[casci]` beside it, against the step called by hand, in
`pipeline_calculation_gpu.cpp`.

## Files

| File | Role |
|------|------|
| `casscf.cppm` | Module interface: `CasscfOutput`, `CasscfStep` |
| `casscf.cpp` | `run`: the selection, `mcscf::solve`, the Fock at the solution, the canonicalization, the Molden file |
| `CMakeLists.txt` | The module library |
