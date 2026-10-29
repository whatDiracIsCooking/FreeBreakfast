# Pipeline: Casscf

State-averaged CASSCF from the SCF's orbitals: the start is an active space
named by index or picked by AVAS over MINAO, as the Casci step picks its own;
`mcscf.solve`'s `mcscf::solve` optimizes the orbitals and the CI of the lowest
singlets at fixed weights, integral-direct or fully density-fitted; the core
and the virtuals are then canonicalized, and the state-averaged natural
orbitals optionally written as Molden.

## Module

`free_breakfast.pipeline.casscf`, target `free_breakfast::pipeline::casscf`.
It re-exports `free_breakfast.pipeline.scf` (which re-exports every product in
`run`'s signature, the `EriOutput` among them),
`free_breakfast.pipeline.active_space` (the selection, `CasscfStep`'s base),
and `free_breakfast.mcscf.solve` (`mcscf::Iteration`, in the product, and
`mcscf::Options`, whose defaults the step's are). `run` is defined in the
implementation unit, so `integrals.active_space`, `response.fock`,
`diagonalize` and `io.molden` are PRIVATE links.

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
                                EriOutput, CoreHamiltonianOutput, ScfOutput> in,
                           DeviceResources &res) const;
};
```

`run`, all on `res.stream()`:

1. **The start**: `ActiveSpaceSelection::select`'s C R, as the Casci step's.
2. **The two-electron integrals' input**: the `EriOutput`'s
   (`pipeline/eri`), J and K classes under `IntegralDirect`, or B under
   `DensityFitted` (below) and no classes.
3. **The optimization**: `mcscf::solve` from it, at the step's `nroots`,
   `weights`, `conv_tol`, `conv_tol_grad` and `max_cycle`, and in
   `mcscf::DensityFitting::Full` over a fitted `EriOutput`'s B, every other
   option at `mcscf::Options`' default. Each macro iteration's weighted energy and
   ‖g‖ is logged at info once it returns.
4. **The state-averaged Fock** F = F^I + F^A at the solution, PySCF's
   `mc.get_fock`: `response::fock` over one more `casscf_integrals`, since the
   solution keeps neither, both over B when fitted. One macro
   iteration's integrals, no Hessian.
5. **Canonicalization**, PySCF's default (`canonicalization = True`,
   `natorb = False`): the core and the virtual columns each rotated to F's
   eigenvectors within their space, by ascending eigenvalue, by
   `diagonalize`; C ← C R in one gemm. The active orbitals are left as
   optimized, so the CI vectors and 1-RDMs stay over them, and their
   `mo_energies` are F's diagonal there. R and the energies are the
   exported `canonicalize_core_virtual(res, nmo, ncore, nocc, fock, r,
   energies)`, which the Nevpt2 step calls again per root over that root's
   own Fock (`nevpt2/README.md`).
6. **The Molden file**, if `molden` is set: the canonical C with its active
   columns rotated to the weighted γ's eigenvectors by descending occupation;
   the core at occupation 2, the virtuals at 0. Each energy is the orbital's
   expectation in F: the canonical eigenvalue for the core and the
   virtuals.

## Density fitting

`[eri]` chooses it for the whole pipeline (`eri/README.md`), so the SCF the
CASSCF starts from is fitted over the same B:

```toml
[eri]
mode = "density_fitting"      # default "integral_direct"
aux_basis = "cc-pvdz-jkfit"   # optional: <basis>-acCD-6-6 when carried
[casscf]
ncas = 4
nelecas = 4
```

With it the step runs `mcscf::DensityFitting::Full`: every build of the loop
is fitted, and so is the Fock it canonicalizes over, so the product is the
DF-CASSCF (PySCF's `mcscf.CASSCF(mf, ...).density_fit()`), its orbital
energies the fitted Fock's eigenvalues. Its J and K are never
integral-direct, and the `EriOutput` holds no J or K classes. A fitted SCF
with a direct CASSCF, or the reverse, is not one pipeline, by decision: one
aux basis per pipeline, and a mix is two runs. `ApproxHessian` (the exact
CASSCF, only the Hessian fitted) is not a config choice.

**B is the pipeline's**, built once by the Eri step and shared with the SCF;
it lives until the run ends.

**Memory.** B, naux nmo² doubles, is refused past the device's free memory
by the Eri step (`Unsupported`, naming the bytes). With it held, the step
refuses, again `Unsupported`, a loop whose fitted
`casscf_integrals` scratch is past what B leaves free:

| Region | Doubles |
|---|---|
| L, the half transform | naux · ncas · nmo |
| X, B's first half transform | naux · ncas · nmo |
| J_AO, every (u, v)'s Bᵀ γ_uv, in ppaa's buffer | nmo² · ncas² |
| the intermediate of Cᵀ J_AO C | nmo² · ncas² |
| papa | nmo² · ncas² |

(`../../integrals/active_space/README.md`'s scratch table; nao = nmo.) Both
checks are against `wwrMemGetInfo`'s free bytes, and both are needed: past
the device's memory an allocation aborts (`CreatePolicy`) rather than
returning. The rest of the loop (the CI, the Hessian's products, the core's
three nmo² matrices) is not counted; it is small beside these at the sizes
this step runs.

**One B.** The SCF and the CASSCF read the same B, so the peak is one B
plus the scratch above, where each step building its own once held two.

What it settled:

- **Singlets only**, for the Casci step's reason: the SCF is closed-shell,
  and an open-shell ScfOutput (`n_alpha != n_beta`) is refused `Unsupported`
  before anything is queued.
- **The weights, tolerances and `max_cycle` are refused when the config is
  read**, since none needs the molecule (`config/README.md`); `mcscf::solve`
  checks them again for a step built by hand. The active space's ranges are
  the selection's, when the step runs.
- **`weights` in the product is never empty**: equal ones are filled in.
- **The product carries `mo_energies`**: the canonical orbital energies are
  what canonicalization is for, and recomputing F from C costs a set of
  integrals.

Errors: `ActiveSpaceSelection::select`'s; under a fitted `EriOutput`,
`Unsupported` for the scratch above;
`mcscf::solve`'s (InvalidConfig for
`nroots` past the determinant count, MaxIterationsExceeded); the module
calls' own (`casscf_integrals`', `response::fock`'s, `diagonalize`'s,
`write_molden`'s); GpuRuntime for a failed copy or gemm.

## Tests

`test/gtest/pipeline_casscf_gpu.cpp`, through the real registry from a config
naming `[casscf]`, its Scf step at the defaults, against PySCF's converged,
polished `mcscf.CASSCF` in `test/gtest/data/casscf/casscf.json`
(`tools/emit/casscf_reference.py`): H₂O cc-pVDZ CAS(4,4) [2, 3, 5, 6], N₂
cc-pVDZ CAS(6,6), H₂CO cc-pVDZ CAS(4,3) at [0.5, 0.5] and [0.7, 0.3], and H₂O
cc-pVDZ by AVAS, each at `conv_tol_grad` 1e-9 (see `mcscf/solve/README.md`
for why):

- each root's energy and the weighted one to 1e-8, ⟨S²⟩ to 1e-8;
- the state-averaged natural occupations to 1e-6;
- the canonical core and virtual orbital energies to 1e-6;
- each CI vector of unit norm, and the weights as written;
- the Molden file: every orbital, at 2, the natural occupations, then 0.

Each case runs again with `[eri] mode = "density_fitting"` and `aux_basis =
"cc-pVDZ-JKFIT"` (compiled in), against PySCF's DF-CASSCF in
`test/gtest/data/casscf/casscf_df.json` (fitted over this tree's own
`data/aux_basis/cc-pVDZ-JKFIT.json`), to the same bars. The reference starts
from the exact RHF and this run from the DF-RHF, since one `[eri]` serves
both steps; the converged DF-CASSCF does not depend on the start, only on
which stationary point it reaches.

Also `max_cycle` running out failing the step. The `[casscf]` parsing and its
refusals are in `pipeline_config_host.cpp`, and an end-to-end run with
`[casci]` beside it, against the step called by hand, in
`pipeline_calculation_gpu.cpp`, which also runs an `[scf]` and a `[casscf]`
over one `EriOutput` in each mode.

## Files

| File | Role |
|------|------|
| `casscf.cppm` | Module interface: `CasscfOutput`, `CasscfStep`, `canonicalize_core_virtual` |
| `casscf.cpp` | `run`: the selection, the fitted scratch's memory check, `mcscf::solve`, the Fock at the solution, the canonicalization, the Molden file |
| `CMakeLists.txt` | The module library |
