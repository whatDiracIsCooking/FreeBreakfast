# Pipeline: Cis

CIS (the Tamm–Dancoff approximation) over the SCF's closed-shell RHF: the
lowest singlet or triplet excitations, as the lowest roots of the A matrix,
found by `davidson_solve` over `cis::Sigma`. The numerics are all in
`../../cis/` (the sigma, the Δε diagonal, the guess and the preconditioner);
the step wires them over the pipeline's products, picks the Davidson's shape,
and copies the roots off the device.

## Module

`free_breakfast.pipeline.cis`, target `free_breakfast::pipeline::cis`. It
re-exports `free_breakfast.pipeline.scf` (which re-exports every product in
`run`'s signature, the `EriOutput` among them) and `free_breakfast.cis.sigma`
(`cis::Options` and `cis::Spin`, which the step holds). `run` is defined in
the implementation unit, so `cis.precondition`, `recipes.davidson` and
`core.wwr` are PRIVATE links.

## The step

```cpp
struct CisState {
  double excitation_energy;                 // ω (hartree)
  double oscillator_strength;               // zero until #201
  std::array<double, 3> transition_dipole;  // zero until #201
  std::vector<double> amplitudes;           // unit norm, i * n_virt + a
};

struct CisOutput {
  int32_t iterations;          // the Davidson's
  int32_t n_occ, n_virt;
  double reference_energy;     // the ScfOutput's (hartree)
  std::vector<CisState> states;  // ascending
};

struct CisStep {
  static constexpr std::string_view kName = "cis";

  cis::Options options{};      // n_states, spin, davidson
  int32_t max_subspace = 40;

  Result<CisOutput> run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput,
                             EriOutput, ScfOutput> in,
                        DeviceResources &res) const;
};
```

`run`, all on `res.stream()`:

1. **The refusals**, before anything is allocated (below).
2. **Δε** = ε_a − ε_i, `cis::diagonal` over the ScfOutput's orbital energies
   uploaded, n_occ = N / 2 from the molecule.
3. **The sigma**, over the ScfOutput's C: `cis::Sigma::create` with the
   `SphericalTransformOutput`'s U and an `IntegralDirect` `EriOutput`'s J and
   K classes, or `cis::DfSigma::create` with a `DensityFitted` one's B.
4. **The Davidson.** `cis::guess`'s min(n_ov, 2 · n_states) unit vectors on
   the lowest Δε, `cis::Preconditioner`'s r / (θ − Δε), and `davidson_solve`
   at `options.davidson` with `max_subspace` clamped to
   [2 · n_states, n_ov]. `davidson_solve`'s floor wins when the two cross
   (n_ov < 2 · n_states): it needs room for a full block, and the extra
   columns are never filled.
5. **The roots**, values and Ritz vectors copied to the host, one
   `CisState` each. The Ritz vectors are column-major n_ov × n_states, so a
   root's column is already its amplitudes in (i, a) row-major order.

What it settled:

- **`CisOutput` absorbs the closed #193's result types**, per
  `docs/style.md` "Pipeline steps and products": the step owns its product,
  and there is no separate result module or JSON writer, since no other
  step's output writes one.
- **No `converged` flag.** `davidson_solve` returns `MaxIterationsExceeded`
  (or `Divergence`) as an error, so a product is always converged.
- **The amplitudes are unit-norm**, as the Davidson leaves them: Σ_ia X_ia² =
  1 over the n_ov spatial excitations, whatever the spin. A spin-adapted
  normalization (1/√2 per spin orbital) is the reader's to apply. Their
  overall sign is arbitrary, and so is the rotation within a degenerate
  level.
- **`cis::Options` is the step's member**, not three of its own: it is the
  method's vocabulary (`../../cis/sigma/`), and its Davidson tolerance is the
  one that was reasoned for CIS (a relative 1e-6, an energy good to ~1e-11).
  `max_subspace` is the step's, as `davidson_solve` takes it beside the
  options rather than in them.
- **The oscillator strengths and transition dipoles are zero** until #201
  computes them.
- **The `EriOutput`'s alternative picks the sigma**: `cis::Sigma` over the
  J and K classes, or `cis::DfSigma` over B. The fitted one is created for a
  block of the guess's min(n_ov, 2 · n_states) columns, the widest
  `davidson_solve` applies (the guess, then at most n_states corrections
  per iteration), so its scratch is sized once and no apply slices.
  `davidson_solve` is the same call over either, through a generic lambda,
  rather than a variant or a wrapper type around the two.

Errors: InvalidConfig, before anything is allocated, for `n_states` below 1;
a molecule that is not a closed-shell singlet (odd electrons, or a
multiplicity other than 1); an ScfOutput whose `n_alpha`, `n_beta` are not the
molecule's N / 2 each; no virtual orbital, or an n_ov past `int32_t`; or more
states than n_ov. Then the module calls' own: `cis::Sigma::create`'s,
`cis::DfSigma::create`'s (Unsupported, naming the bytes, for MO blocks and
scratch past the device's free memory), `davidson_solve`'s
MaxIterationsExceeded or Divergence, and GpuRuntime for a failed copy.

## Tests

`test/gtest/cis_step_gpu.cpp`, against PySCF's RHF + TDA in this tree's basis
files (`test/gtest/data/cis/`, `tools/emit/cis_reference.py`): H₂O STO-3G
and 6-31G exact, and H₂O cc-pVDZ density-fitted over cc-pVDZ-JKFIT against
PySCF's DF RHF + TDA over the same set, singlets and triplets, the five
lowest of each, through the real
registry from a config naming `[cis]` (its `[scf]` converged tightly), and
the step again by hand over the same products. The excitation energies and
the reference energy to 1e-6 hartree, the states ascending, and their
amplitudes orthonormal. Then n_states = n_ov and `max_subspace = 1`, both
clamped, still the reference's roots; the refusals above; and `[cis]` under a
density-fitted `[eri]` planned when the config is read, with nothing refused.
The `[cis]` parsing is in `pipeline_config_host.cpp`.

## Files

| File | Role |
|------|------|
| `cis.cppm` | Module interface: `CisState`, `CisOutput`, `CisStep` |
| `cis.cpp` | `run`: the refusals, the sigma, the Davidson, the roots |
| `CMakeLists.txt` | The module library |
