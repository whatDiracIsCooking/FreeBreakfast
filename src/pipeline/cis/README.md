# Pipeline: Cis

CIS (the Tamm–Dancoff approximation) over the SCF's closed-shell RHF: the
lowest singlet or triplet excitations, as the lowest roots of the A matrix,
found by `davidson_solve` over `cis::Sigma`, and a singlet's transition
dipoles and oscillator strengths. The numerics are all in `../../cis/` (the
sigma, the Δε diagonal, the guess and the preconditioner) and
`../../integrals/one_electron/dipole/`; the step wires them over the
pipeline's products, picks the Davidson's shape, contracts the roots with the
dipole's ov block, and copies the roots off the device.

## Module

`free_breakfast.pipeline.cis`, target `free_breakfast::pipeline::cis`. It
re-exports `free_breakfast.pipeline.scf` (which re-exports every product in
`run`'s signature, the `EriOutput` among them) and `free_breakfast.cis.sigma`
(`cis::Options` and `cis::Spin`, which the step holds). `run` is defined in
the implementation unit, so `cis.precondition`, `recipes.davidson`,
`recipes.blas`, `integrals.one_electron.dipole`, `core.log` and `core.wwr`
are PRIVATE links.

## The step

```cpp
struct CisState {
  double excitation_energy;                 // ω (hartree)
  double oscillator_strength;               // (2/3) ω |μ|²; zero for a triplet
  std::array<double, 3> transition_dipole;  // <0|r|n>; zero for a triplet
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

  cis::Options options{};      // n_states, spin, davidson, cluster_gap, ris
  int32_t max_subspace = 40;

  Result<CisOutput> run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput,
                             PrimitivePairClassesOutput, EriOutput, ScfOutput> in,
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
   the lowest Δε, `cis::Preconditioner`'s r / (θ − Δε), clustered over the
   guess's entries at `options.cluster_gap` (`Preconditioner::cluster`, over
   the sigma; at the default 0, nothing) — or, when `options.ris` is set,
   `cis::RisPreconditioner`'s exact (θ − Δε − 2 L Lᵀ)⁻¹ r over its own fit
   of the RIS basis, whichever `EriOutput` the sigma has, refined by GMRES
   towards (θ − Δε − 2 L Lᵀ + K_RI)⁻¹ r when `options.ris->exchange` is set
   (`../../cis/ris_precondition/`) — and `davidson_solve`
   at `options.davidson` with `max_subspace` clamped to
   [2 · n_states, n_ov]. `davidson_solve`'s floor wins when the two cross
   (n_ov < 2 · n_states): it needs room for a full block, and the extra
   columns are never filled.
5. **The transition dipoles**, a singlet's: `dipole`'s three Cartesian
   matrices over the `PrimitivePairClassesOutput`'s classes, carried
   spherical, and the ov block of each, μ^ov_c = C_virtᵀ μ_c C_occ, formed
   once by two gemms (n_virt × n_occ, so in the amplitudes' order). The three
   side by side are the n_ov × 3 matrix M, and one more gemm gives every
   root's μⁿ = √2 Mᵀ Xₙ at once. f = (2/3) ω |μ|² on the host.
6. **The roots**, values, Ritz vectors and dipoles copied to the host, one
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
- **A triplet's oscillator strength and transition dipole are zero**, set
  without forming anything: spin orthogonality, not a numerical zero.
- **Past d, the dipoles are skipped, not the step.** `dipole` refuses an f
  shell (`kDipoleMaxL`), so in such a basis a singlet's f and μ are left zero
  with a warning logged, and the energies and amplitudes are still the
  product. Dropping the check follows FelizJueves#627 reaching f.
- **μ is the electronic position, origin 0**, √2 Σ_ia X_ia <i|r|a>, the
  convention of `dipole` and of PySCF's `transition_dipole` up to the
  electron's charge; its sign is the amplitudes', so only |μ| and f are
  defined. Both are origin-independent: <i|a> = 0. Within a degenerate level
  only the summed f is.
- **The dipole needs no ERI**, so it is the same under either `EriOutput`.
- **The `EriOutput`'s alternative picks the sigma**: `cis::Sigma` over the
  J and K classes, or `cis::DfSigma` over B. The fitted one is created for a
  block of the guess's min(n_ov, 2 · n_states) columns, the widest
  `davidson_solve` applies (the guess, then at most n_states corrections
  per iteration), so its scratch is sized once and no apply slices.
  `davidson_solve` is the same call over either, through a generic lambda,
  rather than a variant or a wrapper type around the two.

Errors: InvalidConfig, before anything is allocated, for `n_states` below 1;
a `cluster_gap` negative or NaN, or positive beside `ris`;
a molecule that is not a closed-shell singlet (odd electrons, or a
multiplicity other than 1); an ScfOutput whose `n_alpha`, `n_beta` are not the
molecule's N / 2 each; no virtual orbital, or an n_ov past `int32_t`; or more
states than n_ov. Then the module calls' own: `cis::Sigma::create`'s,
`cis::DfSigma::create`'s (Unsupported, naming the bytes, for MO blocks and
scratch past the device's free memory), `Preconditioner::cluster`'s (the
sigma's, or `diagonalize`'s), `cis::RisPreconditioner::create`'s (the RIS
basis's, its fit's, or a RIS-JK GMRES budget it refuses), the RIS correction's
(`lu_solve`'s, or the GMRES's BLAS), `davidson_solve`'s
MaxIterationsExceeded or Divergence, and GpuRuntime for a failed copy,
dipole launch or gemm.

## Tests

`test/gtest/cis_step_gpu.cpp`, against PySCF's RHF + TDA in this tree's basis
files (`test/gtest/data/cis/`, `tools/emit/cis_reference.py`): H₂O STO-3G
and 6-31G exact, and H₂O cc-pVDZ density-fitted over cc-pVDZ-JKFIT against
PySCF's DF RHF + TDA over the same set, singlets and triplets, the five
lowest of each, through the real
registry from a config naming `[cis]` (its `[scf]` converged tightly), and
the step again by hand over the same products. The excitation energies and
the reference energy to 1e-6 hartree, the states ascending, and their
amplitudes orthonormal; each singlet's (and H₂O cc-pVDZ exact's, singlets
only) oscillator strengths, summed per degenerate group, against the
reference's to 1e-6, and each triplet's f and μ exactly zero. HF in cc-pVTZ,
with fluorine's f shell, gives its singlets' energies with f and μ zero:
against the reference integral-direct (~55 s, skipped in fast mode), and
fitted over cc-pVTZ-JKFIT, shape only, in the fast set. The f differences
measured are up to 7e-7 (6-31G's and cc-pVDZ's brightest roots): f is first
order in the vector, which the Davidson's relative 1e-6 residual leaves good
to ~1e-6 over the gap (`../../cis/sigma/README.md`), while the energies are
second order and land at ~1e-10.
Then n_states = n_ov and `max_subspace = 1`, both
clamped, still the reference's roots; with `cluster_gap = 0.1`, three states of
each spin over H₂O STO-3G integral-direct and cc-pVDZ fitted, the guess
narrower than the space and the gap checked to group some of its Δε, against
the reference to 1e-6; with `ris = true`, and again with `ris_exchange = true`
beside it, the same three states of each spin over the same two, against the
reference to 1e-6; the refusals above; and `[cis]` under a
density-fitted `[eri]` planned when the config is read, with nothing refused.
The `[cis]` parsing is in `pipeline_config_host.cpp`.

`test/gtest/e2e_cis_gpu.cpp` is the production-scale run, slow and
`RUN_SERIAL`: benzene in cc-pVDZ from a TOML, fitted over cc-pVDZ-JKFIT
(n_ov = 1953), the lowest five singlets (six roots) with RIS-J through the
config and again by hand with the diagonal preconditioner and RIS-JK, each
against the fitted reference's energies to 1e-6 and summed f to 1e-5, with
every run's Davidson iteration count printed. The exact benzene reference is
not run: the integral-direct sigma takes the three solves to 282 s on the
RTX 3080, past an entry's 120 s.

## Files

| File | Role |
|------|------|
| `cis.cppm` | Module interface: `CisState`, `CisOutput`, `CisStep` |
| `cis.cpp` | `run`: the refusals, the sigma, the Davidson, the roots |
| `CMakeLists.txt` | The module library |
