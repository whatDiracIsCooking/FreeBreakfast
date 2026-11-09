# Pipeline: config

The TOML front end: a config file read into the pipeline it asks for, and the
registry of every step a config may name.

## Module

`free_breakfast.pipeline.config`, target `free_breakfast::pipeline::config`. It
re-exports `pipeline.pipeline` and every step module, since `Calculation`
names them all. It is the one library that imports every step, so only it and
`main` pay that build fan-in.

```cpp
struct Calculation : Pipeline<MoleculeStep, BasisSetStep, /* ... every step ... */ HessianStep> {
  size_t max_device_mem_per_gpu = 0;   // [memory]; 0: no cap
  size_t max_pinned_mem_per_gpu = 0;   // [memory]; 0: no cap
};

Result<Calculation> parse_config(std::string_view toml_text, const std::filesystem::path &input_dir,
                                 const std::filesystem::path &output_dir);
Result<Calculation> read_config(const std::filesystem::path &path,
                                const std::filesystem::path &input_dir,
                                const std::filesystem::path &output_dir);
```

`Calculation` is a struct over the registry's pipeline rather than an alias
of it, so the `[memory]` caps travel with the plan they bound. It is that
pipeline, so everything taking one takes it; the inherited
`Calculation::create` returns the bare pipeline, with no caps, for a plan
built by hand.

A new step joins the registry here: its type in `Calculation`, and, if it has
options, a branch in `read_step` reading them. A step with options and no
branch fails a `static_assert`, rather than having every key refused.

## The config file

One table per step, named by the step's `kName`. Its keys are the step's
options, spelled as the step's members:

```toml
[molecule]
xyz = "water.xyz"   # required
charge = 0          # default 0
multiplicity = 1    # default 1

[basis_set]
basis = "sto-3g.json"   # required, a BSE JSON basis

[eri]                     # every option has a default
mode = "integral_direct"  # default "integral_direct"; or "density_fitting"

[orthogonalizer]          # every option has a default
eigenvalue_floor = 1e-10  # default 1e-10

[core_hamiltonian]        # no options: the table is the request

[guess]
type = "sad"              # default "sad", the only kind so far

[scf]                     # every option has a default
max_iterations = 100

[localization]            # every option has a default
mode = "pipek_mezey"      # default "pipek_mezey"; or "ibo"
exponent = 4              # 2 or 4; default 2, or 4 for "ibo"
livvo = true              # default false; the LIVVOs too, in either mode

[casci]
ncas = 4                  # required
nelecas = 4               # required
nroots = 3                # default 1
active = [2, 4, 5, 7]     # 0-based MO indices; default the frontier orbitals
orbitals = "s0/casscf_orbitals.npy"  # optional: start from these, not the SCF's
```

`orbitals` is a path, relative to the input directory, to a `.npy` of
another run's orbitals, nao × nmo; `active` then indexes its columns
(`active_space/README.md`, "A start from another run's orbitals"). Or, in
place of `ncas`, `nelecas` and `active`, AVAS picks the active space from
MINAO AO labels, and refuses any of the three, and `orbitals`, beside it:

```toml
[casci]
avas = ["O 2p", "H 1s"]   # "<symbol> <n><l>"; every m, every atom of the element
avas_threshold = 0.2      # default 0.2; needs avas
avas_max_ncas = 12        # optional, at least 1; needs avas. A pick past it is
                          # refused (InvalidConfig), never truncated
```

`[eri]` picks the two-electron integrals for the **whole pipeline**: the
`EriStep` builds them once, and `[scf]` and `[casscf]` both read its product
(`eri/README.md`). `mode = "integral_direct"`, the default, is J's and K's
pair classes. `"density_fitting"` fits J and K over an aux basis:

```toml
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"   # optional: a compiled-in fitting set, or a BSE JSON path
```

Left out, `aux_basis` is `<basis>-acCD-6-6` when the basis is a compiled-in
set that has one carried; `EriStep::run` picks it from `BasisSet::name()`, so
this front end never loads a basis, and a basis with none is `InvalidConfig`
then, asking for `aux_basis`. Written, it follows `[basis_set]`'s rule over
the fitting sets: joined to the input directory only when it is not a
compiled-in fitting set's name, and an input file like any other. A missing
`[eri]` is filled in, integral-direct, as every defaulted table is.

Each mode refuses the other's options: `aux_basis` under `"integral_direct"`
is `InvalidConfig`, and so is `[scf] fock_rebuild_interval` beside `mode =
"density_fitting"`, since every DF build is full. That one spans two tables,
so `parse_config` checks it once the pipeline is planned (`[eri]` may be
filled in, or written after `[scf]`). `[scf]` and `[casscf]` take no `eri` or
`aux_basis` of their own, and there is no `[density_fitting]`: one pipeline
cannot mix a fitted SCF with a direct CASSCF (one aux basis per pipeline; a
mix is two runs).

`[localization]`'s method options sit in the table beside `mode`, not in
a subtable named after `LocalizationStep::pipek_mezey`: both modes run the
same Pipek-Mezey sweep, IBO over intrinsic-atomic-orbital populations, so they
take the same options. `livvo` adds the localized valence virtual orbitals
(`pipeline/localization`'s README) in either mode.

`[casci]`'s keys are read for their types only. The active-space keys
(`ncas`, `nelecas`, `active`, `avas`, `avas_threshold`, `avas_max_ncas`) and their refusals are
`read_active_space`'s, one reader for every step whose config derives from
`ActiveSpaceSelection`. Whether the active space fits needs the molecule's
electrons and the basis's orbitals, so `ActiveSpaceSelection::orbital_order`
checks the ranges when the step runs (`active_space/README.md`).

```toml
[casscf]
ncas = 3                  # the start: the same selection keys as [casci], avas too
nelecas = 4
nroots = 2                # default 1
weights = [0.5, 0.5]      # default equal; nroots of them, >= 0, summing to 1
conv_tol = 1e-10          # default 1e-10: |dE| at convergence
conv_tol_grad = 1e-6      # default 1e-6: ||g|| at convergence
max_cycle = 50            # default 50 macro iterations
ci = "s0/casscf_ci.npy"   # optional: the CI start, dim x nroots, beside orbitals
write_npy = true          # default false: casscf_orbitals.npy and casscf_ci.npy
```

`[casscf]` reads the selection as `[casci]` does, and its own keys are checked
when the file is read: `weights` of the wrong count, with a negative entry or
not summing to 1 within 1e-12 (`mcscf::solve`'s own bar), a tolerance not
positive, or `max_cycle` below 1 is InvalidConfig. Unlike the active space's
ranges, none of these needs the molecule. `ci` is a path as `orbitals` is;
whether either file fits the molecule, the basis and `nroots` is checked
when the step runs (`casscf/README.md`, "Starting from another run's
orbitals"). `write_npy` writes the files a later run starts from into the
output directory, as `[overlap]`'s does. `orbitals` and `ci` are input
files, so neither may be in the output directory.

`[casscf]` runs the DF-CASSCF under a density-fitted `[eri]`
(`casscf/README.md`).

```toml
[eri]
mode = "density_fitting"  # required by [nevpt2]
[casscf]
ncas = 6
nelecas = 6
nroots = 3                # quasi_degenerate needs 2 or more
[nevpt2]
quasi_degenerate = true   # default false: SC and PC per root only
frozen_core = "auto"      # default 0: every core orbital correlated
```

`[nevpt2]` runs SC- and PC-NEVPT2 on every root of the `[casscf]` it reads
(`nevpt2/README.md`), and only over density fitting. Under an
integral-direct `[eri]`, written so or filled in, it is `Unsupported`,
naming `[eri]`: across two tables, so `parse_config` checks it once the
pipeline is planned, before a GPU is touched rather than after the SCF and
the CASSCF have converged. The step refuses it again, for a pipeline built
by hand.

`quasi_degenerate` (a boolean, default `false`) adds SC- and PC-QD-NEVPT2
of the roots together beside the per-root results: the effective
Hamiltonians over the CASSCF's state-averaged orbitals, and their
eigenvalues and mixing vectors (`nevpt2/README.md` "QD-NEVPT2"). Over a
`[casscf]` of fewer than two roots, `nroots` written as 1 or left at its
default, it is `InvalidConfig` when the config is read, pointing at plain
`[nevpt2]`: with one root the state-averaged Fock is the root's own, and QD
would only repeat the state-specific result. The step refuses it again, for
a pipeline built by hand.

`frozen_core` (default 0) leaves that many of the lowest canonical core
orbitals uncorrelated, per root and in QD (#859; `nevpt2/README.md` "Frozen
core"): a count of at least 0, or `"auto"`, each element's chemical core
(PySCF's `chemcore`: the 1s for B to Mg). A negative count, another string
or another type is `InvalidConfig` when the file is read; a count past the
CASSCF's core needs the molecule and the active space, so the step refuses
it, `InvalidConfig`, when it runs.

```toml
[fci]
nroots = 3                # default 1
```

`[fci]` takes no active space: it is every orbital and every electron. Whether
the determinant space fits is the device's to say, so `FciStep::run` estimates
it against the free memory when the step runs (`fci/README.md`).

```toml
[cis]
n_states = 5              # default 5: the lowest excitations
spin = "singlet"          # default "singlet"; or "triplet"
method = "full"           # default "full"; "ris": TDA-ris, the RIS approximation as the operator
theta_j = 0.2             # default 0.2; TDA-ris's J (sp) fit exponent scale (needs method = "ris")
theta_k = 0.2             # default 0.2; TDA-ris's K (s) fit exponent scale (needs method = "ris")
residual_tolerance = 1e-6 # default 1e-6, relative (cis::Options)
max_iterations = 100      # default 100
max_subspace = 40         # default 40; clamped to [2 n_states, n_ov] when run
cluster_gap = 0.0         # default 0 (off); hartree, the clustered preconditioner's
ris = false               # default false; true: the RIS-J preconditioner instead
ris_theta = 0.2           # default 0.2; the RIS basis's exponent scale (needs ris)
ris_shells = "sp"         # default "sp"; or "s", "spd": a heavy atom's (needs ris)
ris_exchange = false      # default false; true: RIS-JK, the RI exchange added (needs ris)
ris_k_max_inner_iters = 10 # default 10; RIS-JK's GMRES cap, 0 or more (needs ris_exchange)
ris_k_inner_tol = 1e-2    # default 1e-2; RIS-JK's GMRES tolerance (needs ris_exchange)
```

`[cis]` is CIS over the RHF (`cis/README.md`). Its keys are `CisStep`'s
`options` (`cis::Options`: `n_states`, `spin`, `method` with `theta_j` and
`theta_k`, two of its `davidson`'s,
`cluster_gap`, and `ris`, set by `ris = true` with its `RisOptions` from
`ris_theta`, `ris_shells`, `ris_exchange`, `ris_k_max_inner_iters` and
`ris_k_inner_tol`),
flattened into the table as `[localization]`'s are, and `max_subspace`.
`n_states` below 1, a `residual_tolerance` not positive, `max_iterations`
below 1, a negative `cluster_gap`, an unknown `method`, `theta_j` or `theta_k`
without `method = "ris"` or not finite and positive, a `ris_*` key without
`ris = true`,
`ris_k_max_inner_iters` or `ris_k_inner_tol` without `ris_exchange = true`, a
`ris_theta` not finite and positive, an unknown `ris_shells`, a negative
`ris_k_max_inner_iters`, a `ris_k_inner_tol` not positive, or `ris` beside a
positive `cluster_gap` is `InvalidConfig` when the file is
read; `n_states` past the single
excitations needs the molecule and the basis, so the step refuses it when it
runs. It runs over either `[eri]`: integral-direct, or density-fitted with
the fitted sigma over B's MO blocks; `method = "ris"` reads neither, fitting
its own two RIS bases. `method` picks the operator and `ris` the
preconditioner, independently: `method = "ris"` beside `ris = true` is the
RIS-J correction over the TDA-ris operator, and is not refused.

```toml
[gradient]                # no options
```

`[gradient]` is the RHF nuclear gradient at the SCF's orbitals
(`gradient/README.md`), written to `gradient.json`. It takes no keys. It is
the SCF energy's gradient only as far as the SCF converged, so a config that
wants it to 1e-8 tightens `[scf]` (`energy_tolerance = 1e-12`,
`commutator_tolerance = 1e-10`). Whether the ERIs and the SCF are ones it
takes is the step's to say when it runs: an integral-direct `[eri]` (the
filled-in default) and an ROHF SCF are each `Unsupported` there, naming the
reason.

```toml
[eri]
mode = "density_fitting"  # required by [casscf_gradient]
[casscf]
ncas = 4
nelecas = 4
conv_tol_grad = 1e-9      # the gradient is the energy's only at convergence
[casscf_gradient]
root = 0                  # the default
```

`[casscf_gradient]` is the DF-CASSCF nuclear gradient of one root of the
`[casscf]` it reads (`casscf_gradient/README.md`), written to
`casscf_gradient.json`. `root` must be at least 0; whether the CASSCF has it,
and whether a state average is one the gradient takes (today: a single root
only), is the step's to say when it runs. Like `[nevpt2]` it runs over
density fitting only, and under an integral-direct `[eri]`, written or filled
in, it is `Unsupported` when the file is read, naming `[eri]`; the step
refuses it again, for a pipeline built by hand.

```toml
[eri]
mode = "density_fitting"  # required by [nevpt2_gradient]
[casscf]
ncas = 4
nelecas = 4
conv_tol_grad = 1e-9      # the gradient is the energy's only at convergence
[nevpt2_gradient]
root = 0                  # the default, and the only root it takes
variant = "sc"            # the default; or "pc"
```

`[nevpt2_gradient]` is the SC- or PC-NEVPT2 nuclear gradient of the single-state
`[casscf]` it reads (`nevpt2_gradient/README.md`), written to
`nevpt2_gradient.json`. `variant` knows `"sc"` and `"pc"`, anything else
`InvalidConfig`. `root` must be 0, and `InvalidConfig` otherwise. Its
other refusals are `[nevpt2]`'s and the gradient's, each `Unsupported` when
the file is read, across the tables they name: an integral-direct `[eri]`,
written or filled in; a `[casscf]` with `nroots` above 1 (a state average);
and a `[molecule]` `multiplicity` other than 1 (an open shell, the RHF under
the CASSCF being closed-shell exactly then), under `[optimize]` its start's.
The step refuses each again when it runs, for a pipeline built by hand.
`frozen_core` is `[nevpt2]`'s key, read the same way (#875): a count, or
`"auto"`; whether it fits the CASSCF's core is the step's to say when it
runs. Under `[optimize]` it is the optimized energy's frozen core, the QD
MECI's included. `[nevpt2_nac]` refuses it (`InvalidConfig`, naming the
key): the coupling is all-electron.

**`quasi_degenerate = true`** differentiates one SC- or PC-QD-NEVPT2 state
of the state-averaged `[casscf]` instead, `state` (default 0) indexing
H_eff's eigenvalues ascending:

```toml
[casscf]
ncas = 3
nelecas = 2
nroots = 2
conv_tol_grad = 1e-10
[nevpt2_gradient]
quasi_degenerate = true
state = 1                 # the default 0, the lowest
variant = "pc"            # the default "sc"
```

Each of these is `InvalidConfig` when the file is read: `quasi_degenerate`
over a `[casscf]` of fewer than two roots, written or filled in; a `state`
below 0 or not below `[casscf]` `nroots`; a `state` without
`quasi_degenerate = true`; a `root` beside it (`state` names what is
differentiated). The state average is then not refused; the spin and the
ERI mode are, as above. An `[optimize]` over it is taken, and follows the
state K from geometry to geometry (`optimize/README.md`).

```toml
[eri]
mode = "density_fitting"  # required by [nac]
[casscf]
ncas = 3
nelecas = 4
nroots = 2                # at least 2, at any weights
conv_tol_grad = 1e-10     # the couplings are the CASSCF's only at convergence
[nac]
pairs = [[0, 1]]          # (bra, ket); left out, every I < J
mult_ediff = false        # the default
use_etfs = false          # the default
```

`[nac]` is the nonadiabatic couplings ⟨I|∂J/∂R⟩ between pairs of roots of
the state-averaged `[casscf]` it reads (`nac/README.md`), written to
`nac.json`. `pairs` is an array of two-integer arrays, each two distinct
roots, at least 0; an empty one is `InvalidConfig` (leave it out for every
pair I < J). `mult_ediff` and `use_etfs` are `mcscf::NacOptions`'. Across
tables, when the file is read: a `[casscf]` with `nroots` below 2, or a
pair naming a root at or past it, is `InvalidConfig`; and like `[nevpt2]`
it runs over density fitting only, `Unsupported` under an integral-direct
`[eri]`, written or filled in. The step refuses each again when it runs,
for a pipeline built by hand. Any `[casscf]` `weights` are taken.

```toml
[nevpt2_nac]
pairs = [[1, 2]]          # (bra, ket) QD states; left out, every P < Q
variant = "pc"            # the default "sc"
mult_ediff = false        # the default
use_etfs = false          # the default
```

`[nevpt2_nac]` is the SC- or PC-QD-NEVPT2 derivative couplings ⟨Ψ_P|∂Ψ_Q/∂R⟩
between pairs of QD states, indices into H_eff's eigenvalues ascending, over
the state-averaged `[casscf]` it reads (`nevpt2_nac/README.md`), written to
`nevpt2_nac.json`. `pairs`, `mult_ediff` and `use_etfs` are `[nac]`'s, a
pair naming two distinct QD states; `variant` is `[nevpt2_gradient]`'s.
Across tables, when the file is read: a `[casscf]` with `nroots` below 2,
or a pair naming a state at or past it (H_eff has one per root), is
`InvalidConfig`; an integral-direct `[eri]`, written or filled in, and a
`[molecule]` `multiplicity` other than 1 are `Unsupported`. The step refuses
each again when it runs, for a pipeline built by hand; a PC refusal (a
class's metric with no positive eigenvalue, or a pole: an intruder or a
denominator at round-off) can only
be seen there, once H_eff is built, and is the effective Hamiltonian's
`Numerical` error. No `[nevpt2]` is planned for it.

```toml
[optimize]                # Gaussian's thresholds by default
max_force = 4.5e-4        # hartree / bohr
rms_force = 3.0e-4
max_displacement = 1.8e-3 # bohr
rms_displacement = 1.2e-3
max_steps = 100           # evaluations after the first
# meci = [0, 1]           # the MECI of these two [casscf] roots instead
# max_gap = 1e-5          # hartree, the MECI's fifth criterion (meci only)
```

`[optimize]` is a geometry optimization (`optimize/README.md`). Each
threshold must be positive, and `max_steps` at least 0. It reruns the chain
from the molecule to the gradient at every geometry, and that chain is **the
config's own tables**: `parse_config` copies `[molecule]` (its `xyz` the
start), `[basis_set]`, `[eri]`, `[orthogonalizer]`, `[guess]` and `[scf]`
into the step, each read as the pipeline reads it, from an empty table if the
config has none. **With a `[casscf_gradient]` in the config the chain ends in
the CASSCF instead** (`optimize/README.md` "The CASSCF chain"), and
`[casscf]` and `[casscf_gradient]` are copied in too; the optimized energy is
then that root's. **With a `[nevpt2_gradient]` it ends in the SC- or PC-NEVPT2
gradient** (`optimize/README.md` "The NEVPT2 chain"), `[casscf]` and
`[nevpt2_gradient]` copied in, and the energy optimized is the NEVPT2 total
energy, or with `quasi_degenerate` the QD state's E_K. Any two of `[gradient]`, `[casscf_gradient]` and `[nevpt2_gradient]`
are refused, and under an integral-direct `[eri]` either CASSCF chain is its
gradient's `Unsupported`, as above, as are `[nevpt2_gradient]`'s other
refusals (a state average without `quasi_degenerate`, the start's
multiplicity). One loading `from` a file is refused: the chain is rerun at every
geometry. **Then `[molecule]` is planned loaded from the `final.xyz`
the step writes** (`MoleculeStep::load`, ordered after the optimization by
the `OptimizeOutput` it reads), so every other step in the config -- the
outer `[scf]`, a `[localization]` -- runs at the optimum. That is the only
way the molecule is ever loaded: `[molecule] from` is refused like a step
with no `load`.

**With `meci = [I, J]` it finds the minimum-energy conical intersection of
roots I and J of the `[casscf]`** (`optimize/README.md` "The MECI"): the
chain ends in `[casscf]`, copied in, and no gradient table is read into it.
`meci` must be two distinct non-negative roots, and `max_gap` is refused
without it. Once `[casscf]` is copied in, `OptimizeStep::check_meci` refuses
a `[casscf]` of one root or a pair past its `nroots` (`InvalidConfig`), and
unequal `weights` on I and J or an integral-direct `[eri]`, written or
filled in (`Unsupported`). A `[gradient]` or `[casscf_gradient]` beside it
is a later step, run at the MECI.

**With a `[nevpt2_gradient]` beside `meci`, P and Q are QD-NEVPT2 states**
(`optimize/README.md` "The QD-NEVPT2 MECI"): the table must be
`quasi_degenerate = true`, its `variant` picks SC or PC, and it is copied into
the chain and taken out of the steps (at the MECI it would differentiate one
of two degenerate states). Refused: a `state` key in it, or a table that is
not quasi-degenerate (`InvalidConfig`); a `[casscf]` of one root or a state
past its `nroots` (`InvalidConfig`); a molecule of multiplicity other than 1
or an integral-direct `[eri]` (`Unsupported`). Unequal `weights` are
allowed.

```toml
[hessian]
method = "analytic"       # the default; or "numerical"
displacement = 1e-3       # bohr, the numerical method's only
```

`[hessian]` is the RHF Cartesian Hessian and its harmonic frequencies
(`hessian/README.md`). `method` is `"analytic"` by default, `hessian::df_rhf`
at the SCF of one run of the chain to `[scf]`, density-fitted only: over an
integral-direct `[eri]` (the default) it is refused as Unsupported, naming
`[eri] mode = "density_fitting"` and `method = "numerical"` as the ways out.
`"numerical"` is central differences of the analytic gradient, over either
ERI kind; its `displacement` must be finite and positive, and is refused
beside the analytic method. Like `[optimize]` the step reruns the chain, and
`parse_config` copies the config's own `[basis_set]`, `[eri]`,
`[orthogonalizer]`, `[guess]` and `[scf]` into it the same way, refusing one
that loads `from` a file. The molecule is not copied: the step reads the
pipeline's `MoleculeOutput`, so beside an `[optimize]` it runs at the
optimum. **With a `[casscf_gradient]` in the config the Hessian is that
root's DF-CASSCF Hessian instead**, by either method (`hessian/README.md`
"The CASSCF chain"), and `[casscf]` and `[casscf_gradient]` are copied in
too. **With a `[nevpt2_gradient]` the Hessian is that SC- or PC-NEVPT2
energy's** (`hessian/README.md` "The NEVPT2 chain"), `[casscf]` and
`[nevpt2_gradient]` copied in: analytic by default (`hessian::df_sc_nevpt2`
or `df_pc_nevpt2`, the variant's, and the one analytic method that takes
`coordinates`), or numerical, differencing the gradient. With
`quasi_degenerate = true` it is QD state `state`'s over the state-averaged
`[casscf]`, by either method (`hessian/README.md` "The QD-NEVPT2 chain"):
analytic by default (`hessian::df_qd_sc_nevpt2` or `df_qd_pc_nevpt2`), or
numerical, the state tracked by its eigenvector. Without it a state average
is `[nevpt2_gradient]`'s own refusal (Unsupported). By either method it is
all-electron (a `frozen_core` other than 0, `"auto"` included, refused as
InvalidConfig) and density-fitted (an integral-direct `[eri]` refused as
Unsupported), each before anything runs, since the analytic calls take none
of them. Under PC, `pc_floor` (default 1e-10, `HessianStep::kPcFloor`; 0
or more, 0 the round-off cut) is the PC conditioning floor of the whole
chain the Hessian differentiates, and of the outer `[nevpt2_gradient]`
beside it (`hessian/README.md` "The PC floor", #923); every other PC path,
a `[nevpt2_gradient]` with no `[hessian]` among them, stays at 0. Two of
`[gradient]`, `[casscf_gradient]` and
`[nevpt2_gradient]` beside a `[hessian]` are refused as InvalidConfig,
since one energy's gradient is differenced.

**A table is a request, and the steps producing what it reads are filled
in.** The file above plans all twelve steps: `[orthogonalizer]` reads the
overlap, which reads the spherical transform and the pair classes. Each
filled-in step is **read from an empty table**, so its defaults apply and a
required option it lacks is the same error a written table would give:
`[overlap]` alone is refused with `[molecule] needs 'xyz'`. So only a step
with a required option ever has to be written out for its own sake, and there
is no second list of which options are required. Nothing is pruned: every
planned step is upstream of a request.

The filling in is `Calculation::create`'s insert hook, called with the
missing producer's type. `Pipeline::create` without the hook stays strict, so
a pipeline built by hand lists every step.

**`from` loads a product instead of making it.** For a step with a `load`
(`step/README.md`), `from` names the file its product is read from, and the
step is planned by its load, so only what the load reads is filled in:

```toml
[guess]
from = "water_d.npy"   # the density, read in place of SAD

[scf]                  # starts from it
```

`from` stands alone: the step's options configure a run that does not happen,
so any other key beside it is refused. A step with no `load` refuses `from`.
It is a path, resolved like every other.

**Every path in a config is an input, relative to the input directory**:
`[molecule] xyz`, a `[basis_set] basis` that is not a compiled-in name, a
`[eri] aux_basis` that is not a compiled-in fitting set's name
(`is_builtin_aux_basis`, the other table), and every `from`. The input directory is the run's, not the config file's:
`free-breakfast -i <dir>`, default `.`, so a config can be shared between
runs that each bring their own inputs. An empty `input_dir` leaves the paths
relative to the working directory. The steps document that their paths
"arrive resolved", and this is where that happens.

**Outputs are named by the run, not the config.** Every output a planned step
can write is written, into the output directory (`free-breakfast -o <dir>`,
default `fbf_output`) under the step's `kName`. A pipeline holds one of each
step, so no two collide:

| Step | File |
|------|------|
| `[spherical_transform]` with `write_npy = true` | `spherical_transform.npy`: U, `nao_cart` × `nao_sphe`, float64 (the identity when no shell is above p) |
| `[overlap]` with `write_npy = true` | `overlap.npy`: S, `nao_sphe` square, float64 |
| `[scf]` | `scf.molden`: every MO, the occupied at 2 |
| `[localization]` | `localization.molden`: the localized occupied orbitals in ascending energy, then the canonical virtuals |
| `[localization]` with `livvo = true` | `localization_livvo.molden`: the LIVVOs, at 0 |
| `[casci]` | `casci.molden`: the lowest root's natural orbitals |
| `[casscf]` | `casscf.molden`: the state-averaged natural orbitals |
| `[casscf]` with `write_npy = true` | `casscf_orbitals.npy`: the orbitals, nao × nmo; `casscf_ci.npy`: the CI vectors, dim × nroots; both float64, what `orbitals` and `ci` read back |
| `[gradient]` | `gradient.json`: the SCF energy, the gradient `total` and each of its `terms` (hartree / bohr, one row of three per atom) |
| `[casscf_gradient]` | `casscf_gradient.json`: the root, its energy, the gradient `total` and each of its `terms`, as `gradient.json` |
| `[nevpt2_gradient]` | `nevpt2_gradient.json`: the root, the variant, the NEVPT2, CASSCF and correlation energies, the gradient `total` and each of its `terms`, max \|z^c\|, the smallest inactive gap, the PCG iterations and residual, and X^L's residue per block; for PC, the modes each class cut; with `quasi_degenerate`, the state, the variant, E_K and H_eff's eigenvalues, the CASSCF energies, ρ, the gradient `total`, the `df_gradient` pass's `terms`, the `transition` pass's total, the diagnostics and the gaps |
| `[nac]` | `nac.json`: the options, the CASSCF's root energies, and per pair the roots, the gap, h_IJ and d^CSF (unscaled) and `total`, each one row of three per atom, and the Z-vector's products and residual |
| `[nevpt2_nac]` | `nevpt2_nac.json`: the variant and the options, the CASSCF's root energies, H_eff's eigenvalues and R, and per pair the QD states, E_Q − E_P, h_PQ, d^CAS, d^PT2 and the CSF part (unscaled) and `total`, each one row of three per atom, and the diagnostics |
| `[optimize]` | `optimize.xyz`: every evaluated geometry (Angstrom); `final.xyz`: the optimum, which `[molecule]` then loads; `optimize.json`: the energy, and per evaluation the energy, the four norms, the trust radius and whether it was accepted (under `meci`: the mean energy, the lower and upper energies and the gap, and per evaluation those, the branching plane's dimension, the pair overlap and the CASSCF's `gradient_norm` and `ci_residual_norm`; over QD states also `heff_gap`, `third_state_margin` and `root_gap`) |
| `[hessian]` | `hessian.json`: the method, the numerical method's displacement, the CASSCF chain's root, the NEVPT2 chain's variant (and PC's `pc_floor`), under `quasi_degenerate` its `qd` report (`hessian/README.md` "The QD-NEVPT2 chain"), the symmetrized Hessian (hartree / bohr², 3N rows of 3N), its asymmetry before symmetrizing, its translational sum, the harmonic frequencies (cm⁻¹, ascending, an imaginary one negative) and the normal modes (one row of 3N per frequency) |

There is no `molden` key, nor a `livvo_molden`, `npy` or `json_path` one: a config names
no output, so a later config can rely on what an earlier run wrote
(`final.xyz` is the one not named after its step).
`write_npy` is a switch, off by default, not a path. A file already there is
overwritten; a directory per run is what `-o` is for. The steps keep their
`molden` member (and `LocalizationStep` its `livvo_molden`, `SphericalTransformStep` and
`OverlapStep` their `npy`, `CasscfStep` its `orbitals_npy` and `ci_npy`, `GradientStep`, `CasscfGradientStep`, `Nevpt2GradientStep`, `NacStep` and `Nevpt2NacStep` their `json_path`), empty meaning not written, for a pipeline built by hand.

**The outputs are kept apart from the inputs.** Both are `InvalidConfig`,
comparing the directories named rather than their spelling (`.`, `''` and the
working directory spelt out are one; a trailing `/` or a `..` changes
nothing):

- an output directory that is the input directory, which would mix a run's
  outputs into its inputs;
- an input file directly in the output directory, which a step's output could
  overwrite. An absolute path, or one through `..`, reaches there however the
  two directories differ. A file in a directory *below* the output directory
  is apart, and so is an output directory below the input one, as the
  defaults are. A `[casci]` or `[casscf]` `orbitals` and a `[casscf]` `ci`
  are inputs too. `[optimize]`'s start and its chain's basis files are inputs;
  the `final.xyz` the molecule is then loaded from is this run's own output,
  read after it is written, and is not.

`read_config` then creates the output directory, before anything runs: a
directory that cannot be made is `IO`, refused before a GPU is touched rather
than after the SCF has converged. `parse_config` touches no disk.

**Strict.** Each of these is `InvalidConfig`, naming the table or key:

- a document naming no step;
- a table that names no step, or a step written as a key instead of a table;
- a missing required key, in a written table or in a filled-in step's empty one;
- a key the step has no option for (a step with no options has no keys);
- an enumerated option naming no value it has (`[guess] type`, which knows
  only `"sad"`; `[nevpt2_gradient] variant` and `[nevpt2_nac] variant`, which know `"sc"` and `"pc"`; `[hessian] method`, which knows `"analytic"` and `"numerical"`; `[eri] mode`, which knows `"integral_direct"` and `"density_fitting"`; `[localization] mode`, which knows `"pipek_mezey"` and `"ibo"`;
  `[localization] exponent`, 2 or 4; and `[cis] spin`, which knows
  `"singlet"` and `"triplet"`);
- `from` beside another key, on a step with no `load`, or on `[molecule]`;
  and under `[optimize]` or `[hessian]`, on a table of the chain it reruns;
- an `[optimize]` threshold that is not positive, or a `max_steps` below 0,
  or an `[optimize]` beside two of `[gradient]`, `[casscf_gradient]` and
  `[nevpt2_gradient]`; a `meci` that is not two distinct roots, a `max_gap`
  without it, or `meci` beside a `[nevpt2_gradient]` that is not
  quasi-degenerate or names a `state`, over a `[casscf]` of one root, or
  naming a root past its `nroots`;
- a `[hessian]` `displacement` that is not finite and positive, or one
  beside the analytic method; `ladder` or `pairs` beside the analytic
  method, or `coordinates` beside it without a `[nevpt2_gradient]`; a
  `[hessian]` beside two of `[gradient]`,
  `[casscf_gradient]` and `[nevpt2_gradient]`; or beside a
  `[nevpt2_gradient]` with a `frozen_core` other than 0; a `[hessian]`
  `pc_floor` that is negative or not finite, or one beside anything but a
  `[nevpt2_gradient]` of variant `"pc"`;
- an option of the other ERI mode (above): `[eri] aux_basis` under
  `"integral_direct"`, or `[scf] fock_rebuild_interval` under
  `"density_fitting"`; or an empty `aux_basis`;
- a `[memory]` key other than its two (`max_mem_per_node` included), a
  `memory` that is not a table, or a cap that is not a byte size of at least
  1 (below, "The `[memory]` table");
- a value of the wrong type. A boolean (`[localization] livvo`) is `true` or
  `false`, not `1` or `"true"`. Integers are read exactly, so `charge = 1.5` is
  refused, not truncated, and so is a value outside `int32_t`. A `double`
  option accepts an integer, so `eigenvalue_floor = 0` reads as `0.0`. An
  integer-array option (`[casci] active`) holds integers only, each read the
  same way, a string-array option (`[casci] avas`) strings only, and a
  pair-array option (`[nac] pairs`, `[nevpt2_nac] pairs`) two-integer arrays only;
- a `[nac]` `pairs` that is empty or holds a pair of equal or negative
  roots, or one over a `[casscf]` of fewer than two roots or naming a root
  past them;
- a `[nevpt2_nac]` `pairs` that is empty or holds a pair of equal or
  negative states, or one over a `[casscf]` of fewer than two roots or
  naming a state past them;
- a `[nevpt2]` `quasi_degenerate = true` over a `[casscf]` of fewer than
  two roots; a `[nevpt2]` or `[nevpt2_gradient]` `frozen_core` below 0 or a
  string other than `"auto"`; any `frozen_core` under `[nevpt2_nac]`;
- a `[nevpt2_gradient]` `quasi_degenerate = true` over a `[casscf]` of
  fewer than two roots; a `state` below 0, not
  below `nroots`, or without `quasi_degenerate`; or a `root` beside it;
- what `Pipeline::create` refuses (a cycle).

Some refusals are not `InvalidConfig` but `Unsupported`, since the request
is well formed and the method is what is missing: `[nevpt2]`,
`[casscf_gradient]`, `[nevpt2_gradient]`, `[nac]`, `[nevpt2_nac]` or the analytic `[hessian]` under
an integral-direct `[eri]`, `[cis]` under a
density-fitted one, `[nevpt2_gradient]`'s state
average (without `quasi_degenerate`) and multiplicity other than 1, and
`[nevpt2_nac]`'s multiplicity other than 1.

A misspelt `multiplicty` would otherwise run silently as a singlet. (The
`io.config` this replaced ignored unknown keys, and read a flat `Config`
that restated every step option a second time.) A syntactically invalid
document is `IO`, as `io.toml` reports it.

toml++ keeps a table's keys in a sorted `std::map`, so this module never sees
the order the tables were written in. Pipeline order comes from each step's
`View`, not the file.

## The `[memory]` table

The one table that names no step: it caps what a run may hold, per GPU, and
sets `Calculation`'s two fields. Both keys are optional and independent; a
missing key, or a missing table, leaves that cap off (0), which is the
uncapped run every config had before the table.

```toml
[memory]
max_device_mem_per_gpu = "10GiB"   # one device's share, its context and the
                                   # libraries' reserve included
max_pinned_mem_per_gpu = "16GiB"   # one device's pinned host memory, as its
                                   # DeviceResources counts it
```

`max_device_mem_per_gpu` is one device's memory, everything on it counted:
the CUDA or HIP context and the libraries' reserve as well as the pool.
`max_pinned_mem_per_gpu` is the pinned host memory one device's
`DeviceResources` owns: its pinned workspace and the counted persistent
pinned allocations (offloaded products), the total its `pinned_cap` bounds.
It has no default: unset, pinned memory is uncapped.

**There is no host-RAM key.** Memory is capped as device and pinned memory:
the pageable host allocations in the tree are O(nao²) or smaller (the
largest active-space one O(ncas⁴)), and the large host-side data, staging
and offloaded products, is pinned and capped above. `max_mem_per_node` was
considered and dropped, and is refused as an unknown key. A pageable host
allocation that grows to matter moves to pinned memory, or gets a key of its
own then.

A value is a byte size, in one of two forms:

- an integer, of bytes: `max_device_mem_per_gpu = 8000000000`;
- a string of a number and a unit, a space between them allowed: `"512MiB"`,
  `"10GiB"`, `"1.5GB"`, `"1.5 GB"`. The units are `B`; `kB`, `MB`, `GB` and
  `TB`, powers of 1000; and `KiB`, `MiB`, `GiB` and `TiB`, powers of 1024,
  each spelt exactly (`"10gib"` and `"10Gb"` are refused). A fraction is
  rounded down to a whole byte.

Anything else is `InvalidConfig` naming the key: a size below 1 byte
(`0`, `-1`, `"0GiB"`, `"0.5B"`), a string with no unit (`"1024"`), a unit not
in that list, an exponent (`"1e9B"` reads as the unit `e9B`), a size past
2⁶³ bytes, and a float or a boolean. A cap is never guessed.

The table is only read and stored here. Nothing in this module checks a plan
against it, and `Pipeline::run` does not read it.

## Tests

`test/pipeline/config_host.cpp`, host-only: every option read, the step
defaults kept, path resolution against the input directory, the output
names, the inputs kept apart from the outputs, which steps are filled in and
which are not, `from`, the `[memory]` caps in both forms and every unit, and
each refusal above. Nothing is run.
`test/gtest/pipeline_calculation_gpu.cpp` runs a config end to end, through
the API. `test/examples/` runs `examples/rhf.toml` and `examples/hessian.toml`
through the `free-breakfast` binary (`examples/README.md`): exit status,
the files written, and `hessian.json`'s keys.

## Files

| File | Role |
|------|------|
| `config.cppm` | Module interface: `Calculation` and its caps, `parse_config`, `read_config` |
| `config.cpp` | The document walk, each step's `read_step`, `read_active_space`, and the `[memory]` table's `read_memory` |
| `CMakeLists.txt` | The module library |
