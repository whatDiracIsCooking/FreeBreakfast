# Pipeline

The calculation pipeline: steps that each read a declared set of products and
write one, planned into a task graph and run on a `NodeResources`.

```
step/       free_breakfast.pipeline.step: the step contract -- View, RunShape, PipelineStep, and what a step reads, writes and holds, read off its run
pipeline/   free_breakfast.pipeline.pipeline: Pipeline<Steps...>, the planner -- steps wired by their Views, run on Taskflow with the device steps chained
molecule/   free_breakfast.pipeline.molecule: MoleculeStep, the root step -- an XYZ file read into a Molecule, or after an [optimize] the final.xyz it wrote; and OptimizeOutput, which that load reads
basis_set/  free_breakfast.pipeline.basis_set: BasisSetStep -- a BSE JSON basis read and placed over the Molecule
spherical_transform/  free_breakfast.pipeline.spherical_transform: SphericalTransformStep -- the basis's Cartesian-to-spherical U, uploaded once
primitive_pair_classes/  free_breakfast.pipeline.primitive_pair_classes: PrimitivePairClassesStep -- the basis's primitive-pair records, on the device
eri/        free_breakfast.pipeline.eri: EriStep -- the pipeline's two-electron integrals, once for every reader: J's and K's pair classes, or its own aux basis placed, then the metric, the three-center tensor and the fit B = L^-1 (mn|P)
overlap/    free_breakfast.pipeline.overlap: OverlapStep -- the spherical AO overlap S
orthogonalizer/  free_breakfast.pipeline.orthogonalizer: OrthogonalizerStep -- X = S^{-1/2}, the symmetric orthogonalizer
core_hamiltonian/  free_breakfast.pipeline.core_hamiltonian: CoreHamiltonianStep -- the spherical core Hamiltonian H = T + V
guess/      free_breakfast.pipeline.guess: GuessStep -- the initial-guess density (SAD), host work beside the device chain
scf/        free_breakfast.pipeline.scf: ScfStep -- closed-shell RHF to self-consistency over the EriOutput (integral-direct or density-fitted), its Fock build, DIIS and loop private to the step
localization/  free_breakfast.pipeline.localization: LocalizationStep -- the SCF's occupied orbitals localized, by Pipek-Mezey or as intrinsic bond orbitals
active_space/  free_breakfast.pipeline.active_space: ActiveSpaceSelection -- not a step: the active space by index or by AVAS over MINAO, and C R, shared by the steps that take one
casci/      free_breakfast.pipeline.casci: CasciStep -- CASCI over the SCF's orbitals: the active space by index or by AVAS over MINAO, the active-space Hamiltonian, the CI over the GPU sigma, each root's 1-RDM, and the natural orbitals as Molden
casscf/     free_breakfast.pipeline.casscf: CasscfStep -- state-averaged CASSCF from the SCF's orbitals, the start picked as Casci picks its space, mcscf::solve over the EriOutput (integral-direct, or fully density-fitted), the core and virtuals canonicalized, and the natural orbitals as Molden
fci/        free_breakfast.pipeline.fci: FciStep -- FCI over every SCF orbital, CASCI's calls with no core, refused up front past the device's memory budget (`DeviceResources::memory_budget()`)
nevpt2/     free_breakfast.pipeline.nevpt2: Nevpt2Step -- SC- and PC-NEVPT2 per root of the CASSCF, over the density-fitted EriOutput only: each root's core and virtuals canonicalized over its own Fock, its RDMs, dm3 and f3, and the eight classes of each; with quasi_degenerate, SC- and PC-QD-NEVPT2 over the state-averaged orbitals, pair by pair, and each effective Hamiltonian's eigenpairs
cis/        free_breakfast.pipeline.cis: CisStep -- CIS (TDA) singlet or triplet excitations over the SCF's RHF: davidson_solve over cis::Sigma, or cis::DfSigma under a density-fitted EriOutput, from cis::guess with cis::Preconditioner; a singlet's transition dipoles and oscillator strengths
gradient/   free_breakfast.pipeline.gradient: GradientStep -- the closed-shell RHF nuclear gradient at the SCF's orbitals, gradient::df_rhf over a density-fitted EriOutput's B, L and aux basis or gradient::rhf over an integral-direct one, every term of it, written as gradient.json
casscf_gradient/  free_breakfast.pipeline.casscf_gradient: CasscfGradientStep -- the DF-CASSCF nuclear gradient of one root of the CASSCF, mcscf::df_gradient over the density-fitted EriOutput's B, L and aux basis only, every term of it, written as casscf_gradient.json
nevpt2_gradient/  free_breakfast.pipeline.nevpt2_gradient: Nevpt2GradientStep -- the SC- or PC-NEVPT2 nuclear gradient of the single-state DF-CASSCF, nevpt2::sc_gradient or nevpt2::pc_gradient, or with quasi_degenerate of one SC- or PC-QD-NEVPT2 state of the DF-SA-CASSCF, nevpt2::qd_sc_gradient or nevpt2::qd_pc_gradient, over the density-fitted EriOutput only, every term of it with max |z^c|, the smallest inactive gap and the PCG iterations, written as nevpt2_gradient.json
nac/        free_breakfast.pipeline.nac: NacStep -- the nonadiabatic couplings between pairs of roots of the equal-weight DF-SA-CASSCF (every pair by default), mcscf::df_nac over the density-fitted EriOutput only, each with its numerator h_IJ and CSF term, written as nac.json
nevpt2_nac/  free_breakfast.pipeline.nevpt2_nac: Nevpt2NacStep -- the SC- or PC-QD-NEVPT2 derivative couplings between pairs of QD states of the DF-SA-CASSCF (every pair by default), nevpt2::qd_sc_nac or nevpt2::qd_pc_nac over the density-fitted EriOutput only, each with h_PQ, d^CAS, d^PT2 and its CSF part, and H_eff's eigenvalues and R, written as nevpt2_nac.json
optimize/   free_breakfast.pipeline.optimize: OptimizeStep -- an RHF geometry optimization over either ERI kind, or a DF-CASSCF root's or its SC- or PC-NEVPT2's, or one SC- or PC-QD-NEVPT2 state's followed by its overlaps, with its orbitals carried from geometry to geometry, geometry::optimize over the chain from the molecule to the gradient, rerun as a nested Pipeline at every geometry; the config then loads the molecule from its final.xyz, so every other step runs at the optimum
hessian/    free_breakfast.pipeline.hessian: HessianStep -- the RHF's or, beside a [casscf_gradient], that DF-CASSCF root's, or beside a [nevpt2_gradient] that SC- or PC-NEVPT2 energy's (or with quasi_degenerate one QD state's) Cartesian Hessian at the pipeline's molecule, analytic by default (hessian::df_rhf at the SCF of one nested Pipeline run of the chain to [scf], or hessian::df_casscf / df_sacasscf, df_sc_nevpt2 / df_pc_nevpt2 or df_qd_sc_nevpt2 / df_qd_pc_nevpt2 at the CASSCF of one to [casscf]; density-fitted only) or by central differences of the analytic gradient, the chain to the gradient rerun at each of the 6N displaced geometries (a QD state tracked by its eigenvector); symmetrized, its asymmetry and translational sum reported, its harmonic frequencies and normal modes logged, written as hessian.json
config/     free_breakfast.pipeline.config: the TOML front end and the registry, Calculation -- a config file read into the pipeline it asks for
shape/      free_breakfast.pipeline.shape: plan_shape and preflight -- not steps: a planned Calculation's bases placed and its FootprintDimensions filled, and the plan's footprint held to the [memory] caps, before the first step runs
footprint/  free_breakfast.pipeline.footprint: FootprintDimensions, step_adaptive_unit and the plan fold, and check_plan, the fold held to the caps
```

`src/main.cpp` reads a config into a `Calculation`, holds it to its
`[memory]` caps (`preflight`, below), runs it, and prints the products. [`examples/`](../../examples/README.md) holds a small config per
step family (RHF, CASSCF, NEVPT2 and QD-NEVPT2, `[nac]`, `[nevpt2_nac]`, `[hessian]`) to run
through it; `test/examples/` runs two of them.

## Density fitting

One table chooses the two-electron integrals for the whole pipeline:

```toml
[eri]
mode = "density_fitting"      # default "integral_direct"
aux_basis = "cc-pvdz-jkfit"   # optional: <basis>-acCD-6-6 when carried
```

`EriStep` builds them once, B or the J and K pair classes, and `[scf]`,
`[casscf]`, `[nevpt2]`, `[cis]`, `[gradient]`, `[casscf_gradient]`,
`[nevpt2_gradient]`, `[nac]` and `[nevpt2_nac]` all read
that one `EriOutput`
(`[optimize]` reruns its own copy of `[eri]` at every geometry): `[scf]`
fits its Fock build, `[casscf]` runs `mcscf::DensityFitting::Full`, the
DF-CASSCF, `[nevpt2]` its MO blocks and Fock over the same B, and `[cis]`
its sigma over B's MO blocks (`cis::DfSigma`), and `[gradient]` the DF-RHF
gradient over B, L and the aux basis the product keeps (under integral-direct
ERIs, the exact RHF gradient), and `[casscf_gradient]` the DF-CASSCF
gradient over the same three, and `[nevpt2_gradient]` the SC- or PC-NEVPT2
gradient (or, with `quasi_degenerate`, one QD-NEVPT2 state's) over B, L and
the aux basis too, and `[nac]` the SA-CASSCF couplings
over the same three, and `[nevpt2_nac]` the QD-NEVPT2 couplings over them
too. `[nevpt2]`, `[casscf_gradient]`, `[nevpt2_gradient]`, `[nac]`
and `[nevpt2_nac]` run over density fitting only, and are `Unsupported` under an integral-direct `[eri]`. A missing
`[eri]` is filled in, integral-direct. The rules and refusals are
`config/README.md`'s; what the CASSCF's scratch costs past B, and the memory
check before its loop, `casscf/README.md`'s.

**The device chain is what makes the pair classes shareable.** `coulomb` and
`exchange` write each class's scratch, so `IntegralDirect` holds them
`mutable`; every reader is a device step, and device steps run one at a time
in plan order, so no two builds write them at once.

## Memory caps

A config's `[memory]` table caps what a run may hold, per GPU
(`config/README.md`, "The `[memory]` table"):

```toml
[memory]
max_device_mem_per_gpu = "10GiB"   # the device: context, libraries and pool
max_pinned_mem_per_gpu = "16GiB"   # the pinned host memory one device owns
```

[`examples/memory.toml`](../../examples/memory.toml) is a capped RHF. `main`
calls `preflight` (`shape/`) after it builds the node and before the first
step, capped or not, since it sizes every device's workspaces: each
`DeviceResources` holds a device and a pinned workspace sized at construction
and never resized (`core/resources/README.md`), so the node is rebuilt with
them at the plan's maximum. With neither key the plan is not checked against
anything but the device: the workspaces are the most any planned step
borrows, an AVAS worst case past the device's budget is narrowed as below
against that budget, and a device workspace past it is refused. With either
key, in order:

1. **The caps reach the devices.** Each `DeviceResources` is rebuilt (once the
   plan is checked, with the workspaces of step 4) with its
   pool capped at `max_device_mem_per_gpu` less the 256 MiB reserve the
   context and the library handles hold outside the pool
   (`DeviceResources::kDeviceReserve`), the cap also the pool's release
   threshold, and its pinned total capped at `max_pinned_mem_per_gpu`
   (#1130's cap, which refuses a counted pinned buffer past it at run time). A
   device cap that leaves no pool past the reserve and the headroom is
   refused.
2. **The machine check.** Pinned memory is not pageable, so the node's
   devices times `max_pinned_mem_per_gpu` must fit `/proc/meminfo`'s
   `MemAvailable`; a plan past it is refused naming both. This stands in for
   a node-wide host cap, which has no key.
3. **The fold.** The plan is shaped (`plan_shape`) and each planned step's
   footprint folded in plan order (`footprint/README.md`, "The peak model"):
   at step k, every persistent byte up to k, step k's scratch and the plan's
   maximum workspace. The device peak must fit the pool's cap less the
   fragmentation headroom (`kPoolHeadroomBytes`, 64 MiB), less what the
   `DeviceResources` holds in the pool itself from construction
   (`DeviceResources::kOwnPoolBytes`, its 4-byte solver `info` flag, which no
   step counts) and, on a pool that rounds a reused block (ROCm's), an eighth
   of the peak; the pinned peak
   must fit the pinned cap. A plan past either is refused, `InvalidConfig`
   naming the step that reaches the peak, the peak and the cap, before any
   step runs.
4. **The workspaces are sized** at the plan's maximum: the devices of step 1
   are built with them, held for the run. Nothing grows them, so a borrow
   past them, a footprint that undercounted, is refused at the step.
5. **The pre-warm.** Under a device cap, the pool's cap less the headroom is
   allocated and freed once. The pool keeps it, its release threshold being
   the cap, and a card that cannot supply it, because another process holds
   part of it, is refused there (`Allocation`, naming the free memory and
   the cap) rather than at a later step.

Pre-flight logs each kind's peak and the step reaching it, and the workspace
gap: the workspace is held at the plan's maximum throughout, and the peak
step uses only its own part of it, so a refused plan with a large gap is
fixed by moving a one-off carving to scratch. For an adaptive step (one with
an `adaptive_unit`, whose footprint's scratch is the least it runs with) it
logs what the cap leaves it above that least, at warn level when that is
under one more unit: the plan fits, but the step runs at its narrowest.

**A step without a footprint is refused under a cap.** The fold would count
it as zero and could pass a plan that does not fit, so it names the step
and says to drop `[memory]` to run uncapped. Today that is every step from
`[localization]` on but `[casci]`, `[casscf]`, `[fci]`, `[gradient]`,
`[casscf_gradient]` and `[nac]`: localization and CIS (#1159), the Hessians
(#1053), the NEVPT2 steps (#1152) and the nested ones. Each joins the capped plans as its footprint
lands, with no change here.

**AVAS under a cap.** An AVAS step's active space is unknown until it runs,
so the plan is shaped at its worst case (`shape/README.md`). When every
active space planned is AVAS's and that worst case does not fit, pre-flight
finds the most orbitals that do (the fold grows with the space, so by
bisection), sets each AVAS step's `avas_max_ncas` to it and warns; a pick of
more is then refused at the step (#1042's check), never truncated. When the
worst case fits, it logs that the plan fits any AVAS pick up to it. When not
even one orbital fits, the refusal says to raise the cap. A plan that also
holds an FCI or an active space by index is not narrowed, since its worst
case sizes those too: it is checked as shaped.

**What is not capped.** Pageable host memory: every pageable allocation in
the tree is O(nao²), or O(ncas⁴) in the active space, and large host-side
data is pinned, where the cap counts it. Device memory outside the pool, the
context and the handles, is the reserve: a floor measured on the 3080, which
grows as kernel modules load. ROCm 7.2.4 enforces the pool's `maxSize` as
CUDA does (`core/wwr/README.md`).
