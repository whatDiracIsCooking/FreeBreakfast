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
fci/        free_breakfast.pipeline.fci: FciStep -- FCI over every SCF orbital, CASCI's calls with no core, refused up front past the device's free memory
nevpt2/     free_breakfast.pipeline.nevpt2: Nevpt2Step -- SC- and PC-NEVPT2 per root of the CASSCF, over the density-fitted EriOutput only: each root's core and virtuals canonicalized over its own Fock, its RDMs, dm3 and f3, and the eight classes of each
cis/        free_breakfast.pipeline.cis: CisStep -- CIS (TDA) singlet or triplet excitations over the SCF's RHF: davidson_solve over cis::Sigma, or cis::DfSigma under a density-fitted EriOutput, from cis::guess with cis::Preconditioner; a singlet's transition dipoles and oscillator strengths
gradient/   free_breakfast.pipeline.gradient: GradientStep -- the closed-shell RHF nuclear gradient at the SCF's orbitals, gradient::df_rhf over a density-fitted EriOutput's B, L and aux basis or gradient::rhf over an integral-direct one, every term of it, written as gradient.json
casscf_gradient/  free_breakfast.pipeline.casscf_gradient: CasscfGradientStep -- the DF-CASSCF nuclear gradient of one root of the CASSCF, mcscf::df_gradient over the density-fitted EriOutput's B, L and aux basis only, every term of it, written as casscf_gradient.json
nevpt2_gradient/  free_breakfast.pipeline.nevpt2_gradient: Nevpt2GradientStep -- the SC-NEVPT2 nuclear gradient of the single-state DF-CASSCF, nevpt2::sc_gradient over the density-fitted EriOutput only, every term of it with max |z^c|, the smallest inactive gap and the PCG iterations, written as nevpt2_gradient.json
optimize/   free_breakfast.pipeline.optimize: OptimizeStep -- an RHF geometry optimization over either ERI kind, or a DF-CASSCF root's with its orbitals carried from geometry to geometry, geometry::optimize over the chain from the molecule to the gradient, rerun as a nested Pipeline at every geometry; the config then loads the molecule from its final.xyz, so every other step runs at the optimum
config/     free_breakfast.pipeline.config: the TOML front end and the registry, Calculation -- a config file read into the pipeline it asks for
```

`src/main.cpp` reads a config into a `Calculation`, runs it, and prints the
products.

## Density fitting

One table chooses the two-electron integrals for the whole pipeline:

```toml
[eri]
mode = "density_fitting"      # default "integral_direct"
aux_basis = "cc-pvdz-jkfit"   # optional: <basis>-acCD-6-6 when carried
```

`EriStep` builds them once, B or the J and K pair classes, and `[scf]`,
`[casscf]`, `[nevpt2]`, `[cis]`, `[gradient]`, `[casscf_gradient]` and
`[nevpt2_gradient]` all read
that one `EriOutput`
(`[optimize]` reruns its own copy of `[eri]` at every geometry): `[scf]`
fits its Fock build, `[casscf]` runs `mcscf::DensityFitting::Full`, the
DF-CASSCF, `[nevpt2]` its MO blocks and Fock over the same B, and `[cis]`
its sigma over B's MO blocks (`cis::DfSigma`), and `[gradient]` the DF-RHF
gradient over B, L and the aux basis the product keeps (under integral-direct
ERIs, the exact RHF gradient), and `[casscf_gradient]` the DF-CASSCF
gradient over the same three, and `[nevpt2_gradient]` the SC-NEVPT2
gradient over B, L and the aux basis too. `[nevpt2]`, `[casscf_gradient]`
and `[nevpt2_gradient]` run over density fitting only, and are `Unsupported` under an integral-direct `[eri]`. A missing
`[eri]` is filled in, integral-direct. The rules and refusals are
`config/README.md`'s; what the CASSCF's scratch costs past B, and the memory
check before its loop, `casscf/README.md`'s.

**The device chain is what makes the pair classes shareable.** `coulomb` and
`exchange` write each class's scratch, so `IntegralDirect` holds them
`mutable`; every reader is a device step, and device steps run one at a time
in plan order, so no two builds write them at once.
