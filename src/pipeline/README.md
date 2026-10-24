# Pipeline

The calculation pipeline: steps that each read a declared set of products and
write one, planned into a task graph and run on a `NodeResources`.

```
step/       free_breakfast.pipeline.step: the step contract -- View, RunShape, PipelineStep, and what a step reads, writes and holds, read off its run
pipeline/   free_breakfast.pipeline.pipeline: Pipeline<Steps...>, the planner -- steps wired by their Views, run on Taskflow with the device steps chained
molecule/   free_breakfast.pipeline.molecule: MoleculeStep, the root step -- an XYZ file read into a Molecule
basis_set/  free_breakfast.pipeline.basis_set: BasisSetStep -- a BSE JSON basis read and placed over the Molecule
spherical_transform/  free_breakfast.pipeline.spherical_transform: SphericalTransformStep -- the basis's Cartesian-to-spherical U, uploaded once
primitive_pair_classes/  free_breakfast.pipeline.primitive_pair_classes: PrimitivePairClassesStep -- the basis's primitive-pair records, on the device
overlap/    free_breakfast.pipeline.overlap: OverlapStep -- the spherical AO overlap S
orthogonalizer/  free_breakfast.pipeline.orthogonalizer: OrthogonalizerStep -- X = S^{-1/2}, the symmetric orthogonalizer
core_hamiltonian/  free_breakfast.pipeline.core_hamiltonian: CoreHamiltonianStep -- the spherical core Hamiltonian H = T + V
guess/      free_breakfast.pipeline.guess: GuessStep -- the initial-guess density (SAD), host work beside the device chain
scf/        free_breakfast.pipeline.scf: ScfStep -- closed-shell RHF to self-consistency, its Fock build, DIIS and loop private to the step
localization/  free_breakfast.pipeline.localization: LocalizationStep -- the SCF's occupied orbitals localized, by Pipek-Mezey or as intrinsic bond orbitals
casci/      free_breakfast.pipeline.casci: CasciStep -- CASCI over the SCF's orbitals: the active space by index or by AVAS over MINAO, the active-space Hamiltonian, the CI over the GPU sigma, each root's 1-RDM, and the natural orbitals as Molden
fci/        free_breakfast.pipeline.fci: FciStep -- FCI over every SCF orbital, CASCI's calls with no core, refused up front past the device's free memory
config/     free_breakfast.pipeline.config: the TOML front end and the registry, Calculation -- a config file read into the pipeline it asks for
```

`src/main.cpp` reads a config into a `Calculation`, runs it, and prints the
products.
