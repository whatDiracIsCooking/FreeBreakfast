# Pipeline

The calculation pipeline: steps that each read a declared set of products and
write one, planned into a task graph and run on a `NodeResources`.

```
step/       free_breakfast.pipeline.step: the step contract -- View, RunShape, PipelineStep, and what a step reads, writes and holds, read off its run
molecule/   free_breakfast.pipeline.molecule: MoleculeStep, the root step -- an XYZ file read into a Molecule
basis_set/  free_breakfast.pipeline.basis_set: BasisSetStep -- a BSE JSON basis read and placed over the Molecule
spherical_transform/  free_breakfast.pipeline.spherical_transform: SphericalTransformStep -- the basis's Cartesian-to-spherical U, uploaded once
primitive_pair_classes/  free_breakfast.pipeline.primitive_pair_classes: PrimitivePairClassesStep -- the basis's primitive-pair records, on the device
overlap/    free_breakfast.pipeline.overlap: OverlapStep -- the spherical AO overlap S
orthogonalizer/  free_breakfast.pipeline.orthogonalizer: OrthogonalizerStep -- X = S^{-1/2}, the symmetric orthogonalizer
core_hamiltonian/  free_breakfast.pipeline.core_hamiltonian: CoreHamiltonianStep -- the spherical core Hamiltonian H = T + V
config/     free_breakfast.pipeline.config: the TOML front end -- a config file read into the steps it names
```

The `Pipeline<Steps...>` class that plans and runs these steps comes next,
then SAD as the first host step, each its own module here. Until the planner,
`src/main.cpp` runs them by hand in the order it will choose.
