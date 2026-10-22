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
config/     free_breakfast.pipeline.config: the TOML front end -- a config file read into the steps it names
```

The rest of the setup steps (`Orthogonalizer`, `CoreHamiltonian`, ...)
and the `Pipeline<Steps...>` class that plans and runs them come next, each its
own module here.
