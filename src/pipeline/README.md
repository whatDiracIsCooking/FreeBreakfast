# Pipeline

The calculation pipeline: steps that each read a declared set of products and
write one, planned into a task graph and run on a `NodeResources`.

```
step/       free_breakfast.pipeline.step: the step contract -- View, RunShape, PipelineStep, and what a step reads, writes and holds, read off its run
molecule/   free_breakfast.pipeline.molecule: MoleculeStep, the root step -- an XYZ file read into a Molecule
basis_set/  free_breakfast.pipeline.basis_set: BasisSetStep -- a BSE JSON basis read and placed over the Molecule
```

The rest of the setup steps (`SphericalTransform`, `Overlap`, `CoreHamiltonian`, ...)
and the `Pipeline<Steps...>` class that plans and runs them come next, each its
own module here.
