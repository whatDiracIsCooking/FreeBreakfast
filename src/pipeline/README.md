# Pipeline

The calculation pipeline: steps that each read a declared set of products and
write one, planned into a task graph and run on a `NodeResources`.

```
step/  free_breakfast.pipeline.step: the step contract -- View, RunShape, PipelineStep, and what a step reads, writes and holds, read off its run
```

Only the contract exists so far. The setup steps (`Molecule`, `BasisSet`,
`Overlap`, `CoreHamiltonian`, ...) and the `Pipeline<Steps...>` class that
plans and runs them come next, each its own module here.
