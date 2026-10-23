# Pipeline: Guess

Build the initial-guess density. SAD is the only kind so far.

## Module

`free_breakfast.pipeline.guess`, target `free_breakfast::pipeline::guess`. It
re-exports `free_breakfast.basis` (and through it `free_breakfast.molecule`)
and Taskflow, so an importer can name everything in `run`'s signature without
a second import. It links no GPU library: the step is host work.

## The step

```cpp
enum class GuessType : uint8_t { Sad };

struct Guess {
  std::vector<double> d;  // nao_sphe square, column-major, spherical, on the host
};

struct GuessStep {
  static constexpr std::string_view kName = "guess";

  GuessType type = GuessType::Sad;

  Result<Guess> run(View<Molecule, BasisSet> in, tf::Subflow &subflow) const;
};
```

A thin wrapper over `sad`'s task form, `sad_density(tf::FlowBuilder&, …)`, with
the same errors:

- a basis not placed over these atoms, or one too small to seat an atom's
  occupied subshells, is `InvalidConfig`;
- an atom too large for the atomic SCF is `Unsupported`;
- a non-converged atomic SCF is `MaxIterationsExceeded`.

**One step, with a `type` option**, not one step per kind of guess. Several
steps producing one `Guess` would be two producers of a product, which the
registry refuses. `run` dispatches on `type`, and its `View` carries the inputs
of every kind. SAD needs only the `Molecule`'s atoms and the `BasisSet`. A kind
that reads more, such as a projection from a previous geometry, widens the
`View` for all of them. In TOML the option is `[guess] type = "sad"`, the
default, and `pipeline.config` refuses any other value before anything runs.

**`HostSubflow`-shaped.** `run` takes a `tf::Subflow&` and no device resource,
so the pipeline runs it as free host work beside the device chain. On a config
asking for `[guess]` and the device products, the atomic SCFs run on worker
threads while S, X and H are queued. Inside the step, `sad_density` adds one
task to the subflow. That task spawns one subtask per distinct (Z, shells)
stamp.

**The subflow is joined before `run` returns.** The task writes a `Result`
local to `run`'s frame. Taskflow's implicit join would run the subflow only
after the frame is gone. This is the same lesson as `sad_density`'s own
explicit join, one level up.

**The product is host data**, a `std::vector<double>`, not a
`SphericalMatrixAO`. The rule that every matrix product is a
`SphericalMatrixAO` is about the layout of device matrices. A host step never
touches the device (`step/README.md`), so whichever device step first reads
the guess uploads it. That is the Fock build, which does not exist yet.

**`Molecule` is read for its atoms only**, never its device nuclei, as
`BasisSet` reads it.

## Tests

`test/gtest/pipeline_guess_gpu.cpp`:

- The step's contract is checked by `static_assert`.
- The step's density, run on a subflow, is checked bitwise against a direct
  serial `sad_density`.
- A basis placed over other atoms is checked to be `InvalidConfig`.

The step itself never touches the device, but its input `Molecule` does, so the
tests need a GPU.

`test/gtest/pipeline_calculation_gpu.cpp` runs the step through the real
registry, beside the device chain:

- The planned guess is checked to equal the step called by hand.
- trace(D S) is checked to equal Σ Z_A to 1e-10, with S being the overlap the
  same pipeline built on the device.
- A failed root is checked to fail the guess with the root's error.

`test/gtest/pipeline_config_host.cpp` has the `[guess]` table's cases.

## Files

| File | Role |
|------|------|
| `guess.cppm` | Module interface: `GuessType`, `Guess`, `GuessStep` |
| `CMakeLists.txt` | The module library |
