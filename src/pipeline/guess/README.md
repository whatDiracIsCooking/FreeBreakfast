# Pipeline: Guess

Build the initial-guess density, or read one back from a `.npy` file. SAD is
the only kind built so far.

## Module

`free_breakfast.pipeline.guess`, target `free_breakfast::pipeline::guess`. It
re-exports `free_breakfast.basis` (and through it `free_breakfast.molecule`)
and Taskflow, so an importer can name everything in `run`'s signature without
a second import. It links no GPU library: the step is host work. It links
`sad` and `io::sad` publicly, since `run`'s inline body names both, and
`io::npy` for `load`'s.

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
  static Result<Guess> load(const std::filesystem::path &from, View<BasisSet> in);
};
```

A thin wrapper over `sad`'s task form, `sad_density(tf::FlowBuilder&, …)`,
passing it `io.sad`'s `builtin_sad_density`. So a basis resolved by name
(`BasisSet::name`) takes each element's density from the stamps compiled into
the binary from `data/sad/`, and runs an atomic SCF only for an element they
lack. A basis read from a path runs every one. The errors are `sad_density`'s:

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

**`load` reads a saved density** (`[guess] from = "d.npy"`, `step/README.md`):
a float64 `.npy` file, `nao_sphe` square, in either layout, since `read_npy`
returns column-major. It reads only the `BasisSet`, for that count, and runs on
the host like `run`, with no subflow. The file carries only the numbers, so
only the shape is checked. A file made over another basis of the same size, or
the same atoms in another order, is accepted. A file that cannot be read is
`IO`; another shape is `InvalidConfig`.

**`Molecule` is read for its atoms only**, never its device nuclei, as
`BasisSet` reads it.

## Tests

`test/gtest/pipeline_guess_gpu.cpp`:

- The step's contract is checked by `static_assert`.
- The step's density, run on a subflow, is checked bitwise against a direct
  serial `sad_density`.
- A compiled-in set (cc-pVDZ) is checked to arrive named, and the step's
  density is checked bitwise against `sad_density` with the baked stamps.
- A basis placed over other atoms is checked to be `InvalidConfig`.
- `load` is checked to read a saved density back bitwise. The wrong shape is
  checked to be `InvalidConfig`, and a missing file `IO`.

The step itself never touches the device, but its input `Molecule` does, so the
tests need a GPU.

`test/gtest/pipeline_calculation_gpu.cpp` runs the step through the real
registry, beside the device chain:

- The planned guess is checked to equal the step called by hand.
- trace(D S) is checked to equal Σ Z_A to 1e-10, with S being the overlap the
  same pipeline built on the device.
- A failed root is checked to fail the guess with the root's error.
- An SCF from a loaded guess is checked to match one from its own.

`test/gtest/pipeline_config_host.cpp` has the `[guess]` table's cases.

## Files

| File | Role |
|------|------|
| `guess.cppm` | Module interface: `GuessType`, `Guess`, `GuessStep` and its `load` |
| `CMakeLists.txt` | The module library |
