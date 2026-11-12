# Rust rebuild: design

## Programming model

The programming model has its own document,
[`programming_model.md`](programming_model.md). In short: a feature is a DAG
of tasks. A task declares its device, its layout (every buffer it holds, as a
function of its dimensions), its boundary and, optionally, a partition. An
edge declares its size, its kind (broadcast, reduce, gather, point-to-point),
its reduction order and its lifetime. The orchestrator derives fork and sync
points, concurrency, placement, admission and transport, and lays every
buffer out in an arena at planning, so a run allocates nothing and the
footprint is the plan rather than a formula beside the code. A DAG is planned
once and run many times, and the same DAG runs unchanged on one GPU, a node
of GPUs or many nodes. The C++ pipeline is the model's one-device, one-order
instance.

The sections below lean on three of its terms: the *sharing* an edge's kind
declares, the *layout* a task declares and the *arena* the orchestrator lays
it out in, and *workers*, which are streams, GPUs or nodes.

## Testing

Testing has its own document, [`testing.md`](testing.md). In short: testing
absorbs contracts, a check the code carries with it. A check lives in the
cheapest tier that can express it, from compile time down to multi-GPU, and
covers one node, a sub-DAG or a whole run. A PR runs the device-free tiers in
full and the GPU tier for the crates its diff affects, which is why the
workspace graph is designed as the selection graph. Redundancy is measured by
mutation testing, performance regression reads the orchestrator's task
timings and each plan's arena, the planner is fuzzed as a pure function, and
the sanitizers run on a schedule from the start.

## Diagnostics

Diagnostics has its own document, [`diagnostics.md`](diagnostics.md). In
short: timing is RAII scopes on two clocks, host and stream, with profiler
ranges behind a cargo feature forwarded to `wwr-sys/tx`; the orchestrator
times every task and every sync point itself, resolves the stream scopes at
the sync points it owns, and writes a timing record beside the run's
products. Logging keeps the C++ levels and line, and a line from a task names
its task and worker from a record the orchestrator sets. Errors are values
with the C++ kinds and exit codes; a GPU error keeps the vendor's code, a
sticky fault stops launches on its device, context accrues at task
boundaries, a failed run returns the first error in topological order, and
`main` is the one place an error is logged. Release aborts on a panic through
`critical!`, and test unwinds for the contract tests.

## GPU layer

The GPU layer has its own document, [`gpu_layer.md`](gpu_layer.md). In
short: the kernels stay in C++ and CUDA behind the bridge layer the tree
already has, 42 bridge headers and 114 launchers, none of which allocates,
synchronizes or sets the device; the port changes the ABI the bridge is
spoken in, not what it says. The bridge header is the port line, one module
at a time, with the C++ pipeline green throughout, since C++ calls the same
`extern "C"` header. A launch wrapper stays in its `.cu`, becomes `extern "C"`, takes
the C ABI's stream, and returns one `int32_t` status in three ranges:
success, the C ABI's `wwr_status` from its own `wwr_get_last_error()`, and
the wrapper's refusals, kept apart from launch errors. The launch shape and
the angular-momentum dispatch table stay with the kernel. Scratch is sized
by a `*_workspace_bytes(dims)` query, pure in its dimensions, that the
task's layout calls at planning. Shared types are plain-C headers, the view
headers the bridges already have, mirrored by bindgen and layout-checked on
both sides; the constants split into a macro header both sides read. The
templated `recipes` bridges are monomorphised one entry per instantiation,
as WarpWraps' C ABI does. The libraries are one shared object per module,
bindings are committed and diffed by a test, and FelizJueves is asked to
emit the mechanical fixed-block launchers.

## Documentation

### Code: rustdoc on GitHub Pages

- **The API reference is rustdoc**, built with `cargo doc --no-deps` and
  published to GitHub Pages by a workflow on merge to main.
- **rustdoc warnings are errors** (`-D warnings`), as the crate already
  requires, so a broken intra-doc link fails the build.
- **Doc examples are tests.** `cargo test` compiles and runs them, so an example
  cannot drift from the API it shows.
- **A module's doc comment carries its spec.** That means what it does, its
  contracts, and, while the port lasts, its C++-to-Rust table, as the crate's
  modules already do.

### Equations

rustdoc does not render math on its own. KaTeX, loaded through
`--html-in-header` (`RUSTDOCFLAGS`), renders `$...$` and `$$...$$` in doc
comments. It is one HTML file and adds no build dependency.

An equation in a doc comment cites its source by equation number, the way the
NEVPT2 code cites Park 2019 today. A reader can then check the code against the
paper term by term.

### References

References are tracked in one bibliography file (BibTeX) in the repo, with one
entry per source and a stable key. Docs cite by key, and the key links to a
generated references page. There are five kinds of entry:

| Kind | What the entry records |
|---|---|
| Article | DOI. The equations and sections the code uses. |
| Textbook | Edition, and the chapters and sections the code uses. |
| Other codebase | URL, the version or commit compared against, and its license. For example, PySCF as the reference for regression values. |
| Data | Source and version, such as the BSE basis sets |
| License obligation | Code or data derived from someone else's, and what it requires: a notice, attribution, or a compatible license |

Two rules come with the bibliography:

- **Consulted and derived are different.** Reading a paper or a codebase owes
  nothing. Porting code from one carries its license. The entry says which it
  was, so an obligation such as a BSD notice is recorded when the code arrives,
  not discovered later.
- **Metadata only.** The bibliography is public, so it holds no PDFs (they are
  copyrighted) and no references to private material.

### Diagrams and animations

- **Diagrams are source, not pictures.** They are checked in as text, either
  Mermaid or hand-written SVG, and rendered when the docs build.
- **DAG diagrams are generated from the DAGs themselves.** The task DAG type
  can emit Mermaid or DOT, so a feature's diagram is produced from the same
  structure the orchestrator runs, and cannot drift from it.
- **Animations are generated from real runs**, never drawn by hand. Examples:
  a task timeline built from a run's timing record
  ([`diagnostics.md`](diagnostics.md)), SCF convergence,
  an IRC path, and orbitals along a reaction coordinate. Scripts make them
  while the site builds, so binary media stays out of the repo.

### Open questions

- **Narrative docs.** rustdoc is for the API. Design and theory pages, this
  document included, could go in an mdBook, published beside the rustdoc on the
  same Pages site with the same KaTeX. Or they could stay as Markdown in the
  repo.
- **Publishing while experimental.** Should the Pages site go up now, or once
  the rebuild leaves `experimental/`?
- **Where the bibliography lives.** It could be one file for the whole tree,
  shared with the C++ docs, or one for the rebuild alone.
