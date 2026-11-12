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

## C++/GPU layer

### Layering

```
 Rust      orchestrator: DAG, fork, join, wait, sync; streams, events, memory
   │
 Rust      safe wrapper: owns the arguments' types and lifetimes, checks contracts
   │
 Rust      -sys: FFI bindings, generated from the C header
   │  C ABI
 C++ .cpp  launch wrapper: extern "C", picks the launch shape, launches, returns a status
   │
 C++ .cu   kernels: hand-written and FelizJueves-generated
```

**C++ wraps kernels and nothing else.** A launch wrapper takes a stream, device
pointers, sizes and parameters. It picks the grid and block, launches, and
returns a status. It never:

- allocates,
- syncs, waits or records events,
- sets the current device,
- holds state between calls.

Fork, join, wait and sync belong to Rust, and so does every allocation. A
wrapper therefore only enqueues work. It is safe to call from any thread, and
its cost is all on the stream it was given.

**The launch shape stays with the kernel.** The wrapper picks the grid and
block from the problem's dimensions, because block sizes like `kFourWarps`
belong next to the kernel that assumes them. Rust passes the dimensions, never
a block size.

**Scratch is sized by the wrapper and allocated by Rust.** A kernel that needs
scratch space exposes a query, `*_workspace_bytes(dims)`. The task's layout
calls it at planning, the arena carves that amount, and Rust passes the
buffer in. The query is the kernel's share of the task's layout, so the
programming model's sizes are numbers the kernels own rather than guesses.

### Error codes

A launch wrapper returns one status type, a C enum in the module's header:

- **Success.**
- **The backend's launch error,** from the vendor's get-last-error call, carried
  as `wwr_status` with the vendor's code. The Rust side then reaches the same
  `gpu_status` path as every other wwr call, and the vendor's names survive the
  crossing.
- **The wrapper's own refusals**, such as an angular momentum the generated
  kernels do not cover. These map to `ErrorKind::Unsupported` or
  `ErrorKind::InvalidConfig`.

On the Rust side, the status arrives as an integer and is converted with
`TryFrom`. It is never made into a Rust `enum` directly. A value C sends that
Rust does not know becomes an error, not undefined behaviour. A tier-3 test
checks that the C and Rust lists match, so neither can add a code the other
lacks.

### Shared types: the compatibility layer

Some types have to cross the boundary as data: a shell's exponents and
coefficients, a pair list, or the constants device code shares (today's
`constants.h`). These live in a C header, which is their single source:

- **The header is plain C.** No templates, no vendor types, no layouts that
  depend on the backend. That last rule comes from the crate's experience:
  `wwrDeviceProp` differs between CUDA and ROCm, and so it could not be
  mirrored. `MemPoolProps` happens to match on both, and that had to be checked
  by hand.
- **Rust's `#[repr(C)]` mirror is generated** from the header by bindgen, not
  written by hand.
- **Layout is checked on both sides.** Rust gets `size_of`, `align_of` and
  `offset_of` assertions in `const` (tier 1). C++ gets `static_assert`s.
- **Only some modules need this.** A module whose wrappers take only pointers
  and integers has no compatibility layer. The layer exists where a type must
  cross, and nowhere else.

A `.cuh` that is C++ only (templates, device functions) stays on the C++ side
and is never mirrored.

### Build

- **The backend is chosen once.** The C++ layer is built by CMake for exactly
  one backend, as today. `build.rs` links the result, and fails if the backend
  is not stated, rather than defaulting to one.
- **`devtools/cross-backend-check.sh` still applies** to every `.cu`, because
  nvcc accepts code that `clang -x hip` rejects.
- **Library calls (BLAS, solvers) go straight from Rust through `wwr-sys`.**
  Only our own kernels need a C++ wrapper.

### Open questions

- **Granularity of the C++ libraries.** One library per module, mirroring
  `src/`, or one for everything? One per module keeps rebuilds and links small.
  One for everything keeps `build.rs` simple.
- **When bindings are generated.** bindgen at build time needs libclang
  wherever cargo runs. Committed bindings, with a check that regenerates and
  diffs them, avoid that and make drift visible in review.
- **Can FelizJueves emit the wrappers?** Its kernels are generated, so their
  launch wrappers and their `*_workspace_bytes` queries could be generated
  beside them, instead of written once per kernel family.

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
