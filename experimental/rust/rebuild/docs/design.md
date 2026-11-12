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
`extern "C"` header; the C++ modules stay beside their Rust crates and go
together at the cut-over, when the Rust pipeline matches the C++ one. A
launch wrapper stays in its `.cu`, becomes `extern "C"`, takes
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

Documentation has its own document, [`documentation.md`](documentation.md).
In short: the API reference is rustdoc, with `missing_docs` and broken links
denied in the manifest, every doc example a test, `compile_fail` ones being
the tier-1 suite, and no README beside a module; the C header's comments are
the launch wrappers' documentation, carried across by bindgen; the
configuration reference is emitted from the config type and the examples are
run; equations are KaTeX, cited by number from a published
paper and by label from the tree's own documents; references are one
tree-wide BibTeX file, with derived work's license obligations as entries and
keys checked against citations; the narrative pages are Markdown rendered by
mdBook beside the rustdoc; diagrams are emitted from the plan and committed
as tier-3 goldens; animations are made by the nightly from real runs and
published from the metrics branch; and the Pages site goes up now, built on a
runner once `build.rs` learns it is only documenting.

## Order of work

The documents above describe the finished shape. The crate today is `core`
alone, in one package, and nothing runs cargo in the gates. The first steps,
in order, each one PR:

1. **The section-citation check**, as the first tier-3 test
   ([`documentation.md`](documentation.md), "Narrative: mdBook beside the
   rustdoc"). It is a few lines, needs no card, and these documents have
   drifted once already.
2. **The workspace.** One crate per module and `core` split
   ([`testing.md`](testing.md), "Selection: the workspace graph is the
   selection graph"), done while the crate is four thousand lines rather than
   forty.
3. **The cargo half of the gates**, tiers 1 to 3 for a diff under the crate
   ([`testing.md`](testing.md), "What runs when").
4. **One task through the whole model**, before any module is ported: the
   overlap, which `experimental/rust/s_overlap` already launches, or the
   Coulomb build. Declared with a layout, laid out in an arena, planned once
   and run on one stream, with its arena size committed as a golden and its
   stream scope timed. Everything in these documents leans on the
   orchestrator, and this is where the model meets its hard cases, an
   adaptive width and a data-dependent size, on one task rather than twenty.
5. **The per-module shared objects**, a C++-tree change
   ([`gpu_layer.md`](gpu_layer.md), "Build").

Then the port, bottom-up from `core`, one module per crate, with the C++
tree green throughout ([`gpu_layer.md`](gpu_layer.md), "What the port
changes").
