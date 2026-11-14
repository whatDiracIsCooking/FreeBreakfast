# Rust rebuild: rules

The other documents state rules: what must hold, what never happens, what
every crate or task does. This one is the register of them, one row per
rule, each with where it is stated and what checks it. The reason it exists
is a measurement: every rule in these documents that had a mechanical check
held, and every rule that was prose alone drifted. A row whose check is
**unenforced** is allowed, and is expected for a rule about review or about
code not yet written; what is not allowed is a rule whose lack of a check
nobody can see.

A new rule joins the register in the PR that states it, with its check or
an **unenforced** entry. A check that lands later edits the row.

## Reading the register

Each table below has three columns:

- **Rule**, briefly. The document cited is the full statement.
- **Stated**, as a section citation, the form
  `doc_check/tests/section_citations.rs` checks, so a renamed heading
  breaks the row as it breaks any other citation.
- **Checked by**, a list joined by `;`, each item one of:
  - ``lint `name` ``, a lint configured in `[workspace.lints]` of
    `Cargo.toml` or in `clippy.toml`;
  - ``type `Name` ``, a struct, enum, trait or type alias the workspace
    defines, whose shape is the check (misuse does not build);
  - a path in backticks, a file under `rust/` or else the repository root,
    optionally `::name` for one test or `const` block in it;
  - **unenforced**, alone.

  An item may end in a parenthesised note saying what the check does not
  cover.

`doc_check/tests/rules.rs` reads these tables. It fails when a named file
does not exist, a named test is not in its file, a named lint is not
configured, a named type is not defined, a cell is in none of the forms
above, or a document beside this one is cited by no row. It checks that a
check exists, not that it checks the rule; that is the review's, row by row.

The lints the tree configures are in `[workspace.lints]` of `Cargo.toml`:
`missing_docs`, `disallowed_methods` (its list in `clippy.toml`: spawning,
the host syncs a task may not make, and the GPU rules' kernel launch,
managed memory and free-memory read),
`undocumented_unsafe_blocks`, `multiple_unsafe_ops_per_block` and
`missing_panics_doc`. They gate every member: `devtools/test.py cargo-lint`
the link-free ones at commit, and tier 4 of `test.py push` and `pr` the
ones that link ([`testing.md`](testing.md), "Selecting GPU tiers in cargo").
`unsafe_code` is forbidden per crate root, which
`doc_check/tests/unsafe_code.rs` holds, not a manifest line.

## Enforced at birth

A row marked **unenforced** with a note that begins `not built:` waits for
a feature, not for a check nobody can write. The PR that builds the feature
ships that row's check and edits the row to name it, in the same PR, as a
new rule's PR ships its own. The feature is where the check is cheapest:
its author knows what to assert, and the code it holds has no callers yet
to fix. A feature built without its check turns a row that waited honestly
into a rule that is prose alone, which is what this register exists to
prevent.

So a row waiting for a feature says so in that form, naming what is
missing, and every other unenforced row is a rule nothing can check yet,
or one only review can.

## Design

| Rule | Stated | Checked by |
|---|---|---|
| A question the scoping issue cannot answer is settled before the PR | [`design.md`](design.md), "Where a design lives" | **unenforced** |
| No design document or README beside a module; the doc comment is the spec | [`design.md`](design.md), "Where a design lives", [`documentation.md`](documentation.md), "Code: rustdoc" | `doc_check/tests/source_rules.rs::no_markdown_sits_beside_a_module` (every `.md` under `rust/` but the workspace `README.md` and `CLAUDE.md`, `docs/`, a crate `README.md` its `src/lib.rs` includes, and a golden's provenance `tests/data/README.md`; not a design written into a doc comment beside a module, nor one outside `rust/`) |
| A reference computes the same approximation; its bar is fixed by analysis, compared in one run, never loosened, and atomics never raise it | [`design.md`](design.md), "What it computes", [`testing.md`](testing.md), "Correctness", "Determinism and placements", [`programming_model.md`](programming_model.md), "Reproducibility across placements" | **unenforced** |
| A unit test of a node takes its inputs from a fixture, never from the sub-DAG upstream | [`design.md`](design.md), "What it computes", [`testing.md`](testing.md), "Levels" | **unenforced** |
| Every example config runs through the binary under `cargo test` | [`testing.md`](testing.md), "Levels", [`documentation.md`](documentation.md), "Configuration" | `pipeline/config/tests/examples.rs` (each example named by hand; a new one is not picked up) |
| An adaptive width is chosen at planning, never from free memory at run time | [`design.md`](design.md), "What it holds", [`memory.md`](memory.md), "Adaptive widths" | lint `disallowed_methods` (`wwr_mem_get_info`; every read is an `allow` naming its exception, which review holds to admission, a refusal's message or a test) |
| Host work that would fan out is CPU nodes, never a pool inside a task | [`design.md`](design.md), "What it holds", [`programming_model.md`](programming_model.md), "CPU workers" | lint `disallowed_methods`; `core/thread_pool/tests/thread_pool_threads.rs`; `pipeline/guess/tests/guess_threads.rs` |
| A task reaches only the rows its layout named, by checks always on | [`design.md`](design.md), "What it holds", [`programming_model.md`](programming_model.md), "Memory: layouts and arenas", [`memory.md`](memory.md), "The Rust shape" | `core/arena/src/tests.rs::views_give_a_node_its_rows_and_no_others`; type `Bufs` |
| Nobody writes a footprint; a plan's arena size is a committed tier-3 golden | [`design.md`](design.md), "What it holds", [`testing.md`](testing.md), "Performance", [`memory.md`](memory.md), "What is checked" | `core/arena/src/tests.rs::scf_fixture_golden`; `core/planner/src/tests.rs::scf_plan_golden` (one fixture each, not one per example config) |
| An ordered reduce is bitwise at every worker count, an unordered one meets its declared tolerance, one worker against many | [`design.md`](design.md), "Whether it splits", [`testing.md`](testing.md), "Determinism and placements" | **unenforced** (not built: no task splits yet) |
| A computation that must wait for part of itself is two tasks; a task has no internal host sync | [`programming_model.md`](programming_model.md), "A task declares", "Granularity" | lint `disallowed_methods` (the stream, event and device syncs and the blocking copy, in `clippy.toml`; each allowed site says why: the executor's edges, a destroy, instantiation, a test reading back. A wait spelled otherwise, a polling loop on `wwr_stream_query`, or a second sync inside an allowed function is unseen) |
| One crate per module, no umbrella crate, and a crate that only logs, or re-exports the logging macros, does not depend on the handle types | [`design.md`](design.md), "What it depends on, and what depends on it", [`testing.md`](testing.md), "Selection" | `doc_check/tests/workspace_graph.rs` (the reverse-dependents bar, two thirds of the other members, outside the two global crates it names; `pub use` umbrellas; `fb-core-log`, `fb-core-error-handling` and any crate that `pub use`s `log`'s macros or `fb_core_log::critical` reach neither `fb-core-wwr` nor `wwr-sys`; one crate *per module* is review's) |
| Every direct dependency is named in its crate's sources, a normal one in `src/`; an edge no source shows is listed with its reason | [`design.md`](design.md), "What it depends on, and what depends on it", [`testing.md`](testing.md), "Selection" | `doc_check/tests/workspace_graph.rs::every_direct_dependency_is_named_in_its_crates_sources` (by name in code or an intra-doc link, with `UNNAMED` the exceptions; not that the use is more than nominal) |
| A new kernel is a FelizJueves emit or a `.cu`, and Rust never launches a kernel itself | [`design.md`](design.md), "What already exists", [`gpu_layer.md`](gpu_layer.md), "Layering" | lint `disallowed_methods` (`wwr_launch_kernel_ex_c`, the launch half; where a new kernel comes from is review's) |
| A port forks the device code it uses and no more; a fix to one copy is owed to the other | [`design.md`](design.md), "What already exists", [`gpu_layer.md`](gpu_layer.md), "What the port changes" | **unenforced** |
| Every fallible path returns `Result`, and nothing between the detection site and `main` logs the error | [`design.md`](design.md), "What it says about itself", [`diagnostics.md`](diagnostics.md), "Values, never panics", "The one boundary" | `doc_check/tests/code_rules.rs::nothing_but_the_boundary_logs_an_error` (the second half: every line at error or critical is at a listed site, counted per file; a renamed import of a logging macro is unseen. That every fallible path returns `Result` is review's) |
| A write is a CPU node after a device-to-pinned copy; a dump switched on is a different plan, not a branch in a task | [`design.md`](design.md), "What it writes", [`programming_model.md`](programming_model.md), "CPU workers" | `pipeline/config/tests/plan.rs::plan_golden` (the one write path, `add_write_npy`'s stream download then CPU write, in the overlap example's plan; not that every write goes through it, nor the dump half, since no dump exists) |
| Doc examples compile, and run unless `no_run` | [`design.md`](design.md), "What it documents", [`documentation.md`](documentation.md), "Code: rustdoc" | `devtools/test.py` (tier 3's `cargo test` runs the doctests) |
| A doc example is never `ignore`; a device example is `no_run`; a `compile_fail` carries its error code | [`documentation.md`](documentation.md), "Code: rustdoc" | `doc_check/tests/source_rules.rs::every_doc_example_carries_its_marker` (the `///` and `//!` blocks rustdoc reads as Rust; a device example only by name, a runnable block that calls an associated function of a device type, `Stream::new(…)`, `Block::new(…)`, or a `…_sys::` function: not one that reaches a card through a function whose name does not say so, nor a `#[doc]` attribute or an included Markdown file) |
| `missing_docs` is denied in the manifest | [`design.md`](design.md), "What it documents", [`documentation.md`](documentation.md), "Code: rustdoc" | lint `missing_docs` |
| Broken and private intra-doc links fail the docs build | [`documentation.md`](documentation.md), "Code: rustdoc" | `devtools/test.py` (the gate's `RUSTDOCFLAGS=-D warnings`, not the manifest the document asks for) |
| A config key is a documented field with its default and unit, the reference is generated from the type, and an example sets the key | [`design.md`](design.md), "What it documents", [`documentation.md`](documentation.md), "Configuration" | lint `missing_docs` (each key is a documented public field of `Config`, or of `ScfStep` for `[scf]`, whose rustdoc is the reference; not that the comment gives the default and unit); `pipeline/config/tests/plan.rs::every_key_is_set_by_an_example` (the keys read from the type's own refusal of an unknown key, each set by an example, or, for a key that excludes one an example sets, `[guess] from`, shown as `key =` in an example's comment; the reference generated from the type is rustdoc's) |
| A plan's diagram is a tier-3 golden | [`design.md`](design.md), "What it documents", [`documentation.md`](documentation.md), "Diagrams and animations" | **unenforced** (not built: no diagram emitter yet) |
| The doc comment carries a C++-to-Rust table while the port lasts | [`design.md`](design.md), "Porting a module" | **unenforced** |
| Nothing crosses from Rust back into C++ | [`design.md`](design.md), "Porting a module", [`gpu_layer.md`](gpu_layer.md), "What the port changes" | `doc_check/tests/source_rules.rs::nothing_is_exported_to_c` (`no_mangle` and `export_name` in any attribute, a `pub` `extern` fn, a `cdylib` or `staticlib` crate type; a private `extern "C" fn` is allowed as a callback a C API takes, `fb-core-log`'s `atexit` hook and `panic_hook.rs`'s case today, though a pointer to one could still be handed to C++) |

## Contracts and capabilities

| Rule | Stated | Checked by |
|---|---|---|
| Contracts are debug-only, and a test that exists only to trip one is `#[cfg(debug_assertions)]` | [`testing.md`](testing.md), "Contracts are part of the spec", [`diagnostics.md`](diagnostics.md), "Panics and profiles" | `doc_check/tests/source_rules.rs::a_test_that_trips_a_contract_is_debug_only` (a `#[should_panic(expected = …)]` whose text is in, or holds, a `debug_*` contract's message, under a `cfg` naming `debug_assertions` on the test, a module, `impl` or file around it; not a bare `#[should_panic]`, nor an expected text reworded from the message; that contracts compile out is the `contracts` crate's) |
| A check soundness rests on is an always-on `assert!`, named in the item's `# Panics` | [`design.md`](design.md), "The spec is the contracts", [`testing.md`](testing.md), "Contracts are part of the spec" | lint `missing_panics_doc` (a public item that can panic says so; not that a soundness check is an `assert!` rather than a contract) |
| The outcome of a GPU call that can fail is a status or a `Result`, never a contract | [`design.md`](design.md), "The spec is the contracts", [`testing.md`](testing.md), "Contracts are part of the spec" | **unenforced** |
| A contract complements a dedicated test and never replaces one | [`design.md`](design.md), "The spec is the contracts", [`testing.md`](testing.md), "Contracts are part of the spec" | **unenforced** |
| A doc comment never restates a contract: rustdoc renders it | [`design.md`](design.md), "The spec is the contracts", [`documentation.md`](documentation.md), "Code: rustdoc" | `doc_check/tests/contracts.rs` |
| A `debug_assert!` is an internal invariant, marked `// INVARIANT:`; a caller's precondition is a `#[debug_requires]` | [`testing.md`](testing.md), "Contracts are part of the spec" | `doc_check/tests/contracts.rs` |
| `unsafe_code` is forbidden at every crate root but the listed crates', and the list only shrinks | [`programming_model.md`](programming_model.md), "Capabilities are enforced by lints", [`design.md`](design.md), "The spec is the contracts" | `doc_check/tests/unsafe_code.rs` |
| Spawning is the worker module's alone | [`programming_model.md`](programming_model.md), "Capabilities are enforced by lints" | lint `disallowed_methods` |
| Every `unsafe` block carries a `// SAFETY:` comment | [`programming_model.md`](programming_model.md), "Capabilities are enforced by lints" | lint `undocumented_unsafe_blocks` |
| An `unsafe` block holds one operation | [`programming_model.md`](programming_model.md), "Capabilities are enforced by lints" | lint `multiple_unsafe_ops_per_block` |
| A crate above `core` reaches a library through a safe wrapper, not `wwr-sys` | [`programming_model.md`](programming_model.md), "Capabilities are enforced by lints" | `doc_check/tests/unsafe_code.rs` (an FFI call needs `unsafe`, so an unlisted crate cannot make one; a listed one still can) |
| A checked library call (`gemm`, `dgmm`, `syevd`) holds each extent to its `DeviceSlice` in every build | [`programming_model.md`](programming_model.md), "Capabilities are enforced by lints", [`gpu_layer.md`](gpu_layer.md), "The safe wrapper" | `core/wwr/src/blas.rs::an_extent_one_past_the_slice_panics`; `core/wwr/src/solver/syevd.rs`; `recipes/blas/tests/gemm.rs`; `recipes/blas/tests/dgmm.rs`; `recipes/blas/tests/gemmwich.rs::a_short_c_panics_before_anything_is_enqueued`; `recipes/blas/tests/gmmwich.rs::a_short_c_panics_before_anything_is_enqueued`; type `DeviceSlice` |
| What a launch trusts in device data reaches a safe call only as a type the path that wrote it alone makes (`ClassRows`, from `add_record_tasks`, over tables checked on the host) | [`gpu_layer.md`](gpu_layer.md), "The safe wrapper" | type `ClassRows` (private fields); `integrals/sys/src/records.rs::a_block_past_the_matrix_panics`; `integrals/one_electron/overlap/tests/launcher.rs::a_matrix_short_of_n_cart_squared_panics` |
| No thread-pool crate is a dependency | [`programming_model.md`](programming_model.md), "Capabilities are enforced by lints", "CPU workers" | `doc_check/tests/workspace_graph.rs::no_member_depends_on_a_thread_pool_crate` (the pool crates it names, through any edge; one not on its list, or a pool a linked C library opens, is `core/thread_pool/tests/thread_pool_threads.rs`'s to notice) |

## Orchestration

| Rule | Stated | Checked by |
|---|---|---|
| A task allocates nothing of its own: `run` holds no handle that allocates | [`programming_model.md`](programming_model.md), "A task declares", "Memory: layouts and arenas" | type `Bufs`; `doc_check/tests/unsafe_code.rs` (the raw allocation bindings are `unsafe`) |
| A failed GPU allocation aborts; retry is re-admission | [`programming_model.md`](programming_model.md), "A task declares", [`diagnostics.md`](diagnostics.md), "Allocation is a planning-time error" | `core/arena/tests/arena_block.rs::an_allocation_the_card_cannot_hold_aborts`; `core/diagnostics/tests/log.rs::gpu_must_aborts_at_the_caller` |
| A failed node fails its successors and nothing else, and the run returns the first error in topological order | [`programming_model.md`](programming_model.md), "The orchestrator derives", [`diagnostics.md`](diagnostics.md), "One worker fails" | `core/thread_pool/src/tests.rs::a_failure_skips_its_descendants_and_reports_the_lowest_rank`; `core/planner/src/tests.rs::a_failure_skips_its_readers` |
| Every node runs exactly once, after its predecessors, within the plan's worker counts | [`programming_model.md`](programming_model.md), "CPU workers" | `core/thread_pool/src/tests.rs::every_node_runs_once_after_its_predecessors_within_the_worker_counts` |
| A run uses exactly the threads its plan says, and none survive it | [`programming_model.md`](programming_model.md), "CPU workers", [`design.md`](design.md), "What it holds" | `core/thread_pool/tests/thread_pool_threads.rs`; `pipeline/guess/tests/guess_threads.rs` |
| `n_cpu` defaults to the available parallelism less the stream threads, floored at one | [`programming_model.md`](programming_model.md), "CPU workers" | `core/thread_pool/src/tests.rs::default_n_cpu_leaves_the_stream_threads_and_keeps_one` |
| A plan needing more workers than the budget is refused before anything runs | [`programming_model.md`](programming_model.md), "CPU workers" | `core/thread_pool/src/tests.rs::a_plan_the_workers_cannot_run_is_refused_before_anything_runs` |
| A panicking worker fails the run rather than hanging it, and the join re-raises the panic | [`programming_model.md`](programming_model.md), "CPU workers" | `core/thread_pool/src/tests.rs::a_panicking_node_fails_the_run_instead_of_hanging_it` |
| One run at a time per instance and per backend; within it the workers share both by reference | [`programming_model.md`](programming_model.md), "Plan once, run many" | type `Instance` (its `run` takes the instance and the backend by `&mut`, as `thread_pool::run` takes the plan and the backend) |
| A backend is on several threads only as one run's workers: no backend is shared across threads outside a run | [`programming_model.md`](programming_model.md), "Plan once, run many" | type `Backend` (sealed, and not `Sync`); type `Device` (not `Sync`; `core/thread_pool/src/backend.rs` has a `compile_fail,E0277` doctest of it); type `RunBackend` (`thread_pool::run`'s private wrapper, the one `unsafe impl Sync`) |
| Sub-DAGs borrow their parent's workers; only top-level runs own threads, and a loop's replay spawns none | [`programming_model.md`](programming_model.md), "CPU workers", "Plan once, run many" | `core/thread_pool/tests/thread_pool_threads.rs` (a loop's replay, which a planner loop is) |
| Sub-DAGs borrow their parent's arena; only top-level plans own blocks | [`programming_model.md`](programming_model.md), "Blocks, plans and the process", [`memory.md`](memory.md), "Nested plans" | `core/arena/src/tests.rs::scf_loop_golden` (the block is the parent's rows live across the loop plus the body's peak, and admission names the region); `core/arena/src/tests.rs::loop_bodies_lie_in_their_regions`; `core/planner/tests/planner_device.rs::a_device_loop_to_its_fixed_point` (free memory unchanged from the first iteration to the last; a loop's body only, the region as the max over several inner shapes, the Hessian's, is not built) |
| A row a loop carries is produced before the loop, read in the body and written by its last user there; every other user is ordered against the writer, and it is live across the whole loop | [`memory.md`](memory.md), "Nested plans" | `core/arena/src/tests.rs::a_carried_row_is_refused_where_no_loop_carries_it` (contracts of `TableBuilder::build`); `core/arena/src/tests.rs::scf_loop_golden`; `core/planner/src/tests.rs::a_loop_carries_its_rows_to_the_fixed_point`; `core/planner/src/tests.rs::every_loop_carries_what_the_model_does` (random bodies, carried rows and iteration counts through the executor against a serial model; the parent's rows' extension across the loop dropped fails it, though the carried row's extension past its writer alone is held only by the goldens: no other row can take its bytes there) |
| A loop's decision is a CPU node holding host rows alone | [`programming_model.md`](programming_model.md), "Plan once, run many" | type `Decision` (`core/planner/src/task.rs` has a `compile_fail,E0053` doctest of a device layout) |
| A loop replays its body on the run's workers until its decision, after every body node and the nodes it is given, stops it, at most its planned bound, past which the run fails with `MaxIterationsExceeded`; a failed body node ends the loop and skips what follows it | [`programming_model.md`](programming_model.md), "Plan once, run many" | `core/thread_pool/src/tests.rs::a_loop_replays_its_body_to_the_fixed_point`; `core/thread_pool/src/tests.rs::a_loop_that_reaches_its_bound_fails_the_run`; `core/thread_pool/src/tests.rs::a_failed_body_node_ends_the_loop_and_the_run`; `core/thread_pool/src/tests.rs::a_decision_runs_after_its_own_predecessors` |
| What follows a loop depends on its decision, never on a body node | [`programming_model.md`](programming_model.md), "Plan once, run many" | `core/thread_pool/src/tests.rs::a_node_after_a_loop_cannot_read_its_body` (a debug contract of `Plan::add`); `core/arena/src/tests.rs::loop_bodies_lie_in_their_regions` (no read or handoff a table returns leaves a body but from its decision) |
| Every edge's sync precedes its first read | [`design.md`](design.md), "What it depends on, and what depends on it", [`testing.md`](testing.md), "Fuzzing" | `core/planner/src/tests.rs::every_dag_reads_what_was_written` |
| The other planner invariants: deterministic, the same footprint for one worker and many, placement within the declared device and budget, sub-DAG offsets inside their region | [`testing.md`](testing.md), "Fuzzing" | `core/arena/src/tests.rs::loop_bodies_lie_in_their_regions` (a loop body's offsets inside its region only; no property test holds the other three) |
| A library handle is the stream worker's, made before the run, never in one | [`gpu_layer.md`](gpu_layer.md), "Layering" | `core/thread_pool/src/tests.rs::device_library_handles_are_the_workers_made_before_the_run` |
| A task that changes a shared handle's state sets it back before it returns | [`gpu_layer.md`](gpu_layer.md), "Layering" | **unenforced** |
| Each handle's library workspace is a worker row of the arena, which the handle owns while bound and no task can name | [`gpu_layer.md`](gpu_layer.md), "Layering", [`memory.md`](memory.md), "The table" | `core/planner/src/tests.rs::a_blas_workspace_is_a_worker_row_admission_counts`; `core/planner/src/device_tests.rs::the_backend_binds_each_instances_workspace_row`; `core/arena/tests/arena_block.rs::a_held_worker_row_is_alone_and_keeps_its_block`; type `HeldRow` (a plan that does not call `Dag::blas_workspace` still runs, on the library's own workspace) |

## Memory

| Rule | Stated | Checked by |
|---|---|---|
| An output's bytes are declared once, in the producer's layout; the edge reads them | [`memory.md`](memory.md), "The table" | type `RowId`; `core/arena/src/tests.rs::builder_refuses_misordered_reads` (contracts of `TableBuilder::read`: an output of its table, read after its producer, once per node) |
| Two rows share bytes only when one's end precedes the other's start, and every reuse is a handoff the plan orders | [`memory.md`](memory.md), "Lifetimes", [`programming_model.md`](programming_model.md), "Memory: layouts and arenas" | `core/arena/src/tests.rs::assign_invariants`; `core/arena/src/tests.rs::handoffs_order_every_reuse` |
| Every row is placed at `FB_DEVICE_ALIGNMENT` | [`memory.md`](memory.md), "The two kinds", [`gpu_layer.md`](gpu_layer.md), "Scratch" | `core/arena/src/tests.rs::assign_invariants`; `core/planner/tests/planner_device.rs` |
| Admission refuses a cap one byte under the peak, naming the rows live at it, and every row over the cap on its own, and admits one at the peak | [`memory.md`](memory.md), "Admission", "What is checked" | `core/arena/src/tests.rs::admit_is_exact_and_names_the_peak`; `core/arena/src/tests.rs::admit_names_every_row_over_the_cap_alone`; `pipeline/eri/tests/plan.rs::admission_counts_b_and_l` (B, not live at the DF step's peak, named under a cap one byte short of it) |
| An uncapped plan is admitted against the card's free memory less a measured reserve and the stack | [`memory.md`](memory.md), "Admission" | **unenforced** (not built: admission takes a cap only) |
| An adaptive `bytes(width)` is monotone, bisected once at planning; a width of 1 is a `warn` | [`memory.md`](memory.md), "Adaptive widths", [`diagnostics.md`](diagnostics.md), "Levels" | **unenforced** (not built: no adaptive task yet) |
| A block is one plain allocation per worker per kind, not the stream-ordered pool; pinned is portable, never mapped or write-combined, and managed memory is not used | [`memory.md`](memory.md), "The two kinds", [`programming_model.md`](programming_model.md), "The arena is a table and a block" | lint `disallowed_methods` (`wwr_malloc_managed`, `wwr_stream_attach_mem_async`: the managed half only; plain, not pooled, and the pinned flags are not checked) |
| Every block is filled with `0xFF`: the device's in debug builds, the pinned one in every build | [`memory.md`](memory.md), "The two kinds" | `core/arena/tests/arena_block.rs::blocks_are_filled` |
| A row's element type is one for which every bit pattern is a value | [`memory.md`](memory.md), "The two kinds" | type `Element` |
| The memory kind is in the type, so a device row cannot be handed to a host write | [`memory.md`](memory.md), "The Rust shape", [`programming_model.md`](programming_model.md), "Memory: layouts and arenas" | type `Block`; type `RowId` |
| The raw allocation calls sit on a sealed trait nothing outside the arena can name | [`memory.md`](memory.md), "The Rust shape" | type `Sealed` |
| A stream node does not touch a pinned row from the host | [`memory.md`](memory.md), "The Rust shape" | **unenforced** (a stream copy holds its pinned rows' borrows, so the enqueuing node cannot touch them after; nothing stops a stream node's `Bufs::get` of a pinned row, and no test trips the held borrow) |
| Declared equals allocated: free memory before and after instantiation differs by the block | [`memory.md`](memory.md), "What is checked" | `core/arena/tests/arena_block.rs::declared_equals_allocated` |
| No library allocates behind the arena: free memory never falls across a run (a fall, not any change: the reading, `wwr_mem_get_info`, is the device's, so another tenant's free raises it mid-run, #1553, and nothing wwr wraps counts one process alone. An allocation of this process can only lower it, so the rule loses nothing. What goes unseen is a library's free of memory it took before the first sample, and, as before, an allocation freed again between two samples) | [`memory.md`](memory.md), "What is checked", [`programming_model.md`](programming_model.md), "Memory kinds" | `core/planner/tests/planner_device.rs::a_round_trip_through_the_planner` (copies only); `core/planner/tests/planner_device.rs::a_device_loop_to_its_fixed_point` (a gemm on the plan's BLAS workspace, from a loop's first iteration to its last); `pipeline/scf/tests/scf.rs::water_converges_to_pyscf` (the DF-RHF loop's BLAS, eigensolver and DIIS solve, from its second iteration to its last: the first launches no DIIS solve, whose code loads onto the card once per process) |
| A kernel is handed exactly its workspace query's bytes under `memcheck` | [`memory.md`](memory.md), "What is checked", [`testing.md`](testing.md), "The C++ boundary", [`gpu_layer.md`](gpu_layer.md), "Scratch" | **unenforced** (not built: no gate runs `memcheck`) |
| Large host-side data is pinned, where it is counted, not pageable | [`programming_model.md`](programming_model.md), "Memory kinds" | **unenforced** |

## Testing

| Rule | Stated | Checked by |
|---|---|---|
| The arena, the thread pool and the planner build with no wwr library under `--no-default-features` | [`testing.md`](testing.md), "Tiers" | `test/pytest/test_test_py.py::test_link_free_crates_leave_out_a_linking_graph` |
| One cache per test process serves the computed fixtures | [`testing.md`](testing.md), "Levels" | **unenforced** (not built: no computed fixture is cached yet) |
| A diff under `rust/` fires the cargo half and nothing else; tier 4 runs only once tiers 1 to 3 pass | [`testing.md`](testing.md), "What runs when" | `test/pytest/test_test_py.py::test_dry_run_names_tier4_crates`; `test/pytest/test_test_py.py::test_a_failing_nightly_cargo_tier3_runs_no_tier4` |
| A commit touching `rust/` runs `cargo fmt --check` and clippy with warnings denied | [`testing.md`](testing.md), "What runs when" | `.pre-commit-config.yaml`; `devtools/test.py` |
| `Cargo.lock` is tracked, and every cargo call of the gates and `cargo-lint` passes `--locked`, so a manifest change without its lock update fails | [`testing.md`](testing.md), "What runs when" | `test/pytest/test_test_py.py::test_every_cargo_call_of_the_gates_is_locked`; `test/pytest/test_test_py.py::test_a_stale_lock_fails_the_cargo_half_and_says_so`; `rust/Cargo.lock` (an untracked lock fails a fresh checkout's `--locked`, not a worktree that still has one) |
| A gate's skips are visible: one list of every case that did not run, with its reason | [`testing.md`](testing.md), "What runs when", "Selecting GPU tiers in cargo" | `test/pytest/test_test_py.py::test_cargo_skips_lists_each_ignored_case_with_its_reason`; `test/pytest/test_test_py.py::test_tier4_without_a_device_lists_every_device_case_as_skipped` |
| The PR gates stay under five minutes; a case past about 1.5 s is nightly-only unless its family's representative | [`testing.md`](testing.md), "What runs when" | **unenforced** (no cargo case time is recorded) |
| Each crate's tier 4 is one process on one test thread, and the gate passes `--include-ignored` only when wwr sees a device | [`testing.md`](testing.md), "Running on a card: the process model", "Selecting GPU tiers in cargo" | `test/pytest/test_test_py.py::test_tier4_with_a_device_runs_them_and_skips_none`; `devtools/test.py` |
| A failure prints what reproduces it alone: the seed, the shapes, the task and the command | [`testing.md`](testing.md), "Running on a card: the process model", "Determinism and placements" | **unenforced** |
| A device case says `#[ignore = "tier 4: needs a GPU"]`, and `#[ignore]` means nothing else | [`testing.md`](testing.md), "Selecting GPU tiers in cargo", "Slow cases" | `doc_check/tests/source_rules.rs::every_ignore_is_a_device_case` (every `ignore` in an attribute, `cfg_attr` included, carries exactly that reason; not that the case needs a device, nor that a device case is ignored at all) |
| Tier 4 runs the changed crates and their reverse dependents, by the path rules | [`testing.md`](testing.md), "Selecting GPU tiers in cargo" | `test/pytest/test_test_py.py::test_tier4_selects_a_crate_and_its_reverse_dependents`; `doc_check/tests/workspace_graph.rs::the_path_rules_match_the_layout` (the rules against `rust/`'s layout: every `kernels/` covered and wrapped by a `-sys` crate that declares `links`, every source and target present) |
| A slow case returns at once under `FREE_BREAKFAST_FAST_TESTS`, which the gates set for both tiers and the nightly clears | [`testing.md`](testing.md), "Slow cases" | `test/pytest/test_test_py.py::test_nightly_cargo_runs_the_slow_cases_in_both_tiers` (the gate's switch, not each case's reading of it) |
| Selection is replayed against #988's bar before it is believed | [`testing.md`](testing.md), "Selection" | **unenforced** |
| A test's doc comment names the defect it alone catches; no `tests/` case repeats a doc example; no host and GPU twins | [`testing.md`](testing.md), "Redundancy" | **unenforced** |
| A reference value is committed beside its crate with the reference's version and the command that made it | [`testing.md`](testing.md), "Correctness", "Reference data" | **unenforced** |
| A comparison with PySCF loads the tree's basis JSON, goes through the overlap, and uses CODATA 2018 | [`testing.md`](testing.md), "Reference data" | **unenforced** |
| Seeds are fixed and printed | [`testing.md`](testing.md), "Determinism and placements" | **unenforced** |
| The weekly run: compute-sanitizer with `--error-exitcode=1`, Miri, the fuzzers, `cargo-mutants` and tier 5 | [`testing.md`](testing.md), "What runs when", "Sanitizers", "Fuzzing" | **unenforced** (not built: no weekly run exists) |

## The GPU layer

| Rule | Stated | Checked by |
|---|---|---|
| A launch wrapper only enqueues: no allocation, sync, event, copy, device change or profiler range | [`gpu_layer.md`](gpu_layer.md), "Layering" | `device_check/tests/launch_wrappers.rs` |
| A kernel library's ABI version and backend are checked at load, and a mismatch is refused naming both | [`gpu_layer.md`](gpu_layer.md), "What the port changes" | `kernel_abi/src/lib.rs::another_version_is_refused_naming_both`; `kernel_abi/src/lib.rs::another_backend_is_refused_naming_both`; `kernel_lib.rs` |
| The ABI version is bumped with any change to a declaration the header reaches | [`gpu_layer.md`](gpu_layer.md), "What the port changes" | **unenforced** |
| Emitted units are emitted again from the generator, byte for byte, never copied by hand | [`gpu_layer.md`](gpu_layer.md), "Build", "What the port changes" | `device_check/tests/kernel_sources.rs::every_emitted_unit_equals_its_cpp_twin` (that the two trees agree, not that the C++ copy matches a fresh run of the generator; a unit that lost its `generated by felizjueves` line is unseen) |
| The current device is the stream's; the safe wrapper checks it in debug builds | [`gpu_layer.md`](gpu_layer.md), "Layering", "The safe wrapper" | **unenforced** (no safe wrapper carries the contract yet) |
| Rust passes dimensions, never a block size, and never learns the warp width | [`gpu_layer.md`](gpu_layer.md), "Launch shape and dispatch" | `doc_check/tests/code_rules.rs::no_rust_source_names_a_warp_width_or_a_launch_shape`; `device_check/tests/kernel_sources.rs::no_header_the_bindings_read_names_a_launch_shape` (names, not values: a bare `32` passed as a count is unseen) |
| Each launcher asserts its dispatch table's extent against the momentum limit its header exports | [`gpu_layer.md`](gpu_layer.md), "Launch shape and dispatch" | `integrals/kernels/mmd/mmd_launch.cu`; `integrals/kernels/one_electron/overlap/overlap_launch.cu`; `integrals/kernels/one_electron/kinetic/kinetic_launch.cu`; `integrals/kernels/one_electron/potential_en/potential_en_launch.cu` |
| A launcher that takes scratch has a workspace query, pure in its dimensions | [`gpu_layer.md`](gpu_layer.md), "Scratch" | `recipes/sys/tests/merge_sort.rs::workspace_query_is_the_two_halves`; `integrals/sys/tests/overlap.rs::mmd_order_workspace_is_the_sort_with_values` (the two queries that exist, pinned on the host in a test that needs no device; not that every launcher taking scratch has one) |
| A launch status is an `int32_t` converted by value, never a Rust `enum`; an unknown value is an error | [`gpu_layer.md`](gpu_layer.md), "Statuses", [`testing.md`](testing.md), "The C++ boundary" | `kernel_abi/src/lib.rs::every_status_maps_to_its_kind` |
| The C and Rust status lists match member by member | [`gpu_layer.md`](gpu_layer.md), "Statuses", [`testing.md`](testing.md), "The C++ boundary" | `recipes/sys/src/lib.rs`; `integrals/sys/src/lib.rs` (`const` assertions of each value against the bindings) |
| An unsupported momentum reaches Rust as `ErrorKind::Unsupported` through the whole stack | [`testing.md`](testing.md), "The C++ boundary", [`gpu_layer.md`](gpu_layer.md), "Statuses" | `kernel_abi/src/lib.rs::every_status_maps_to_its_kind` (the mapping alone; no test sends one through a wrapper) |
| A shared type's header is plain C, its Rust mirror generated, its layout asserted on both sides | [`gpu_layer.md`](gpu_layer.md), "Shared types", [`testing.md`](testing.md), "The C++ boundary" | `device_check/tests/bindings.rs`; `integrals/sys/src/lib.rs`; `integrals/kernels/mmd/mmd.h` |
| Bindings are committed, and a tier-3 test regenerates them and fails on a difference | [`gpu_layer.md`](gpu_layer.md), "Build", [`testing.md`](testing.md), "The C++ boundary" | `device_check/tests/bindings.rs` |
| The header's `///` comment is the wrapper's contract, carried into the `-sys` rustdoc by the bindings script | [`documentation.md`](documentation.md), "The C header" | `device_check/tests/bindings.rs` |
| A safe wrapper checks non-overlap from offsets and each pointer's alignment | [`gpu_layer.md`](gpu_layer.md), "The safe wrapper", "Scratch" | `core/wwr/src/device_slice.rs::the_parts_tile_the_slice`; `core/wwr/src/device_slice.rs::a_cut_past_the_end_panics`; `recipes/sys/tests/merge_sort.rs::the_safe_call_refuses_a_misaligned_workspace` (non-overlap within one row: `split_at_mut` cuts it at an offset into parts that tile it, and refuses a cut past its end; slices from two borrows rest on `DeviceSliceMut`'s `unsafe` promise, not offsets; alignment: `merge_sort` checks its workspace's, and no other wrapper checks one) |
| The backend has no default, and the crates refuse to choose between two builds | [`gpu_layer.md`](gpu_layer.md), "Build" | `CMakeLists.txt`; `capi_dir.rs` |

## Diagnostics

| Rule | Stated | Checked by |
|---|---|---|
| `scope!` binds its guard to a name, and a range guard is `!Send` | [`diagnostics.md`](diagnostics.md), "Scopes" | **unenforced** (not built: no `scope!` yet) |
| The orchestrator times every task on its own, and no scope adds a sync | [`diagnostics.md`](diagnostics.md), "What the orchestrator times on its own" | `core/thread_pool/src/tests.rs::every_node_is_timed_by_its_worker` (the times; not the absence of a sync) |
| The timing record names its profile, and a failed run still returns its times | [`diagnostics.md`](diagnostics.md), "The timing record" | `core/thread_pool/src/tests.rs::a_record_reads_back_from_its_json`; `core/thread_pool/src/tests.rs::a_failed_run_still_returns_its_times` |
| A loop's body nodes have an entry per iteration with its index, the record carries each loop's iteration count, and each iteration's stream scopes are read by its decision before the replay | [`diagnostics.md`](diagnostics.md), "What the orchestrator times on its own", "The timing record" | `core/thread_pool/src/tests.rs::a_loop_replays_its_body_to_the_fixed_point`; `core/thread_pool/src/tests.rs::device_loop_reads_each_iterations_stream_scope` |
| Each level is used for what its row of the table says | [`diagnostics.md`](diagnostics.md), "Levels" | **unenforced** |
| A warning counts when it is written, and filtered ones do not | [`diagnostics.md`](diagnostics.md), "Levels" | `core/diagnostics/tests/log.rs::levels_format_counter_and_file_sink` |
| A task's log lines and errors name its task and worker, set by the orchestrator | [`diagnostics.md`](diagnostics.md), "The task context", "Context accrues at task boundaries" | `core/thread_pool/src/tests.rs::a_node_runs_inside_its_task_record`; `core/thread_pool/src/tests.rs::an_error_names_its_task_and_worker_and_carries_the_others` |
| A GPU error keeps the vendor's code name, number and backend, and out-of-memory is `Allocation` | [`diagnostics.md`](diagnostics.md), "GPU codes" | **unenforced** (not built: `gpu_status` maps every code to `GpuRuntime` under wwr's frozen name, which `core/wwr/src/status.rs`'s own test pins; no vendor name, number or backend is kept) |
| Exit codes are the C++ ones | [`diagnostics.md`](diagnostics.md), "GPU codes" | `core/error_handling/src/lib.rs::exit_codes_are_pinned` |
| `Allocation` is a value only at admission | [`diagnostics.md`](diagnostics.md), "Allocation is a planning-time error" | `doc_check/tests/code_rules.rs::allocation_is_a_value_only_at_admission` (the spellings `Error::allocation` and `ErrorKind::Allocation` outside test code; a kind passed through a variable or a glob import is unseen) |
| A sticky fault stops launches on its device and is reported once | [`diagnostics.md`](diagnostics.md), "Sticky faults" | **unenforced** (not built: no fault is told sticky) |
| A switch syncs after every task, so a fault names one task | [`diagnostics.md`](diagnostics.md), "Asynchronous errors surface at sync points" | **unenforced** (not built: no such switch) |
| A failed destroy is logged at `error` and counted, never raised | [`diagnostics.md`](diagnostics.md), "`Drop` cannot fail" | `core/diagnostics/tests/log.rs::levels_format_counter_and_file_sink` (`gpu_log_error_at`, which returns nothing to raise, writes an `error` line; not that every `Drop` calls it, and not the count: nothing counts errors, and `main`'s `completed: N errors` counts only the run's own) |
| `main` logs an error once and prints `completed: N errors, N warnings` | [`diagnostics.md`](diagnostics.md), "The one boundary" | `pipeline/config/tests/examples.rs` |
| A config refusal names the key, the file and the line | [`documentation.md`](documentation.md), "Configuration" | `pipeline/config/tests/examples.rs` (an unknown key names the file and line; a refused value names the key and the path, not the line) |
| The release profile aborts on a panic, and dev and test unwind | [`diagnostics.md`](diagnostics.md), "Panics and profiles" | `doc_check/tests/panic_profile.rs::the_release_profile_alone_aborts_on_a_panic` (the manifest's line; not a release binary run) |
| The panic hook logs at `critical`, flushed before the abort, and prints the backtrace | [`diagnostics.md`](diagnostics.md), "Panics and profiles" | `core/log/tests/panic_hook.rs::a_panic_is_a_critical_line_in_the_file_sink_before_the_abort` (an abort by a panic that cannot unwind, in a test build) |
| A panic's message is never lost: when the hook's critical line is dropped (level off), it goes to stderr instead, and when it is written there is no stderr copy | [`diagnostics.md`](diagnostics.md), "Panics and profiles" | `core/log/tests/panic_hook.rs::at_level_off_a_panic_is_still_written_to_stderr`; `core/log/tests/panic_hook.rs::a_panic_is_a_critical_line_in_the_file_sink_before_the_abort` (the no-copy half) |
| A panic in a test binary goes through the same hook, and `#[should_panic]` still passes | [`diagnostics.md`](diagnostics.md), "Panics and profiles" | `core/log/tests/panic_hook.rs::the_hook_leaves_should_panic_tests_passing` (a test binary has it once it calls into `fb-core-log`) |
| `check_or_abort` logs at `critical` and aborts at the caller | [`diagnostics.md`](diagnostics.md), "Levels" | `core/diagnostics/tests/log.rs::check_or_abort_logs_at_critical_and_aborts` |

## Documentation

| Rule | Stated | Checked by |
|---|---|---|
| A section cited by name is a heading of the file cited | [`documentation.md`](documentation.md), "Narrative" | `doc_check/tests/section_citations.rs` |
| A rule in these documents has a row here | [`rules.md`](rules.md), "Reading the register" | `doc_check/tests/rules.rs` (that each named check exists and each document is cited; not that every rule is listed) |
| The PR that builds a feature whose row is `not built:` ships the row's check and flips it | [`rules.md`](rules.md), "Enforced at birth" | **unenforced** (review's: the reviewer reads the unenforced rows first, and a PR that defines a `not built:` row's feature without editing the row is the departure) |
| Every bibliography key is cited and every citation has an entry | [`documentation.md`](documentation.md), "References" | **unenforced** (not built: no bibliography yet) |
| A derived source's license obligation is recorded when the code arrives | [`documentation.md`](documentation.md), "References" | **unenforced** |
| A published equation is cited by number and bibliography key, the tree's own by label | [`documentation.md`](documentation.md), "Equations", [`design.md`](design.md), "What it documents" | **unenforced** |
| A reproduction note's numbers come from a committed run product | [`documentation.md`](documentation.md), "Narrative" | **unenforced** |
| Diagrams are checked-in text; animations come from real runs; binary media stays out of `main` | [`documentation.md`](documentation.md), "Diagrams and animations" | **unenforced** (the commit hook's `check-added-large-files` refuses a file over 3 MB, not a small binary) |
| The safe wrapper's doc comment links down to the `-sys` item and adds only what is new | [`documentation.md`](documentation.md), "The C header" | `doc_check/tests/source_rules.rs::every_safe_wrapper_links_down_to_its_binding` (in a crate with a `src/bindings.rs`, a `pub fn` links each binding it calls, or, calling none itself, one of those its private functions reach by name; not a binding reached through a task's trait method, as `add_record_tasks`' are; "adds only what is new" is review's) |
| The book builds, its links checked, and the site needs no GPU | [`documentation.md`](documentation.md), "Narrative", "Publishing" | **unenforced** (not built: no book yet) |

## What is not here

Left out as not normative: the documents' history and measurements (the
C++ tree's counts, #988's replay, the 3080's numbers), the examples (the
Coulomb build, the semi-numerical Hessian), the open questions, and what a
section only describes, such as the tables of what the C++ code becomes.
Left out as one rule stated twice: a rule cited from several documents has
one row, citing each.
