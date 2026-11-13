# rust — free_breakfast's core, rebuilt in Rust

Can free_breakfast's C++ layers be rebuilt in Rust, one at a time, over
WarpWraps' C ABI (`wwr-sys`), keeping each one's contract? This Cargo
workspace is that rebuild. Its tree mirrors `src/`, one crate per module:
`core/wwr/` is the crate `fb-core-wwr` (`fb_core_wwr` in a path), and it is
`src/core/wwr`. It stands alone: nothing here uses the crates under
`experimental/rust/`.

## What it depends on

Outside `rust/`, the tree reads four things, and nothing under `src/`:

- **`deps/WarpWraps`**: `wwr-sys`, its C ABI crate, and the sources
  `CMakeLists.txt` here builds `libwwr_capi_runtime_api.so` from.
- **`deps/FelizJueves`' `include/`**: `felizjueves/runtime.h`, which every
  emitted kernel unit includes.
- **`data/basis/`**: the BSE JSON `fb-io-basis` compiles in.
- **`tools/emit/`**: the FelizJueves generators, run by hand with
  `--out-root` to emit the kernel units `integrals/kernels/` carries ("Build
  and test").

The device code the crates call is forked into `rust/` (`device_headers/`,
`recipes/kernels/`, `integrals/kernels/`) and built by `CMakeLists.txt`
here, which configures nothing of the C++ tree
([`gpu_layer.md`](docs/gpu_layer.md), "What the port changes", "Build").
The fork's cost: its hand-written device code, the launchers, the merge
sort and the shared device headers, exists twice, once here and once in the
C++ tree, and is kept in step by hand until the C++ side is retired. The
emitted units are not a cost, since the one generator writes both trees.
The devtools scripts that drive the gates and the bindings
(`devtools/test.py`, `devtools/rust-bindings.py`) are tooling around the
tree, not inputs to its build.

## What is here

| Crate or module | Ports | What |
|---|---|---|
| `fb_core_wwr::common` | wwr's `src/extension/common`, and the `src/extension/error_handling` primitives it re-exports | `ErrorCode` (`success_code` / `error_name` / `error_string`), `wwr_status` as an error code, `ScopedDeviceIndex` |
| `fb_core_wwr::handle` | wwr's `src/extension/handle` | `BaseHandle` over a `HandleKind` (CRTP's `create`/`destroy` hooks), `DeviceBoundHandle`, `StreamBoundHandle` over a `StreamBoundKind`, each with a fallible `try_new`; the three views; the `DeviceHandle` / `DeviceHandleStream` traits |
| `fb_core_wwr::runtime` | wwr's `src/extension/runtime` | `Stream`, `Event`, `MemPool` (with `MemPoolProps`), `Graph`, `GraphExec`, each a newtype over the handle it is built on, whose constructors abort on failure; stream capture through to a launched graph |
| `fb_core_wwr` (`gpu_status`) | `src/core/wwr`'s `:status`, and the rule its `:policy` encodes | `gpu_status` / `gpu_status_at`, and `gpu_log_error_at`, what a `Drop` does with a failed destroy |
| `fb_core_log` | `src/core/log` | The one backend behind the [`log`](https://crates.io/crates/log) facade: spdlog's line format (local time, OS thread id, `file:line`), the seven levels, `set_level` / `should_log` / `flush`, one file sink, the warning counter, and `log_at` for a location already held |
| `fb_core_error_handling` | `src/core/error_handling` | `ErrorKind` (with the pinned exit codes as `exit_code`), `Error` and its `#[track_caller]` factories, `Result<T>` / `Status`, `check_or_abort` |
| `fb_core_diagnostics` | `src/core/diagnostics` (`diagnostics.h`) | `gpu_must` (`FBF_GPU_MUST`), `critical!` (`FBF_LOG_CRITICAL`), and `log`'s `trace!` ... `error!` re-exported (`FBF_LOG_TRACE` ... `FBF_LOG_ERROR`); `FBF_TRY` is `?` |
| `fb_core_arena` | the peak model of `src/core/footprint` and `src/pipeline/footprint`, and `check_plan`'s fit | The plan's memory laid out once (`docs/memory.md`): `TableBuilder` (rows declared node by node as scratch, outputs and their readers, or the worker's; lifetimes derived), `assign` (the pure greedy best-fit offset assigner), `Table` (offsets, total, `peak`, and `handoffs`, the orderings reusing bytes needs; its text is the committed golden `core/arena/tests/golden/arena_scf_fixture.txt`, rewritten by `FB_BLESS=1`), `admit` (the fit against a cap, refusing with the rows live at the peak), `Block<K>` (the table's one plain allocation, `Device` or portable `Pinned`, filled with `0xFF`, aborting on failure with the bytes and the free memory) and `Block::partition` (a node's `Bufs`: its own rows as `&[T]` / `&mut [T]` on the host and pointer views on the device, behind always-on borrow flags). The one module that allocates |
| `fb_core_planner` | nothing: the C++ steps allocate as they go | The glue between the two below: `Task` (`layout`, which declares every row through a `Layout`, a node's rows of one table, and `run`, which computes through them), `Dag` (tasks added in plan order, each on a GPU worker's stream or the CPU pool; `add` returns the task's rows, whose output ids the reading tasks take), `Planned` (each worker's device and host tables, and each task's dependencies, the reads and the handoffs, derived; its text is the committed golden `core/planner/tests/golden/planner_scf.txt`) and `Instance` (the blocks, and a `thread_pool::Plan` whose nodes hand each task its rows; run any number of times, each run returning the executor's `Record` with every task under its own name). No `unsafe` |
| `fb_core_thread_pool` | nothing: the C++ pipeline runs on one thread | The run's workers (`docs/programming_model.md`, "CPU workers"): `Plan` (nodes added in topological order, each placed on the CPU pool or one stream worker), `run` (one scope, `n_cpu` CPU workers and a thread per stream worker, readiness counts, failure skipping descendants, the lowest-ranked error returned, a panic re-raised rather than hung), and the `Backend` the edges leaving a stream wait through: `Device` (a stream per worker, an event per GPU node) or `HostOnly` for device-free tests. `fb_core_log::TaskScope` tags each node's lines `[task@worker]`, and `Error::context` carries it out. Every run returns a `Record` (`docs/diagnostics.md`, "The timing record"): its result, its wall time, and a `TaskTime` per node that ran, failed ones included, with its name, worker, host time around the callable and, under `Device`, its stream scope's device time (an event pair on its stream, read after the run). `clippy.toml` makes it the one place that spawns |
| `fb_core_types` | `src/core/types`, what the host modules name | `Double3`, `Vec<double, 3>`: `#[repr(C)]` with the backend's `double3` layout. The fixed-width integers and `make_vec` are the language's |
| `fb_core_primitives` | `src/core/primitives`, what the host modules name | `double_factorial_odd`, `factorial`, `binomial` as `const fn`, bitwise the C++ loops. `idivup` / `align_up` are `div_ceil` / `next_multiple_of`, and the element-wise functions `f64`'s methods |
| `fb_core_constants` | `src/core/constants`, what the host modules name | `BOHR_PER_ANGSTROM`, `kBohrPerAngstrom<double>`, asserted against the C++ double's bit pattern |
| `fb_atom` | `src/atom` | `Element` (`#[repr(u8)]`, Z as the discriminant, `Sentinel` = 0), `z` / `from_z`, `symbol` / `from_symbol` (case-insensitive, `Sentinel` for none), and `Atom`, a `Double3` position in bohr and an `Element` |
| `fb_molecule` | `src/molecule`, its host half | `Molecule` as host data (atoms, charge, multiplicity; no atoms aborts), `distance`, `nuclear_charge`, `electron_count`, and the nuclear repulsion energy, gradient and Hessian. The nuclei on the device (`view`, `NucleiView`, `molecule_footprint`) become an upload task when nuclear attraction is ported |
| `fb_basis` | `src/basis` | `Shell` and `Primitive`, the counts (`function_count_cart` / `function_count_sphe`) and `MAX_ANGULAR_MOMENTUM` (3, the C++ `kMaxAngularMomentum`), `cartesian_powers` and `Ao`, `solid_harmonic_block`, `BasisSet` (its Cartesian and spherical AO layouts, the per-atom shell grouping, the name) and `spherical_transform_matrix`, `ShellStamp` with `primitive_normalized` and `place`, and `BasisStamp` with `place` onto a `Molecule`. Contractions are not renormalized, as in the C++ |
| `fb_recipes_sys` | `src/recipes/sort`'s merge sort, forked to `recipes/kernels/` (with the device headers it includes, trimmed, in `device_headers/`) | The raw FFI over `libfb_recipes_kernels.so`, built by `CMakeLists.txt` here: `fb_merge_sort`, an `extern "C"` launch wrapper that only enqueues, its `fb_merge_sort_workspace_bytes` query, `fb_launch_status` and `FB_DEVICE_ALIGNMENT` (`docs/gpu_layer.md`, "Layering"). The bindings are generated from `recipes/kernels/fb_recipes.h` by `devtools/rust-bindings.py` and committed |
| `fb_integrals_sys` | `src/integrals/mmd`'s launchers (order, records, bound, dressed bound) and `src/integrals/one_electron/overlap`'s, forked to `integrals/kernels/` with the FelizJueves units they launch (`pair_bound`, `record_bound` and the overlap's ten classes, emitted there by `tools/emit/*.py --out-root`) | The raw FFI over `libfb_integrals_kernels.so`, which links `libfb_recipes_kernels.so` for the order's sort: `fb_mmd_order` (with `fb_mmd_order_workspace_bytes`), `fb_mmd_records`, `fb_mmd_bound`, `fb_ordered_pair_bound` and `fb_overlap_class`, each an `extern "C"` launch wrapper that only enqueues and refuses a class outside its table as `FB_LAUNCH_UNSUPPORTED`; the tables' limits (`FB_*_MAX_MOMENTUM`); and the three structs that cross by pointer, `fb_primitive_pair_class_view`, `fb_primitive_pair_source` and `fb_primitive_pair_records`, `#[repr(C)]` with their layout asserted on both sides. The bindings are generated from `integrals/kernels/fb_integrals.h` and committed |
| `fb_device_check` | nothing: a check on the kernel libraries | Two tier-3 tests: no launch wrapper under `rust/` allocates, synchronizes, touches an event, copies, sets the device or opens a profiler range (a grep of every `.cu`), and every `-sys` crate's committed bindings are what the script generates now |
| `fb_io_xyz` | `src/io/xyz` | `parse_xyz` / `read_xyz`: a standard XYZ geometry into a `Molecule`, Angstrom to bohr by `BOHR_PER_ANGSTROM`, every refusal (too short, a bad count, a malformed line, an unknown symbol, a count the lines disagree with, a file that cannot be read) an `Io` error. A field must parse as a whole token, where the C++'s `operator>>` reads a prefix |
| `fb_io_basis` | `src/io/basis`, its orbital sets | `parse_basis` / `read_basis`: a BSE JSON document into a `BasisStamp` (sp blocks and general contractions split, spherical shells only: one Cartesian or ambiguous shell refuses the document), over [`serde_json`](https://crates.io/crates/serde_json); `is_builtin_basis` and `resolve_basis`, a name (spelling folded) or else a path, over the eleven sets of `data/basis/` compiled in with `include_str!`, only that branch naming the stamp. The fitting sets of `data/aux_basis/` are not ported yet |

Two more crates hold no port: `fb-test-support` (`test_support/`), what
more than one crate's unit tests use, and a dev-dependency only; and
`fb-doc-check` (`doc_check/`), the section-citation check. There is no
umbrella crate. The crate graph is the selection graph
([`testing.md`](docs/testing.md), "Selection: the workspace graph is the
selection graph"): `log` at the bottom, `error_handling` on it, `wwr` on
both, then `arena` and `thread_pool` on `wwr`, `planner` on those two, and
`diagnostics` at the top with nothing in `core` depending on it, since
`log`'s macros are imported from `log` directly and `critical!` lives in
`fb-core-log`. Above `core`, `atom` stands on `types`, and `molecule` on
`atom`, `types` and `error_handling`, and `basis` on `molecule`, `types`,
`primitives` and `error_handling`; `io/xyz` on `molecule`, `atom`, `types`,
`constants` and `error_handling`, and `io/basis` on `basis` and
`error_handling`.

Each module's doc comment has the full C++-to-Rust table. Some C++ has no
Rust counterpart, because the language already does what that code exists
to force:

- `NonCopyable`. A Rust type is move-only unless it opts in to `Clone`.
- The error policies (`error_policy`, `policy_map`, `policy_slot`,
  `gpu_check`, the C++ tree's `CreatePolicy` / `DestroyPolicy`) and wwr's
  `kit`. A C++ constructor cannot return a failure, so what it does with
  one is injected as a type parameter. A Rust constructor returns `Result`,
  and this tree's answer is fixed: a wrapper's constructor aborts on an
  `Err` (`check_or_abort`), and a `Drop` logs a failed destroy
  (`gpu_log_error_at`). `fb_core_wwr::common`'s doc has the whole reasoning.
- The move constructors, `release()`, the moved-from state and the
  null-handle sentinel. A Rust move leaves nothing behind to drop, and a
  failed create yields an `Err`, not an empty handle.
- `device_handle_pool` and `kit::DeviceHandle`, the rung a buffer allocates
  from a stream-ordered pool through. The arena allocates one plain block
  per plan ([`memory.md`](docs/memory.md), "The two kinds").
- Dangling views. A view borrows the handle it came from.

- The `FBF_LOG_*` macros, and the rule that formatting goes only through
  them. `log`'s macros check the level before they format.
- `LocatedFormat` and the `err_*` helpers. A `#[track_caller]` factory
  captures its caller even with a format string, so
  `Err(Error::io(format!(...)))` does what `err_io(...)` does.
- `FBF_TRY`. It is `?`, which moves the error's origin through unchanged.

Two things go the other way, where Rust asks for more than the C++ does:

- wwr's borrow-safe operations (`sync`, `record`, `launch`, ...) are free
  functions on a raw handle. A destroyed handle there is a use-after-free in
  the runtime, so the raw forms are `unsafe fn`, and each owner has methods
  that call them safely.
- `wwrMemPoolProps` reaches the C ABI as a bare `void*`. `MemPoolProps`
  mirrors the vendors' layout, which CUDA 13 and ROCm share (88 bytes,
  identical offsets; checked with a C program against both headers).
  `wwrDeviceProp`, whose layout differs between the backends, is not
  mirrored.

## Contracts

One precondition is checked with the [`contracts`](https://crates.io/crates/contracts)
crate, in debug builds only: a device-bound view's `dev_idx` is -1
(unknown) or more.

Outcomes of GPU calls are deliberately not contracts. A call that can fail
returns a status or a `Result`, and a failure is an expected result, not a
broken promise; a contract would turn it into a panic.

## Build and test

```bash
cd rust
cmake --preset cuda                    # or hip; the backend has no default
cmake --build --preset cuda            # libwwr_capi_runtime_api.so and the kernel libraries
cargo test --workspace -- --include-ignored  # debug, device cases and contract tests
cargo test --workspace --release       # the contract tests are skipped
cargo test -p fb-core-arena            # one crate, with only its own features
```

A device case is `#[ignore = "tier 4: needs a GPU"]`, so a plain `cargo
test` lists it as ignored, and `-- --include-ignored` runs it; `#[ignore]`
means nothing else here ([`testing.md`](docs/testing.md), "Selecting GPU
tiers in cargo"). The gates run it for the crates a diff affects, on the
card they hold, and print every case that did not run.

Tier 3, everything that needs neither card nor library, builds with no
CMake step at all:

```bash
cargo test -p fb-core-log -p fb-core-error-handling -p fb-doc-check \
  -p fb-core-arena -p fb-core-thread-pool -p fb-core-planner \
  -p fb-core-types -p fb-core-primitives -p fb-core-constants \
  -p fb-atom -p fb-molecule -p fb-basis -p fb-device-check \
  -p fb-io-xyz -p fb-io-basis \
  --no-default-features
```

`CMakeLists.txt` here is a CMake project of its own, which the root
`CMakeLists.txt` does not add: it configures `deps/WarpWraps` alone (its
tests, examples and install rules off) and builds `wwr.capi.runtime_api`,
so building what the crates link configures nothing of the C++ tree. Its
presets are `cuda` and `hip`, into `build-cuda/` and `build-hip/` here,
with the root's sm_80 floor checked. It also builds the kernel shared
objects, one per module from sources forked into `rust/` (nothing under
`src/` is compiled or included): `libfb_recipes_kernels.so`, from
`recipes/kernels/` and the trimmed device headers in `device_headers/`, at
`<build>/recipes/kernels`, with its own device link and the shared vendor
runtime, and `libfb_integrals_kernels.so`, from `integrals/kernels/`, at
`<build>/integrals/kernels`, which links the first. Its `-sys` crate's `build.rs` includes `kernel_lib.rs`, which links
it from the same `<build>` `capi_dir.rs` chose. After editing a kernel
library's C header, run `devtools/rust-bindings.py` (stdlib Python, no
bindgen: it reads the header's small plain-C grammar as text, as WarpWraps'
C ABI generator does) and commit the bindings; `fb-device-check` fails
until then. `devtools/cross-backend-check.sh` compiles this project for the
other backend too. `capi_dir.rs`, which each linking crate's `build.rs`
includes, finds `libwwr_capi_runtime_api.so` under
`<build>/deps/WarpWraps/src/capi`, where `<build>` is whichever of the two
holds it. With both built it refuses to choose: set `FB_RUST_BUILD_DIR` to
one, which also takes any other CMake build directory, the root's
`build-experimental` among them. `devtools/doctor.sh` reports which are
built. Only the `runtime_api` surface of `wwr-sys` is linked.

The GPU tests (the ignored ones) use device 0 and need room on it for a
context. The host-only
tests (the fake handle kind, error codes and contracts) need no GPU, and
`fb-core-log`, `fb-core-error-handling`, `fb-core-types`,
`fb-core-primitives`, `fb-core-constants`, `fb-atom`, `fb-molecule`,
`fb-basis`, `fb-io-xyz`, `fb-io-basis`, `fb-test-support`, `fb-doc-check`
and `fb-device-check` do not link the library at all: they build and pass
with it missing.
`fb-core-arena`, `fb-core-thread-pool` and `fb-core-planner` link it
through their `device` feature, which is on by default: without it
(`--no-default-features`) their device-free tests, on `Heap` blocks and
`HostOnly` workers, build and pass with no wwr crate in the graph. There
`Device` and `Pinned` are still kinds, so tables, admission and planning
are whole, but a block of either of non-zero size aborts, and the
thread pool's `DeviceStream` is an uninhabited placeholder with no
`Device` backend to run it. `device-tests` (also default) adds what the
crates' own device tests need beyond `device`; a dependent asks for
`device` alone, and the workspace's dependency entries for the arena and
the thread pool turn their defaults off, so nothing turns them on by
accident. Only `fb-core-wwr`, `fb-core-diagnostics`, `fb-recipes-sys`
and `fb-integrals-sys` always link. Use a
separate `CARGO_TARGET_DIR` per backend, and another for
`--no-default-features`: each `build.rs` bakes the library's path into the
rpath.

Allocating from a memory pool is tested in `core/wwr/tests/pool_allocation.rs`, a
binary of its own. Under ROCm 7.2.4 on the gfx1200,
`hipMallocFromPoolAsync` / `hipFreeAsync` on one thread beside other
threads' stream work intermittently deadlocked inside `libamdhip64` (about
one run in two of the unit tests); that file has the evidence. Cargo runs
each test binary as its own process, so they never overlap.

The logger is process-global, so `core/diagnostics/tests/log.rs` is a
binary of its own too,
and its in-process checks are one test. Its abort and exit cases re-run
that binary as a child and read the child's stdout: what they check (the
critical line flushed before `abort`, the buffered tail flushed at exit) is
what survives the process ending.

The arena's blocks on a card are `core/arena/tests/arena_block.rs`, a
binary of its
own whose cases take turns, since two of them measure the card's free
memory: another process allocating on the card between two readings fails
them, so hold the card, and expect a shared one to break them now and then.
Its abort case re-runs the binary as a child, as the log test does. The
arena's device-free tests are unit tests, over a test-only heap kind, and so
are the planner's, run on `HostOnly` workers; the heap kind reaches the
planner's tests through `fb-core-arena`'s `test-support` feature, enabled
as a dev-dependency only, and `fb_core_wwr::runtime::tests`' device helpers
reach `thread_pool`'s the same way. The planner on a card, a round trip
through pinned and device rows, is `core/planner/tests/planner_device.rs`,
which measures free memory too.

The merge sort on a card is `recipes/sys/tests/merge_sort.rs`. It copies
its inputs up on the stream it sorts on, never with the synchronous
`wwr_memcpy`: that runs on the legacy stream, which the crate's
non-blocking `Stream` does not wait for, and from pageable memory it
returns before its DMA lands. Its first version did that, and from 22 529
keys up the sort read keys the DMA was still writing, and failed with wrong
keys or a fault (`WWR_STATUS_UNKNOWN` at the next synchronization) that the
kernel was not at fault for; `compute-sanitizer` serialized the two and hid
it. A kernel library's test copies on its own stream.

The overlap on a card is `integrals/sys/tests/overlap.rs`: one (s|s)
record against the closed-form primitive overlap, within 4 ulp, and the
refusals of every wrapper in `fb-integrals-sys`. The FelizJueves units in
`integrals/kernels/` are emitted, not written: after bumping
`deps/FelizJueves`, emit both tools' whole trees into a scratch directory
and copy the three directories this library carries, byte for byte what
the defaults write under `src/integrals/`:

```bash
# from the repository root, tmp=$(mktemp -d)
OMP_NUM_THREADS=8 tools/emit/one_electron_kernels.py --out-root "$tmp"
OMP_NUM_THREADS=8 tools/emit/exchange_kernels.py --out-root "$tmp"
cp "$tmp"/mmd/kernels/pair_bound/* rust/integrals/kernels/mmd/kernels/pair_bound/
cp "$tmp"/mmd/kernels/record_bound/* rust/integrals/kernels/mmd/kernels/record_bound/
cp "$tmp"/one_electron/overlap/kernels/* rust/integrals/kernels/one_electron/overlap/kernels/
```

`doc_check/tests/section_citations.rs` checks that every section the
README, `docs/` and every member's Rust sources cite by name is a heading of
the file cited ([`documentation.md`](docs/documentation.md), "Narrative:
mdBook beside the rustdoc", has the rules). It reads files and nothing else,
in a crate that links nothing, so it needs neither card nor library:
`cargo test -p fb-doc-check`. It finds the members from the workspace
manifest, so a new crate is scanned without being named there.

Tests run in parallel, so a capture test must not use global capture mode.
A global capture forbids other threads' unsafe calls, and on the 3080 it
made a neighbouring pool's create abort or its destroy segfault.

Nothing in ctest runs cargo. Both gates do, for a diff under `rust/`:
`devtools/test.py push` and `pr` run tiers 1 to 3, the tier-3 command above
over every member whose graph declares no `links` under
`--no-default-features` (found from `cargo tree`, so a new link-free crate
is gated without being named), then `RUSTDOCFLAGS='-D warnings' cargo doc
--no-deps` over the members whose default features link nothing, in
`target/nodef` ([`docs/testing.md`](../docs/testing.md), "Gates", at the
root). A commit touching `rust/` runs `devtools/test.py cargo-lint`, `cargo
fmt --check` and clippy with warnings denied in both feature sets, and so
does CI's `lint` job. Tier 4, on a card, runs in both gates for the crates
a diff affects (above).

## Design

`docs/` holds the design. [`design.md`](docs/design.md) is the questions a
feature's design answers and where each answer lands; the six beside it
hold the mechanics: the [programming model](docs/programming_model.md),
[memory](docs/memory.md), the [GPU layer](docs/gpu_layer.md),
[testing](docs/testing.md), [diagnostics](docs/diagnostics.md) and
[documentation](docs/documentation.md).
They describe the finished shape; the section below is the order the tree
gets there in.

## Order of work

Before this milestone: the section-citation check
(`doc_check/tests/section_citations.rs`), the workspace split into one
crate per module with no umbrella, and `arena`, `thread_pool` and `planner`
building without the wwr library. Milestone 32, the overlap through the
whole model, in waves (each one PR; done means merged):

| Wave | Issues |
|---|---|
| 0 | #1362 these docs; #1363 the cargo half of the gates, done; #1364 this directory's CMake project, done; #1365 core types, primitives and constants, done; #1366 stream scopes and the timing record, done |
| 1 | #1367 atom and molecule, done; #1368 tier-4 skips, done; #1369 the forked device layer and the merge sort, done |
| 2 | #1370 basis, done; #1371 the forked mmd and overlap kernels, done |
| 3 | #1372 io/xyz and io/basis, done; #1373 the primitive-pair records as tasks |
| 4 | #1374 the overlap: declared with a layout, laid out in an arena, planned once and run on one stream, its arena size a golden and its stream scope timed |

Open beside them: #1382, the kernel libraries' ABI version and backend
check at load ([`gpu_layer.md`](docs/gpu_layer.md), "What the port
changes"), and #1380, the basis's angular-momentum ceiling asserted against
a forked device header.

Then the port, bottom-up from `core`, one module per crate, its device code
forked with it ([`gpu_layer.md`](docs/gpu_layer.md), "What the port
changes").

## Verified

2026-11-12, on both cards: on the RTX 3080 (CUDA) and on the gfx1200 (HIP),
33 unit tests and the 2 pool-allocation tests pass in a debug build, and 30
and 2 in release. The debug unit-test binary passed 20 runs in a row on
each card, and the pool binary 20 on each. Clippy, rustdoc and `cargo fmt
--check` are clean.

2026-11-12, `core::log`, `core::error_handling`, `core::diagnostics` and
`core::wwr`'s status and policies, on the RTX 3080 (CUDA) only: 39 unit
tests, the 5 log tests and the 2 pool-allocation tests pass in a debug
build, and 36, 5 and 2 in release. The log binary passed 20 runs in a row.
Clippy, rustdoc (`-D warnings`) and `cargo fmt --check` are clean. Nothing
added calls a backend-specific path, but it has not been run on the
gfx1200.

2026-11-13, the error policies, `kit` and the pool rung removed, on the RTX
3080 (CUDA) only: 44 unit tests, the 5 log tests, the pool-allocation test
and the thread-count test pass in a debug build, and 42, 5, 1 and 1 in
release. Clippy, rustdoc (`-D warnings`) and `cargo fmt --check` are clean.
Not run on the gfx1200.

2026-11-13, `core::arena`'s tables (the assigner, the table, handoffs,
admission), on the RTX 3080 (CUDA) only: 50 unit tests (6 of them the
arena's, 2 of those proptest properties at 256 cases), the 5 log tests,
the pool-allocation test, the thread-count test and 2 doctests pass in a
debug build, and 48, 5, 1, 1 and 2 in release. Clippy (`-D warnings`),
rustdoc (`-D warnings`) and `cargo fmt --check` are clean. The arena's
tests touch no device.

2026-11-13, `core::arena`'s blocks and views, on the RTX 3080 (CUDA): 53
unit tests, the 5 arena-block tests, the 5 log tests, the pool-allocation
test, the thread-count test and 3 doctests pass in a debug build, and 51,
5, 5, 1, 1 and 3 in release. The arena-block binary passed 79 of 80
repeats while another process held part of the card, and its one failure
was not caught (see "Build and test" on the free-memory cases); it is
clean under `compute-sanitizer --tool memcheck`. Clippy (`-D warnings`),
rustdoc (`-D warnings`) and `cargo fmt --check` are clean. On the gfx1200
(HIP), with the card to itself, the same counts pass in both profiles, and
the arena-block binary passed 20 repeats in a row.

2026-11-13, `core::planner` (the `Task` trait, `Dag`, `Planned`,
`Instance`), with the two hand-wired executor tests moved onto it, on the
RTX 3080 (CUDA) only: 59 unit tests (6 of them the planner's, one a
proptest property at 64 random DAGs), the 4 arena-block tests, the planner
device test, the 5 log tests, the pool-allocation test, the thread-count
test and 4 doctests pass in a debug build, and 57, 4, 1, 5, 1, 1 and 4 in
release. With the handoffs dropped from the derived dependencies, the
property and the executor test both fail, so they see a missing ordering.
The planner device binary and the planner's unit tests each passed 20
repeats in a row, and the device binary is clean under `compute-sanitizer
--tool memcheck`. Clippy (`-D warnings`), rustdoc (`-D warnings`) and
`cargo fmt --check` are clean. Not run on the gfx1200.

2026-11-13, the workspace split (one crate per module, no umbrella), on the
RTX 3080 (CUDA) only, holding the card: the same tests as the entry above,
redistributed. In a debug build the unit tests are 30 (`wwr`), 10
(`thread_pool`), 9 (`arena`), 6 (`planner`) and 4 (`error_handling`), 59 in
all as before, with the 4 arena-block tests, the planner device test, the 5
log tests, the pool-allocation test, the thread-count test, the 8
section-citation tests and 4 doctests (1 more `ignore`d); in release 57,
with the rest unchanged. Each crate passes `cargo test -p` on its own. The
section-citation counts are unchanged (117 links, 21 backticked paths, 3
self-references). `fb-core-log`, `fb-core-error-handling`,
`fb-test-support` and `fb-doc-check` have no `libwwr_capi` under `NEEDED`,
and the first, second and fourth pass with the library moved aside.
`FB_BLESS=1` rewrites each golden in its own crate, unchanged. Clippy (`-D
warnings`), rustdoc (`-D warnings`) and `cargo fmt --check` are clean. Not
run on the gfx1200.

2026-11-13, a link-free tier 3 (the `device` feature on the arena, the
thread pool and the planner), on the RTX 3080 (CUDA) only, holding the
card: with default features the counts are the workspace split's entry,
unchanged in both profiles, and each of the three crates passes `cargo test
-p` alone. With `--no-default-features`, in a worktree where the library had
never been built, the three build and pass 9, 9 and 6 unit tests (the
thread pool's one device test is the tenth), the thread-count test and 4
doctests, alone and together; `cargo tree` shows no wwr crate, and no test
binary has `libwwr_capi` under `NEEDED`. A non-zero `Device` block there
aborts naming the missing feature (checked by hand). Clippy (`-D warnings`)
is clean in both feature sets; rustdoc (`-D warnings`) and `cargo fmt
--check` are clean with default features. Not run on the gfx1200.

2026-11-13, `core::types`, `core::primitives` and `core::constants` (what
`atom`, `molecule`, `basis`, `io/xyz` and `io/basis` name), on the gfx1200
(HIP), holding the card: with default features the whole workspace passes in
a debug build, the three new crates 2 unit tests each and every other count
the link-free tier 3 entry's. With `--no-default-features` the tier-3 set,
the three included, passes 9 (`arena`), 2 (`constants`), 4
(`error_handling`), 6 (`planner`), 2 (`primitives`), 9 (`thread_pool`), 2
(`types`) and the 8 section-citation tests, with the thread-count test and 4
doctests; `cargo tree` for the three shows no wwr crate, nor any dependency.
`BOHR_PER_ANGSTROM`'s bit pattern is the one a C++ program printed from
`constants.h` itself. Clippy (`-D warnings`) is clean in both feature sets;
rustdoc (`-D warnings`) and `cargo fmt --check` are clean. Not run on the
RTX 3080; the three crates touch no device.

2026-11-13, `rust/`'s own CMake project (`CMakeLists.txt`, the `cuda` and
`hip` presets) and `capi_dir.rs` defaulting to its build directories, on
the RTX 3080 (CUDA) only, holding the card: in a checkout with no
`build-experimental/`, `cmake --preset cuda` and `cmake --build --preset
cuda` built `libwwr_capi_runtime_api.so` alone (15 steps), and `cargo test
--workspace` with no `FB_RUST_BUILD_DIR` linked it from `build-cuda/` and
passed the previous entry's debug counts, unchanged: 65 unit tests (the
three newest crates 2 each), the 4 arena-block tests, the planner device
test, the 5 log tests, the pool-allocation test, the thread-count test,
the 8 section-citation tests and 4 doctests (1 more `ignore`d). The `hip`
preset configures and builds too (compile only, not run on the gfx1200);
with both built, `capi_dir.rs` refuses to choose, and
`FB_RUST_BUILD_DIR=build-hip` links that one. The `--no-default-features`
tier 3, the three newest crates included, passes the previous entry's
counts. The root's ctest listing is unchanged. Clippy (`-D warnings`, both
feature sets), rustdoc (`-D warnings`) and `cargo fmt --check` are clean.

2026-11-13, `atom` and the host half of `molecule`, on the gfx1200 (HIP),
holding the card, linking the library the `hip` preset built (`cmake
--preset hip`, `cmake --build --preset hip`): with default features the
whole workspace passes in a debug build, the two new crates 14 (`atom`) and
12 (`molecule`) unit tests and the 2 tests of `molecule_abort` (the
no-atoms abort, re-run as a child), every other count the CMake project's
entry's. With `--no-default-features` the tier-3 set, the two included,
passes those counts plus 14, 12 and 2; `cargo tree` for the two shows no
wwr crate, and no test binary of theirs has `libwwr_capi` under `NEEDED`.
Clippy (`-D warnings`) is clean in both feature sets; rustdoc (`-D
warnings`) and `cargo fmt --check` are clean. Not run on the RTX 3080; the
two crates touch no device.

2026-11-13, `basis`, on the gfx1200 (HIP), holding the card, linking the
library the `hip` preset built: with default features the whole workspace
passes in a debug build, the new crate 28 unit tests and the 5 tests of
`basis_abort` (the fourteen aborts of `test/basis/`'s death tests, each
re-run as a child), every other count the `atom` and `molecule` entry's.
With `--no-default-features` the tier-3 set, `basis` included, passes those
counts plus 28 and 5; `cargo tree` for it shows no wwr crate, and neither
of its test binaries has `libwwr_capi` under `NEEDED`. With the partition
check's `is_sorted` removed, `basis_abort` fails on the decreasing case.
Clippy (`-D warnings`) is clean in both feature sets; rustdoc (`-D
warnings`) and `cargo fmt --check` are clean. Not run on the RTX 3080; the
crate touches no device.

2026-11-13, the forked device layer (`device_headers/`, the merge sort in
`recipes/kernels/` built as `libfb_recipes_kernels.so`, `fb-recipes-sys`,
`fb-device-check`), on the RTX 3080 (CUDA), holding the card: with default
features the whole workspace passes in a debug build, the new binaries 4
merge-sort tests (keys with values and keys alone at 32 sizes from 1 to
100 000, equal bitwise and in permutation to the host's stable sort; the
workspace query; the refusals), 1 bindings test and 2 launch-wrapper tests,
every other count the `basis` entry's; in release the same, but for `wwr`'s
28 unit tests. The merge-sort binary is clean under `compute-sanitizer
--tool memcheck`, each sort handed exactly the bytes its query asked for.
With `--no-default-features` the tier-3 set, `fb-device-check` included,
passes. The bindings test fails on a header edited without regenerating,
and the grep on a planted `wwrDeviceSynchronize` (both checked by hand).
The `hip` preset builds the library too, and `devtools/cross-backend-check.sh
--device-only` (HIP) compiles it and passes; neither was run on the
gfx1200. Clippy (`-D warnings`, both feature sets), rustdoc (`-D warnings`)
and `cargo fmt --check` are clean.

2026-11-13, `io/xyz` and `io/basis` (its orbital sets), on the gfx1200
(HIP), holding the card, linking the library the `hip` preset built: with
default features the whole workspace passes in a debug build, the new
crates 7 tests (`xyz`, the cases of `test/gtest/io_xyz_gpu.cpp`, with the
whole-token and non-finite refusals beside them) and 15 unit tests (`io/basis`:
the orbital-set cases of `test/io/basis_host.cpp` and of
`test/gtest/io_basis_gpu.cpp`, an element the set lacks, and the table
checked against `data/basis/`), every other count the `basis` entry's, and
the forked device layer's 4 merge-sort, 1 bindings and 2 launch-wrapper
tests, here on the gfx1200 too, with the `hip` preset's
`libfb_recipes_kernels.so`. With `--no-default-features` the tier-3 set,
the two included, passes those counts plus 7 and 15; `cargo tree` for them shows no wwr crate, and none of
their test binaries has `libwwr_capi` under `NEEDED`. With the Cartesian
refusal removed, or the Angstrom-to-bohr conversion, a test fails.
Clippy (`-D warnings`) is clean in both feature sets; rustdoc (`-D
warnings`) and `cargo fmt --check` are clean. Not run on the RTX 3080; the
crates touch no device.

2026-11-13, the orchestrator's stream scopes and timing record
(`thread_pool`, `planner`), on the gfx1200 (HIP), holding the card, linking
the library the `hip` preset built: with default features the whole
workspace passes in a debug build, `fb-core-thread-pool` 12 unit tests (2
new: every node timed by its worker, at least its sleep, and a failed run
returning the times of the nodes that ran) and `fb-core-planner` 6 unit
tests and 2 device tests (1 new: six GPU tasks on one stream, one stream
scope each, summing to at most the run's wall time), `fb-core-wwr` 30, every
other count the `io/xyz` entry's. With `--no-default-features` the tier-3
set passes, the thread pool's 11 unit tests among them. Clippy (`-D
warnings`) is clean in both feature sets; rustdoc (`-D warnings`) and `cargo
fmt --check` are clean. Not run on the RTX 3080.

2026-11-13, the integrals kernel library (`integrals/kernels/` built as
`libfb_integrals_kernels.so`, `fb-integrals-sys`), on the RTX 3080 (CUDA),
holding the card: with default features the whole workspace passes in a
debug build and in release, the new binary 4 tests (one (s|s) record
through `fb_overlap_class` against the closed-form primitive overlap within
4 ulp at both mirror positions, the diagonal untouched; every wrapper's
refusals, a class outside a table `FB_LAUNCH_UNSUPPORTED`; the order's
workspace query), every other count unchanged. Two of the three debug
runs of the workspace failed `arena_block`'s `declared_equals_allocated`
(a free-memory reading, "the drop gave back 100663296 of 268435456 bytes";
"Build and test" has why a shared card breaks it); the third passed, and
the binary passes run alone. Nothing here touches the arena. The overlap binary is clean under `compute-sanitizer --tool
memcheck`. With `--no-default-features` the tier-3 set passes, `bindings`
and `launch_wrappers` covering the new library: the first fails on its
header edited without regenerating, the second on a planted
`wwrDeviceSynchronize` in `overlap_launch.cu` (both checked by hand). The
emitted units are byte for byte those under `src/integrals/` (39 files,
`cmp`). The `hip` preset builds the library and
`devtools/cross-backend-check.sh --device-only` (HIP) passes; neither was
run on the gfx1200. Clippy (`-D warnings`, both feature sets), rustdoc
(`-D warnings`) and `cargo fmt --check` are clean.
