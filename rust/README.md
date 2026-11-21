# rust — free_breakfast's core, rebuilt in Rust

Can free_breakfast's C++ layers be rebuilt in Rust, one at a time, over
WarpWraps' C ABI (`wwr-sys`), keeping each one's contract? This Cargo
workspace is that rebuild. Its tree mirrors `src/`, one crate per module:
`core/wwr/` is the crate `fb-core-wwr` (`fb_core_wwr` in a path), and it is
`src/core/wwr`. It stands alone: nothing here uses the crates under
`experimental/rust/`.

## What it depends on

Outside `rust/`, the tree reads seven things, and nothing under `src/`:

- **`deps/WarpWraps`**: `wwr-sys`, its C ABI crate, and the sources
  `CMakeLists.txt` here builds `libwwr_capi_runtime_api.so` from.
- **`deps/FelizJueves`' `include/`**: `felizjueves/runtime.h`, which every
  emitted kernel unit includes.
- **`data/basis/`**: the BSE JSON `fb-io-basis` compiles in, and whose
  cc-pVDZ document `fb-pipeline-guess`'s tests also read by path.
- **`data/aux_basis/`**: the density-fitting sets, BSE JSON too, that
  `fb-io-basis` compiles in as a table of their own.
- **`tools/emit/`**: the FelizJueves generators, run by hand with
  `--out-root` to emit the kernel units `integrals/kernels/` carries ("Build
  and test").
- **`test/sad/data/atomic_scf.json`**: the PySCF reference `fb-sad`'s
  tests read from where the C++ test reads it, so the two trees hold one.
- **`data/sad/`**: the SAD stamps `fb-io-sad` compiles in, and whose
  `STO-3G/` files `fb-io-npy`'s tests read and write back byte for byte,
  and `fb-tools-sad`'s compare its own output with; `fb-tools-sad` writes
  them.

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
| `fb_core_wwr::common` | wwr's `src/extension/common`, and the `src/extension/error_handling` primitives it re-exports | `ErrorCode` (`success_code` / `error_name` / `error_string`), `wwr_status` as an error code, `wwr_status_from_code` (a launch wrapper's positive status read back as one), `ScopedDeviceIndex` |
| `fb_core_wwr::handle` | wwr's `src/extension/handle` | `BaseHandle` over a `HandleKind` (CRTP's `create`/`destroy` hooks), `DeviceBoundHandle`, `StreamBoundHandle` over a `StreamBoundKind`, each with a fallible `try_new`; the three views; the `DeviceHandle` / `DeviceHandleStream` traits |
| `fb_core_wwr::runtime` | wwr's `src/extension/runtime`, and `src/core/wwr`'s `Device::stack_growth` | `Stream`, `Event`, `MemPool` (with `MemPoolProps`), `Graph`, `GraphExec`, each a newtype over the handle it is built on, whose constructors abort on failure; stream capture through to a launched graph; `device_count()`, the devices the runtime sees (`wwr_get_device_count`); `stack_growth(dev_idx, per_thread)`, what the kernels' stack can still take, measured across raising the stack limit and setting it back, 0 under HIP |
| `fb_core_wwr::blas` | wwr's `src/extension/blas` | `wwr_blas_status` as an `ErrorCode`, and `BlasHandle`, a `StreamBoundHandle` over a shared `Stream` whose constructor aborts on failure. `bind_workspace` (`wwr_blas_set_workspace`): the handle takes its workspace by value, a `'static` `DeviceSliceMut` (the planner's worker row, as an arena `HeldRow`) it owns until it is destroyed or handed another, since the library keeps the pointer beyond any borrow; a previous one is released only after the library has the new pointer and the stream has drained, and `Drop` syncs the stream first, so the call's one `unsafe` block rests on ownership, field order and `&mut self`. `BlasScalar`, wwr's typed dispatch (`usual_fp`), sealed over `f32`, `f64` and the two complex types, with each type's `ONE` and `ZERO` (the C++ `kOne` and `kZero`, by value) and its `unsafe` `gemm_raw`, `gemm_strided_batched_raw`, `dgmm_raw`, `trsm_raw`, `syrk_raw`, `gemv_raw` and `geam_raw` the letter's C ABI calls; and `gemm`, `gemm_strided_batched`, `dgmm`, `trsm`, `syrk`, `gemv` and `geam`, the safe calls over them, every operand a `DeviceSlice` whose extent (a matrix's at its op and leading dimension, `dgmm`'s diagonal at its stride) is checked against its length in every build, a short one a panic before anything is enqueued; `gemm_strided_batched` checks every matrix of the batch at its stride, takes a stride of 0 for A or B (one matrix shared by every product) and refuses a C stride under one C's extent, so no two products write one element. `dgmm` and `gemv` take strides of at least 1 (the library's 0 and negative ones are refused) and has no in-place form: C is a slice of its own, never A; nor do `trsm` (X written over B, A triangular on either side) and `syrk` (C's `fill` triangle alone), whose B and C are slices apart from A, nor `geam` (`C = alpha op_a(A) + beta op_b(B)`), whose C is apart from both |
| `fb_core_wwr::ccl` | wwr's `src/extension/ccl` | `wwr_ccl_result` as an `ErrorCode`; `Comm`, one rank of a clique `Comm::init_all(devices)` makes over a device list, a rank per device, destroyed on drop; and the f64 collectives through it, each on a stream of the rank's device and each failure a `Status`: `broadcast_send` / `broadcast_recv`, `reduce_send` / `reduce_recv` (a sum to a root), `all_reduce`, `all_gather`, `send`, `recv`, and `group` |
| `fb_core_wwr::device_slice` | nothing: the C++ hands a library call raw pointers | `DeviceSlice` and `DeviceSliceMut`, `unsafe` traits for device memory with its length, which a safe call takes in place of a pointer. The arena's `DeviceRef` and `DeviceMut` implement them (with its `device` feature), and so does the test helper `DeviceBytes`. `split_at_mut(slice, mid)` cuts a `DeviceSliceMut` into two disjoint `DevicePart`s, each its memory's only user while it lives, as `<[T]>::split_at_mut` does: how the blocks of one matrix reach two operands of one call, non-overlap decided from offsets; `split_at(slice, mid)` cuts a `DeviceSlice` into two `DeviceView`s to read, as `<[T]>::split_at`: how the dense ERI's sub-blocks reach separate calls |
| `fb_core_wwr::copy` | wwr's `src/extension/memory_buffer` `:copy` | `d2d(dst, src, stream)`, a device-to-device `wwr_memcpy_async` between a `DeviceSliceMut` and a `DeviceSlice` of equal length (asserted in every build): a run of elements inside a row, such as a column of DIIS's ring, which only a `split_at_mut` part names; a whole row is the arena's `memcpy_d2d` |
| `fb_core_wwr::solver` | wwr's `src/extension/solver` | `wwr_solver_status` as an `ErrorCode` (two values: the C ABI maps every vendor failure to `WWR_SOLVER_STATUS_UNKNOWN`), and `SolverHandle`, the dense solver's `StreamBoundHandle` over a shared `Stream`, as `BlasHandle` is. `SolverScalar` (`solver/syevd.rs`), the typed dispatch sealed over `f32` and `f64`, its `unsafe` `syevd_raw` and `syevd_lwork_raw` the letter's C ABI calls; `syevd_lwork`, the legacy query; and `syevd`, the safe call, A, w, the work and the flag `DeviceSliceMut`s whose extents are checked in every build, the work against the query, which the call asks again |
| `fb_core_wwr` (`gpu_status`) | `src/core/wwr`'s `:status`, and the rule its `:policy` encodes | `gpu_status` / `gpu_status_at`, and `gpu_log_error_at`, what a `Drop` does with a failed destroy |
| `fb_core_log` | `src/core/log` | The one backend behind the [`log`](https://crates.io/crates/log) facade: spdlog's line format (local time, OS thread id, `file:line`), the seven levels, `set_level` / `should_log` / `flush`, one file sink, the warning counter and the error counter (`error_count`, a failed destroy or free and the boundary's line, which `main` prints), and `log_at` for a location already held; the value checks `le!`, `lt!`, `eq!` and `fits_i32!` (`check`), which a contract's condition is written with so the panic hook ends its line with the operands' values |
| `fb_core_error_handling` | `src/core/error_handling` | `ErrorKind` (with the pinned exit codes as `exit_code`), `Error` and its `#[track_caller]` factories, `Result<T>` / `Status`, `check_or_abort` |
| `fb_core_diagnostics` | `src/core/diagnostics` (`diagnostics.h`) | `gpu_must` (`FBF_GPU_MUST`), and nothing else: `FBF_TRY` is `?`, `FBF_LOG_TRACE` ... `FBF_LOG_ERROR` are `log`'s macros and `FBF_LOG_CRITICAL` is `fb_core_log::critical!`, each imported from its own crate, since this one depends on `fb-core-wwr` |
| `fb_core_arena` | the peak model of `src/core/footprint` and `src/pipeline/footprint`, and `check_plan`'s fit | The plan's memory laid out once (`docs/memory.md`): `TableBuilder` (rows declared node by node as scratch, outputs and their readers, or the worker's; lifetimes derived; `add_loop(positions)` makes a loop's body rows a table of their own in a region the rest holds for the loop, sized to the body's peak once, `add_counted(positions, depth)` a counted loop's, its region `depth` copies of the body's table a stride apart, iteration `k` in copy `k mod depth`, declared before a decision loop whose body holds it to make its region a row of that body's table, a row it writes and does not own refused unless one body node is its only user there, and `write(pos, row)` a row the loop carries from one iteration to the next, written by its last user in the body), `assign` (the pure greedy best-fit offset assigner, run in three orders and kept at the smallest total; every committed golden's block within 3% of its lower bound, `docs/memory.md`, "Lifetimes"), `Table` (offsets, total, `peak`, a loop's region counted once, `live_at`, the bytes held at one position as `peak` counts them, `rows`, each row's name, `Owner`, bytes and users as a read-only `RowInfo`, and `handoffs` and `reads`, the orderings reusing bytes and the data need, a pair leaving a decision loop's body taken from its decision, one leaving only a counted loop in it from the writer's last trip, a row outside a counted loop's body held against every copy; its text is the committed golden `core/arena/tests/golden/arena_scf_fixture.txt`, and with the SCF iteration as a loop's body `arena_scf_loop.txt`, rewritten by `FB_BLESS=1`), `admit` (the fit against a cap, refusing with every row over the cap on its own and the rows live at the peak), `admit_host` (a node's pinned blocks and the plan's heap block together against the host's available memory) and `host_available` (`MemAvailable`, or under a cgroup v2 limit the smaller of it and what the limit leaves) and, with `device`, `device_memory` (a card's free and total memory, one reading at planning, which bounds a device cap and measures the device's reserve), `Block<K>` (the table's one plain allocation, `Device`, portable `Pinned` or `Heap`, the plan's pageable block from the global allocator, filled with `0xFF`, aborting on failure with the bytes and the free memory; a plan's per-worker host table is of a `StagingKind`, `Pinned` or, under `test-support`, the device-free tests' `PinnedStandIn`, never `Heap`) and `Block::partition(pos, iteration)` (a node's `Bufs` in one iteration, `Bufs::iteration` its index, a counted loop's body row in its iteration's copy with a borrow flag per copy: its own rows as `&[T]` / `&mut [T]` on the host and pointer views on the device, behind always-on borrow flags; a stream node's host rows wrapped in a `CopyOnly`, which yields no slice and only `memcpy_p2d` and `memcpy_d2p` take); with `device`, `Block::hold` (a worker row held as its writer outside any view, a `HeldRow` that keeps its `Arc`'d block alive and is a `DeviceSliceMut`: memory a library keeps beyond a node, a BLAS handle's workspace). The one module that allocates |
| `fb_core_planner` | nothing: the C++ steps allocate as they go | The glue between the two below: `GpuTask` and `CpuTask`, one trait per sort (`layout`, which declares every row through a `Layout`, a node's rows of one table, and `run`, which computes through them: a GPU task's over device rows, its pinned rows as a `CopyOnly` only its copies take, and its stream; a CPU task's over host rows alone, pinned and heap, through a `Layout` and a `Bufs` of each; and a GPU task's `uses_libraries`, yes by default, no for the copy nodes and the mmd records' classes, without which a lane above 0 gets no library handles and no BLAS workspace row; and its `forks`, none by default, the most streams its `run` forks onto, whose maximum over a worker's tasks is the worker's fork streams), `Dag` (tasks added in plan order, each on a GPU worker, `add_gpu` on a lane the caller names (a `LaneId`, or a bare worker index for its lane 0), the worker's tables shared by its lanes so a task reads another lane's row in place, and `add_cpu` on the CPU pool, and with `device` the copy nodes `upload` and `download`, a pinned row to a new device row or back, one node named after the new row (`upload <row>`, `download <row>`) on the lane the caller names, and `download_rows`, several device rows of one element type down to pinned rows the caller names, in one node `download <label>` with one event for them all, and `upload_heap` and `download_heap`, a range of a heap row to a new device row or a device row into a range of a heap row, two nodes each through a pinned window of their own (`stage <row>` on the CPU and `upload <row>`; `download <row>` and `unstage <row>` on the CPU), iteration `k` of a loop taking the range moved on by `k` of its lengths, clamped to the heap row's end, so a counted loop of `total.div_ceil(len)` trips moves a ragged last range and its reader takes `len.min(total - k len)` from its iteration, so no task hand-writes a copy that is its whole work; each returns the task's rows, whose output ids the reading tasks take; `blas_workspace` declares a worker's BLAS workspaces, a worker row of `BLAS_WORKSPACE_BYTES` (32 MiB) for lane 0 and each further lane a task that uses the libraries is placed on, that admission counts and no task can name; `add_loop(name, worker, body, max_iterations, decision)` closes a loop over the tasks from position `body` (`len`) on, with a `Decision`, a CPU node with pinned and heap rows whose `decide` returns `ControlFlow`, the body's rows laid out in a region of the parent's tables and the rows it carries the parent's; `add_counted(body, trips, depth)` closes a counted loop over the tasks from `body` on, no decision, the body's rows held once per iteration in flight, a body task reading its iteration from its views to pick its batch, perhaps inside a decision loop's body, replayed whole every iteration), `Plan` (each worker's device and host tables and its lane count, the plan's one heap table, `heap_table`, `n_lanes`, and each task's dependencies, the reads and the handoffs, derived, a handoff between two lanes an event that serializes them; its text, a lane printed `stream<l>` on a worker of several, is the committed golden `core/planner/tests/golden/planner_scf.txt`, and `planner_scf_loop.txt` with a loop; `graph`, the plan as an owned `Graph` that outlives `instantiate`, whose `dot(record)` is Graphviz: each task's lane, dims and per kind the bytes it produces and those live while it runs, each edge labelled with the rows behind it, an ordering with none drawn only where no other path implies it, each loop a cluster, and with a run's `Record` each task's times, its golden `planner_scf_loop.dot`) and `Instance` (the blocks, each worker's two and the plan's heap block, and a `thread_pool::TaskGraph` whose nodes hand each task its rows in the iteration it runs in; `backend`, the `thread_pool::Device` built over the plan with each lane's declared BLAS workspace bound to its handle, once; `run(&mut self, &mut backend, n_cpu)`, any number of times and one at a time, each returning the executor's `Record` with every task under its own name and its `dims`, and the blocks' sizes). No `unsafe`. Across workers: `Dag::add_collective`, one step with a part on lane 0 of every GPU worker, each worker's collectives issued in plan order, the backend making the clique for a plan that holds one (`Plan::ranks`); `Dag::broadcast`, `Dag::gather` and `Dag::reduce` over it; and `Partition`, rows cut into a range per worker. Adaptive widths: `Adaptive` (`width_max`, `bytes(width)`, a pinned window's `pinned_bytes(width)`, a `warn_below` floor, `power_of_two`), `Dag::width` asked before a part's tasks, `Dag::plan_within(rooms, build)` planning a DAG declared by a fallible closure, each worker's `Room` (device and pinned bytes) asked once of `rooms` when its first part is measured, each part bisected once by `widest` in the room of each kind the schedule leaves at its position, a `width` line of the plan's text per part, logged at `debug`, at `warn` narrowed below its floor |
| `fb_core_thread_pool` | nothing: the C++ pipeline runs on one thread | The run's workers (`docs/programming_model.md`, "CPU workers"): `TaskGraph` (nodes added in topological order, each placed on the CPU pool or a lane of a GPU worker, a `LaneId`, `gpu<w>.<l>` in its log lines and record; `lanes`, the lanes it places per worker; `add_loop(name, body, deps, max_iterations, decide)` closes a loop over the nodes from `body` on, a sub-DAG replayed in the same run on the same workers while its CPU decision node, after every body node and `deps`, returns `ControlFlow::Continue`, and `MaxIterationsExceeded` past the bound; `add_counted(body, trips, depth)` a counted loop, the body run `trips` times with no decision, trip `k` after trip `k - depth` (each source after each sink), each node after itself in the trip before, a node after it free to follow a body node's last trip, perhaps in a decision loop's body, replayed whole every iteration; a node handed its innermost loop's iteration in its `Lane`), `run` (the plan and the backend borrowed `&mut`, so each is in one run at a time; one scope, `n_cpu` CPU workers and a thread per GPU worker that enqueues its ready nodes in rank order across all its lanes, a counted loop's trips back to back, and never waits on the device, an edge between lanes an event wait on the device and no sync point, readiness counts, failure skipping descendants, the lowest-ranked error returned, a panic re-raised rather than hung), and the `Backend` the edges leaving a stream wait through: `Device` (a stream per lane with its library handles bound to it, `DeviceStream`, a BLAS and a dense-solver handle on each lane `Device::new`'s `libraries` marks, made with the backend before any run and reached by a GPU node through its `Lane`; `DeviceStream::is_privileged`, true on lane 0 of worker 0 alone, the run's privileged lane, set by `Device::new` (`docs/programming_model.md`, "The run's devices"); each worker's fork streams, `Device::new`'s `forks`, beside its lanes and with no handles, over which `DeviceStream::fork_join(n, f)` runs a node's `n` items and joins them back into its stream before it returns, every wait on the device (`docs/programming_model.md`, "Forks inside a task"); `bind_blas_workspace`, through `&mut self`, so never during a run; an event pair per GPU node and per trip of its counted loop) or `HostOnly` for device-free tests. `fb_core_log::TaskScope` tags each node's lines `[task@worker]`, and `Error::context` carries it out. Every run returns a `Record` (`docs/diagnostics.md`, "The timing record"): its result, its wall time, and a `TaskTime` per node that ran, failed ones included, with its name, worker, dimensions (`TaskGraph::set_dims`), a loop node's iteration (one entry per iteration; a counted loop's node inside a decision loop, its trip and the decision loop's iteration as `outer`, one per pair), host time around the callable and, under `Device`, its stream scope's device time (an event pair on its stream, read after the run); a `SyncPoint` per host wait (a CPU node's on each GPU predecessor, a GPU worker's exit sync of each lane) naming the node waited for; each decision loop's iteration count, `Record::iterations`; the build profile; and the plan's `ArenaSize`s, which the planner adds. `Record::to_json` is the `timing.json` text, over [`serde`](https://crates.io/crates/serde), its schema in the type's rustdoc. `clippy.toml` makes it the one place that spawns. `Device::make_clique` gives lane 0 of each worker its rank (`DeviceStream::comm`) |
| `fb_core_types` | `src/core/types`, what the host modules name | `Double3`, `Vec<double, 3>`: `#[repr(C)]` with the backend's `double3` layout. The fixed-width integers and `make_vec` are the language's |
| `fb_core_primitives` | `src/core/primitives`, what the host modules name | `double_factorial_odd`, `factorial`, `binomial` as `const fn`, bitwise the C++ loops. `idivup` / `align_up` are `div_ceil` / `next_multiple_of`, and the element-wise functions `f64`'s methods |
| `fb_core_constants` | `src/core/constants`, what the host modules name | `BOHR_PER_ANGSTROM`, `kBohrPerAngstrom<double>`, asserted against the C++ double's bit pattern |
| `fb_atom` | `src/atom` | `Element` (`#[repr(u8)]`, Z as the discriminant, `Sentinel` = 0), `z` / `from_z`, `symbol` / `from_symbol` (case-insensitive, `Sentinel` for none), and `Atom`, a `Double3` position in bohr and an `Element` |
| `fb_molecule` | `src/molecule`, its host half | `Molecule` as host data (atoms, charge, multiplicity; no atoms aborts), `distance`, `nuclear_charge`, `electron_count`, and the nuclear repulsion energy, gradient and Hessian. The nuclei on the device (`view`, `NucleiView`, `molecule_footprint`) are `fb_molecule_nuclei`'s upload task |
| `fb_basis` | `src/basis` | `Shell` and `Primitive`, the counts (`function_count_cart` / `function_count_sphe`) and `MAX_ANGULAR_MOMENTUM` (3, the C++ `kMaxAngularMomentum`), `cartesian_powers` and `Ao`, `solid_harmonic_block`, `BasisSet` (its Cartesian and spherical AO layouts, the per-atom shell grouping, the name) and `spherical_transform_matrix(basis, u)`, U written into a slice the caller hands it (a pinned row, in `fb_recipes_matrix_ao`), `ShellStamp` with `primitive_normalized` and `place`, and `BasisStamp` with `place` onto a `Molecule`. Contractions are not renormalized, as in the C++ |
| `fb_recipes_sys` | `src/recipes/sort`'s merge sort, `src/recipes/orthogonalize`'s `inverse_sqrt`, `src/recipes/symmetrize`'s two tile kernels `src/recipes/cholesky`'s diagonal-tile kernel and `src/recipes/diis`'s bordered solve, forked to `recipes/kernels/` (with the device headers it includes, trimmed, in `device_headers/`) | The raw FFI over `libfb_recipes_kernels.so`, built by `CMakeLists.txt` here: `fb_merge_sort`, an `extern "C"` launch wrapper that only enqueues, its `fb_merge_sort_workspace_bytes` query, `fb_inverse_sqrt` (the eigenvalue map `λ → λ^(-1/2)`, 0 at or below a floor passed in, optionally scaled by a value on the device), `fb_symmetrize` and `fb_mirror_triangle` (the exact average `(A + Aᵀ)/2` and one strict triangle copied onto the other, in place, refusing a negative order, an `lda` below it or a null matrix), `fb_cholesky_diagonal` (one diagonal tile of a blocked Cholesky in shared memory, the flag naming the first non-positive minor, the tile also written packed for the panel's trsm) and its tile order `FB_CHOLESKY_TILE` (64), `fb_diis_solve` (the bordered DIIS system solved by Gauss-Jordan in one block's shared memory, the unit selector at the newest column and a flag when a pivot falls below `1e-14`, refusing the C++ doc comment's bounds and null pointers) and its order `FB_DIIS_SOLVE_MAX_DIM` (65), `fb_launch_status` and `FB_DEVICE_ALIGNMENT` (`docs/gpu_layer.md`, "Layering"), and `check_library()`, the load check of the library's ABI version and backend, compared by `fb-kernel-abi`. The bindings are generated from `recipes/kernels/fb_recipes.h` by `devtools/rust-bindings.py` and committed. Beside them, the safe calls a caller uses: `inverse_sqrt(stream, count, data, floor, floor_scale)`, `merge_sort(stream, keys, values, workspace)`, `merge_sort_keys(stream, keys, workspace)`, `merge_sort_workspace_bytes(n, with_values)`, `symmetrize(stream, n, a, lda)`, `mirror_triangle(stream, from_lower, n, a, lda)`, `cholesky_diagonal(stream, jb, first, a, lda, l11, info)` and `diis_solve(stream, m, ld, newest, gram, coeff, singular)`, the buffers `DeviceSliceMut`s whose lengths (a count within its slice and a `u32`, values as long as the keys, the workspace at least the query and aligned, a matrix holding its `n × n` block at `lda` with both within `i32`) are asserted in every build, the launch status a `Status` through `fb_kernel_abi::launch_status` |
| `fb_recipes_blas` | `src/recipes/blas`: `gemmwich` and `gmmwich`; its packed `gemm`, and `wwr::dgmm` as `gmmwich` calls it, private to the crate | The C++'s named compositions alone (`docs/gpu_layer.md`, "Layering": a single BLAS call is `fb_core_wwr::blas`'s, under that one name, every leading dimension passed, and no public function here shares a name with one, which `fb-doc-check`'s `tests/source_rules.rs` holds). `gemmwich::<OpX, OpF, _>`, the sandwich `C = OpX(X)ᴴ OpF(F) OpX(X)` as two packed gemms through a caller's `work`, and `gmmwich::<OpX, _>`, the same with F = diag(d) as a dgmm and a gemm, the diagonal never formed; each op a type parameter, the marker `N` or the scalar's adjoint (`T` on a real type, `C` on a complex one, `SandwichOp` over `BlasOp`, since stable const generics take no enum), a C too short for m × m a panic before anything is enqueued; the C++ square overloads are the same calls at `m = n`. Their scalars are `fb_core_wwr::blas::BlasScalar`'s `ONE` and `ZERO`, read in the handle's default (host) pointer mode, which no safe call changes, so there is no `ScopedPointerMode` to port. No `unsafe` (`#![forbid(unsafe_code)]`) |
| `fb_integrals_sys` | `src/integrals/mmd`'s launchers (records, bound, dressed bound; not its order, the records left in enumeration order), `src/integrals/one_electron/{overlap,kinetic,potential_en}`'s and `src/integrals/eri/density_fitting/metric`'s and its three-center launcher's shape, forked to `integrals/kernels/`, and the conventional ERI's, rust/'s own with no C++ launcher (`integrals/kernels/eri/conventional/`, the dense one's in its `dense/`), with the FelizJueves units they launch (`pair_bound`, `record_bound`, the overlap's, kinetic energy's and nuclear attraction's ten classes each, and the Boys routine `eval_boys.h` they and the density-fitting units include, emitted there by `tools/emit/*.py --out-root`; the metric's 28 classes, `tools/emit/df_kernels.py`'s; the packed three-center integrals' 70, `speri3<la, lb, lP>`, the aux index in real solid harmonics, `tools/emit/packed_three_center_kernels.py`'s, rust/'s own in `integrals/kernels/eri/density_fitting/three_center/packed/`; the packed full ERI's 21 classes through (dd\|dd) and its exchange gather `eri_kperm`, and the dense ERI's 81 ordered classes in each of its two layouts, `eri_dj` and `eri_dk`, FelizJueves's `eri/full_eri`, `tools/emit/conventional_kernels.py`'s) | The raw FFI over `libfb_integrals_kernels.so`: `fb_mmd_records`, `fb_mmd_bound`, `fb_ordered_pair_bound`, `fb_overlap_class`, `fb_kinetic_class`, `fb_potential_en_class`, `fb_metric_class`, `fb_three_center_packed_class` (the only (mn\|P): the square tensor's kernels, launchers and symmetrization left in #1725), `fb_conventional_eri_class`, `fb_conventional_exchange`, `fb_conventional_pack`, `fb_conventional_unpack`, `fb_dense_eri_class`, `fb_dense_row_dot`, `fb_packed_pack` and `fb_packed_gather`, each an `extern "C"` launch wrapper that only enqueues, a class wrapper refusing a class outside its table as `FB_LAUNCH_UNSUPPORTED`; the tables' limits (`FB_*_MAX_MOMENTUM`, the aux basis's `FB_METRIC_MAX_MOMENTUM` and `FB_THREE_CENTER_AUX_MAX_MOMENTUM` among them); the density-fitting record, pair and cell field counts (`FB_METRIC_*_FIELDS`, `FB_THREE_CENTER_AUX_FIELDS` and `FB_THREE_CENTER_PACKED_CELL_FIELDS` among them), the three-center ones read by its safe layer; and the four structs that cross by pointer, `fb_primitive_pair_class_view` (the one-electron launchers', and the three-center one's for its bra records), `fb_primitive_pair_source`, `fb_primitive_pair_records` and `fb_nuclei_view` (`src/molecule`'s `NucleiView`), `#[repr(C)]` with their layout asserted on both sides; and `check_library()`, as in `fb_recipes_sys`. The bindings are generated from `integrals/kernels/fb_integrals.h` and committed. Beside them, the safe layer: `RecordTables` (a `ShellTable` and the `ClassPairs`, every index the kernels follow asserted when made), `add_record_tasks`, the records as tasks of a `Dag` (`mmd tables` on the CPU, the tables into pinned rows; an `upload <table>` copy node per table on the worker's lane 0; and `mmd records (la,lb)` per class, the `c`-th on the worker's lane `c`, the first user of lanes, reading the uploaded tables for its records, bound and dressed bounds, with no scratch), returning `ClassRows`, each class's record fields as device output rows in enumeration order, each shell pair's records one run whose range is its prefix in the tables, a type only those tasks make, so a class launcher can trust the AO offsets it holds on the device; `RecordTables::mixed` and `add_mixed_record_tasks`, two bases' records, each offset checked inside its own basis's AOs, returned as `MixedClassRows`, which no one-basis launcher takes (`add_record_tasks` refuses two bases' tables, always); `Fields`, the record fields a family's kernels read (`OVERLAP`, `KINETIC`, `POTENTIAL_EN`, `THREE_CENTER`, and their `union`), which a reader's layout declares through `ClassRows::read` and no others, its launcher's view carrying those and null for the rest, each set held to its forked kernels by a scan of their sources; and `overlap_class`, `kinetic_class` and `potential_en_class` over a `ClassRows` read with their family's `Fields`, each adding into the matrix's lower triangle alone (row-major `m >= n`), which the task that launches a build's last class mirrors onto the upper, the output a `DeviceSliceMut` asserted `n_cart²` long and within `i32` in every build, the nuclei two `DeviceSlice`s (three coordinates per charge asserted), the class's momenta a contract, the status a `Status` through `fb_kernel_abi::launch_status`; and the metric's: `AuxTable`, an aux basis's shells and records (centre, exponent and weight `c (π/α)^{3/2}`) as the density-fitting kernels read them, built shell by shell with each block inside `n_cart` and `n_cart²` within `i32` asserted, and `add_metric_tasks`, `metric tables` on the CPU (the records and every aux shell pair P ≤ Q, formed from the table at planning, into pinned rows) and `metric` on the stream (both uploaded into scratch, one `fb_metric_class` per class into the Cartesian metric, an output row), so no device table a caller fills reaches the launch; and the packed three-center integrals': `add_packed_three_center_tasks`, over the same records, tables and aux table and a batch of shell pairs by their index among the tables' (a range of the packed map's kept pairs), `packed three-center tables` on the CPU (the aux records and the cells, each cell's destination its pair's batch-relative column times the aux basis's spherical count plus the aux shell's first spherical function and a diagonal flag, formed at planning) and `packed three-center` on the stream (uploaded, one `fb_three_center_packed_class` per class into an output row of `n_col · naux` doubles, aux fastest and spherical, each pair's Cartesian block in its own columns, a diagonal pair's whole; `n_col · naux` within `i32` asserted); and the conventional ERI's: `add_conventional_tasks`, over a basis's `ClassRows` and the `RecordTables` they were formed from (a contract checked always, as are `n_cart` within `FB_CONVENTIONAL_MAX_AO` = 150 and `n_pp²` within `i32`), `conventional tables` on the CPU (every class's shell pairs, the quartet classes' tasks, every unordered pair of shell pairs once, and the pair index `pidx` with its inverse, formed at planning, into pinned rows), `conventional eri` on the stream (uploaded, the tensor zeroed, one `fb_conventional_eri_class` per quartet class into `ERI_pp`, both sides' fields read in place through `Fields::CONVENTIONAL`) and `exchange gather` (`fb_conventional_exchange` into `X_pp`), each tensor an output row of `n_pp²` doubles, `n_pp = n_cart (n_cart + 1) / 2` (`packed_pairs`); and `pack` and `unpack`, a density packed for the GEMVs and a packed result unpacked, their lengths `#[requires]` in every build; and the dense conventional ERI's: `add_dense_tasks`, over the records between a basis and itself (`MixedClassRows`, every ordered shell pair) and the `RecordTables` they were formed from (a contract checked always, as are `n_cart²` within `i32` and a nonempty shell range), `dense tables` on the CPU (each launch's bra pairs, a batch of a sub-block's pairs of one class, every class's ket pairs and each ket class's task table, about 2^16 tasks however large the basis, formed at planning) and `dense eri` on the stream (both blocks zeroed, one `fb_dense_eri_class` per launch and layout), returning `DenseBlocks`: a worker's rows of `M_J[(m,n),(l,s)] = (mn\|ls)` and `M_K[(m,n),(l,s)] = (ml\|ns)` for its shells of `m`, output rows of `rows · n_cart²` doubles, cut into sub-blocks each within the kernels' 32-bit index; and `dense_row_dot`, a row-major sub-block by a vector, each entry in an order its width alone fixes, lengths `#[requires]`; and density fitting's packed shell-pair map's, rust/'s own (`integrals/kernels/eri/density_fitting/packed/`): `add_packed_tables`, `packed tables` on the CPU (the caller's fill, the map's `Packing::write_tables`, into pinned rows, then every `m` and `n` checked inside `[0, nao)`) and an `upload packed <table>` per table, returning `PackedTables`, a type only those tasks make, read by a task's layout through `PackedTables::read`, with `PackedTables::pack` (`fb_packed_pack`, a symmetric density into the map's columns, each times its weight) and `PackedTables::gather` (the slab zeroed, then `fb_packed_gather`, a packed batch of `b` aux functions into `b` square matrices, every element written, `(m, n)` and `(n, m)` the same bits; with `b = 1` the unpack of a J), lengths `#[requires]` in every build, each bitwise the host map's. The private `launched`, a launch status as a `Status` naming the wrapper and its class's momenta, is the one conversion every launcher here goes through |
| `fb_device_check` | nothing: a check on the kernel libraries | Tier-3 tests: no launch wrapper under `rust/` allocates, synchronizes, touches an event, copies, sets the device or opens a profiler range (a grep of every `.cu`), every `-sys` crate's committed bindings are what the script generates now, no header the bindings read names a block, grid or warp, and every emitted unit equals its C++ twin byte for byte |
| `fb_kernel_abi` | nothing: what the kernel libraries share above their `-sys` crates | The one Rust copy of `device_headers/fb_launch.h`, generated by `devtools/rust-bindings.py` and re-exported by each `-sys` crate's bindings (`docs/gpu_layer.md`, "From bridge to C ABI"): `fb_launch_status` and its values, `FB_DEVICE_ALIGNMENT`, `fb_backend` and its values. `LaunchStatus`, the newtype a launch wrapper returns, and `Backend`, a library's `fb_backend` read by value (`TryFrom`, `Display`, the build's name through `const fn named`). `check`, the load check's comparison: a library's ABI version and `fb_backend` against the ABI version and `Backend` its `-sys` crate expects, a mismatch of either refused naming both values. Each `-sys` crate's `check_library()` calls it. And `launch_status`, the one conversion of a launch wrapper's `fb_launch_status` to a `Status` (`docs/gpu_layer.md`, "Statuses"): a refusal as `Unsupported` or `InvalidConfig`, a positive `wwr_status` handed to the caller's `gpu_status_at`, any other value `GpuRuntime`, each naming the wrapper and the status. Every safe layer calls it (`fb_integrals_mmd::launch` for the integrals' class wrappers). It links nothing, so its tests are tier 3 |
| `fb_io_xyz` | `src/io/xyz` | `parse_xyz` / `read_xyz`: a standard XYZ geometry into a `Molecule`, Angstrom to bohr by `BOHR_PER_ANGSTROM`, every refusal (too short, a bad count, a malformed line, an unknown symbol, a count the lines disagree with, a file that cannot be read) an `Io` error. A field must parse as a whole token, where the C++'s `operator>>` reads a prefix |
| `fb_io_basis` | `src/io/basis` | `parse_basis` / `read_basis`: a BSE JSON document into a `BasisStamp` (sp blocks and general contractions split, spherical shells only: one Cartesian or ambiguous shell refuses the document), over [`serde_json`](https://crates.io/crates/serde_json); `is_builtin_basis` and `resolve_basis`, a name (spelling folded) or else a path, over the eleven sets of `data/basis/` compiled in with `include_str!`, only that branch naming the stamp; `is_builtin_aux_basis` and `resolve_aux_basis`, the same over the thirteen density-fitting sets of `data/aux_basis/`, a table of its own disjoint from the orbital one (about 16 MB, compiled in for the reason the orbital sets are: nothing is installed, so a name cannot be misplaced) |
| `fb_integrals_mmd` | `src/integrals/mmd`'s `primitive_pair_classes` and `mixed_primitive_pair_classes` | `add_records`: the kernel library's load check (`check_library()`, a mismatch the planning's `GpuRuntime` error), then the records of a `BasisSet`, built once per plan and passed to every one-electron reader: the shell table and each class's shell pairs with their record prefixes formed on the host, a `fb_integrals_sys::RecordTables` (`record_tables`, public, for a reader that needs each pair's record range at planning), and handed to `fb_integrals_sys::add_record_tasks`, which adds `mmd tables`, the tables' uploads and one `mmd records (la,lb)` per class on its own lane, and returns each class's `ClassRows`, in enumeration order: the C++ tree's descending-ζ sort is left out, since no reader here needs it (the crate doc has why). No `unsafe` (`#![forbid(unsafe_code)]`). Every size is the basis's, at planning; the plan text for H2O/cc-pVDZ and CO/cc-pVTZ is the committed golden `integrals/mmd/tests/golden/plan_*.txt`. `add_mixed_records`: the records between two bases, every ordered shell pair never swapped (classes with `la > lb`), `off_a` into the bra basis's Cartesian AOs and `off_b` into the ket basis's, over `fb_integrals_sys::RecordTables::mixed` and `add_mixed_record_tasks`, returning `MixedClassRows`; held bitwise to `add_records` where the two coincide (one basis against itself; a record of equal exponents within 4 ulps in its positions, the kernel fusing the bra's term at a tie), and its plan for water in cc-pVDZ against water in STO-3G at another geometry the committed golden `plan_mixed_h2o_ccpvdz_sto3g.txt` |
| `fb_integrals_one_electron_overlap` | `src/integrals/one_electron/overlap`'s `overlap`, and the Overlap step's spherical transform (`MatrixAOPair::cartesian_to_spherical`) | The first whole run through the model. `add_overlap`: S of a `BasisSet` as tasks after the records, their `ClassRows` passed in, `overlap` on the stream (the Cartesian matrix zeroed, one `fb_integrals_sys::overlap_class` per class into its lower triangle, in order, or each on a fork of the stream (`DeviceStream::fork_join`) when the records opt in (`ClassRows::fork`, off by default), reading `ClassRows` as an edge, then that triangle mirrored onto the upper by `fb_recipes_symmetrize::symmetrize_from`, so the Cartesian S is exactly symmetric), then `fb_recipes_matrix_ao`'s spherical transform, U's slot passed in so a plan stages U once. `Run`: an XYZ file and a basis name planned once (the records and their load check, through `add_records`, among it), laid out in one arena per kind, run on one stream, returning S and the timing record. No `unsafe` (`#![forbid(unsafe_code)]`). The plan text for each fixture is the committed golden `integrals/one_electron/overlap/tests/golden/plan_*.txt` |
| `fb_sad` | `src/sad`, its serial form | `sad_density`, the SAD guess: one spherically averaged, fractionally occupied atomic RHF per distinct (Z, shells) over closed-form one-center integrals, each density placed on its atoms' diagonal blocks, and `SadStampLookup`, the baked block that skips an element's SCF (checked by size). Every loop in the C++ order, so its stamps compare bitwise with the C++ generator's. `distinct_stamps` and `stamp_density` are public for the task form, which is `fb_pipeline_guess`'s |
| `fb_io_npy` | `src/io/npy` | `NpyArray` (a shape and its float64 elements, column-major) and `element_count`; `parse_npy` / `read_npy`, versions 1.0 to 3.0, `'<f8'` only, a C-order payload reordered, every refusal of a document (a bad magic, version, length or header field, another dtype, a payload the shape disagrees with, a file that cannot be read) an `Io` error and an overflowing shape `InvalidConfig`; `npy_header`, `npy_to_bytes` and `write_npy(path, shape, data)`, the data borrowed so a CPU node writes its row in place, version 1.0 with `np.save`'s canonical header, byte for byte the C++ writer's (the `data/sad/STO-3G/` stamps written back are their files) |
| `fb_recipes_matrix_ao` | `src/recipes/matrix_ao`'s `upload_spherical_transform_matrix` and `MatrixAOPair::cartesian_to_spherical` | `add_cartesian_to_spherical`: the spherical form `Uᵀ M_cart U` of a Cartesian AO matrix row as tasks of a `Dag`, `harmonics` on the CPU and `upload U`, the planner's copy node (U computed into a pinned row, then copied once into a device row, `add_harmonics`, which a task reading U itself calls too), added by the first call over a basis on a worker and shared through the caller's slot by every later one, and `spherical transform` on the stream (two `fb_core_wwr::blas::gemm`s reading U's device row on the stream worker's BLAS handle, the intermediate scratch). Every one-electron matrix calls it; `spherical_to_cartesian` is not ported yet. The GEMMs run on the worker's BLAS workspace when the plan declares one (`Dag::blas_workspace`). Its tier-4 case runs two plans of water in cc-pVTZ, one with an extra unused stream task, four times in one process and holds them bitwise equal, the guarantee a bitwise comparison of two plans rests on (on the gfx1200 hipBLASLt picks the transposed GEMM's kernel per process, #1575). No `unsafe` (`#![forbid(unsafe_code)]`) |
| `fb_io_sad` | `src/io/sad` | `builtin_sad_density`, `fb-sad`'s `SadStampLookup`: every `data/sad/<basis>/<Z>.npy` compiled in with `include_bytes!` (the table written by `build.rs`), found by the directory name compared exactly with `BasisSet::name` and the atomic number, parsed by `fb-io-npy`; bytes that are not a square matrix abort. `carries_sad_stamp` is the same membership without the parse, which the guess plan asks at planning. A directory naming no `data/basis` set fails a unit test, a stem that is not a three-digit Z the build |
| `fb_integrals_one_electron_kinetic` | `src/integrals/one_electron/kinetic`'s `kinetic` and its bridge's `kinetic_class` | `kinetic_classes`: every class's `fb_integrals_sys::kinetic_class` in order on the task's stream, or each on a fork of it joined back (`DeviceStream::fork_join`) when the records opt in (`ClassRows::fork`, off by default), into the lower triangle of a Cartesian matrix the caller zeroed and mirrors afterwards, callable by a task that launches several families into one matrix. `add_kinetic`: T of a `BasisSet` as tasks after the records, their `ClassRows` passed in, `kinetic` on the stream (the Cartesian matrix zeroed, then `kinetic_classes`, reading `ClassRows` as an edge, then the lower triangle mirrored onto the upper by `fb_recipes_symmetrize::symmetrize_from`), then `fb_recipes_matrix_ao`'s spherical transform, U's slot passed in. No `unsafe` (`#![forbid(unsafe_code)]`). The plan text for each of the overlap's fixtures is the committed golden `integrals/one_electron/kinetic/tests/golden/plan_*.txt` |
| `fb_pipeline_guess` | `src/pipeline/guess`, and `src/sad`'s task form | The Guess step as tasks of a `Dag`: `add_sad_guess`, one CPU node per distinct (Z, shells) stamp known from the basis at planning (`baked stamp <symbol>` when the basis carries a name and `fb-io-sad` carries the stamp, asked at planning by `carries_sad_stamp` without a parse, and parsed once, by its node; `atomic scf <symbol>`, `fb-sad`'s SCF, otherwise, so a path-read basis runs every one), each into a host row, then `sad guess` on the CPU placing them into D, bitwise the serial `sad_density`; the first failing stamp in stamp order is the run's error. `add_loaded_guess`, the step's `load`: one CPU node reading D from a `.npy` file. Both generic over the host kind and stream, so their tier 3 runs on `PinnedStandIn` rows and `HostOnly` workers; `add_guess_upload` copies D into a device row by the planner's copy node `upload D`. `GuessStep` and `GuessKind`, the config's `[guess]` table (serde, `type` or `from`, an unknown key refused), `check` refusing both keys together as `InvalidConfig`. The plan text for formaldehyde in cc-pVDZ, by name and by path, is the committed golden `pipeline/guess/tests/golden/plan_*.txt` |
| `fb_molecule_nuclei` | `src/molecule`, its device half (`Molecule`'s `upload`, `view`, `NucleiView`, `molecule_footprint`) | `add_nuclei`: a `Molecule`'s nuclei as three nodes of a `Dag`, `nuclei` on the CPU (each nucleus's charge and position, in atom order, into two pinned rows, `n` and `3 n` doubles) and the copy nodes `upload nuclei z` and `upload nuclei c` (each copied into a device row of its name), and `NucleiRows`, `n_nuc` and those two device rows, which a reader declares with `NucleiRows::read` and hands its launch wrapper as an `fb_nuclei_view` of their pointers. A crate beside `fb_molecule`, so that one links nothing. Every size is the molecule's, at planning; the plan text for H2O and CO is the committed golden `molecule/nuclei/tests/golden/plan_*.txt` |
| `fb_integrals_one_electron_potential_en` | `src/integrals/one_electron/potential_en`'s `nuclear_attraction` / `add_nuclear_attraction` and its bridge's `potential_en_class` | `potential_en_classes`: every class's `fb_integrals_sys::potential_en_class` in order on the task's stream, or each on a fork of it joined back (`DeviceStream::fork_join`) when the records opt in (`ClassRows::fork`, off by default), into the lower triangle of a Cartesian matrix the caller zeroed and mirrors afterwards, over `fb_molecule_nuclei`'s `NucleiRows`, callable by a task that launches several families into one matrix (after `kinetic_classes`, for T + V). `add_nuclear_attraction`: V of a `BasisSet` over the nuclei's rows as tasks after the records, their `ClassRows` passed in, `potential en` on the stream (the Cartesian matrix zeroed, then `potential_en_classes`, reading `ClassRows` and `NucleiRows` as edges, then the lower triangle mirrored onto the upper by `fb_recipes_symmetrize::symmetrize_from`), then `fb_recipes_matrix_ao`'s spherical transform, U's slot passed in; nuclei with no nucleus refused at planning as `InvalidConfig`, adding nothing. No `unsafe` (`#![forbid(unsafe_code)]`). The plan text for each of the overlap's fixtures is the committed golden `integrals/one_electron/potential_en/tests/golden/plan_*.txt` |
| `fb_integrals_one_electron_core_hamiltonian` | `src/integrals/one_electron/core_hamiltonian`'s `core_hamiltonian` | `add_core_hamiltonian`: H = T + V of a `BasisSet` over `fb_molecule_nuclei`'s `NucleiRows` as tasks after the records, their `ClassRows` passed in, one build task, `core hamiltonian`, on the stream (the Cartesian matrix zeroed, then `kinetic_classes` and `potential_en_classes` on top into its lower triangle, with `ClassRows::fork` set the T forks joined before the V forks are made (off by default: it benches slower on the gfx1200, the crate doc has the table), reading `ClassRows` and `NucleiRows` as edges, with no T or V row of its own, then the lower triangle mirrored onto the upper once, after the last V launch, by `fb_recipes_symmetrize::symmetrize_from`), then `fb_recipes_matrix_ao`'s spherical transform, U's slot passed in; nuclei with no nucleus refused at planning as `InvalidConfig`, adding nothing. The plan text for each of the overlap's fixtures is the committed golden `integrals/one_electron/core_hamiltonian/tests/golden/plan_*.txt` |
| `fb_pipeline_config`, and the `free-breakfast` binary | `src/pipeline/config`, `src/main.cpp` | The binary: `Config`, a serde type read from TOML by `read_config` (`calculation`, `output`, `num_gpus` (default 1, a `NonZeroUsize`, the run's devices `0..num_gpus` by `Config::devices`, a count above the visible devices refused, the runtime not asked at 1; above 1 spreading `[eri] storage = "dense"`'s row blocks over the devices, every other task on device 0, `docs/programming_model.md`, "The run's devices"), `[molecule] xyz`, `charge` and `multiplicity` (default 0 and 1), `[basis_set] basis`, `[guess] type` or `from`; each field's doc comment its default and unit; an unknown key refused `InvalidConfig` as `<file>:<line>: unknown field ...`; the geometry, a basis path and a guess's `from` joined to the config's directory; `[guess]`, `fb_pipeline_guess::GuessStep`, read by `"guess"`, `"rhf"` and `"localization"`, refused with both keys; `[eri] mode` and `aux_basis`, `fb_pipeline_eri::EriStep`, required by `"rhf"` and `"localization"`; `[scf]`, `fb_pipeline_scf::ScfStep`, read by both; `[localization]`, `fb_localization_pipek_mezey::LocalizationStep`, read by `"localization"`, `mode = "ibo"` refused `Unsupported`; which of these optional tables each calculation requires and accepts is one row per calculation, a table set beside a calculation that does not read it refused naming those that do, each table's own values checked by its type's `check`; `[memory]`, `MemoryCaps`, `max_device_mem_per_gpu` and `max_pinned_mem_per_gpu` in the C++ size strings, and the heap block's `max_host_mem`, each defaulted at its floor, 8 GiB, 2 GiB and 16 GiB, and a value under it refused), `plan`, the calculation's tasks as one `Plan` (`"overlap"`, S written as `overlap.npy`; `"guess"`, `fb_pipeline_guess`'s SAD or loaded D written as `d.npy`; `"rhf"`, `fb_pipeline_scf::add_scf` over `fb_pipeline_eri::add_eri` (`[eri] mode = "density_fitting"` or `"conventional"`) from that guess, ROHF at a multiplicity above 1, written as `scf.json` (energy, residual, iterations, `n_alpha`, `n_beta`, ε), `c.npy` and `d.npy`; `"localization"`, that RHF and its products, then `fb_localization_pipek_mezey::add_pm_localize` over its S and each occupied space of its C (an open shell's doubly then singly occupied columns, each among itself), each space sorted by `<i|F|i>` as the C++ step sorts it (`localization order` on the CPU, a stable sort per space written as a permutation matrix, applied by `localization sort`'s one gemm), written as `c_local.npy` (`n × n_alpha`) and `localization.json` (L before and after, the sweeps, summed over the spaces); `tests/open_localization.rs` holds H₂O⁺ and triplet CH₂ at exponents 2 and 4 to the C++ tree's PySCF reference (`tests/data/localization_rohf.json`, copied) at first-order bars from `tests/data/open_localization_reference.py`; `tests/schwarz_default.rs` pins `[eri] schwarz_threshold`'s default, 1e-12: benzene in cc-pVDZ / cc-pVDZ-JKFIT, the tree's largest density-fitted case, its DF-RHF energy at the default within 1e-10 hartree of its energy at 0 (a slow case); `plan` takes syevd's `lwork`, which the binary asks of device 0, and each worker's `Card`, its one reading of the card, from which `Dag::plan_within` sizes each adaptive width in `Card::room`, asked only of a worker holding a part), and the write nodes: `add_write_npy`, `download <file>` on the stream into a pinned row and `write <file>` on the CPU, a float64 `.npy` in Fortran order of the `[rows, cols]` it is given, feeding nothing, and `add_write_host_npy`, the write node alone for a product already a pinned row. `admit_plan` holds a plan to the caps: the workers' pinned caps against the host's available memory (`fb_core_arena::host_available`), then each worker's device table (`fb_core_arena::admit`) against its `Card::room`, the smaller of its cap and the card's total, less the reserve measured from that one reading (`Card::read`: `fb_core_arena::device_memory`, the total less the free memory, at least `DEVICE_RESERVE`) and the kernels' stack (`fb_core_wwr::runtime::stack_growth` for `KERNEL_STACK_BYTES`), and its pinned table against its cap, the pinned room, then the heap table against `max_host_mem` and the pinned blocks and the heap block together against the available memory (`fb_core_arena::admit_host`); the binary reads each card once, with its worker's stream and handles made, and hands the same reading to `plan` and `admit_plan`. `free-breakfast <config.toml>` (`src/main.rs`) plans, makes the output directory and writes the plan there, `plan.txt` its text and `plan.dot` its `Graph` (`Graph::dot`), so a plan admission refuses can be read, then admits, runs on the run's devices (worker `w` on device `w`) and writes `timing.json` (`Record::to_json`) and `timing.dot`, the graph with the run's times on it, beside the products; its `main` is the one place an error is logged, exiting with `ErrorKind::exit_code`. `rust/examples/` holds one config per calculation, `overlap.toml`, `guess.toml`, `rhf.toml`, `rohf.toml` (the water cation's doublet), `localization.toml`, `memory.toml` (the overlap under raised caps), `two_gpus.toml` (the overlap at `num_gpus = 2`, its plan the overlap's golden, run through the binary as a tier-5 case), `conventional.toml` (the RHF over the packed tensors) and `conventional_dense.toml` (over the dense storage at `num_gpus = 2`, a tier-5 case), run through the binary by `tests/examples.rs`; each example's plan text is the committed golden `pipeline/config/tests/golden/plan_*.txt` |
| `fb_tools_sad`, the `free-breakfast-sad-stamps` binary | `tools/sad` | The generator of `data/sad/`: `free-breakfast-sad-stamps <repo>/data (<basis>... \| --all)` resolves each named set through `fb-io-basis`'s compiled-in table (a set the binary does not carry refused `InvalidConfig`; `--all`, every `<data>/basis/*.json` by its document's BSE `name`), runs `fb-sad`'s atomic SCF for each element alone at the origin, one CPU node of a `thread_pool::TaskGraph` per element on `HostOnly`, and writes `<data>/sad/<basis>/<Z>.npy` through `fb-io-npy` after removing each regenerated set's directory (every one under `--all`); an atom too large for the SCF (`Unsupported`) is skipped, any other failure exits with its kind's code before anything is removed. It links `fb-sad`, never `fb-io-sad`, so it is not linked against its own output. Its STO-3G output is the committed `data/sad/STO-3G/` byte for byte, `tests/stamps.rs` |
| `fb_recipes_diagonalize` | `src/recipes/diagonalize` | `add_diagonalize`: the eigendecomposition `A = U diag(w) Uᵀ` of a symmetric device row as two tasks of a `Dag`, `syevd` on the stream (A copied into the output row U, then the legacy syevd through `fb_core_wwr::solver::syevd` on the stream worker's solver handle, lower triangle, eigenvectors over U, ascending eigenvalues into w, its work a scratch row of exactly `lwork` doubles, its flag a device scratch row copied into a pinned row) and `syevd check` on the CPU (a positive flag `Numerical`, a negative one `GpuRuntime`). `syevd_lwork`, `fb_core_wwr::solver::syevd_lwork` at its modes on a `SolverHandle`, which a run calls before planning and hands `add_diagonalize` as a number. `f64` only. No `unsafe` (`#![forbid(unsafe_code)]`). The plan text at `n = 24` is the committed golden `recipes/diagonalize/tests/golden/plan_n24.txt` |
| `fb_recipes_orthogonalize` | `src/recipes/orthogonalize`'s `orthogonalize` | `add_orthogonalize`: the symmetric orthogonalizer `X = S^(-1/2) = U Λ^(-1/2) Uᵀ` of a symmetric device row as tasks of a `Dag`, after `fb-recipes-sys`'s load check at planning: `fb_recipes_diagonalize`'s two (S copied into U, S left as it was), then `inverse root` on the stream (w copied into a scratch row Λ, `fb_recipes_sys::inverse_sqrt` over it, 0 at or below the floor, then `X = U diag(Λ) Uᵀ` into the output row X by `fb_recipes_blas::gmmwich::<T, f64>`, its dgmm through scratch M, on the stream worker's BLAS handle). No `unsafe` (`#![forbid(unsafe_code)]`). `ORTHOGONALIZE_EIGENVALUE_FLOOR`, the C++ `kOrthogonalizeEigenvalueFloor` (1e-10). `f64` only; cond(S) is not logged, and `orthonormalize` is not ported. The plan text at `n = 24` is the committed golden `recipes/orthogonalize/tests/golden/plan_n24.txt` |
| `fb_recipes_symmetrize` | `src/recipes/symmetrize`'s two `symmetrize` overloads | Calls a task makes inside its `run`, on its stream, not tasks: `symmetrize(stream, n, a, lda)`, `A ← (A + Aᵀ)/2` in place, `(i, j)` and `(j, i)` the same bits, `fb_recipes_sys`'s re-exported; and `symmetrize_from(stream, fill, n, a, lda)`, the strict triangle opposite `fill` overwritten with the transpose of `fill`'s, `fill` BLAS's `wwr_blas_fill_mode` so a syrk's passes straight through, over `fb_recipes_sys::mirror_triangle`. No `unsafe` (`#![forbid(unsafe_code)]`) |
| `fb_recipes_cholesky` | `src/recipes/cholesky` | `add_cholesky`: `A = L Lᵀ` of a symmetric positive definite device row as two tasks of a `Dag`, after `fb-recipes-sys`'s load check at planning, never the dense solver's `potrf` (rocSOLVER's small-matrix kernel races on its pivot, #404, #379): `cholesky` on the stream (A copied into the output row L, the flag zeroed, then right-looking and blocked, `CHOLESKY_BLOCK` (64) columns at a time: the diagonal tile by `fb_recipes_sys::cholesky_diagonal`, the panel by `fb_core_wwr::blas::trsm` against the tile's packed copy, the trailing matrix by `syrk`, L cut into disjoint parts by `split_at_mut`; the flag into a pinned row), and `cholesky check` on the CPU, a non-zero flag a `Numerical` error naming the 1-based leading minor. It returns L and the check's `checked` pinned row, which a task that must not run on a failed factor reads. No `unsafe` (`#![forbid(unsafe_code)]`) |
| `fb_recipes_diis` | `src/recipes/diis` | Pulay DIIS over opaque vectors of `len` doubles, its history rows a loop carries: `add_diis(dag, worker, name, len, max_history)`, before the loop, after `fb-recipes-sys`'s load check, adds `diis history` (stream), producing the ring of iterates and the ring of residuals (`len × max_history` each) and the Gram, and returns `Diis`; `Diis::add_push(dag, iterate, residual)`, in the loop's body, adds `diis push` (stream), which writes those rows as their last user (`Layout::write`): the pair copied into the ring's slot (`fb_core_wwr::copy::d2d`; the residual by a `geam` times the history's residual scale when it is not 1), the Gram's changed row and column by two `gemv`s, then with two or more pairs `fb_recipes_sys::diis_solve` and the extrapolating `gemv` into its own output row, else the iterate copied there; the singular flag into a pinned output row a CPU node reads. The ring's cursor is host state of the tasks, emptied by `diis history` on every run and by `Diis::reset`, from a loop's decision, between pushes; `Diis::restart(scale)`, from the same place, keeps the last pair and sets the scale every later residual is multiplied by, the next push moving the kept pair to column 0 and rescaling it first (the C++ SCF's singular restart, chosen over a relative pivot in the solve); `DIIS_MAX_HISTORY_LIMIT` (64) asserted against `FB_DIIS_SOLVE_MAX_DIM`. No `unsafe` (`#![forbid(unsafe_code)]`) |
| `fb_integrals_eri_density_fitting_metric` | `src/integrals/eri/density_fitting/metric`'s `metric` | `add_metric`: the spherical Coulomb metric (P\|Q) of a placed aux `BasisSet` as tasks of a `Dag`, after the kernel library's load check at planning (a shell above i, or `nao_cart²` past `i32`, refused there as `Unsupported`, adding nothing): `fb_integrals_sys::add_metric_tasks` (`metric tables`, `metric`), `fb_recipes_matrix_ao`'s spherical transform (U's slot passed in), and `metric average` on the stream (M copied into its own row and averaged with its transpose by `fb_recipes_symmetrize`, so `M[P][Q]` and `M[Q][P]` are the same bits). No `unsafe` (`#![forbid(unsafe_code)]`). Water in cc-pVDZ-JKFIT and iron beside water in def2-universal-JKFIT (aux through i) against PySCF's `int2c2e` at 1e-12 of max\|M\|; their plan text is the committed golden `integrals/eri/density_fitting/metric/tests/golden/plan_*.txt` |
| `fb_integrals_eri_density_fitting_fit` | `src/integrals/eri/density_fitting/fit`, the Cholesky path | `add_batch_fit(dag, worker, naux, columns, batch, factor, tensor)`: the fit `B = L⁻¹ T` of a batch of `batch` of `columns` columns of a three-center tensor (in a counted loop batch `k` the `k`-th, the last what is left, its count from the iteration) (`P + naux · c`, a packed batch of shell pairs) against a factor `fb_recipes_cholesky::add_cholesky` made once over the metric (`M = L Lᵀ`), as one task of a `Dag`, `density fit` on the stream (the tensor copied into B, an output row of its own, then one `fb_core_wwr::blas::trsm`, left, lower, no transpose), which reads the check's `checked` row beside L, so a metric that is not positive definite is the check's `Numerical`, naming the minor, and neither the fit nor any reader of B runs; each column solved alone, so a B built batch by batch is the one-batch B. The C++ square `density_fit` over `nao²` columns has no counterpart since #1721. `lindep`, the C++ fit over M's eigenmodes above a threshold (#630), is not ported. No `unsafe` (`#![forbid(unsafe_code)]`). The plan text at naux = 41, 25 columns is the committed golden `integrals/eri/density_fitting/fit/tests/golden/plan_naux41_nao5.txt` |
| `fb_integrals_eri_density_fitting_jk` | `src/integrals/eri/density_fitting/jk`, the closed-shell forms | `add_streamed_jk(dag, worker, packing, tables, naux, b, set) -> Result<Vec<RowId>>`: the streamed pass, every `Contraction` of `set` (`Coulomb(d)`, `Exchange { c, nocc }`) from one read of a packed B in a heap row (`fb_integrals_eri_density_fitting_pair_map`'s layout, `tables` its `fb_integrals_sys::PackedTables`), after both kernel libraries' load checks: `df jk pack` (each J's density packed, every accumulator zeroed), the aux batches as one counted loop of `naux.div_ceil(width)` trips, depth 2 (1 for a single trip; `Dag::upload_heap` on lane 1, the last range clamped to B's end, then `df jk batch` over the trip's functions: two gemvs per J over the packed batch, one gather to a square slab and a strided-batched gemm and a syrk per K), and `df jk finish` (each packed J unpacked by the gather, each K's lower triangle copied out and mirrored); results `"J <i>"` and `"K <i>"`, `nao²` each, exactly symmetric. The batch is an adaptive part, `df jk batch`, at most `BATCH_MAX` (64), a power of two, `warn` below 32, its bytes the ring's region. A pass may lie in a decision loop's body. `tests/streamed.rs` holds it on a card to a host J and K over B unpacked (every pair kept, and screened) at 1e-12 of max\|·\|, one pass of two J and two K bitwise two passes, narrowed to 16 and inside a decision loop; its plan text at naux = 150, nao = 9 is the golden `plan_streamed_naux150_nao9_nocc3.txt`; `tests/jk.rs` holds one pass over the committed fixture (naux = 23, nao = 7, nocc = 3) to the host's `Bᵀ (B vec(D))` and `Σ_Q B_Q D B_Q` at `100 naux nao ε max \|·\|`. The C++ resident `df_coulomb` and `df_exchange` have no counterpart since #1721; the two-factor K, the block forms and the symmetrized K are not ported. No `unsafe` (`#![forbid(unsafe_code)]`) |
| `fb_integrals_eri_density_fitting_three_center` | `src/integrals/eri/density_fitting/three_center`'s `three_center`, packed | `add_packed_three_center(dag, worker, basis, classes, aux, map, pairs, batch, u)`: the spherical three-center integrals (mn\|P) of a placed orbital `BasisSet` and a placed aux one over a range of a `PairMap`'s kept pairs, over the orbital records passed in and the plan's slot for U (`harmonics` and `upload U` added if it is empty; no U_aux, the kernels writing the aux index spherical), after the kernel library's load check at planning (an orbital shell above f, an aux shell above i, or a batch past `i32`, refused there as `Unsupported`, adding nothing), or in a counted loop of `pairs.len().div_ceil(batch)` trips batch `k` the `k`-th `batch` of the range, the last what is left, every batch a prefix of the first's classes pair for pair, a short one the front of the first's rows (`fb_integrals_eri_density_fitting_pair_map`), each pair's `(m, n)` once, no average: `fb_integrals_sys::add_packed_three_center_tasks`, then `fb_integrals_eri_density_fitting_transform::add_spherical_transform`, into `naux` × the batch's spherical columns, aux fastest, ready for `add_batch_fit`. Water in cc-pVDZ / cc-pVDZ-JKFIT, and the C++ tree's `df_high_aux` (all 70 classes through (ff\|i), a slow case), every pair as one batch, against PySCF's `int3c2e` at 1e-12 of max\|(mn\|P)\|; water's B over one batch and over two, at threshold 0, within 1e-12 of max\|B\| of the host's forward substitution over the same L and tensor, and the two bitwise the one. The C++ square tensor and its (m, n) average have no counterpart since #1721. `add_packed_b(dag, worker, basis, classes, aux, map, factor, u, u_aux)`: the build loop (`u_aux` neither read nor filled since #1736, kept for its callers), B into a heap row of `naux · n_columns` doubles, pair-fastest, per class a counted loop of ring depth 2 over batches of an adaptive width in Cartesian columns (`packed B batch`, the largest class's ring its bytes), a short last batch the loop's last trip, its count taken from the iteration, each trip the packed three-center tasks, `add_batch_fit` against `add_cholesky`'s factor, a `geam` transpose and `Dag::download_heap_pitched`, so no device row holds B whole; its planned width admitted in four rooms and a room short of one pair refused by admission naming the loops. No `unsafe` (`#![forbid(unsafe_code)]`). Water's plan texts are the committed goldens `integrals/eri/density_fitting/three_center/tests/golden/plan_h2o_packed_two_batches.txt` and `plan_h2o_packed_b.txt` |
| `fb_integrals_eri_conventional` | nothing: rust/'s own, with no C++ counterpart (FelizJueves's packed full ERI, `eri/full_eri`, its two-tensor design) | `add_conventional(dag, worker, basis, classes, u)`: the conventional integrals of a placed `BasisSet` stored once, as tasks of a `Dag`, after both kernel libraries' load checks at planning (a shell above d, or more than 150 Cartesian AOs, refused there as `Unsupported`, adding nothing): `fb_integrals_sys::add_conventional_tasks` over the records passed in and the tables they were formed from, returning `Conventional`, `ERI_pp[pidx(m,n), pidx(l,s)] = (mn\|ls)` and `X_pp[pidx(l,s), pidx(m,n)] = ((ml\|ns) + (ms\|nl))/2`, each an output row of `n_pp²` doubles over the Cartesian AOs (so admission sees both, 1.03 GB each at the cap), beside U's row (the plan's slot, `fb_recipes_matrix_ao::add_harmonics`). `add_conventional_jk(dag, worker, eri, density)`: J and K of a symmetric spherical D as one task, `conventional jk` on the stream: `D_cart = U D Uᵀ` by two gemms, `fb_integrals_sys::pack`, `J_pp = ERI_pp Dp` and `K_pp = X_pp Dp` by two `fb_core_wwr::blas::gemv`s, `unpack`, `Uᵀ · U` by two gemms each and `fb_recipes_symmetrize::symmetrize`, so J's and K's `(m, n)` and `(n, m)` are the same bits; J and K output rows named `"J"` and `"K"`. No `unsafe` (`#![forbid(unsafe_code)]`). H₂ and water in STO-3G and cc-pVDZ (water's with an O d shell) against PySCF's exact `get_jk` of the SAD guess at 1e-10 of max\|·\|, measured within 6e-15; water's plan text is the committed golden `integrals/eri/conventional/tests/golden/plan_h2o_cc-pvdz.txt`. `add_dense(dag, workers, basis, u)`: the dense storage over GPU workers `0..workers`, the shells of `m` cut at the boundaries nearest an even split of the rows, each worker with rows given the records between the basis and itself (`fb_integrals_mmd::add_mixed_records`) and its row blocks (`fb_integrals_sys::add_dense_tasks`), returning `Dense` (each worker's `DenseBlocks`, the rows' `Partition`, U on worker 0); a shell above d, or `n_cart²` past `i32`, refused at planning. `add_dense_jk(dag, eri, density)`: `dense density` on worker 0 (`D_cart = U D Uᵀ`), `broadcast D cartesian`, a `dense jk` per worker (its sub-blocks by `vec(D_cart)`, `fb_integrals_sys::dense_row_dot`), `gather J cartesian` and `gather K cartesian` onto worker 0, and `dense spherical` there (`Uᵀ · U` and `symmetrize`), J and K bitwise the same on one worker or several. Water in cc-pVDZ against PySCF's exact `get_jk` on one card (tier 4) and bitwise one card's on two and four (tier 5), `tests/dense.rs`; its two-worker plan the golden `plan_h2o_cc-pvdz_dense_2.txt` |
| `fb_pipeline_scf` | `src/pipeline/scf`'s density-fitted path, closed or open shell: `scf_loop` with a `DfFockBuilder`, `solve_fock` (the Roothaan step), the densities, `electronic_energy`, `commutator`, and ROHF's `roothaan_fock` and `canonicalize_open` | `add_solve_fock`: `F C = S C ε` through `X = S^(-1/2)` for the rows of a symmetric operator F and of X, as tasks of a `Dag`: `fock transform` on the stream (`F' = X F X`, two gemms through a scratch row), `fb_recipes_diagonalize`'s two on F', and `back transform` on the stream (`C = X C'`), ε and C output rows, so the SCF's Fock matrix takes the core Hamiltonian's place later with no change. One closed-shell iteration's dense nodes, shaped as a loop body: `occupied_orbitals(molecule) -> (n_alpha, n_beta)`, the electron count split by the multiplicity at planning, a count of the wrong parity, a multiplicity out of range or a non-positive count `InvalidConfig`; `add_density(dag, worker, n, nocc, c, d, c_occ)`, `density` on the stream, C's first `nocc` columns copied into the carried factor `c_occ` by a `fb_core_wwr::blas::geam` and `D = 2 C_occ C_occᵀ` into the carried `d` by a gemm, both rows written (`Layout::write`), so it sits in a loop that carries them; `add_fock(dag, worker, n, e_nuc, h, j, k, d, s, x)`, returning F, e and a pinned row `E, |e|`: `fock` on the stream (`F = (H + J) - K/2` by two geams, `tr(D H)/2 + tr(D F)/2` by two gemvs down to a pinned row), `residual` on the stream (`M = F D S` by two gemms, `M - Mᵀ` by a geam, `e = Xᵀ (M - Mᵀ) X` by two gemms, `‖e‖²` by a gemv down to a pinned row), and `energy and norm` on the CPU (`E = tr/2 + E_nuc`, `‖e‖_F` the root). J and K are input rows. `add_scf(dag, worker, step, molecule, n, lwork, eri, [h, s, x], guess) -> (scalars, [ε, C, D])`, the DF-RHF loop, or the RHF over the conventional integrals when `eri` is `fb_pipeline_eri::Eri::Conventional` (each build's J and K one `conventional jk` of the carried D, with no rest's K and no `guess exchange`; an open shell refused at planning, `Unsupported`): before it, the guess's factor (`syevd`, `w` down to a pinned row by the copy node `download guess w host`, which `guess check` reads on the CPU for the floor, 1e-12 of the largest eigenvalue as in the C++, warning of an eigenvalue below minus that floor as the C++ does, then `guess factor`, `(w/2)^(1/2)` on the device through `fb_recipes_sys::inverse_sqrt` at that floor, copied to the device and read by the kernel there), its largest `nocc` columns the carried occupied factor and the other `n - nocc` one K (a streamed pass, `fb_integrals_eri_density_fitting_jk::add_streamed_jk`, over the packed B), DIIS's rings and the decision's carried last energy; the body, one streamed pass (J of D and K of the factor, B crossing the bus once), `guess exchange` (the rest's K added to the first iteration's and zeroed), `fock`, `residual`, `energy and norm`, `diis push`, the Roothaan step on the extrapolated F and `density`; the decision `converged` (`|ΔE| < energy_tolerance` and `‖e‖ < commutator_tolerance` from the second iteration, a singular DIIS subspace restarted from its newest pair at residual scale `1/‖e‖` by `Diis::restart`, each iteration logged at `info`); after it, the Roothaan step on F as built and `published density`; an ion's guess rescaled to its electron count first (`guess rescale`). At a multiplicity above 1, ROHF: the guess's alpha and beta factors and each rest's K, `spin densities` (`Da = Db = D/2`) and `identity` before the loop; in the body, the beta K in the body's pass, `open fock` (Fa, Fb and `tr(D H + Da Fa + Db Fb)/2`) and `roothaan fock` (the Roothaan effective F, its open block `(Fa + 3 Fb)/4`) in place of `fock`, and `open density` in place of `density`; `built densities` keeping each build's Da and Db; after it, the C++ `publish_open`: `open block`, its eigensolve and `canonical open`, the open orbitals rotated to diagonalize `(Fa + Fb)/2` so every energy is PySCF's ROHF `mo_energy`, and `pyscf roothaan fock` (`w = 1/2`, at the last build's densities) with the Roothaan step on it, `open order` (CPU) testing whether the first set's energies ascend and `publish open` keeping PySCF's matrix's orbitals if they do, the first set if not, bit for bit by weights; with no beta electron (H, H₂⁺), no beta factor and no beta K in the body, the first build's Kb the guess's beta rest's and Db zeroed after. `ScfStep`, the C++ step's `max_iterations`, `energy_tolerance`, `commutator_tolerance` and `diis_history`, refused out of range as `InvalidConfig` (`check`), and the config's `[scf]` table (serde, every key at its default, an unknown one refused). `add_scf`'s scalars are `[E, ‖e‖, builds]` of the last Fock build, a heap row the decision carries. `Run`, a molecule, a basis and an `[eri]` step (`fb_pipeline_eri::EriStep`) planned once from the SAD guess over `gpus` GPU workers (setup, X, `add_eri`, the guess and its upload, `add_scf`, the copy nodes `download epsilon`, `download C` and `download D`, and a collect), run on a device per worker (`Run::run(&devices)`), returning the record and an `ScfOutput` (E, ‖e‖, the iteration count, `n_alpha`, `n_beta`, ε, C, D). The plan text for each of the overlap's fixtures' core-Hamiltonian orbitals (a plan the tests build) is the committed golden `pipeline/scf/tests/golden/plan_*.txt`, the iteration's as a loop body at water/cc-pVDZ's shape `plan_iteration_h2o_cc-pvdz.txt`, water's whole DF-RHF run `plan_scf_h2o_cc-pvdz_cc-pvdz-jkfit.txt`, and its cation's DF-ROHF run `plan_rohf_h2o_cation_cc-pvdz_cc-pvdz-jkfit.txt`, and water's RHF over the conventional integrals `plan_scf_h2o_cc-pvdz_conventional.txt`, and over the dense storage on two workers `plan_scf_h2o_cc-pvdz_dense_2.txt`. Over `Eri::Dense` each build's J and K are `fb_integrals_eri_conventional::add_dense_jk`'s tasks, the loop's body spanning every worker, closed shell only; `tests/dense.rs` holds water and glycine chains (gly5 in STO-3G and 6-31G, gly10 in STO-3G, 122 to 237 AOs) to PySCF's exact RHF on one, two and four cards and prints each run's times |
| `fb_pipeline_eri` | `src/pipeline/eri`'s density-fitted branch, `DensityFitted`; and a conventional mode the C++ does not have | `add_eri(dag, worker, gpus, molecule, basis, classes, u, step) -> Eri`: the step `[eri] mode` names, `EriMode::DensityFitting`'s `add_df_eri` or `EriMode::Conventional`'s `fb_integrals_eri_conventional::add_conventional` (named for the stored-integral SCF that integral-direct is the alternative to), or with `[eri] storage = "dense"` (`EriStorage::Dense`; `"packed"` the default, `storage` refused under `"density_fitting"`) `add_dense` over GPU workers `0..gpus`, `Eri::Dense`, after `EriStep::check`. `add_df_eri(dag, worker, molecule, basis, classes, u, aux_basis, schwarz_threshold)`: the ERI step as tasks of a `Dag`, B built once for every reader, over the plan's slot `u` for the orbital basis's U (`fb_recipes_matrix_ao::add_harmonics`). At planning the fitting set is resolved by its compiled-in name (`fb_io_basis::resolve_aux_basis`; any other name `InvalidConfig`, never read as a path) and placed on the molecule as the orbital basis is, and the screen formed: each shell pair of `fb_integrals_mmd::record_tables` bounded by the largest element of its host Schwarz diagonal (`fb_integrals_schwarz::schwarz_diagonal`, not formed at threshold 0) and kept by `PairMap::new` when not below `schwarz_threshold` (a threshold dropping every pair `Unsupported`); then `add_metric`, `fb_recipes_cholesky::add_cholesky` and `fb_integrals_eri_density_fitting_three_center::add_packed_b` (U_aux staged once), and the map's tables once (`fb_integrals_sys::add_packed_tables`), returning `DensityFitted`: B, a heap row of `naux · n_columns` doubles pair-fastest, L (`naux²`, a device output row), the placed aux basis, the `PairMap` and its `PackedTables`. No device row holds B; L and the batches' rings are admission's, the build's batch width the widest that fits the room. `EriStep` and `EriMode`, the config's `[eri]` table (serde, `mode` required, `aux_basis` required under `"density_fitting"` and `schwarz_threshold` optional there, 1e-12 when absent (non-negative, 0 screening nothing; a dropped column of B has 2-norm below it, the fit a Coulomb-metric projection), both refused under `"conventional"` by `check`, an unknown key refused), `check` refusing a fitting set not compiled in as `InvalidConfig`; `EriStep::density_fitting(aux, schwarz_threshold)`, the table that mode reads into. Integral-direct, the `<basis>-acCD-6-6` default and an aux basis by path are not ported. No `unsafe` (`#![forbid(unsafe_code)]`). Water in cc-pVDZ / cc-pVDZ-JKFIT at threshold 0: J and K of the committed SAD guess by one streamed pass over B against PySCF's DF `get_jk` at 1e-10 of max\|·\|; two waters 12 Å apart at 1e-8: the kept pairs exactly those the host bound keeps, and each dropped pair's columns of the unscreened B below the threshold; its plan text is the committed golden `pipeline/eri/tests/golden/plan_h2o_cc-pvdz_cc-pvdz-jkfit.txt` |
| `fb_localization_sys` | `src/localization/pipek_mezey`'s launchers (`pipek_mezey_bridge.h`, `pipek_mezey_kernels.cu`), forked to `localization/kernels/` (with the block and warp reductions it includes, forked into `device_headers/`) | The raw FFI over `libfb_localization_kernels.so`, built by `CMakeLists.txt` here: `fb_pm_populations`, `fb_pm_pair_angles`, `fb_pm_rotate_populations`, `fb_pm_rotate_columns`, `fb_pm_functional` and `fb_pm_identity`, the C++ launchers as `extern "C"` wrappers that only enqueue and refuse a count below 1, an `n_occ` past `FB_PM_MAX_OCCUPIED` (65535), an exponent other than 2 or 4 and a null pointer; and `check_library()`. Three of the launches trust indices in device memory (the atoms' AO offsets, a round's orbital pairs, the partners the pairs give), so the safe layer is two builders whose tasks stage those indices themselves: `add_pm_start(dag, worker, atom_ao_offsets, n_ao, n_occ, exponent, s, c)`, before a loop, checks the offsets at planning and adds `pm tables` (CPU: the offsets and the round-robin schedule into pinned rows), their two uploads and `pm start` (stream: `W = S C`, the population tensor, `U = I`, and `L` into a pinned row), returning a `PmRows` whose index rows are private; `add_pm_sweep(dag, worker, rows, gain_threshold)`, in the body, adds `pm sweep` (stream: every round's three launches between the carried rows and scratch of its own, the result copied back, then `L` into a pinned row) |
| `fb_localization_pipek_mezey` | `src/localization/pipek_mezey`'s `pm_localize`, `PmOptions`, `PmExponent` and `Info` | `add_pm_localize(dag, worker, options, basis, n_occ, s, c)`: Pipek-Mezey localization of the first `n_occ` orbitals of C over Mulliken populations, `L_p = Σ_i Σ_A (Q^A_ii)^p` with `p` 2 or 4 (`PmExponent`), after the kernel library's load check at planning (a basis with no atoms, `n_occ` outside `1..=min(n_ao, 65535)` or `max_sweeps` 0 `InvalidConfig`, adding nothing): `fb_localization_sys::add_pm_start`, `pm before` (CPU: the carried heap row `[L before, L, sweeps]`), the loop of `add_pm_sweep` and `pm converged` (CPU: `L` and the count recorded, converged when a sweep moves `L` by less than `functional_tolerance`, `MaxIterationsExceeded` at `max_sweeps`), no loop for one orbital, and `pm rotate` (stream: `C_local = C U` by a gemm); returns the heap row and the device rows `C_local` (`n_ao × n_occ`) and `U` (`n_occ²`). `PmOptions` holds the C++ defaults (64 sweeps, 1e-10, 1e-14, `p = 2`). `LocalizationStep`, the config's `[localization]` table (`mode`, `max_sweeps`, `functional_tolerance`, `rotation_gain_threshold`, `exponent`, serde, at those defaults), and its `check`: `LocalizationMode::Ibo` `Unsupported`, `max_sweeps` 0 or a negative tolerance `InvalidConfig`; an `exponent` other than 2 or 4 refused as it is read. No `unsafe` (`#![forbid(unsafe_code)]`) |
| `fb_integrals_eri_density_fitting_pair_map` | nothing: rust/'s own, the C++ B being square and resident | `PairMap::new(tables, bounds, threshold)`, built at planning from a basis's `fb_integrals_mmd::record_tables`: the kept shell pairs (`KeptPair`, a pair dropped whole when its bound, the largest Cartesian `sqrt((mn\|mn))` of its block, is below the threshold; every pair with no bounds or at 0), in the tables' class order so each class's blocks (`KeptClass`) are contiguous and of one shape, each pair's `PairBlock` (its shells' AO offsets and sizes and its first column) in the Cartesian and the spherical packing (`Packing::blocks`, a diagonal shell pair's block its full square), and `Packing::write_tables`, the per-column rows, columns and weights (1 on a diagonal pair's block, 2 elsewhere) the device pack and gather read (`fb_integrals_sys::add_packed_tables` takes it as its fill), written into a caller's slices (a CPU task's pinned rows), with the host `pack` and `unpack` they are held to, bitwise, on a card; the crate doc is the one definition of B's heap layout, pair-fastest, `(column, Q)` at `column + n_columns · Q`. Every count fits `i32` by `RecordTables`' own bound, so nothing is refused. No `unsafe` (`#![forbid(unsafe_code)]`). Random bases' maps (pack then unpack the identity bitwise, each covered `(m, n)` one column, every pair kept unscreened, a higher threshold a subset) and the committed golden `integrals/eri/density_fitting/pair_map/tests/golden/map_psd.txt`; tier 4 (`tests/device.rs`): a screened basis's Cartesian and spherical packings through `PackedTables`, pack D, a batch's gather, unpack J and unpack of a host pack each bitwise the host's |
| `fb_integrals_schwarz` | nothing: rust/'s own, with no C++ counterpart (the C++ screens on the device records' bounds) | `schwarz_diagonal(basis, pairs)`: the Cartesian Schwarz diagonal `sqrt((mn|mn))` of a placed `BasisSet`, per shell pair the caller lists (the screen's caller lists `fb_integrals_mmd::record_tables`' `ClassPairs` in order), each pair's `ncart_a · ncart_b` block `m` slower, by McMurchie–Davidson on the host, general in `l` (Hermite E per primitive pair, the Boys function by its series and downward recursion, the R recursion, each primitive quartet and its swap once), in the device kernels' Cartesian normalisation, nothing transformed: a deliberate host-only path, run at planning so the density-fitting screen can size the plan before admission. The diagonal calls into no library; the tier-4 test's device B makes the crate link both kernel libraries, its `build.rs` giving the tests their rpath as its neighbours' do. No `unsafe` (`#![forbid(unsafe_code)]`). Water in cc-pVTZ (every class through (ff\|ff)) against PySCF's `int2e_cart` at 1e-10 relative, measured within 1.6e-14; and on a card, water in cc-pVDZ / cc-pVDZ-JKFIT, the device's fitted B within the diagonal through U, `Σ_Q B_Q,μν² <= (Σ_mn |U_mμ U_nν| sqrt((mn|mn)))²`, the Cartesian projection inequality itself on the s and p shells |
| `fb_integrals_eri_density_fitting_transform` | nothing: rust/'s own, the C++ (mn\|P) being square and its transform `fb_integrals_sys`'s three dense GEMMs | `add_spherical_transform(dag, worker, map, pairs, batch, n_aux, cartesian, u)`: the orbital half of the Cartesian-to-spherical transform of a packed three-centre batch, the aux index arriving spherical from the kernels, of the kept pairs `pairs` of a `PairMap` in batches of `batch` (in a counted loop batch `k` the `k`-th, the last what is left, a prefix of the first's blocks, its counts from the iteration), as one task of a `Dag`, `spherical transform` on the worker's stream: the input `n_aux` × the batch's Cartesian packed columns, aux fastest, an earlier task's output, into an output row `n_aux` × its spherical packed columns, aux fastest; per `(la, lb)` class with a pair in the batch two strided-batched GEMMs over its contiguous blocks (`fb_core_wwr::blas`'s `gemm_strided_batched`), the ket index then the bra, U's block of each momentum read in place from the plan's U row (`fb_recipes_matrix_ao::add_harmonics`'), stride 0; the intermediate a scratch row, so a batch that does not fit is admission's; a GEMM count past `i32` refused at planning as `Unsupported`, adding nothing. No `unsafe` (`#![forbid(unsafe_code)]`). Random symmetric Cartesian input over a basis through f, an aux one through g, a screened map and a batch cutting its first and last classes, against the pack of the dense `Uᵀ X_Q U` at 1e-12 of max\|·\|; an s, p, d basis's plan text is the committed golden `integrals/eri/density_fitting/transform/tests/golden/plan_spd_naux25.txt` |

Two more crates hold no port: `fb-test-support` (`test_support/`), what
more than one crate's unit tests use, and a dev-dependency only; and
`fb-doc-check` (`doc_check/`), the section-citation, `unsafe`, contract and
source-rule checks, the code-rule scans, and the workspace graph test. There is no umbrella crate. The crate graph is the selection graph
([`testing.md`](docs/testing.md), "Selection: the workspace graph is the
selection graph"): `log` at the bottom, `error_handling` on it, `wwr` on
both, then `arena` and `thread_pool` on `wwr`, `planner` on those two, and
`diagnostics` at the top with nothing in `core` depending on it, since
`log`'s macros are imported from `log` directly and `critical!` lives in
`fb-core-log`. Above `core`, `atom` stands on `types`, and `molecule` on
`atom`, `types` and `error_handling`, and `basis` on `molecule`, `types`,
`primitives` and `error_handling`; `io/xyz` on `molecule`, `atom`, `types`,
`constants` and `error_handling`, and `io/basis` on `basis` and
`error_handling`; `integrals/mmd` on `basis`, `planner`, `error_handling` and
`integrals/sys`; `recipes/sys` on `kernel_abi`, and `integrals/sys` on
`kernel_abi`, `planner`, `arena`, `thread_pool`, `wwr` and
`error_handling`, and `kernel_abi` on `error_handling`; `recipes/blas` on `wwr` and `error_handling`; and
`integrals/one_electron/overlap` on `integrals/mmd` and what it stands on,
`io/xyz` and `io/basis`; `sad` on `basis`, `molecule`, `atom`,
`primitives` and `error_handling`; `io/npy` on `error_handling`; and
`recipes/matrix_ao` on `basis`,
`planner`, `arena`, `thread_pool`, `wwr` and `error_handling`, with
`integrals/one_electron/overlap` on it and `recipes/symmetrize` too; `io/sad` on `io/npy` and
`error_handling`; and `integrals/one_electron/kinetic` on `integrals/mmd`,
`recipes/matrix_ao`, `recipes/symmetrize` and what they stand on; `pipeline/guess` on `sad`,
`io/sad`, `io/npy`, `basis`, `atom`, `planner`, `arena`, `thread_pool`,
`wwr` and `error_handling`; and `molecule/nuclei` on `molecule`,
`planner`, `arena`, `thread_pool`, `wwr` and `error_handling`;
`integrals/one_electron/potential_en` on `integrals/mmd`,
`molecule/nuclei`, `recipes/matrix_ao`, `recipes/symmetrize` and what they stand on; and
`integrals/one_electron/core_hamiltonian` on `integrals/one_electron/kinetic`,
`integrals/one_electron/potential_en`, `recipes/symmetrize` and what they stand on; and
`pipeline/config` on `integrals/one_electron/overlap`, `io/xyz`,
`io/basis`, `io/npy`, `log`, `planner`, `arena`, `thread_pool`, `wwr` and
`error_handling`; and `tools/sad` on `sad`, `io/basis`, `io/npy`, `basis`,
`atom`, `types`, `thread_pool` (without `device`) and `error_handling`;
and `recipes/diagonalize` on `planner`, `arena`, `thread_pool`, `wwr` and
`error_handling`; and `recipes/orthogonalize` on `recipes/diagonalize`,
`recipes/blas`, `recipes/sys`, `kernel_abi` and what they stand on; and `pipeline/scf` on
`integrals/one_electron/overlap`, `integrals/one_electron/core_hamiltonian`,
`recipes/orthogonalize` and what they stand on; and `recipes/symmetrize` on
`recipes/sys`, `wwr` and `error_handling`; and `recipes/cholesky` on
`recipes/sys`, `planner`, `arena`, `thread_pool`, `wwr` and
`error_handling`; and
`integrals/eri/density_fitting/metric` on `integrals/sys`,
`recipes/matrix_ao`, `recipes/symmetrize`, `basis` and what they stand on; and
`integrals/eri/density_fitting/fit` on
`recipes/cholesky` and what it stands on; and
`integrals/eri/density_fitting/jk` on `recipes/symmetrize`, `recipes/sys`,
`integrals/eri/density_fitting/pair_map`, `integrals/sys`,
`planner`, `arena`, `thread_pool`, `wwr` and `error_handling`; and
`integrals/eri/density_fitting/three_center` on `integrals/mmd`,
`integrals/sys`, `recipes/matrix_ao`, `basis`,
`integrals/eri/density_fitting/{pair_map,transform,fit}` and what they
stand on; and
`integrals/eri/conventional` on `integrals/mmd`, `integrals/sys`,
`recipes/matrix_ao`, `recipes/symmetrize`, `recipes/sys` and what they
stand on; and `pipeline/eri` on
`integrals/eri/density_fitting/{metric,three_center,fit}`,
`integrals/eri/conventional`, `io/basis`,
`molecule` and what they stand on; and `pipeline/scf` on
`integrals/eri/conventional` and `integrals/eri/density_fitting/jk` too;
and `integrals/eri/density_fitting/pair_map` on `integrals/sys` and what
it stands on; and `integrals/eri/density_fitting/transform` on
`integrals/eri/density_fitting/pair_map`, `planner`, `arena`,
`thread_pool`, `wwr` and `error_handling`.

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
  per plan ([`memory.md`](docs/memory.md), "The three kinds").
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

Preconditions are checked with the [`contracts`](https://crates.io/crates/contracts)
crate, and rustdoc shows each as its item's `§ Contracts` section. A
`#[debug_requires]` runs in debug builds only; a check soundness rests on
is always on, a `#[requires]` on the arguments, its condition written with
`fb_core_log`'s value checks so a tripped one prints its operands
(`#[requires(le!(count, data.len()), "count is within data")]`), and a
body `assert!` named in `# Panics` only for state the body produces, a
recomputation too long for an attribute, or a helper called once per operand,
naming it ([`testing.md`](docs/testing.md), "Contracts are part of the spec").

Outcomes of GPU calls are deliberately not contracts. A call that can fail
returns a status or a `Result`, and a failure is an expected result, not a
broken promise; a contract would turn it into a panic.

## Build and test

```bash
cd rust
cmake --preset cuda                    # or hip; the backend has no default
cmake --build --preset cuda            # libwwr_capi_{runtime_api,blas,solver,ccl}.so and the kernel libraries
cargo test --workspace -- --include-ignored  # debug, device cases and contract tests
cargo test --workspace --release       # the debug contracts' tests are skipped
cargo test -p fb-core-arena            # one crate, with only its own features
```

`Cargo.lock` is tracked, as the repository's `uv.lock` is: it is the one
place the crates' versions are written down, and every cargo call of the
gates and the commit hook passes `--locked`, so a manifest edit that leaves
the lock behind fails there instead of resolving whatever crates.io serves
that day ([`testing.md`](docs/testing.md), "What runs when"). After
changing a requirement in a `Cargo.toml`, run any cargo command without
`--locked` (`cargo metadata` is the quickest) and commit the lock with it.
A `[workspace.dependencies]` requirement is a plain caret one; the lock,
not an `=` pin, holds the exact version.

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
  -p fb-io-xyz -p fb-io-basis -p fb-kernel-abi -p fb-sad \
  -p fb-io-npy -p fb-io-sad -p fb-tools-sad --no-default-features
```

`CMakeLists.txt` here is a CMake project of its own, which the root
`CMakeLists.txt` does not add: it configures `deps/WarpWraps` alone (its
tests, examples and install rules off, its collectives on) and builds
`wwr.capi.runtime_api`, `wwr.capi.blas`, `wwr.capi.solver` and
`wwr.capi.ccl`,
so building what the crates link configures nothing of the C++ tree. Its
presets are `cuda` and `hip`, into `build-cuda/` and `build-hip/` here,
with the root's sm_80 floor checked. It also builds the kernel shared
objects, one per module from sources forked into `rust/` (nothing under
`src/` is compiled or included): `libfb_recipes_kernels.so`, from
`recipes/kernels/` and the trimmed device headers in `device_headers/`, at
`<build>/recipes/kernels`, with its own device link and the shared vendor
runtime, `libfb_integrals_kernels.so`, from `integrals/kernels/`, at
`<build>/integrals/kernels`, which links the first, and
`libfb_localization_kernels.so`, from `localization/kernels/`, at
`<build>/localization/kernels`. Its `-sys` crate's `build.rs` includes `kernel_lib.rs`, which links
it from the same `<build>` `capi_dir.rs` chose, and hands the crate that
build's backend from its `CMakeCache.txt` for the load check. After editing a kernel
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
built. Only the `runtime_api`, `blas`, `solver` and `ccl` surfaces of
`wwr-sys` are linked: `blas`, `solver` and `ccl` are asked for by
`fb-core-wwr` (each stream worker's BLAS and dense-solver handles, made by
the `Device` backend, and the collectives between the run's devices,
`fb_core_wwr::ccl`, so every binary over it needs NCCL on CUDA or RCCL
on HIP), and
by `fb-recipes-blas` (the operation its sandwiches' gemms take and the
side its dgmm takes) and `fb-recipes-diagonalize` (the modes its syevd takes), and a workspace build unifies features, so every crate linking
`wwr-sys` in it links against `libwwr_capi_blas.so` and
`libwwr_capi_solver.so` (only a binary that calls one keeps it under
`NEEDED`).

The GPU tests (the ignored ones) use device 0 and need room on it for a
context; a tier-5 case (`tier 5: needs two GPUs`, named `tier5_*`) uses
devices 0 and 1, or every device it sees, and the gates run it only where
wwr sees two (`docs/testing.md`, "Selecting GPU tiers in cargo"). A
collective between two cards with no peer path the library will take, as
PCIe through a host bridge, goes through shared memory, a `/dev/shm`
segment of tens of MiB per pair; docker's default 64 MiB is too small,
and the clique's first collective fails with `WWR_CCL_RESULT_SYSTEM_ERROR`.
The devcontainers and `docker/compose.yaml` start the container with
`--shm-size=1g`, which a container made before that carries only after
`devtools/devcontainer.sh rebuild` on the host; `devtools/doctor.sh` warns
when more than one card is visible and `/dev/shm` is under 512 MiB, and
the error itself names `/dev/shm`'s size and the fix (`fb_core_wwr::ccl`).
Until the rebuild, `NCCL_P2P_LEVEL=SYS` is the stopgap: it lets NCCL take
the peer path over the bridge, overriding its own judgment of the
topology for every run, so it is not the fix. The host-only
tests (the fake handle kind, error codes and contracts) need no GPU, and
`fb-core-log`, `fb-core-error-handling`, `fb-core-types`,
`fb-core-primitives`, `fb-core-constants`, `fb-atom`, `fb-molecule`,
`fb-basis`, `fb-io-xyz`, `fb-io-basis`, `fb-test-support`, `fb-doc-check`,
`fb-kernel-abi`, `fb-device-check`, `fb-sad`, `fb-io-npy`, `fb-io-sad` and `fb-tools-sad` do not link the library at all: they build and pass
with it missing.
`fb-core-arena`, `fb-core-thread-pool` and `fb-core-planner` link it
through their `device` feature, which is on by default: without it
(`--no-default-features`) their device-free tests, on `PinnedStandIn` and `Heap` blocks and
`HostOnly` workers, build and pass with no wwr crate in the graph. There
`Device` and `Pinned` are still kinds, so tables, admission and planning
are whole, but a block of either of non-zero size aborts, and the
thread pool's `DeviceStream` is an uninhabited placeholder with no
`Device` backend to run it. `device-tests` (also default) adds what the
crates' own device tests need beyond `device`; a dependent asks for
`device` alone, and the workspace's dependency entries for the arena and
the thread pool turn their defaults off, so nothing turns them on by
accident. Only `fb-core-wwr`, `fb-core-diagnostics`, `fb-recipes-sys`,
`fb-recipes-blas`, `fb-recipes-matrix-ao`, `fb-recipes-diagonalize`,
`fb-recipes-orthogonalize`, `fb-recipes-symmetrize`, `fb-recipes-cholesky`,
`fb-recipes-diis`, `fb-integrals-sys`, `fb-integrals-mmd`,
`fb-integrals-one-electron-overlap`, `fb-integrals-one-electron-kinetic`,
`fb-integrals-one-electron-potential-en`,
`fb-integrals-one-electron-core-hamiltonian`,
`fb-integrals-eri-density-fitting-metric`,
`fb-integrals-eri-density-fitting-three-center`,
`fb-integrals-eri-density-fitting-fit`,
`fb-integrals-eri-density-fitting-jk`,
`fb-integrals-eri-conventional`, `fb-molecule-nuclei`,
`fb-pipeline-config`, `fb-pipeline-guess`, `fb-pipeline-scf`,
`fb-pipeline-eri`, `fb-localization-sys`,
`fb-localization-pipek-mezey`,
`fb-integrals-eri-density-fitting-pair-map`, `fb-integrals-schwarz` and
`fb-integrals-eri-density-fitting-transform` always link (`fb-doc-check`'s `doc_lists.rs` holds these
three lists to the package graph); planning the records runs the kernel
library's load check, so even the overlap's and the transform's device-free
tests (the host tables, the plan goldens) need the library built. `fb-molecule-nuclei` links wwr alone,
through the planner, so its device-free tests (the plan goldens, the
staged rows on `HostOnly`) need no kernel library. Use a
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
arena's device-free tests are unit tests, over the heap kind and the
test-only pinned stand-in, and so are the planner's, run on `HostOnly`
workers; the stand-in reaches the planner's tests through `fb-core-arena`'s `test-support` feature, enabled
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
refusals of the mmd and overlap wrappers; the kinetic energy and the
nuclear attraction are `integrals/sys/tests/core_hamiltonian.rs`: one (s|s)
record each against the closed form (the attraction's with one nucleus and
`F0` from libm's `erf`), within `1e-14` and `1e-12` relative, and both
wrappers' refusals. Density fitting's metric and three-center integrals
are `integrals/sys/tests/density_fitting.rs`: one (s|s) aux pair and one
(ss|s) cell, the latter through the packed launcher, against their closed
forms (`F0` from libm's `erf`), within `1e-12` relative, and both
wrappers' refusals. The packed tensor on a card is
`integrals/sys/tests/packed_witness.rs`: every shell pair of a basis
through f on two centres, one centre's contracted, and a screened
sub-range with batch-relative columns, against libcint's Cartesian
`int3c2e_cart` through (ff|i) over an aux basis through i (a committed
reference, `integrals/sys/tests/data/`, with the script that writes it),
within `1e-11` of each class's largest element. The FelizJueves units in
`integrals/kernels/` are emitted, not written: after bumping
`deps/FelizJueves`, emit the tools' whole trees into a scratch directory
and copy the six directories this library carries from there, byte for
byte what the defaults write under `src/integrals/`.
`tools/emit/df_kernels.py` writes nothing else, so its metric family, the
one rust/ carries (`--family metric`), is emitted straight into
`integrals/kernels/`, as is the conventional ERI's one:

```bash
# from the repository root, tmp=$(mktemp -d)
OMP_NUM_THREADS=8 tools/emit/one_electron_kernels.py --out-root "$tmp"
OMP_NUM_THREADS=8 tools/emit/exchange_kernels.py --out-root "$tmp"
cp "$tmp"/mmd/kernels/pair_bound/* rust/integrals/kernels/mmd/kernels/pair_bound/
cp "$tmp"/mmd/kernels/record_bound/* rust/integrals/kernels/mmd/kernels/record_bound/
cp "$tmp"/one_electron/overlap/kernels/* rust/integrals/kernels/one_electron/overlap/kernels/
cp "$tmp"/one_electron/kinetic/kernels/* rust/integrals/kernels/one_electron/kinetic/kernels/
cp "$tmp"/one_electron/potential_en/kernels/* rust/integrals/kernels/one_electron/potential_en/kernels/
cp "$tmp"/shared/* rust/integrals/kernels/shared/
OMP_NUM_THREADS=8 tools/emit/df_kernels.py --out-root rust/integrals/kernels --family metric
tools/emit/conventional_kernels.py
tools/emit/packed_three_center_kernels.py
```

`tools/emit/conventional_kernels.py` writes the conventional ERI's units,
which the C++ tree does not have, straight into
`integrals/kernels/eri/conventional/kernels/` (packed) and
`integrals/kernels/eri/conventional/dense/kernels/` (dense), its default
root. `tools/emit/packed_three_center_kernels.py` writes the packed
three-center units, rust/'s own too, straight into
`integrals/kernels/eri/density_fitting/three_center/packed/kernels/`.

The records on a card are `integrals/mmd/tests/records.rs`: H2O/cc-pVDZ
and CO/cc-pVTZ planned, run once and compared with the C++ tree's records,
committed in `integrals/mmd/tests/data/` (1.3 MB, raw little-endian; that
directory's README has the format). They are dumped on CUDA by
`tools/mmd_records`, a program the root build builds with the C++ tree,
after a change to either basis file, the host normalization or the C++ mmd
module:

```bash
# from the repository root, after cmake --preset default && cmake --build build
build/tools/mmd_records/free-breakfast-mmd-records rust/integrals/mmd/tests/data
```

The overlap on a card, the first whole run, is
`integrals/one_electron/overlap/tests/overlap.rs`: H2O in STO-3G and
cc-pVDZ and CO in cc-pVTZ, each from its XYZ file and basis name, run once
and compared with PySCF's S at `max |S - S_ref| <= 1e-12 * max |S_ref|`, and
the run's timing record checked for a stream scope on every stream task.
The references are committed in `integrals/one_electron/overlap/tests/data/`
with the script that writes them; that directory's README has the
conventions it pins (the same JSON digits, CODATA 2018 bohr, `D S D` for the
unrenormalized contractions).

The kinetic energy on a card is
`integrals/one_electron/kinetic/tests/kinetic.rs`: the overlap's three
fixtures, run once and compared with PySCF's T at `max |T - T_ref| <=
1e-12 * max |T_ref|`, the overlap's bar by the same analysis. Its
references are committed in `integrals/one_electron/kinetic/tests/data/`
with the script that writes them, the overlap's with `int1e_kin`, and the
same conventions.

The nuclear attraction on a card is
`integrals/one_electron/potential_en/tests/potential_en.rs`: the
overlap's three fixtures, the nuclei uploaded by `fb_molecule_nuclei`, run
once and compared with PySCF's V (`int1e_nuc`, point nuclei) at `max |V -
V_ref| <= 1e-12 * max |V_ref|`, the C++ test's bar, two decades above the
rounding of the sums over primitives and nuclei, the Boys evaluation and
the two GEMMs. Its references are committed in
`integrals/one_electron/potential_en/tests/data/` with the script that
writes them, the overlap's with `int1e_nuc`, and the same conventions.

The core Hamiltonian on a card is
`integrals/one_electron/core_hamiltonian/tests/core_hamiltonian.rs`: the
overlap's three fixtures, T and V launched into one matrix by one task,
run once and compared with PySCF's H (`int1e_kin + int1e_nuc`) at
`max |H - H_ref| <= 1e-12 * max |H_ref|`, the bar of T and V, since
adding in place reorders no sum inside either. Its references are committed in
`integrals/one_electron/core_hamiltonian/tests/data/` with the script that
writes them, the overlap's with `int1e_kin + int1e_nuc`, and the same
conventions.

The eigensolver on a card is `recipes/diagonalize/tests/diagonalize.rs`:
the overlap's three committed references, S of order 7, 24 and 60, read
from `integrals/one_electron/overlap/tests/data/`, each run once through
`add_diagonalize` with `syevd_lwork` asked of a solver handle on the device
first, at `max |A U - U diag(w)| <= 1e-12 ‖A‖₂` and `max |Uᵀ U - I| <=
1e-12` with w ascending: syevd's backward-stability bound is a small
multiple of `n ε ‖A‖₂`, 1.3e-14 at n = 60, so two decades of margin. Its
flag check is a tier-3 unit test on `PinnedStandIn` rows, and its plan at a given
`lwork` a golden.

The orthogonalizer on a card is `recipes/orthogonalize/tests/orthogonalize.rs`:
the same three S, each run once through `add_orthogonalize` at the default
floor, at `max |X S X - I| <= 100 n ε κ₂(S)`, the eigendecomposition's
backward error `n ε ‖S‖₂` amplified by `1 / λ_min`, with two decades of
margin; κ₂(S) and the check that no eigenvalue is at or below the floor
come from LAPACK's eigenvalues, committed in
`recipes/orthogonalize/tests/data/` with the script that writes them. Then
the path graph's Laplacian of order 3, rank 2: its null mode dropped, `X S X`
is the projector `I - J / 3` to `100 n ε`. The forked `fb_inverse_sqrt` is
`recipes/sys/tests/inverse_sqrt.rs`, bitwise against the host's
`1.0 / x.sqrt()` (both correctly rounded; on HIP with contraction off in
the kernel, which otherwise folds into an `rsq` an ulp off) on each side of the floor and
across block boundaries, the elements past its count untouched.

The symmetrizer on a card is `recipes/symmetrize/tests/symmetrize.rs`, one
run per call on the committed nonsymmetric 37 × 37
(`recipes/symmetrize/tests/data/`, with the script that writes it) at
leading dimension 40: after the average, `(i, j)` and `(j, i)` are both
the host's `(a + b) / 2`, bitwise (addition commutes and the halving is
exact), and averaging again changes nothing; after either mirror, the
destination triangle is the source's bits and the source with the diagonal
untouched; the padding rows untouched throughout. The wrappers' refusals
are `recipes/sys/tests/symmetrize.rs`, raw, with no device.

The Cholesky factorization on a card is `recipes/cholesky/tests/cholesky.rs`,
one run per committed matrix (`recipes/cholesky/tests/data/`, with the
script that writes them): the SPD orders 1, 63, 64, 65, 128 and 200 held
to `max |L Lᵀ - A| <= 100 n ε max |A|`, the backward error's bound with two
decades of margin, the strict upper triangle A's bits; and two order-100
matrices whose first non-positive leading minor is 3 and 67, each a
`Numerical` error naming it, with the task that reads `checked` not run.
Its plan at n = 200 is the golden `recipes/cholesky/tests/golden/`, and the
tile wrapper's refusals are `recipes/sys/tests/cholesky.rs`, raw, with no
device. `trsm` and `syrk` on a card are `fb_core_wwr::blas`'s own tests,
exact on small integers with the unread triangle NaN.

The density fit on a card is
`integrals/eri/density_fitting/fit/tests/fit.rs`, one run per case on the
committed fixtures (`integrals/eri/density_fitting/fit/tests/data/`, with
the script that writes them): an SPD metric of order 41 (cond(M) 1e4) and
a random tensor of 41 × 5², held to `max |L B - b| <= 100 n ε κ₂(L) max
|b|`, the triangular solve's backward error with two decades of margin,
κ₂(L) from the fixture's committed singular values; and a metric whose
first non-positive leading minor is 17, a `Numerical` error naming it with
neither the fit nor the download of B run. Its plan is the golden
`integrals/eri/density_fitting/fit/tests/golden/`.

J and K on a card are `integrals/eri/density_fitting/jk/tests/jk.rs`, one
run per case on the committed fixtures
(`integrals/eri/density_fitting/jk/tests/data/`, with the script that
writes them): a random B of 23 × 7², each `B_Q` symmetric, a factor C of
7 × 3 and `D = 2 C Cᵀ`. J against the host's `Bᵀ (B vec(D))` and K against
the host's `Σ_Q B_Q D B_Q`, each held to `100 naux nao ε max |·|` (the
products' rounding, two decades of margin), and K's `(m, n)` and `(n, m)`
the same bits. Their plans are the goldens
`integrals/eri/density_fitting/jk/tests/golden/`. `gemv` on a card is
`fb_core_wwr::blas`'s own test, exact on small integers.

The ERI step on a card is `pipeline/eri/tests/jk.rs`, one run each: water
in cc-pVDZ / cc-pVDZ-JKFIT planned by `add_df_eri` at
`schwarz_threshold = 0`, then J of the committed SAD guess D and K of its
factor C (`D = 2 C Cᵀ`) by one streamed pass over the step's packed B,
against PySCF's DF `get_jk` of the same D over the same
documents (`pipeline/eri/tests/data/`, with the script that writes them),
held to `1e-10 max |·|`, the C++ tree's DF bar against the same
approximation; and the screen at 1e-8 over two waters 12 Å apart, each
dropped pair's columns of the unscreened B below the threshold. Its plan
is the golden `pipeline/eri/tests/golden/`.

The conventional integrals on a card are
`integrals/eri/conventional/tests/jk.rs`, one run per case: H₂ and water,
each in STO-3G and cc-pVDZ (water's with an O d shell, 25 Cartesian AOs to
24 spherical), the tensors planned by `add_conventional`, J and K of the
committed SAD guess D by `add_conventional_jk`, against PySCF's exact
`get_jk` of the same D over the same documents
(`integrals/eri/conventional/tests/data/`, with the script that writes
them), held to `1e-10 max |·|`, the tree's bar for a path against the
same approximation (the analysis, in the test's doc, leaves about 1e-13),
and J and K exactly symmetric. It was the FelizJueves kernels' first run
on a card: every case within 6e-15 of max. Water's plan, the refusals of
an f shell and of 175 Cartesian AOs, and admission counting both tensors
are `tests/plan.rs`. The RHF over them is `pipeline/scf/tests/scf.rs`'s
water case over `[eri] mode = "conventional"`, within `1e-9` hartree of
PySCF's exact RHF (`pipeline/scf/tests/data/scf_h2o_cc-pvdz_conventional.npy`,
written by `conventional_reference.py` there), 2.1e-5 hartree from the
fitted energy; and `rust/examples/conventional.toml` runs it through the
binary. The dense storage's J and K are `integrals/eri/conventional/tests/dense.rs`:
water in cc-pVDZ against the same `get_jk` at the same bar on one card,
and on two and four bitwise one card's (tier 5). Its RHF is
`pipeline/scf/tests/dense.rs`, water and, as slow cases, glycine chains
against PySCF's exact RHF at `1e-11 |E_ref|` on one, two and four cards;
`rust/examples/conventional_dense.toml` runs water through the binary on
two (tier 5); and `pipeline/config/tests/admission.rs` has ten glycines
refused by admission on one card and admitted on two.

The core Hamiltonian's orbitals on a card are
`pipeline/scf/tests/core_orbitals.rs`: the overlap's three fixtures, each
planned by the test's own `CoreRun` (`tests/common/`) with `syevd_lwork` asked of a solver
handle on the device and run once, at `max_i |ε_i - ε_ref,i| <= 100 n ε
κ₂(S) ‖H‖₂` (the generalized problem's eigenvalue sensitivity through
`S^(-1/2)`, two decades of margin), `max |Cᵀ S C - I| <= 100 n ε κ₂(S)` and
`max |H C - S C diag(ε)| <= 100 n ε κ₂(S) ‖H‖₂`, the last two with PySCF's
S and H, which the run's own meet decades under those bars. The reference
energies are LAPACK's `scipy.linalg.eigh(H, S)` of the committed S and H,
with κ₂(S) and ‖H‖₂ beside them, in `pipeline/scf/tests/data/` with the
script that writes them. The run's timing record has an entry per task, a
stream scope on each stream task's.

The DF-RHF loop on a card is `pipeline/scf/tests/scf.rs`, one run per
case from the SAD guess: water in cc-pVDZ / cc-pVDZ-JKFIT (the
representative) and benzene (a slow case, skipped under
`FREE_BREAKFAST_FAST_TESTS`), each within `1e-9` hartree of PySCF's DF-RHF
over the same documents and geometry at `conv_tol = 1e-12`
(`pipeline/scf/tests/data/scf_*.npy`, with the script that writes them),
the final residual under `commutator_tolerance`, and the card's free
memory, read at each iteration's log line, the same from the second
iteration to the last; and water at `max_iterations = 2`, which fails the
run with `MaxIterationsExceeded`. The DF-ROHF loop is
`pipeline/scf/tests/rohf.rs`: OH and H₂O⁺ (doublets) and CH₂ (a
triplet) in cc-pVDZ / cc-pVDZ-JKFIT, CH₂ in cc-pVTZ / cc-pVTZ-JKFIT, and
the H atom and H₂⁺, no beta electron, in cc-pVDZ / cc-pVDZ-JKFIT,
against PySCF's DF-ROHF
(`pipeline/scf/tests/data/df_rohf.json`, a copy of the C++ tree's): the
energy within `1e-9` hartree, every orbital energy within `1e-6`, the
occupations, and D the published C's `Da + Db`; and but for OH, whose
²Π minimum is a circle of states, the published C against PySCF's
`mo_coeff` (`pipeline/scf/tests/data/df_rohf_orbitals.json`, with the
script that writes it), each block of degenerate columns's subspace
within `1e-6 √k / gap` of the reference's, the Davis-Kahan bound.

`doc_check/tests/section_citations.rs` checks that every section the
README, `docs/` and every member's Rust sources cite by name is a heading of
the file cited ([`documentation.md`](docs/documentation.md), "Narrative:
mdBook beside the rustdoc", has the rules). It reads files and nothing else,
in a crate that links nothing, so it needs neither card nor library:
`cargo test -p fb-doc-check`. It finds the members from the workspace
manifest, so a new crate is scanned without being named there.
`doc_check/tests/unsafe_code.rs`, in the same crate, checks every member's
crate roots against the list of crates allowed `unsafe`
([`programming_model.md`](docs/programming_model.md), "Capabilities are
enforced by lints"): a root that does not `#![forbid(unsafe_code)]` must be
a listed crate's, and a listed crate must still hold `unsafe`.
`doc_check/tests/rules.rs` reads the rules register,
[`rules.md`](docs/rules.md), and fails when a check a row names, a lint,
a type, a test file or a test in one, does not exist.

`doc_check/tests/contracts.rs` checks that no doc comment restates a
contract, which rustdoc renders on its own, that every `debug_assert!`
carries an `// INVARIANT:` comment, and that a public function's leading
precondition on its arguments is a `#[requires]` whatever its message (an
`assert!`, interpolated or not, an `assert_eq!` or `assert_ne!`, or an
`if … { panic!() }` guard), and
that a value check (`le!`, `lt!`, `eq!`, `fits_i32!`) appears only in a
contract's condition, as a whole `&&` term of it
([`testing.md`](docs/testing.md), "Contracts are part of the spec").
`doc_check/tests/unsafe_contracts.rs` checks that, in a function holding an
`unsafe` block, a `#[debug_requires]` on an integer parameter or a
parameter's `.len()` is always on or carries a `// RELEASE:` comment saying
what holds the bound in a release build.

`doc_check/tests/source_rules.rs` scans the sources for six rules of the
register, each named in its row: every `#[ignore]` is a device case's, no
Markdown sits beside a module, a doc example carries its marker, nothing is
exported to C, a binding crate's safe wrapper links down to the binding it
calls, and a test that trips a contract is `#[cfg(debug_assertions)]`
exactly when the contract is a debug one.

Tests run in parallel, so a capture test must not use global capture mode.
A global capture forbids other threads' unsafe calls, and on the 3080 it
made a neighbouring pool's create abort or its destroy segfault.

Nothing in ctest runs cargo. Both gates do, for a diff under `rust/`:
`devtools/test.py push` and `pr` run tiers 1 to 3, the tier-3 command above
over every member whose graph declares no `links` under
`--no-default-features` (found from `cargo tree`, so a new link-free crate
is gated without being named), then `RUSTDOCFLAGS='-D warnings' cargo doc
--no-deps` over the same members with `--no-default-features`, and over the
members whose default features link nothing with their defaults, in
`target/nodef` ([`docs/testing.md`](../docs/testing.md), "Gates", at the
root). A commit touching `rust/` runs `devtools/test.py cargo-lint`, `cargo
fmt --check` and clippy with warnings denied in both feature sets, over
the link-free members, and so does CI's `lint` job. Tier 4 runs in both
gates for the crates a diff affects that link (above): `cargo clippy
--all-targets -- -D warnings` over them in `target/<backend>`, the one
gate of the workspace lints on those crates, then their tests on a card.

## Design

`docs/` holds the design. [`design.md`](docs/design.md) is the questions a
feature's design answers and where each answer lands; the six beside it
hold the mechanics: the [programming model](docs/programming_model.md),
[memory](docs/memory.md), the [GPU layer](docs/gpu_layer.md),
[testing](docs/testing.md), [diagnostics](docs/diagnostics.md) and
[documentation](docs/documentation.md).
[`rules.md`](docs/rules.md) is the register of the rules they state, each
with what checks it. **A new rule is added to the register in the PR that
states it, with its check or an *unenforced* entry.**
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
| 0 | #1362 these docs, done; #1363 the cargo half of the gates, done; #1364 this directory's CMake project, done; #1365 core types, primitives and constants, done; #1366 stream scopes and the timing record, done |
| 1 | #1367 atom and molecule, done; #1368 tier-4 skips, done; #1369 the forked device layer and the merge sort, done |
| 2 | #1370 basis, done; #1371 the forked mmd and overlap kernels, done |
| 3 | #1372 io/xyz and io/basis, done; #1373 the primitive-pair records as tasks, done |
| 4 | #1374 the overlap: declared with a layout, laid out in an arena, planned once and run on one stream, its arena size a golden and its stream scope timed, done |

Beside them: #1382, the kernel libraries' ABI version and backend check at
load ([`gpu_layer.md`](docs/gpu_layer.md), "What the port changes"),
done; #1380, the basis's angular-momentum ceiling asserted against a forked
device header, done; #1377, the doctor's compiler-drift scan over
`rust/build-*`, done; #1393, three doc leftovers after the fork, done; and
#1400, the load check run by the safe layers at planning, its comparison in
tier 3, done.

Milestone 33, the core Hamiltonian's orbitals and the SAD guess from a
config, in waves:

| Wave | Issues |
|---|---|
| 0 | #1402 one conversion from `fb_launch_status`, done; #1403 library handles per device, done; #1404 `cargo doc` without the library, done; #1405 the kinetic and nuclear-attraction forks, done; #1406 the shared spherical transform, done; #1407 the nuclei upload task, done; #1408 the timing record's JSON, done; #1417 io/npy, done; #1418 sad, done |
| 1 | #1409 the kinetic energy, done; #1410 the nuclear attraction, done; #1411 the symmetric eigensolver, done; #1412 the binary, done; #1419 io/sad, done; #1420 the SAD stamp generator, done |
| 2 | #1413 the core Hamiltonian, done; #1414 the symmetric orthogonalizer, done; #1421 the SAD guess as tasks, done |
| 3 | #1415 the core-Hamiltonian orbitals, done; #1422 the SAD guess from a config, through the binary, done |

Milestone 34, the DF-RHF energy from that guess, through the binary, in
waves:

| Wave | Issues |
|---|---|
| 0 | #1384 the auxiliary basis sets, done; #1505 the executor's replay between host decisions, done; #1506 recipes/cholesky, done; #1507 recipes/diis, done; #1508 recipes/symmetrize, done; #1509 the DF metric and three-center forks, done; #1510 the SCF iteration's dense nodes, done |
| 1 | #1511 a loop in the planner, done; #1512 the DF metric as tasks, done; #1513 the three-center integrals as tasks, done; #1514 the density fit, done; #1515 DF J and K, done |
| 2 | #1516 the density-fitted ERI step, done |
| 3 | #1517 the DF-RHF loop, done |
| 4 | #1518 `calculation = "rhf"` from a config, through the binary, done |

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

2026-11-13, `core::planner` (the `Task` trait, `Dag`, the planned
DAG, since named `Plan`, `Instance`), with the two hand-wired executor
tests moved onto it, on the RTX 3080 (CUDA) only: 59 unit tests (6 of
them the planner's, one a
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

2026-11-13, `integrals/mmd` (the primitive-pair records as two tasks), on
the RTX 3080 (CUDA) only, holding the card: in a debug build the whole
workspace passes, the new crate's host-table unit test, its 2 plan tests
(the two plan goldens and an empty basis) and its tier-4 `records` test, and
every other count unchanged. Against the C++ tree's records, dumped on the
same card by `tools/mmd_records`, every field of both molecules (6 classes
and 1629 records of H2O/cc-pVDZ, 10 and 7684 of CO/cc-pVTZ) is bitwise
equal, inside the issue's bars (integers exact; `zeta`, `ea`, `eb` bitwise;
the rest 1e-13 relative); the lowest bit of one `s` value flipped in the
reference fails the test, naming the field. The `--no-default-features` tier 3 passes its previous
counts. Clippy (`-D warnings`, both feature sets), rustdoc (`-D warnings`)
and `cargo fmt --check` are clean. Not run on the gfx1200.

2026-11-13, `integrals/one_electron/overlap` (the overlap, the first whole
run through the model), on the RTX 3080 (CUDA) only, holding the card: in a
debug build the whole workspace passes with `--include-ignored`, the new
crate's 2 plan tests (the three plan goldens and a refused input) and its
tier-4 `overlap` test, and every other count unchanged. Against PySCF's S
(`D S D`, the tree's JSON digits, CODATA 2018 bohr), `max |S - S_ref|` is
4.4e-16 for H2O/STO-3G (7 AOs), 4.4e-16 for H2O/cc-pVDZ (24) and 1.1e-15
for CO/cc-pVTZ (60), inside the fixed bar `1e-12 * max |S_ref|`; leaving
out the (1,2) class's launch fails it at 0.25 and 0.34. Each run's timing
record has an entry for each of its 7 tasks, a stream scope on each of the
4 stream tasks'. The `--no-default-features` tier 3 passes its previous
counts. Clippy (`-D warnings`, both feature sets), rustdoc (`-D warnings`)
and `cargo fmt --check` are clean. Not run on the gfx1200.

2026-11-13, the kernel libraries' load check (`fb_<module>_abi_version()`
and `fb_<module>_backend()` in each library's `kernels/abi.cu`,
`check_library()` in each `-sys` crate), on the gfx1200 (HIP), holding the
card, linking the libraries the `hip` preset built: in a debug build the
whole workspace passes with `--include-ignored`, `fb-recipes-sys` and
`fb-integrals-sys` 3 new unit tests each (equal values pass; another version
or another backend is refused naming both values, an unknown tag by its
number) and 1 new test each in `merge_sort` (now 5) and `overlap` (now 5),
the check against the library built for this card, and every other count
unchanged. With the bindings' version bumped and the library not rebuilt,
the check fails naming versions 1 and 2. The `--no-default-features` tier
3 passes its previous counts. Clippy (`-D warnings`, both feature sets),
rustdoc (`-D warnings`) and `cargo fmt --check` are clean. The `cuda`
preset builds both libraries and `devtools/cross-backend-check.sh
--device-only` (CUDA) passes; neither was run on the RTX 3080.

2026-11-13, the load check run by the safe layers and its comparison in
tier 3 (`fb-kernel-abi`; the backend read in `kernel_lib.rs`), on the RTX
3080 (CUDA), holding the card, linking the libraries the `cuda` preset
built: in a debug build the whole workspace passes with
`--include-ignored`, `fb_kernel_abi` 3 unit tests (equal values pass;
another version or another backend is refused naming both values, an
unknown tag by its number), the 3 in each of `fb_recipes_sys` and
`fb_integrals_sys` gone with the comparison they tested, and every other
count unchanged. The gate's tier 3 (`devtools/test.py pr --all --only
cargo`) goes from 135 tests over 16 link-free crates to 138 over 17, and
its tier 4 from 96 to 90. With the integrals bindings' version set to 2 and
the library not rebuilt, `add_records` and `Run::new` return the check's
`GpuRuntime` error naming versions 1 and 2 at planning, before anything
launches. Clippy (`-D warnings`, both feature sets), rustdoc (`-D
warnings`), `cargo fmt --check` and `devtools/rust-bindings.py --check` are
clean. Not run on the gfx1200.

2026-11-13, the kinetic and nuclear-attraction wrappers
(`fb_kinetic_class`, `fb_potential_en_class` and `fb_nuclei_view` in
`libfb_integrals_kernels.so`, `FB_INTEGRALS_ABI_VERSION` 2), on the RTX
3080 (CUDA), holding the card, linking the libraries the `cuda` preset
built: `devtools/test.py pr --only cargo --all` passes, tier 3 its 138
tests unchanged and tier 4 94 (was 90), the 4 new in
`integrals/sys/tests/core_hamiltonian.rs`: one (s|s) record through each
wrapper against the closed form at both mirror positions, the kinetic
energy `1.3e-16` relative to the reference (the bar `1e-14`) and the
attraction `2.1e-16` (the bar `1e-12`), the diagonal untouched; and both
wrappers' refusals. The emitted units are byte for byte those under
`src/integrals/` (44 files, `cmp`). Clippy (`-D warnings`, `fb-integrals-sys`
with its tests), rustdoc (`-D warnings`), `cargo fmt --check` and
`devtools/rust-bindings.py --check` are clean. `devtools/cross-backend-check.sh
--device-only` (HIP) passes, building this library for HIP; not run on
the gfx1200.

2026-11-13, one conversion of a launch status (`fb_kernel_abi::launch_status`,
which `fb_integrals_mmd::launch` calls for every class wrapper), on the RTX
3080 (CUDA), holding the card, linking the libraries the `default` preset
built: `devtools/test.py pr --only cargo --all` passes, tier 3 142 tests
(was 138), the 4 new in `fb_kernel_abi`: every `fb_launch_status` value
to its `ErrorKind` (`FB_LAUNCH_UNSUPPORTED` among them, and unknown values
on both sides), each message naming the wrapper and the status, a negative
status never reaching the runtime's half, and every error located at the
caller; tier 4 94 unchanged, the plan goldens among them. Clippy (`-D
warnings`, the linking crates with their tests too), rustdoc (`-D
warnings`) and `cargo fmt --check` are clean. Not run on the gfx1200.

2026-11-14, `recipes/matrix_ao` (the Cartesian-to-spherical transform
moved out of the overlap, U staged once per plan and basis), on the RTX
3080 (CUDA), holding the card, linking the libraries the `default` preset
built: `devtools/test.py pr --only cargo --all` passes, tier 3 166 tests,
none in the new crate, tier 4 95 (was 94), the new one its plan test: two
matrices transformed in one plan share one `harmonics` task, which both
transforms run after. The overlap's three plan goldens are bitwise
unchanged without `FB_BLESS`, and its tier-4 test passes at its bar,
`1e-12 * max |S_ref|`. Clippy (`-D warnings`, the two crates with their
tests too), rustdoc (`-D warnings`) and `cargo fmt --check` are clean. Not
run on the gfx1200.

2026-11-14, `integrals/one_electron/kinetic` (T as tasks, and the safe
layer over `fb_kinetic_class`), on the RTX 3080 (CUDA), holding the card,
linking the libraries the `default` preset built: `devtools/test.py pr
--only cargo --all` passes, tier 3 190 tests, none in the new crate, tier
4 97 (was 95), the 2 new its plan goldens and its PySCF comparison. Against
PySCF's T, `max |T - T_ref|` is `7.1e-15` (H2O/STO-3G), `1.8e-14`
(H2O/cc-pVDZ) and `5.3e-14` (CO/cc-pVTZ), the bar `1e-12 * max |T_ref|`
about `2.9e-11` for each. The script, run on the overlap's CO fixture,
rewrites the overlap's committed S byte for byte. Clippy (`-D warnings`,
the new crate with its tests too), rustdoc (`-D warnings`) and `cargo fmt
--check` are clean. Not run on the gfx1200.

2026-11-14, `molecule/nuclei` (a `Molecule`'s nuclei on the device as two
tasks, `nuclei` and `nuclei upload`), on the RTX 3080 (CUDA), holding the
card, linking the libraries the `default` preset built: `devtools/test.py
pr --only cargo --all` passes, tier 3 195 tests, none in the new crate,
tier 4 100, the 3 new its unit tests: the plan goldens for H2O and CO, the
staged rows equal to the molecule's charges and positions bitwise on heap
blocks and a `HostOnly` worker, and the device rows copied back equal to
them bitwise on the card. `cargo tree -p fb-molecule --no-default-features`
names no wwr crate. Clippy (`-D warnings`, the new crate with its tests
too), rustdoc (`-D warnings`) and `cargo fmt --check` are clean. Not run on
the gfx1200.

2026-11-14, the timing record's remaining fields and its JSON
(`thread_pool`, `planner`: the sync points' host scopes, each task's
dimensions, the plan's block sizes, the build profile, and
`Record::to_json`), on the gfx1200 (HIP), holding the card, linking the
libraries the `hip` preset built: `devtools/test.py pr --only cargo --all`
passes, tier 3 197 tests, `fb-core-thread-pool` 13 (2 new: a fixed DAG's
record reads back equal from its JSON, with one entry per task, each sync
point naming the node waited for, and the profile `cfg!(debug_assertions)`
names; a failed record reads back with its error's kind, message and
context); tier 4 106, none skipped, the planner's device round trip now
also checking the record's device and pinned block sizes against the
tables' and a host wait for every CPU task's GPU predecessor. Clippy (`-D
warnings`, the two crates with their tests and device features too),
rustdoc (`-D warnings`) and `cargo fmt --check` are clean. Not run on the
RTX 3080.

2026-11-14, `integrals/one_electron/potential_en` (V as tasks over the
nuclei's rows, and the safe layer over `fb_potential_en_class`), on the RTX
3080 (CUDA), holding the card, linking the libraries the `default` preset
built: `devtools/test.py pr --only cargo --all` passes, tier 3 195 tests,
none in the new crate, tier 4 110, the 3 new its plan goldens, its refusal
of nuclei with no nucleus at planning (`InvalidConfig`, nothing added) and
its PySCF comparison. Against PySCF's V, `max |V - V_ref|` is `4.8e-14`
(H2O/STO-3G), `8.8e-14` (H2O/cc-pVDZ) and `3.7e-13` (CO/cc-pVTZ), the bar
`1e-12 * max |V_ref|` about `6.2e-11`, `6.2e-11` and `6.4e-11`. Clippy (`-D
warnings`, the new crate with its tests too), rustdoc (`-D warnings`) and
`cargo fmt --check` are clean. Not run on the gfx1200.

2026-11-14, `integrals/one_electron/core_hamiltonian` (H = T + V as one
task over both kernel families), on the RTX 3080 (CUDA), holding the card,
linking the libraries the `default` preset built: `devtools/test.py pr
--only cargo --all` passes, tier 3 197 tests, none in the new crate, tier
4 115, none skipped, 3 of them new: its plan goldens (one build task,
`core hamiltonian`, and one `H cartesian` row, none for T or V), its
refusal of nuclei with no nucleus at planning (`InvalidConfig`, nothing
added) and its PySCF comparison. Against PySCF's H, `max |H - H_ref|` is
`4.8e-14` (H2O/STO-3G), `8.8e-14` (H2O/cc-pVDZ) and `3.7e-13`
(CO/cc-pVTZ), the bar `1e-12 * max |H_ref|` about `3.3e-11`, `3.3e-11`
and `3.5e-11`. Each reference equals the kinetic's T plus the nuclear
attraction's V to `1.4e-14`. Clippy (`-D warnings`, the new crate with its
tests too), rustdoc (`-D warnings`) and `cargo fmt --check` are clean. Not
run on the gfx1200.

2026-11-14, `pipeline/config` (the `free-breakfast` binary: its config
type, `main` and write nodes, running the overlap), on the gfx1200 (HIP),
holding the card, linking the libraries the `hip` preset built:
`devtools/test.py pr --only cargo --all` passes, tier 3 197 tests, none in
the new crate (it links); tier 4 118, none skipped, the 6 new its own:
`rust/examples/overlap.toml` through the binary exits 0 and its
`overlap.npy` meets the overlap's bar against PySCF (`max |S - S_ref|`
`4.4e-16`, the bar `1e-12 * max |S_ref|`), its `timing.json` parses with one
entry per task, the two write nodes among them; an unknown key, a missing
geometry file and an unknown basis name each exit with their kind's code
(1, 4, 4) naming the key and the path, no panic; the example's plan text
is a golden; and the config's paths. Clippy (`-D warnings`, the new crate
with its tests), rustdoc (`-D warnings`) and `cargo fmt --check` are clean.
Not run on the RTX 3080.

2026-11-14, the SAD guess from a config (`calculation = "guess"`, `[guess]
type` or `from`, and `add_write_host_npy`'s `write d.npy`), on the gfx1200
(HIP), holding the card, linking the libraries the `hip` preset built:
`devtools/test.py pr --only cargo --all` passes, tier 3 197 tests; tier 4
125, none skipped, `fb-pipeline-config`'s 10 among them, 3 new:
`rust/examples/guess.toml` through the binary exits 0 and its `d.npy`
equals `fb-sad`'s serial SAD density of water in cc-pVDZ bitwise (0 of
576 elements differ), its `timing.json` one entry per task, the two baked
stamps, `sad guess` and `write d.npy` among them; a second run loading
that file with `[guess] from` writes the same bytes; the example's plan
text is the golden `pipeline/config/tests/golden/plan_guess.txt`; and
`[guess]` read and refused (an unknown kind, `type` beside `from`, an
unknown key, the table beside `"overlap"`, each `InvalidConfig`). Clippy
(`-D warnings`, the crate with its tests), rustdoc (`-D warnings`) and
`cargo fmt --check` are clean. Not run on the RTX 3080.

2026-11-14, `recipes/diagonalize` (the symmetric eigensolver: `syevd` and
its flag check as tasks, `lwork` a dimension asked of the device before
planning) and the dense-solver handle in each stream worker's
`DeviceStream`, on the RTX 3080 (CUDA), holding the card, linking the
libraries the `cuda` preset built: `devtools/test.py pr --only cargo --all`
passes, tier 3 197 tests; tier 4 134, none skipped, the new crate's 4
among them: S of H2O/STO-3G, H2O/cc-pVDZ and CO/cc-pVTZ (order 7, 24, 60)
diagonalized once each, `max |A U - U diag(w)|` `4.7e-16`, `1.3e-15` and
`2.6e-15` against the bar `1e-12 ‖A‖₂` (`1.9e-12`, `4.4e-12`, `4.7e-12`),
`max |Uᵀ U - I|` `8.9e-16`, `1.6e-15` and `1.6e-15` against `1e-12`, w
ascending, the vendor's `lwork` 71106, 85715 and 122475; the flag check
(`Numerical` for a positive flag, `GpuRuntime` for a negative one, on
`Heap` rows); the plan golden at `lwork` 1234; and an empty matrix refused.
The device binary is clean under `compute-sanitizer --tool memcheck`, the
work row exactly `lwork` doubles (inside the arena's block, whose bounds
are what memcheck sees). Clippy (`-D warnings`, the changed crates with
their tests), rustdoc (`-D warnings`) and `cargo fmt --check` are clean.
Not run on the gfx1200; `wwr.capi.solver` compiles for HIP.

2026-11-14, `recipes/orthogonalize` (the symmetric orthogonalizer
`X = S^(-1/2)`: the eigensolver's two tasks, then `inverse root` on the
stream, the forked `fb_inverse_sqrt` and a `dgmm` and a `gemm`) and
`wwr_status_from_code` moved into `fb-core-wwr`, on the RTX 3080 (CUDA),
holding the card, linking the libraries the `cuda` preset built:
`devtools/test.py pr --only cargo --all` passes, tier 3 204 tests; tier 4
141, none skipped, 6 new: S of H2O/STO-3G, H2O/cc-pVDZ and CO/cc-pVTZ
orthogonalized once each, `max |X S X - I|` `2.2e-15`, `9.2e-15` and
`8.0e-14` against `100 n ε κ₂(S)` (`8.8e-13`, `1.3e-10`, `2.9e-9`, κ₂
5.6, 252 and 2176); the order-3 path Laplacian's null mode dropped,
`max |X S X - (I - J/3)|` `1.2e-15` against `6.7e-14`; the plan golden at
`lwork` 1234 and an empty matrix refused; `fb_inverse_sqrt` bitwise
against the host on each side of the floor, and a null array refused.
Clippy (`-D warnings`, the changed crates with their tests), rustdoc
(`-D warnings`) and `cargo fmt --check` are clean. Then on the gfx1200
(HIP), the `hip` preset: the same gate passes, tier 3 204, tier 4 141, none
skipped, `fb_inverse_sqrt` bitwise there too once its kernel turned
contraction off (contracted, clang folds `1 / sqrt` into a refined `rsq`
an ulp off).

2026-11-14, `pipeline/scf` (the Roothaan step `F C = S C ε` through X as
tasks, `fock transform`, the eigensolver's two and `back transform`, and
`Run`, the core Hamiltonian's orbitals from an XYZ file and a basis name),
on the RTX 3080 (CUDA), the `default` preset: `devtools/test.py pr --only
cargo --all` passes, tier 3 204 tests; tier 4 144, none skipped, 3 new: the
three plan goldens at `lwork` 1234 and the refused inputs; and H2O/STO-3G,
H2O/cc-pVDZ and CO/cc-pVTZ run once each, `max |ε - ε_ref|` `1.4e-13`,
`1.5e-12` and `2.1e-11` against `100 n ε κ₂(S) ‖H‖₂` (`3.1e-11`, `5.2e-9`,
`1.8e-7`), `max |Cᵀ S C - I|` `2.0e-15`, `1.0e-14` and `1.7e-13` against
`100 n ε κ₂(S)` (`8.8e-13`, `1.3e-10`, `2.9e-9`), `max |H C - S C ε|`
`6.8e-14`, `3.2e-13` and `1.7e-12` against the first bar; each run's timing
record has an entry for each of its 20 tasks, a stream scope on each of the
13 stream tasks'. Clippy (`-D warnings`, the new crate with its tests),
rustdoc (`-D warnings`) and `cargo fmt --check` are clean. Not run on the
gfx1200.

2026-11-14, density fitting's metric and three-center kernels forked
(`fb_metric_class`, `fb_three_center_class` and
`fb_three_center_symmetrize` in `libfb_integrals_kernels.so`, the 28
metric and 70 three-center FelizJueves units, `FB_INTEGRALS_ABI_VERSION`
3), on the RTX 3080 (CUDA), the `default` preset: `devtools/test.py pr
--only cargo --all` passes, tier 3 243 tests; tier 4 196, none skipped, 5
new in `integrals/sys/tests/density_fitting.rs`: one (s|s) aux pair and
one (ss|s) cell against their closed forms at both mirror positions,
within `5.2e-16` relative (the bar `1e-12`), the diagonal untouched; the
symmetrization of a 3 x 3 x 2 tensor bit for bit; and the three wrappers'
refusals. The emitted units are byte for byte those under
`src/integrals/eri/density_fitting/` (100 files, `cmp`); the slowest to
compile, read from the build's `.ninja_log` at 8 jobs, is `metric_45.cu`
at 3.7 s. Clippy (`-D
warnings`), rustdoc (`-D warnings`), `cargo fmt --check` and
`devtools/rust-bindings.py --check` are clean. Not run on the gfx1200.

2026-11-14, `recipes/symmetrize` (the exact average `(A + Aᵀ)/2` and the
triangle mirror, forked as `fb_symmetrize` and `fb_mirror_triangle`, the
ABI version 3), on the gfx1200 (HIP), the `hip` preset:
`devtools/test.py pr --only cargo --all` passes, tier 3 243 tests; tier 4
195, none skipped, 4 new: the committed 37 × 37 at leading dimension 40
averaged, `(i, j)` and `(j, i)` both the host's `(a + b) / 2` bit for bit
and a second average changing nothing; mirrored from each triangle, the
destination the source's bits and the rest untouched; the padding untouched
throughout; a short slice a panic; and the wrappers' refusals, raw.
`devtools/cross-backend-check.sh --device-only` compiles the fork for CUDA.
Not run on a CUDA card.

2026-11-14, `io/basis`'s fitting sets (`is_builtin_aux_basis`,
`resolve_aux_basis`, the thirteen documents of `data/aux_basis/` compiled
in), on the RTX 3080 (CUDA), the `default` preset: `devtools/test.py pr
--only cargo --all` passes, tier 3 544 tests, `fb-io-basis` 29 unit tests
(14 new: `test/io/basis_host.cpp`'s ten `IoBasisAux` cases,
`test/gtest/io_basis_gpu.cpp`'s two, which need no device here, every
fitting set resolved by name, and the table checked against
`data/aux_basis/`) and 2 doctests; tier 4 191, none skipped. Clippy (`-D
warnings`), rustdoc (`-D warnings`) and `cargo fmt --check` are clean. Not
run on the gfx1200; the crate touches no device.

2026-11-14, `recipes/cholesky` (the blocked factorization as tasks, its
diagonal tile forked as `fb_cholesky_diagonal`, ABI version 4, and
`fb_core_wwr::blas`'s checked `trsm` and `syrk`), on the RTX 3080 (CUDA),
the `default` preset: `devtools/test.py pr --only cargo --all` passes,
tier 3 270 tests; tier 4 215, none skipped, among them the SPD fixtures
at orders 1 to 200 within `1.8e-15` of A (the bar `100 n ε max |A|`,
`1.0e-11` at n = 200), the strict upper triangle A's bits; the minors 3
and 67 each named in a `Numerical` error with the reader of `checked` not
run; and `trsm` and `syrk` exact on small integers, the unread triangle
NaN. `devtools/cross-backend-check.sh --device-only` compiles the fork
for HIP. Not run on the gfx1200.

2026-11-14, `integrals/eri/density_fitting/fit` (`add_density_fit`, the
Cholesky path: `fb-recipes-cholesky`'s tasks over the metric, then one
trsm into B behind the check), on the RTX 3080 (CUDA), the `default`
preset: `devtools/test.py pr --only cargo --all` passes, tier 3 291 tests;
tier 4 225, none skipped, among them the committed order-41 metric and
41 × 5² tensor fitted with `max |L B - b| = 2.6e-15` (the bar
`100 n ε κ₂(L) max |b|`, `2.8e-10`, κ₂(L) = 100), and the metric whose
minor 17 is not positive a `Numerical` naming it with neither the fit nor
the download of B run. Not run on the gfx1200.

2026-11-14, `integrals/eri/density_fitting/jk` (`add_df_coulomb` and
`add_df_exchange`, the closed-shell J and one-factor K over B, and
`fb_core_wwr::blas`'s checked `gemv`), on the RTX 3080 (CUDA), the
`default` preset: `devtools/test.py pr --only cargo --all` passes, tier 3
291 tests; tier 4 245, none skipped, among them J of the committed
23 × 7² B and `D = 2 C Cᵀ` within `2.3e-13` of the host's (the bar
`100 naux nao ε max |J|`, `2.4e-9`) and K within `3.4e-13` (bar `3.2e-9`),
its `(m, n)` and `(n, m)` the same bits; and `gemv` exact on small
integers under both ops. Not run on the gfx1200.

2026-11-14, `pipeline/eri` (`add_df_eri`, the density-fitted ERI step:
the fitting set resolved by name and placed, then (mn|P), M and the fit
as tasks, B and L output rows; and `[eri]` read by `fb-pipeline-config`),
on the gfx1200 (HIP), the `hip` preset: `devtools/test.py pr --only cargo
--all` passes, tier 3 291 tests; tier 4 255, none skipped, among them
J and K of the committed SAD guess over the step's B for water in
cc-pVDZ / cc-pVDZ-JKFIT against PySCF's DF `get_jk`, `max |ΔJ| = 3.0e-13`
(1.8e-14 of max |J|) and `max |ΔK| = 1.3e-14` (1.3e-15 of max |K|), the
bar 1e-10 of each. Not run on the RTX 3080 (CUDA).

2026-11-14, `pipeline/scf`'s DF-RHF loop (`add_scf`, one iteration's
sub-DAG replayed to convergence from the SAD guess, its factor entering
the first K; `ScfStep`; `Run` the whole run), on the gfx1200 (HIP), the
`hip` preset: `devtools/test.py pr --only cargo --all` passes, tier 3 298
tests; tier 4 285, none skipped, among them water in cc-pVDZ /
cc-pVDZ-JKFIT within `1.7e-13` hartree of PySCF's DF-RHF (the bar 1e-9),
11 iterations, `‖e‖ = 9.1e-8`, free memory unchanged from the second
iteration to the last, and `max_iterations = 2` a `MaxIterationsExceeded`.
The slow case, benzene in the same sets, run by hand: `1.5e-12` hartree,
11 iterations. Not run on the RTX 3080 (CUDA).

2026-11-14, `calculation = "rhf"` from a config (`[eri]` required beside
it, `[scf]` the `ScfStep` table, the write nodes of `scf.json`, `c.npy`
and `d.npy`), on the gfx1200 (HIP), the `hip` preset: `devtools/test.py pr
--only cargo --all` passes, tier 3 304 tests; tier 4 294, none skipped, the
slow case returning at once under the fast mode, among them
`rust/examples/rhf.toml` through the binary, its `scf.json` energy
`2.6e-13` hartree from PySCF's DF-RHF (the bar 1e-9), 11 iterations, its
`timing.json` the record with the loop's entries per iteration; a run
`[guess] from` its `d.npy` converged in 2 iterations; `max_iterations = 2`
exits 6 and two nuclei at one point (a singular metric) exits 8, each
through the binary. Not run on the RTX 3080 (CUDA).

2026-11-14, the `[memory]` caps (`MemoryCaps`, each key's default its
floor, 8 GiB and 2 GiB, a key under it refused) and admission in the
binary (`admit_plan`: the pinned caps against `MemAvailable`, each
device table against the smaller of its cap and the card's free memory
less the 256 MiB reserve, each pinned table against its cap), on the
gfx1200 (HIP), the `hip` preset: `devtools/test.py pr --only cargo
--all` passes, tiers 1 to 3 321 tests; tier 4 332, none skipped, the
slow cases skipped under the fast mode, among them `rust/examples/memory.toml`
through the binary, its log naming worker 0's device and pinned blocks
admitted; a cap under its floor and a pinned cap past `MemAvailable`
refused by the binary before the output directory is made; and under a
cap of 2^62 bytes a PiB device block refused by device 0's free memory.
Not run on the RTX 3080 (CUDA).

2026-11-14, `GpuTask` and `CpuTask` in place of `Task` (#1595): a GPU
task's pinned rows a `CopyOnly` only `memcpy_p2d` and `memcpy_d2p` take,
a CPU task's layout and run host rows alone, each held by a
`compile_fail` doctest; `guess factor` copies the SCF guess's floor to the
device, where `fb_inverse_sqrt` reads it as a scale on its floor
(`FB_RECIPES_ABI_VERSION` 6), bitwise the floor by value. On the RTX 3080
(CUDA), the `default` preset: `devtools/test.py pr --only cargo --all`
passes, tier 3 321 tests; tier 4 322, none skipped, every plan golden
unchanged. HIP compiled by `devtools/cross-backend-check.sh --device-only`
(rust/'s kernel libraries with preset `hip`); not run on the gfx1200.

2026-11-14, admission's measured device reserve and the kernels' stack
term (`admit_plan`: `block + reserve + stack <= min(cap, total)`, the
reserve the card's total less its free memory from one reading made with
the worker's stream and handles alive, at least 256 MiB, and the stack
`fb_core_wwr::runtime::stack_growth` for 16,912 bytes per thread), on the
gfx1200 (HIP), the `hip` preset: `devtools/test.py pr --only cargo
--all` passes, tiers 1 to 3 321 tests; tier 4 335, none skipped, the slow
cases skipped under the fast mode, among them device 0's measured
reserve at least the floor and short of the card, and its stack term
exactly 0 with the stack limit as it was. Not run on the RTX 3080 (CUDA): the CUDA branch of the
stack term is compiled on every build, but never run.

2026-11-14, lanes (#1600): a GPU worker is one card, one thread and
several streams, a GPU task placed on a lane (`LaneId`), the worker's
tables and blocks its lanes', an edge between two lanes an event on the
device and no sync point, a handle pair and a BLAS workspace row per
lane. Two lanes sharing a row on device 0, a read and a handoff between
them, each shown load-bearing by dropping its wait. On the RTX 3080
(CUDA), the `default` preset: `devtools/test.py pr --only cargo --all`
passes, tier 3 325 tests; tier 4 339, none skipped, every plan golden
unchanged. HIP compiled by `devtools/cross-backend-check.sh --device-only`;
not run on the gfx1200.

2026-11-14, #1596, the planner's copy nodes: `Dag::upload` and
`Dag::download`, each a pinned row to a new device row or back as one node
named after the row, on the lane the caller names. Every hand-written
upload (`harmonics`'s U, the nuclei, the guess) and every readback that
was a task's whole work (the SCF's `guess w`, its products, the write
nodes') moved to them, and U is computed into its pinned row
(`fb_basis::spherical_transform_matrix` writes into a slice). The plan
goldens move in node names and positions only, and in the offsets the
split nuclei upload implies. On the gfx1200 (HIP), the `hip` preset:
`devtools/test.py pr --only cargo --all` passes, tiers 1 to 3 328
tests, tier 4 343, none skipped. CUDA not run.

2026-11-14, #1608, the mmd records on one lane per class, the first user
of lanes: `mmd records` split into an `upload <table>` copy node per
table on the worker's lane 0 and one `mmd records (la,lb)` per class, the
`c`-th on lane `c`, its sort's scratch sized to the class and packed with
the others', so the handoffs chain the classes as the C++ forks do. The
plan goldens holding the records move in their task lists and lanes; every
device table, less the `blas workspace lane <l>` row each new lane adds
where the plan declares a BLAS workspace (#1614), is the same size or
smaller, peak too. The overlap's device case asserts each class on a lane
of its own and no sync point for an edge between two lanes. On the
gfx1200 (HIP), the `hip` preset: `devtools/test.py pr --only cargo --all`
passes, tiers 1 to 3 328 tests, tier 4 343, none skipped; the records
still match the C++ tree's at their bars, `zeta`, `ea` and `eb` bitwise.
CUDA not run.

2026-11-14, counted replay (#1601): a counted loop runs its body for a
trip count and a depth `R` fixed at planning, with no decision, and
iteration `k` waits on iteration `k - R` alone (the ring). Each body node
runs after itself in the iteration before. The body's rows are `R`
copies of its table, and each GPU body node has an event pair per trip.
A batched tensor uploaded on one lane and reduced on another came out
bitwise equal to the ordered reduce, and the record held no host wait
for the loop. Dropping the ring by hand changed every element. On the
RTX 3080 (CUDA), the `default` preset: `devtools/test.py pr --only cargo
--all` passed, with 335 tier 3 tests and 353 tier 4 tests, none skipped,
and every plan golden unchanged. On the gfx1200 (HIP), the `hip` preset:
the same gate passed with the same counts, none skipped. On both cards, a
wait on an event was released by the recording that came before it was
enqueued.

2026-11-14, a tracked heap kind (#1585): `Heap` is a production kind,
one block per plan from the global allocator, filled with `0xFF` in every
build. A CPU task and a decision get a pinned and a heap `Layout` and
`Bufs`; a GPU task gets none, and `StagingKind` keeps the heap off a
worker's table. The device-free tests use `PinnedStandIn`. Admission
holds the heap block to `[memory] max_host_mem` (16 GiB, default and
floor), and the node's pinned blocks plus the heap block to
`MemAvailable` within the cgroup v2 limit. The SCF decision's
`last build` moved to the heap, and `write_npy` borrows its slice. In
the moved goldens, `last build` leaves the pinned table and the stand-in
prints as `pinned`; no table's total or peak changes. On the RTX 3080
(CUDA), the `default` preset: `devtools/test.py pr --only cargo --all`
passes, tier 3 339 tests, tier 4 360, none skipped. HIP not run.

2026-11-14, a counted loop in a decision loop's body (#1613): a decision
loop's body may hold a counted loop, replayed whole every iteration, its
trips overlapped with no host wait between them. The counted loop's
region is a row of the decision loop's body table, and a counted body
node's record entries carry the decision loop's iteration as `outer`.
On a card, a decision loop of three iterations zeroed an accumulator,
uploaded and reduced eight batches on two lanes at depth 2, and
downloaded the sum. Every iteration came out bitwise equal to the host's
ordered reduce, with the arena's block count unchanged, and the only
sync points were the decision's on the download and the lanes' exits. On
the RTX 3080 (CUDA), the `default` preset: `devtools/test.py pr --only
cargo --all` passed, tiers 1 to 3 342 tests, tier 4 365, none skipped,
and no plan golden moved. HIP not run.

2026-11-14, ancestor cgroup limits and a settled reclaimable term
(#1622): `host_available` walks from the process's cgroup up to the
mount's root and takes the smallest headroom, stopping without an error
at a directory it cannot list. A cgroup's reclaimable memory is half its
file LRU page cache and half its reclaimable slab, rounded up: the
kernel's `si_mem_available` rule with the low watermark at its cap of
half. A synthetic tree pins the walk: an ancestor's lower limit wins,
`max` everywhere leaves `MemAvailable`, and an unlistable parent and the
mount's root stop the walk. On the RTX 3080 (CUDA), the `default`
preset: `devtools/test.py pr --only cargo --all` passed, tiers 1 to 3
343 tests, tier 4 366, none skipped. HIP not run.

2026-11-14, heap rows to the device and back (#1619): `Dag::upload_heap`
and `Dag::download_heap` move a range of a heap row through a pinned
window of their own, a CPU node and a stream node each (`stage <row>`,
`upload <row>`; `download <row>`, `unstage <row>`). In a loop, iteration
`k` takes the range moved on by `k` of its lengths. A 32 MiB heap row cut
into eight ranges went up on lane 1 and came back on lane 0 in a counted
loop of depth 2, bitwise equal to the source, with no allocation on a
second run. The record's only host waits were each trip's `unstage` on
its download, plus the lanes' exits. The copy-node scan now also sees a
CPU task whose whole work is a heap/pinned copy. On the gfx1200 (HIP),
the `hip` preset: `devtools/test.py pr --only cargo --all` passed on
this tree rebased over #1613, tiers 1 to 3 342 tests, tier 4 368, none
skipped, every plan golden unchanged. CUDA not run.

2026-11-15, Pipek-Mezey localization (`fb-localization-sys`,
`fb-localization-pipek-mezey`): `src/localization/pipek_mezey`'s kernels
forked into `libfb_localization_kernels.so`, and `add_pm_localize`, the
sweeps a planner loop to convergence. Against PySCF's Pipek-Mezey over
Mulliken populations (`pyscf.lo.PM`, PySCF 2.14.0, driven to a maximum
by its stability check), for water (five occupied orbitals) and borane
(four) in cc-pVDZ at both exponents: L of the orbitals as given within
2e-16 relative, the localized L within 2e-14 absolute (bar 2e-12), and
every orbital's population on every atom within 4e-9 of PySCF's (bar
1e-5), U orthogonal and `C_local = C U` to round-off. On the RTX 3080
(CUDA), the `default` preset: `devtools/test.py pr --only cargo --all`
passed, tiers 1 to 3 352 tests, tier 4 376, none skipped. The fork also
compiles under HIP (rust/'s kernel library built with the `hip` preset);
not run on the gfx1200.

2026-11-15, the localization calculation (`fb-pipeline-config`'s
`calculation = "localization"`, `rust/examples/localization.toml`): the
DF-RHF of water in cc-pVDZ over cc-pVDZ-JKFIT through the binary, then
`add_pm_localize` over its S and C at exponent 2. Against PySCF's
Pipek-Mezey maximum of its own DF-RHF of the same water and sets (PySCF
2.14.0, `tests/data/localization_reference.py`), the localized L within
3.2e-10 (bar 1.3e-8: the reference's first-order sensitivity of the
maximum to the SCF's residual, 0.13, times the commutator tolerance 1e-7,
plus both localizations' stopping tolerances), in 4 sweeps after 11 SCF
iterations, `c_local.npy` 24 × 5. The RHF's plan golden unchanged by the
split into `plan_rhf`. On the RTX 3080 (CUDA), the `default` preset:
`devtools/test.py pr --only cargo --all` passed, tier 4 382, none
skipped. Not run on the gfx1200.

2026-11-16, the plan as Graphviz (`fb-core-planner`'s `Graph`,
`Plan::graph` and `Graph::dot`; `fb-core-arena`'s `Table::rows` and
`Table::live_at`): the binary writes `plan.txt` and `plan.dot` before
admission and `timing.dot`, the graph with the run's times, after the run.
The RHF example's graph: 77 tasks, 217 edges drawn (178 with rows behind
them, 39 orderings with none), 822 orderings implied by other paths left
out. On the RTX 3080 (CUDA), the `default` preset: `devtools/test.py pr
--only cargo` passed at 8c23f12b, tiers 1 to 3 373 tests, tier 4 271, none
skipped, the examples' 16 among them. Not run on the gfx1200.

2026-11-17, the conventional ERI (`[eri] mode = "conventional"`,
`fb-integrals-eri-conventional`, `rust/examples/conventional.toml`) over
FelizJueves's packed full ERI at 5af6614, its first run on a card: J and K
of the SAD guess against PySCF's exact `get_jk` (bar 1e-10 of max) within
5.6e-15 (H₂ STO-3G 4.5e-15 and 1.1e-15, H₂ cc-pVDZ 5.6e-15 and 3.6e-15,
water STO-3G 1.8e-15 and 1.1e-15, water cc-pVDZ 3.7e-15 and 1.5e-15, J
and K), and water's RHF in cc-pVDZ within 1.7e-13 hartree of PySCF's exact
RHF (bar 1e-9), 11 iterations, through the binary within 2.1e-13. After
the FelizJueves bump every unit both trees carry, emitted again, is byte
for byte the same. On the RTX 3080 (CUDA), the `default` preset:
`devtools/test.py pr --only cargo` passed at 471159fa, tiers 1 to 3 374
tests, tier 4 445, none skipped. Not compiled for HIP: the cross-backend
check could not run on this host (no `hipconfig`, no docker). Not run on
the gfx1200.

2026-11-17, the dense conventional ERI over the run's GPUs (`[eri] mode =
"conventional"`, `storage = "dense"`, `rust/examples/conventional_dense.toml`):
FelizJueves's dense J- and K-layout units at 03c2357, s through d, their
first run on cards (four RTX A6000s, CUDA, `NCCL_P2P_LEVEL=SYS`). J and K
of water's cc-pVDZ SAD guess against PySCF's exact `get_jk` (bar 1e-10 of
max) within 3.7e-15 (J) and 1.3e-15 (K), and on two and four cards bitwise
one card's. The RHF against PySCF's exact RHF (bar `1e-11 |E_ref|`): water
in cc-pVDZ 1.3e-13 to 2.1e-13 hartree on one, two and four cards; five
glycines in STO-3G (122 AOs, 3.5 GB of `M_J` and `M_K`) 2.7e-12 on one,
two and four, and in 6-31G (223 AOs, 39.6 GB) 3.2e-12 on two; ten in
STO-3G (237 AOs, 50.5 GB) 6.4e-12 on two and 7.3e-12 on four, refused by
admission on one. Each through the binary (release, `timing.json`, every card held by this run alone, all four card locks and FelizJueves's taken; J+K
is worker 0's stream time over `dense density`, the broadcast, `dense
jk`, the gathers and `dense spherical`, the mean over the iterations, and
the product the slowest worker's `dense jk`):

| Case | GPUs | ERI build | J+K per iteration | of it the product | SCF wall | Iterations |
|---|---|---|---|---|---|---|
| gly5 / STO-3G | 1 | 7.10 s | 5.34 ms | 5.01 ms | 7.60 s | 18 |
| gly5 / STO-3G | 2 | 3.58 s | 4.77 ms | 2.57 ms | 4.03 s | 18 |
| gly5 / STO-3G | 4 | 1.84 s | 6.43 ms | 1.34 ms | 2.35 s | 18 |
| gly10 / STO-3G | 2 | 52.3 s | 40.2 ms | 38.1 ms | 53.6 s | 18 |
| gly10 / STO-3G | 4 | 26.6 s | 24.0 ms | 19.2 ms | 27.6 s | 18 |

The build halves with each doubling of cards, the rows split evenly. The
product reads each card's blocks at about 660 GB/s (gly10 on two: 25.1 GB
in 38 ms). At gly5's size the collectives' fixed cost, a fraction of a
millisecond each and more in the first iteration, is what four cards add
over two. The SCF's wall is the build's plus 0.5 to 1.3 s.
On an RTX A6000 (CUDA), the `default` preset, `NCCL_P2P_LEVEL=SYS`:
`devtools/test.py pr --only cargo` passed at 6e80a56b, tier 4 478 with
four cards visible, none skipped, the tier-5 cases among them (the slow
glycine cases returned at once under the gate's switch and were run by
hand, above). Not compiled for HIP. Not run on the gfx1200.

2026-11-17, the packed three-center kernels (`fb_three_center_packed_class`
over FelizJueves's `peri3<la, lb, lP>` at 826b69e, its 70 classes through
(ff|i) emitted by `tools/emit/packed_three_center_kernels.py`, and
`add_packed_three_center_tasks`), on an RTX A6000 (CUDA), the `default`
preset: `devtools/test.py pr --only cargo --all` passes, tier 4 473 passed,
the 13 tier-5 cases not run (one card visible). Every shell pair of a basis
through f on two centres (158,928 elements) and a screened sub-range with
batch-relative columns (40,656) are bitwise the square kernels' Cartesian
tensor over an aux basis through i, all 70 classes launched. The units
compile alone in at most 4.7 s each (`peri3_222`,
`devtools/kernel-compile-watchdog.sh --build-dir rust/build-cuda`, 179 s for
all 70 at four jobs); the largest stack is `peri3<3,3,6>`'s 8,256 B, the
square `eri3<3,3,6>`'s, under `KERNEL_STACK_BYTES`. Not compiled for HIP.

2026-11-17, packed B for a batch of shell pairs
(`add_packed_three_center` over a `PairMap` range, then `add_batch_fit`
against one L), on an RTX A6000 (CUDA), the `default` preset:
`devtools/test.py pr --only cargo --all` passes, tier 4 499 passed, the 13
tier-5 cases not run (one card visible). Water in cc-pVDZ /
cc-pVDZ-JKFIT at threshold 0: every packed column within 5.5e-15 of max|B|
of the resident fit (bar 1e-12), and B over two batches, a class cut
between them, bitwise the one-batch B. Not compiled for HIP.

2026-11-17, a heap copy's ragged last trip (#1712): `Dag::upload_heap`
and `Dag::download_heap` clamp a range past the heap row's end, so a
counted loop of `total.div_ceil(len)` trips streams a row whose length is
no multiple of its ranges', and `add_streamed_jk`'s batches are one such
loop. On an RTX A6000 (CUDA), the `default` preset:
`devtools/test.py pr --only cargo --all` passes, tier 4 512 passed, the 13
tier-5 cases not run (one card visible). A row of 4337 doubles in ranges
of 1000 summed on the card in five trips of depth 2 is the whole row's sum
exactly and comes back bitwise, the same bits in a decision loop of three
iterations; the streamed J/K tests pass unchanged at their bars. Not
compiled for HIP.

2026-11-18, the DF build loop's ragged class trip (#1717): each class of
`add_packed_b` is one counted loop of `pairs.div_ceil(per)` trips, the
one-trip remainder loops gone. The packed tables write a short last
trip's cells and launch its counts, its classes a prefix of the first
trip's (the always-on assert relaxed to that); the spherical transform,
the fit's trsm and the transpose take the trip's count from the
iteration over the front of the first trip's rows; and
`Dag::download_heap_pitched` takes the columns and a batch width and
clamps the last trip to the columns. Water's build plan in its room
went from 99 tasks in 10 loops to 71 in 6, the arena unchanged. On an
RTX A6000 (CUDA), the `default` preset: `devtools/test.py pr --only
cargo --all` passes, tier 4 521 passed, the 13 tier-5 cases not run (one
card visible). Water's B in batches of 27 Cartesian columns, four
classes ending in a short trip: all 17 batches bitwise their own
planned apart, and within 1.06e-14 of max|B| of the one-batch B (bar
1e-12). Not compiled for HIP.

2026-11-21, the spherical-aux packed three-center kernels (#1736):
`fb_three_center_packed_class` over FelizJueves's `speri3<la, lb, lP>` at
6e1c341 (#743), its 70 classes through (ff|i) in the low-stack form, the
aux index written in real solid harmonics, so `add_spherical_transform`
lost its dense `U_aux` GEMM and its `transform aux` row. On an RTX 3080
(CUDA), the `default` preset: `devtools/test.py pr --only cargo` passes at
597bf1af, tier 4 520 passed, the 13 tier-5 cases not run (one card
visible). Every pair of the witness basis (46,354 elements) and its
screened sub-range within 1.04e-13 of each class's `|U|`-weighted scale of
libcint's `int3c2e_cart` contracted by `U` (bar 1e-11); water and
`df_high_aux` (run by hand, a slow case) within 2.2e-14 and 1.6e-14 of
max|(mn|P)| of PySCF (bar 1e-12); and, in an A/B against 00c4d498's
Cartesian kernels then `U_aux` GEMM, both fixtures' packed tensors within
4.6e-16 and 1.9e-15 of max|(mn|P)| of the old path's, 71% and 31% of
elements bitwise. Water's B within 1.4e-14 of max|B| of the host's
substitution. The units compile alone in at most 4.6 s each (`speri3_136`,
`devtools/kernel-compile-watchdog.sh --build-dir rust/build-cuda`, 122 s
for all 70 at six jobs); the largest stack is `speri3<3,3,6>`'s 7,072 B,
down from `peri3<3,3,6>`'s 8,256 B. Compiled for HIP
(`devtools/cross-backend-check.sh --device-only`, PASS); not run on the
gfx1200.
