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
- **`data/basis/`**: the BSE JSON `fb-io-basis` compiles in.
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
| `fb_core_wwr::handle` | wwr's `src/extension/handle` | `BaseHandle` over a `HandleKind` (CRTP's `create`/`destroy` hooks), `DeviceBoundHandle`, `StreamBoundHandle` over a `StreamBoundKind`, each with a fallible `try_new`, holding its stream owner in an `Arc`; the three views; the `DeviceHandle` / `DeviceHandleStream` traits |
| `fb_core_wwr::runtime` | wwr's `src/extension/runtime`, and `src/core/wwr`'s `Device::stack_growth` | `Stream`, `Event`, `MemPool` (with `MemPoolProps`), `Graph`, `GraphExec`, each a newtype over the handle it is built on, whose constructors abort on failure; `Stream` and `Event` `Send` and `Sync` (`unsafe impl`, the runtime's thread safety), so a step's stream moves with it between workers; stream capture through to a launched graph; `device_count()`, the devices the runtime sees (`wwr_get_device_count`); `stack_growth(dev_idx, per_thread)`, what the kernels' stack can still take, measured across raising the stack limit and setting it back, 0 under HIP |
| `fb_core_wwr::blas` | wwr's `src/extension/blas` | `wwr_blas_status` as an `ErrorCode`, and `BlasHandle`, a `StreamBoundHandle` over a shared `Stream` (`Arc<Stream>`) whose constructor aborts on failure, `Send` but not `Sync` (`unsafe impl`: the library allows a handle on any thread, one call at a time; a `compile_fail,E0277` doctest of `Sync`). `bind_workspace` (`wwr_blas_set_workspace`): the handle takes its workspace by value, a `'static` `DeviceSliceMut` (the planner's worker row, as an arena `HeldRow`) it owns until it is destroyed or handed another, since the library keeps the pointer beyond any borrow; a previous one is released only after the library has the new pointer and the stream has drained, and `Drop` syncs the stream first, so the call's one `unsafe` block rests on ownership, field order and `&mut self`. `BlasScalar`, wwr's typed dispatch (`usual_fp`), sealed over `f32`, `f64` and the two complex types, with each type's `ONE` and `ZERO` (the C++ `kOne` and `kZero`, by value) and its `unsafe` `gemm_raw`, `gemm_strided_batched_raw`, `dgmm_raw`, `trsv_raw`, `scal_raw`, `syrk_raw`, `gemv_raw` and `geam_raw` the letter's C ABI calls; and `gemm`, `gemm_rows`, `gemm_strided_batched`, `dgmm`, `syrk`, `gemv`, `geam`, `trsv` and `scal`, the safe calls over them, every operand a `DeviceSlice` whose extent (a matrix's at its op and leading dimension, `dgmm`'s diagonal at its stride) is checked against its length in every build, a short one a panic before anything is enqueued; `gemm_strided_batched` checks every matrix of the batch at its stride, takes a stride of 0 for A or B (one matrix shared by every product) and refuses a C stride under one C's extent, so no two products write one element. `dgmm` and `gemv` take strides of at least 1 (the library's 0 and negative ones are refused) and has no in-place form: C is a slice of its own, never A; nor does `syrk` (C's `fill` triangle alone), whose C is a slice apart from A, nor `geam` (`C = alpha op_a(A) + beta op_b(B)`), whose C is apart from both. `gemm_rows` is the gemm whose B and C are two blocks of rows of one matrix, the k rows from `src` read and the m from `dst` written, held apart by an always-on contract since no cut of the slice parts them: a left-sided blocked triangular solve's update. `trsv` (`op(A) x = b` over a vector at a stride of at least 1, no workspace on either backend) and `scal` are the one-right-hand-side path of that solve. The library's `trsm` is not bound: its workspace grows with the right-hand sides on ROCm, and the tree's own is `fb_core_device::GpuResources::trsm` |
| `fb_core_wwr::ccl` | wwr's `src/extension/ccl` | `wwr_ccl_result` as an `ErrorCode`; `Comm`, one rank of a clique `Comm::init_all(devices)` makes over a device list, a rank per device, destroyed on drop; and the f64 collectives through it, each on a stream of the rank's device and each failure a `Status`: `broadcast_send` / `broadcast_recv`, `reduce_send` / `reduce_recv` (a sum to a root), `all_reduce`, `all_gather`, `send`, `recv`, and `group` |
| `fb_core_wwr::device_slice` | nothing: the C++ hands a library call raw pointers | `DeviceSlice` and `DeviceSliceMut`, `unsafe` traits for device memory with its length, which a safe call takes in place of a pointer. The arena's `DeviceRef` and `DeviceMut` implement them (with its `device` feature), and so does the test helper `DeviceBytes`. `split_at_mut(slice, mid)` cuts a `DeviceSliceMut` into two disjoint `DevicePart`s, each its memory's only user while it lives, as `<[T]>::split_at_mut` does: how the blocks of one matrix reach two operands of one call, non-overlap decided from offsets; `split_at(slice, mid)` cuts a `DeviceSlice` into two `DeviceView`s to read, as `<[T]>::split_at`: how the dense ERI's sub-blocks reach separate calls |
| `fb_core_wwr::copy` | wwr's `src/extension/memory_buffer` `:copy` | `d2d(dst, src, stream)`, a device-to-device `wwr_memcpy_async` between a `DeviceSliceMut` and a `DeviceSlice` of equal length (asserted in every build): a run of elements inside a row, such as a column of DIIS's ring, which only a `split_at_mut` part names; a whole row is the arena's `memcpy_d2d` |
| `fb_core_wwr::solver` | wwr's `src/extension/solver` | `wwr_solver_status` as an `ErrorCode` (two values: the C ABI maps every vendor failure to `WWR_SOLVER_STATUS_UNKNOWN`), and `SolverHandle`, the dense solver's `StreamBoundHandle` over a shared `Stream`, as `BlasHandle` is, `Send` but not `Sync` as it is. `SolverScalar` (`solver/syevd.rs`), the typed dispatch sealed over `f32` and `f64`, its `unsafe` `syevd_raw` and `syevd_lwork_raw` the letter's C ABI calls; `syevd_lwork`, the legacy query; and `syevd`, the safe call, A, w, the work and the flag `DeviceSliceMut`s whose extents are checked in every build, the work against the query, which the call asks again |
| `fb_core_wwr` (`gpu_status`) | `src/core/wwr`'s `:status`, and the rule its `:policy` encodes | `gpu_status` / `gpu_status_at`, and `gpu_log_error_at`, what a `Drop` does with a failed destroy |
| `fb_core_device` | nothing in the C++ tree: the v0.0.4 prototype's `device.rs` but its kernel launches, the v0.0.4 layer beside `fb_core_wwr` (`docs/gpu_layer.md`, "The v0.0.4 runtime") | Over `wwr-sys`, `fb-recipes-sys`'s raw kernel bindings and the device-free crates: `ScopedDevice` (a device current on this thread, restored on drop, not `Send`), `device_count()`, `memory(physical)` (free and total bytes, for the budget's cap), `single_thread_host_blas()` (a linked host OpenBLAS, ROCm's on a HIP build, set to one thread and its load-time pool joined, once per process, by the steps' setup); `Stream` (non-blocking, on one device, `Send` and `Sync`) with `raw`, `physical`, `sync` (counted and timed, `sync_meter()`), `wait(event)`, `zero`, `fill`, and the asynchronous copies `upload`, `download` and `copy_device`, each making its stream's device current and each length, the device end's, an always-on contract; `upload` and `download` take the host end by value and return `Pending`, which holds it until a sync of the stream has covered the copy (`Pending::sync` hands it back, without blocking again after a covering sync; an upload's token, `Pending<'_, H>`, takes `and_upload(stream, dst)`, the same host end to another stream, joined through an event, so one token covers every GPU's copy; a download's, `Pending<'_, H, false>`, has none); `Event` (`record(stream)`, `sync`); `DeviceSlice` / `DeviceSliceMut` and the cuts `part`, `part_mut`, `split_at_mut` into `Part` / `PartMut`, each cut an always-on contract; and `HostSlice` / `HostSliceMut`, the host end of a copy, an `unsafe` promise its owner makes that only the value reaches the bytes and a leaked one leaves them valid; `GpuResources::new(gpu, physical)`, a GPU's stream with a BLAS and a dense-solver handle made on it, inside a `Mutex`, its methods reached through the guard: `bind_workspace` (once, `HANDLE_WORKSPACE_BYTES`), the checked `gemm`, `gemm_strided_batched`, `gemv`, `geam`, `dgmm`, `syrk`, `syevd_lwork` and `syevd` (its work held to the repeated query at the caller's line), the recipes `trsm` and `cholesky` (`CHOLESKY_TILE`) and the recipes' `inverse_sqrt` (0 at or below a floor), `diis_solve` (the bordered DIIS system, the unit selector at `newest` when singular) and `mirror_triangle` (the strict triangle opposite a `fill` written from it, a `syrk`'s other half) on the guard's one stream, the kernels through `fb-recipes-sys`'s raw bindings; the BLAS mode enums those calls take (`wwr_blas_operation`, `wwr_blas_side_mode`, `wwr_blas_fill_mode`, `wwr_blas_diag_type`), re-exported. A create aborts; every other runtime failure is a `Status` |
| `fb_core_budget` | the v0.0.4 prototype's `budget.rs` and `host.rs`; the C++ tree's pre-flight and footprints, which it replaces by counting the allocations themselves (`docs/memory.md`, "The budget") | The v0.0.4 layer's memory budget, over `fb-core-device`, `fb-core-pack` and `wwr-sys`: `Ledger` (per-kind atomic counters, device and pinned per gpu and the heap for the host, the `over` flag, the per-step table, `finish`, the budget DAG's final node, which packs each gpu and kind's scratch by the closure, adds the packed peaks and the handle workspace (`HANDLE_WORKSPACE_BYTES`) per gpu, and refuses the plan with the whole `table` as an `Allocation` error or allocates the slabs and workspaces, `take_workspace` for a `GpuResources` to bind, and `lease`), `Budget` (`device`, `pinned`, `heap`, `scratch`, `locks`), the buffers `DeviceBuf`, `PinnedBuf` and `HeapSlab` (a `Box<[T]>`, huge pages asked for and every page touched), `PinnedBuf`, `HeapSlab` and `HostPart` each the host end of a copy (`HostSlice`); `Layout` and `Span`, `Lease` (`fill_host`, and `fill_device` on a stream, the wrapper's NaN fill) with its tenants `ScratchBuf` and `HostPart`, each carve recorded so no two meet; `PinnedBuf::tenant` and `DeviceBuf::tenant`, typed tenants of an `Output`'s own buffer by a `Span` (`PinnedTenant`, the host end of a copy; `DeviceTenant`), each carve recorded in the buffer and cleared by the tenant's drop, a leaked one's kept, each tenant holding the allocation alive, the whole buffer refusing to be reached while a carve is recorded (`docs/memory.md`, "Tenants of an Output's buffers"); `Cap` / `Caps` (`resolve`, `read`, every cap clamped to the reading, `device_margin`, `host_margin`), `host_available` (cgroup-aware), `Backing` (`Real`, `Host`, `Dry`), `Kind`, `Plain` (sealed), `LockId` / `ManagerId` (rank by derived `Ord`), `bytes`; `AllocationRecord`, the `#[global_allocator]` a test binary installs, `watch` around a `run` and `RECORD_THRESHOLD` (1 MiB), with `Ledger::measure_execute` / `heap_growth`, the table's unbudgeted row |
| `fb_core_flow` | nothing in the C++ tree: the v0.0.4 prototype's `flow.rs` and `flow_tuples.rs`, with the step wrapper that sat on the prototype's `MainBus` (`docs/programming_model.md`, "Steps on rustflow") | The v0.0.4 layer's step DAG on rustflow, over `fb-core-budget` and `fb-core-device`: `Step` (`NAME`, `Output`, `Reads`, `STREAMS`, `slot`, `placement`, `mem_budget`, `run`), `Slot` (the `Output` behind an `RwLock`, the step's `Lane` beside it), `View` (`shared`, `read`, which compiles only for a step in `Reads`: `ReadSet`, `Has`, `At`), `Read`, `Placement` (`Host`, `Gpu0`, `AllGpus`), `Bus` (`shared`, `ledger`, `locks`), `Locks` (each GPU's `GpuResources` in its `Mutex` and each manager's lock, no `Mutex` reachable, `query` given a `Budget`), `Held` (the guards the wrapper took in rank order: `gpu`, `all_gpus`, `taken`), `Lane` / `GpuLane` (a step's streams per GPU), `Dag` (`add`, `build`, refusing a read the config did not ask for, naming both, a step added twice, a cycle), `Plan` (`budget`, every `mem_budget` then `Ledger::finish`; `execute`, every `run` inside the wrapper, a failed flag per node, the first error in topological order, replayable; `output`, `edges`), `StepTime` (a step's times and its lock acquisitions: the locks, whether it waited, when it held and released them), `fan_out` (one scoped thread per GPU), rustflow's `Executor` re-exported, and under the `test-support` feature `controls` (`skip_step_end_sync`, `delay_host_steps`), the step-end sync's negative control for a dependent's tests |
| `fb_steps` | the v0.0.4 prototype's `bus.rs`, its seven steps (`overlap`, `core_hamiltonian`, `metric`, `guess`, `orthogonalizer`, `eri`, `scf`) and its `run`; the C++ tree's `src/pipeline` steps of the same names (`docs/programming_model.md`, "The MainBus and its steps") | The v0.0.4 layer's MainBus and steps, over `fb-core-flow`: `Config` (`parse`, `read`, a refusal naming the file and the line, an unknown key among them (the old binary's `[basis_set]`, `num_gpus` and `[memory]` too); `calculation` (`Calculation`: `"overlap"`, `"guess"`, `"rhf"`, `"localization"`, one row per calculation of the steps it adds, the tables it reads, filled at their defaults, and its products; a table it does not read refused at the table's line, `"rhf"` and `"localization"` with no `[eri]` at `calculation`'s; absent, the tables present pick the steps); `[molecule]` (`MoleculeTable`: `xyz`, `charge` and `multiplicity`, default 0 and 1, `occupations(electrons) -> (n_alpha, n_beta)`, an impossible multiplicity `InvalidConfig`), `[basis]`, `[guess]` (`GuessTable`: `type` (`GuessType`, `"sad"`) or `from`, a float64 `.npy` D joined to the config's directory and loaded in place of the SAD density, its staging and budget the SAD guess's; refused in setup before anything is allocated, naming the file and the line: both keys together and a shape other than `n_ao × n_ao` `InvalidConfig`, an unreadable file `Io`, an open shell `Unsupported`, since `d.npy` is ROHF's spin densities summed; `tests/guess_device.rs` restarts water from the `d.npy` a run wrote, its first build within `energy_tolerance` of the converged energy), `[eri]`, `[scf]` (`ScfTable`: `max_iterations`, `energy_tolerance`, `commutator_tolerance`, `diis_history`, `stage_sync`, `check`), `[localization]` (`LocalizationTable`: `mode` (`LocalizationMode`, `"ibo"` refused `Unsupported`), `exponent` 2 or 4, `max_sweeps`, `functional_tolerance`, `rotation_gain_threshold`, `check`), `[resources]`), `MainBus::setup(config, backing)` (the geometry and bases on the host, `Dims::count` filling `Dims` (`n_alpha`, `n_beta` and `n_occ`, the doubly occupied) and its `ClassDim`s by `for_each_pair`, a walk that builds no array the host integrals' `HostPool` of `[resources] host_threads`, each GPU's `GpuResources`, the caps), `Shared` (`host_pool`, `syevd_lwork`, 0 with no device), `plan(bus)` (`"overlap"` the overlap alone, `"guess"` the guess alone); the host steps `OverlapStep`, `CoreHamiltonianStep`, `MetricStep` (every GPU, one staging under one token) and `GuessStep` (the SAD density, or `[guess] from`'s D read into its staging by `fb_io_npy::read_npy_into`), each built inside the host pool through the host integral crates into its pinned staging and uploaded, no lock; `OrthogonalizerStep`, X = S^(-1/2) on gpu0 under its lock (`EIGENVALUE_FLOOR`); `EriStep` (every GPU, `[eri] mem_per_gpu` in bytes, `DEFAULT_MEM_PER_GPU`, `ring`, 2 or 1, and `schwarz_threshold`, default 1e-12, a shell pair under it skipped in B's build and its columns zero, `EriManager::dropped_pairs`; `EriStep::least_mem_per_gpu`, under which the plan is refused) and its `Output`, the `EriManager` (L on every GPU by the tree's Cholesky, B on the host heap built over pair batches through each GPU's ring of `ssperi3` launches and blocked triangular fits, each GPU scattering into its own contiguous columns of B, and `build_jk(shared, held, &[Density], j, k)`, a J/K pass from D or the occupied C, one density or an open shell's two in one pass, through each GPU's ring of aux batches on the manager's own streams, reduced onto gpu0 through the host, under the manager's lock and every GPU's, which the reader declares; `passes()`, the passes since B was last built); `ScfStep`, the DF-RHF loop with DIIS, DF-ROHF at a multiplicity above 1 (both spins' factors in one J/K pass, the Roothaan effective Fock matrix, the published open orbitals canonical in `(Fa + Fb)/2`), a plain loop in `run` on gpu0 in its scratch, every J/K pass from a factor, the first from the guess's eigen-factor above 1e-12 of its largest eigenvalue (`ScfOutput`: `energy`, `residual`, `iterations`, `converged`, `guess_rank`, `meter` (a `LoopMeter`), C, ε and D on the host, D on gpu0, and an open shell's effective Fock matrix `fock` on gpu0); `LocalizationStep`, Pipek-Mezey on gpu0 under its lock over each block of the SCF's occupied columns, a closed shell's doubly occupied one, an open shell's doubly then singly occupied ones each among itself, each block's arrays sized by its width after the previous block's (`Dims::n_atom` beside the occupations), a plain sweep loop in `run` whose turn read carries L, `C U` and the sort key `diag(Uᵀ diag(ε) U)`, each block's converged columns stably sorted by `<i|F|i>` (`LocalizationOutput`: `c` on the host, `c_device`, `n_cols`, `functional_before`, `functional_after`, `sweeps`, each summed over the blocks; `tests/open_localization_device.rs` holds H₂O⁺ and triplet CH₂ at exponents 2 and 4 to PySCF's DF-ROHF localization, `tests/data/localization_rohf.json`, at the first-order bars of `tests/data/open_localization_reference.py`); `run(config, backing)`, setup, budget mode and, unless dry, execute mode over the run's one executor, into a `Run` (the plan, the table, the times) whose `replay` executes again in the same buffers, and whose `write(dir)` puts the run on disk on the host after execute mode: the record, `budget.txt` (the table) and `timing.json` (the run's and each step's times), and with `[scf]` `scf.json` (`energy`, `residual`, `iterations`, `converged`, `n_alpha`, `n_beta`, ε, each double shortest round-trip) and `c.npy` and `d.npy` (float64, Fortran order, through `fb-io-npy`), and with `[localization]` `localization.json` (`mode`, `exponent`, L before and after, `sweeps`) and `c_local.npy` (`n_ao × n_alpha`); `overlap.npy` for `"overlap"` and `d.npy` for `"guess"`, from a host copy (`OverlapOutput::s_host`, `GuessOutput::d_host`) budgeted only for that calculation. The `free-breakfast` binary (`src/main.rs`): `free-breakfast <config.toml>` with `--output` (default `fbf_output`), `--budget-only`, the exact table with no device, `--devices`, `--threads`, `--mem-per-gpu-mib` and `--stage-sync`; its `main` the one place an error is logged, printing `completed: N errors, N warnings` and exiting with `ErrorKind::exit_code`; no old-layer crate in its graph. `rust/examples/` holds `overlap.toml`, `guess.toml`, `rhf.toml`, `rohf.toml`, `localization.toml`, `resources.toml` (the RHF with `[resources]`, `[eri] mem_per_gpu` and `ring` written) and `two_gpus.toml` (the RHF on devices 0 and 1, tier 5), every one run through the binary by `tests/examples.rs`, which also holds that every key of `Config` is set by an example |
| `fb_core_log` | `src/core/log` | The one backend behind the [`log`](https://crates.io/crates/log) facade: spdlog's line format (local time, OS thread id, `file:line`), the seven levels, `set_level` / `should_log` / `flush`, one file sink, the warning counter and the error counter (`error_count`, a failed destroy or free and the boundary's line, which `main` prints), and `log_at` for a location already held; the value checks `le!`, `lt!`, `eq!` and `fits_i32!` (`check`), which a contract's condition is written with so the panic hook ends its line with the operands' values |
| `fb_core_error_handling` | `src/core/error_handling` | `ErrorKind` (with the pinned exit codes as `exit_code`), `Error` and its `#[track_caller]` factories, `Result<T>` / `Status`, `check_or_abort` |
| `fb_core_diagnostics` | `src/core/diagnostics` (`diagnostics.h`) | `gpu_must` (`FBF_GPU_MUST`), and nothing else: `FBF_TRY` is `?`, `FBF_LOG_TRACE` ... `FBF_LOG_ERROR` are `log`'s macros and `FBF_LOG_CRITICAL` is `fb_core_log::critical!`, each imported from its own crate, since this one depends on `fb-core-wwr` |
| `fb_core_arena` | the peak model of `src/core/footprint` and `src/pipeline/footprint`, and `check_plan`'s fit | The plan's memory laid out once (`docs/memory.md`): `TableBuilder` (rows declared node by node as scratch, outputs and their readers, or the worker's; lifetimes derived; `add_loop(positions)` makes a loop's body rows a table of their own in a region the rest holds for the loop, sized to the body's peak once, `add_counted(positions, depth)` a counted loop's, its region `depth` copies of the body's table a stride apart, iteration `k` in copy `k mod depth`, declared before a decision loop whose body holds it to make its region a row of that body's table, a row it writes and does not own refused unless one body node is its only user there, and `write(pos, row)` a row the loop carries from one iteration to the next, written by its last user in the body), `assign` (the pure greedy best-fit offset assigner, run in three orders and kept at the smallest total; every committed golden's block within 3% of its lower bound, `docs/memory.md`, "Lifetimes"), `Table` (offsets, total, `peak`, a loop's region counted once, `live_at`, the bytes held at one position as `peak` counts them, `rows`, each row's name, `Owner`, bytes and users as a read-only `RowInfo`, and `handoffs` and `reads`, the orderings reusing bytes and the data need, a pair leaving a decision loop's body taken from its decision, one leaving only a counted loop in it from the writer's last trip, a row outside a counted loop's body held against every copy; its text is the committed golden `core/arena/tests/golden/arena_scf_fixture.txt`, and with the SCF iteration as a loop's body `arena_scf_loop.txt`, rewritten by `FB_BLESS=1`), `admit` (the fit against a cap, refusing with every row over the cap on its own and the rows live at the peak), `admit_host` (a node's pinned blocks and the plan's heap block together against the host's available memory) and `host_available` (`MemAvailable`, or under a cgroup v2 limit the smaller of it and what the limit leaves) and, with `device`, `device_memory` (a card's free and total memory, one reading at planning, which bounds a device cap and measures the device's reserve), `Block<K>` (the table's one plain allocation, `Device`, portable `Pinned` or `Heap`, the plan's pageable block from the global allocator, filled with `0xFF`, aborting on failure with the bytes and the free memory; a plan's per-worker host table is of a `StagingKind`, `Pinned` or, under `test-support`, the device-free tests' `PinnedStandIn`, never `Heap`) and `Block::partition(pos, iteration)` (a node's `Bufs` in one iteration, `Bufs::iteration` its index, a counted loop's body row in its iteration's copy with a borrow flag per copy: its own rows as `&[T]` / `&mut [T]` on the host and pointer views on the device, behind always-on borrow flags; a stream node's host rows wrapped in a `CopyOnly`, which yields no slice and only `memcpy_p2d` and `memcpy_d2p` take); with `device`, `Block::hold` (a worker row held as its writer outside any view, a `HeldRow` that keeps its `Arc`'d block alive and is a `DeviceSliceMut`: memory a library keeps beyond a node, a BLAS handle's workspace). The one module that allocates |
| `fb_core_planner` | nothing: the C++ steps allocate as they go | The glue between the two below: `GpuTask` and `CpuTask`, one trait per sort (`layout`, which declares every row through a `Layout`, a node's rows of one table, and `run`, which computes through them: a GPU task's over device rows, its pinned rows as a `CopyOnly` only its copies take, and its stream; a CPU task's over host rows alone, pinned and heap, through a `Layout` and a `Bufs` of each; and a GPU task's `uses_libraries`, yes by default, no for the copy nodes and the mmd records' classes, without which a lane above 0 gets no library handles and no BLAS workspace row; and its `forks`, none by default, the most streams its `run` forks onto, whose maximum over a worker's tasks is the worker's fork streams), `Dag` (tasks added in plan order, each on a GPU worker, `add_gpu` on a lane the caller names (a `LaneId`, or a bare worker index for its lane 0), the worker's tables shared by its lanes so a task reads another lane's row in place, and `add_cpu` on the CPU pool, and with `device` the copy nodes `upload` and `download`, a pinned row to a new device row or back, one node named after the new row (`upload <row>`, `download <row>`) on the lane the caller names, and `download_rows`, several device rows of one element type down to pinned rows the caller names, in one node `download <label>` with one event for them all, and `upload_heap` and `download_heap`, a range of a heap row to a new device row or a device row into a range of a heap row, two nodes each through a pinned window of their own (`stage <row>` on the CPU and `upload <row>`; `download <row>` and `unstage <row>` on the CPU), iteration `k` of a loop taking the range moved on by `k` of its lengths, clamped to the heap row's end, so a counted loop of `total.div_ceil(len)` trips moves a ragged last range and its reader takes `len.min(total - k len)` from its iteration, so no task hand-writes a copy that is its whole work; each returns the task's rows, whose output ids the reading tasks take; `blas_workspace` declares a worker's BLAS workspaces, a worker row of `BLAS_WORKSPACE_BYTES` (32 MiB) for lane 0 and each further lane a task that uses the libraries is placed on, that admission counts and no task can name; `add_loop(name, worker, body, max_iterations, decision)` closes a loop over the tasks from position `body` (`len`) on, with a `Decision`, a CPU node with pinned and heap rows whose `decide` returns `ControlFlow`, the body's rows laid out in a region of the parent's tables and the rows it carries the parent's; `add_counted(body, trips, depth)` closes a counted loop over the tasks from `body` on, no decision, the body's rows held once per iteration in flight, a body task reading its iteration from its views to pick its batch, perhaps inside a decision loop's body, replayed whole every iteration), `Plan` (each worker's device and host tables and its lane count, the plan's one heap table, `heap_table`, `n_lanes`, and each task's dependencies, the reads and the handoffs, derived, a handoff between two lanes an event that serializes them; its text, a lane printed `stream<l>` on a worker of several, is the committed golden `core/planner/tests/golden/planner_scf.txt`, and `planner_scf_loop.txt` with a loop; `graph`, the plan as an owned `Graph` that outlives `instantiate`, whose `dot(record)` is Graphviz: each task's lane, dims and per kind the bytes it produces and those live while it runs, each edge labelled with the rows behind it, an ordering with none drawn only where no other path implies it, each loop a cluster, and with a run's `Record` each task's times, its golden `planner_scf_loop.dot`) and `Instance` (the blocks, each worker's two and the plan's heap block, and a `thread_pool::TaskGraph` whose nodes hand each task its rows in the iteration it runs in; `backend`, the `thread_pool::Device` built over the plan with each lane's declared BLAS workspace bound to its handle, once; `run(&mut self, &mut backend, n_cpu)`, any number of times and one at a time, each returning the executor's `Record` with every task under its own name and its `dims`, and the blocks' sizes). No `unsafe`. Across workers: `Dag::add_collective`, one step with a part on lane 0 of every GPU worker, each worker's collectives issued in plan order, the backend making the clique for a plan that holds one (`Plan::ranks`); `Dag::broadcast`, `Dag::gather` and `Dag::reduce` over it; and `Partition`, rows cut into a range per worker. Adaptive widths: `Adaptive` (`width_max`, `bytes(width)`, a pinned window's `pinned_bytes(width)`, a `warn_below` floor, `power_of_two`), `Dag::width` asked before a part's tasks, `Dag::plan_within(rooms, build)` planning a DAG declared by a fallible closure, each worker's `Room` (device and pinned bytes) asked once of `rooms` when its first part is measured, each part bisected once by `widest` in the room of each kind the schedule leaves at its position, a `width` line of the plan's text per part, logged at `debug`, at `warn` narrowed below its floor |
| `fb_core_thread_pool` | nothing: the C++ pipeline runs on one thread | The run's workers (`docs/programming_model.md`, "CPU workers"): `TaskGraph` (nodes added in topological order, each placed on the CPU pool or a lane of a GPU worker, a `LaneId`, `gpu<w>.<l>` in its log lines and record; `lanes`, the lanes it places per worker; `add_loop(name, body, deps, max_iterations, decide)` closes a loop over the nodes from `body` on, a sub-DAG replayed in the same run on the same workers while its CPU decision node, after every body node and `deps`, returns `ControlFlow::Continue`, and `MaxIterationsExceeded` past the bound; `add_counted(body, trips, depth)` a counted loop, the body run `trips` times with no decision, trip `k` after trip `k - depth` (each source after each sink), each node after itself in the trip before, a node after it free to follow a body node's last trip, perhaps in a decision loop's body, replayed whole every iteration; a node handed its innermost loop's iteration in its `Lane`), `run` (the plan and the backend borrowed `&mut`, so each is in one run at a time; one scope, `n_cpu` CPU workers and a thread per GPU worker that enqueues its ready nodes in rank order across all its lanes, a counted loop's trips back to back, and never waits on the device, an edge between lanes an event wait on the device and no sync point, readiness counts, failure skipping descendants, the lowest-ranked error returned, a panic re-raised rather than hung), and the `Backend` the edges leaving a stream wait through: `Device` (a stream per lane with its library handles bound to it, `DeviceStream`, a BLAS and a dense-solver handle on each lane `Device::new`'s `libraries` marks, made with the backend before any run and reached by a GPU node through its `Lane`; `DeviceStream::is_privileged`, true on lane 0 of worker 0 alone, the run's privileged lane, set by `Device::new` (`docs/programming_model.md`, "The run's devices"); each worker's fork streams, `Device::new`'s `forks`, beside its lanes and with no handles, over which `DeviceStream::fork_join(n, f)` runs a node's `n` items and joins them back into its stream before it returns, every wait on the device (`docs/programming_model.md`, "Forks inside a task"); `bind_blas_workspace`, through `&mut self`, so never during a run; an event pair per GPU node and per trip of its counted loop) or `HostOnly` for device-free tests. `fb_core_log::TaskScope` tags each node's lines `[task@worker]`, and `Error::context` carries it out. Every run returns a `Record` (`docs/diagnostics.md`, "The timing record"): its result, its wall time, and a `TaskTime` per node that ran, failed ones included, with its name, worker, dimensions (`TaskGraph::set_dims`), a loop node's iteration (one entry per iteration; a counted loop's node inside a decision loop, its trip and the decision loop's iteration as `outer`, one per pair), host time around the callable and, under `Device`, its stream scope's device time (an event pair on its stream, read after the run); a `SyncPoint` per host wait (a CPU node's on each GPU predecessor, a GPU worker's exit sync of each lane) naming the node waited for; each decision loop's iteration count, `Record::iterations`; the build profile; and the plan's `ArenaSize`s, which the planner adds. `Record::to_json` is the `timing.json` text, over [`serde`](https://crates.io/crates/serde), its schema in the type's rustdoc. `clippy.toml` makes it the one place that spawns. `Device::make_clique` gives lane 0 of each worker its rank (`DeviceStream::comm`) |
| `fb_core_types` | `src/core/types`, what the host modules name | `Double3`, `Vec<double, 3>`: `#[repr(C)]` with the backend's `double3` layout. The fixed-width integers and `make_vec` are the language's |
| `fb_core_primitives` | `src/core/primitives`, what the host modules name | `double_factorial_odd`, `factorial`, `binomial` as `const fn`, bitwise the C++ loops; `lower_index`, the host integral crates' packed lower triangle, rust/'s own. `idivup` / `align_up` are `div_ceil` / `next_multiple_of`, and the element-wise functions `f64`'s methods |
| `fb_core_constants` | `src/core/constants`, what the host modules name | `BOHR_PER_ANGSTROM`, `kBohrPerAngstrom<double>`, asserted against the C++ double's bit pattern |
| `fb_core_pack` | nothing: the C++ steps allocate as they go; the v0.0.4 prototype's `src/pack.rs` | Step scratch packed by the DAG (`docs/memory.md`, "Step scratch"): `closure` (the transitive closure, a `Reach` bitset of successors per step), `ordered`, `pack` (first fit, largest first and then by step name, each `Region` at the lowest offset clear of every unordered region placed) into a `Packed` with its peak, plain sum and `lower_bound`, the heaviest antichain and the regions that make it. Device-free |
| `fb_atom` | `src/atom` | `Element` (`#[repr(u8)]`, Z as the discriminant, `Sentinel` = 0), `z` / `from_z`, `symbol` / `from_symbol` (case-insensitive, `Sentinel` for none), and `Atom`, a `Double3` position in bohr and an `Element` |
| `fb_molecule` | `src/molecule`, its host half | `Molecule` as host data (atoms, charge, multiplicity; no atoms aborts), `distance`, `nuclear_charge`, `electron_count`, and the nuclear repulsion energy, gradient and Hessian. The nuclei on the device (`view`, `NucleiView`, `molecule_footprint`) are not ported: the v0.0.4 layer forms V on the host |
| `fb_basis` | `src/basis` | `Shell` and `Primitive`, the counts (`function_count_cart` / `function_count_sphe`) and `MAX_ANGULAR_MOMENTUM` (3, the C++ `kMaxAngularMomentum`), `cartesian_powers` and `Ao`, `solid_harmonic_block`, `BasisSet` (its Cartesian and spherical AO layouts, the per-atom shell grouping, the name) and `spherical_transform_matrix(basis, u)`, U written into a slice the caller hands it, `ShellStamp` with `primitive_normalized` and `place`, and `BasisStamp` with `place` onto a `Molecule`. Contractions are not renormalized, as in the C++ |
| `fb_recipes_sys` | `src/recipes/sort`'s merge sort, `src/recipes/orthogonalize`'s `inverse_sqrt`, `src/recipes/symmetrize`'s two tile kernels, `src/recipes/cholesky`'s diagonal-tile kernel, `src/recipes/diis`'s bordered solve and `src/recipes/trsm`'s diagonal-block kernel, forked, and `src/recipes/trsm`'s blocked solve over it, ported to `recipes/kernels/` (with the device headers it includes, trimmed, in `device_headers/`) | The raw FFI over `libfb_recipes_kernels.so`, built by `CMakeLists.txt` here: `fb_merge_sort`, an `extern "C"` launch wrapper that only enqueues, its `fb_merge_sort_workspace_bytes` query, `fb_inverse_sqrt` (the eigenvalue map `λ → λ^(-1/2)`, 0 at or below a floor passed in, optionally scaled by a value on the device), `fb_symmetrize` and `fb_mirror_triangle` (the exact average `(A + Aᵀ)/2` and one strict triangle copied onto the other, in place, refusing a negative order, an `lda` below it or a null matrix), `fb_cholesky_diagonal` (one diagonal tile of a blocked Cholesky in shared memory, the flag naming the first non-positive minor, the tile also written packed for the panel's trsm) and its tile order `FB_CHOLESKY_TILE` (64), `fb_diis_solve` (the bordered DIIS system solved by Gauss-Jordan in one block's shared memory, the unit selector at the newest column and a flag when a pivot falls below `1e-14`, refusing the C++ doc comment's bounds and null pointers) and its order `FB_DIIS_SOLVE_MAX_DIM` (65), `fb_trsm_diagonal_f32` and `fb_trsm_diagonal_f64` (`M X = s B` in place for one triangular diagonal block of a blocked solve and every right-hand side, the triangle packed and 48 columns of B to a thread block in shared memory, either triangle, M or B read transposed, a unit diagonal unread, `s = 0` zeroing B with neither read; refusing a leading dimension under its column and null pointers beside the C++ launcher's arguments) and the block's order `FB_TRSM_TILE` (64), `fb_launch_status` and `FB_DEVICE_ALIGNMENT` (`docs/gpu_layer.md`, "Layering"), and `check_library()`, the load check of the library's ABI version and backend, compared by `fb-kernel-abi`. The bindings are generated from `recipes/kernels/fb_recipes.h` by `devtools/rust-bindings.py` and committed. No `fb-core-wwr` edge: the safe calls over the bindings are `fb-core-device`'s, which calls them directly (`docs/gpu_layer.md`, "The safe wrapper"). It declares the library's `links` key, so a change under `recipes/kernels/` selects it and every crate above it |
| `fb_integrals_sys` | `src/integrals/mmd`'s launchers (records, bound, dressed bound; not its order, the records left in enumeration order), `src/integrals/one_electron/{overlap,kinetic,potential_en}`'s and `src/integrals/eri/density_fitting/metric`'s and its three-center launcher's shape, forked to `integrals/kernels/`, and the conventional ERI's, rust/'s own with no C++ launcher (`integrals/kernels/eri/conventional/`, the dense one's in its `dense/`), with the FelizJueves units they launch (`pair_bound`, `record_bound`, the overlap's, kinetic energy's and nuclear attraction's ten classes each, and the Boys routine `eval_boys.h` they and the density-fitting units include, emitted there by `tools/emit/*.py --out-root`; the metric's 28 classes, `tools/emit/df_kernels.py`'s; the packed three-center integrals' 70, `speri3<la, lb, lP>`, the aux index in real solid harmonics, `tools/emit/packed_three_center_kernels.py`'s, rust/'s own in `integrals/kernels/eri/density_fitting/three_center/packed/`, and the fully spherical ones' 70, `ssperi3<la, lb, lP>` (FelizJueves #745), all three indices in real solid harmonics, from the same tool, in `.../three_center/spherical/`; the packed full ERI's 21 classes through (dd\|dd) and its exchange gather `eri_kperm`, and the dense ERI's 81 ordered classes in each of its two layouts, `eri_dj` and `eri_dk`, FelizJueves's `eri/full_eri`, `tools/emit/conventional_kernels.py`'s) | The raw FFI over `libfb_integrals_kernels.so`: `fb_mmd_records`, `fb_mmd_bound`, `fb_ordered_pair_bound`, `fb_overlap_class`, `fb_kinetic_class`, `fb_potential_en_class`, `fb_metric_class`, `fb_three_center_packed_class` (the only Cartesian-bra (mn\|P): the square tensor's kernels, launchers and symmetrization left in #1725), `fb_three_center_spherical_class` (the same over `ssperi3`), `fb_conventional_eri_class`, `fb_conventional_exchange`, `fb_conventional_pack`, `fb_conventional_unpack`, `fb_dense_eri_class`, `fb_dense_row_dot`, `fb_packed_pack` and `fb_packed_gather`, each an `extern "C"` launch wrapper that only enqueues, a class wrapper refusing a class outside its table as `FB_LAUNCH_UNSUPPORTED`; the tables' limits (`FB_*_MAX_MOMENTUM`, the aux basis's `FB_METRIC_MAX_MOMENTUM` and `FB_THREE_CENTER_AUX_MAX_MOMENTUM` among them); the density-fitting record, pair and cell field counts (`FB_METRIC_*_FIELDS`, `FB_THREE_CENTER_AUX_FIELDS` and `FB_THREE_CENTER_PACKED_CELL_FIELDS` among them), the three-center ones read by its safe layer; and the four structs that cross by pointer, `fb_primitive_pair_class_view` (the one-electron launchers', and the three-center one's for its bra records), `fb_primitive_pair_source`, `fb_primitive_pair_records` and `fb_nuclei_view` (`src/molecule`'s `NucleiView`), `#[repr(C)]` with their layout asserted on both sides; and `check_library()`, as in `fb_recipes_sys`. The bindings are generated from `integrals/kernels/fb_integrals.h` and committed. Beside them, the safe layer, the v0.0.4 layer's, over `fb-core-device`'s `Stream` and slices: `three_center_spherical_class`, one class of the fully spherical (μν\|P) into a batch of `n_col · naux` doubles (`n_col` spherical bra columns, aux fastest), every slice's length and `n_col · naux` within `i32` a `#[requires]` naming the kernel, and the cells handed in on the host by value, each one's record ranges and block checked inside the slices it indexes and then uploaded by the call itself, which returns the upload's token, so no device table a caller fills reaches the launch; held to the experiment `df_batched/single_gpu_spherical`'s output on water at 1e-12 of the largest element (`tests/three_center_spherical.rs`). Beside it, on the same terms, `mmd_records` (one class's records through `fb_mmd_records` into `RECORD_F64_FIELDS` and `RECORD_I32_FIELDS` fields of `n_rec`, field-major, the shell and pair tables handed in on the host, each shell's primitive range, each pair's shells and record prefix checked, and uploaded by the call) and `packed_gather` (the slab zeroed, then `fb_packed_gather`, each column's row and column handed in on the host, checked inside the matrix and uploaded by the call), each returning the upload's token and each refusing a short slice or a stray index before anything is enqueued (`tests/records_and_gather.rs`); and `record_bound` (a class's dressed Schwarz bounds through `fb_ordered_pair_bound` from those records, every length a contract; an s pair's against its closed form, `tests/record_bound.rs`), the eri step's screen. The other launchers have no safe call, their old-layer task builders deleted (#1874); `tests/overlap.rs`, `tests/core_hamiltonian.rs` and `tests/density_fitting.rs` hold the overlap, kinetic, nuclear-attraction, metric and packed three-center ones raw, one s class each against its closed form. The private `launched`, a launch status as a `Status` naming the wrapper and its class's momenta, is the one conversion every launcher here goes through |
| `fb_device_check` | nothing: a check on the kernel libraries | Tier-3 tests: no launch wrapper under `rust/` allocates, synchronizes, touches an event, copies, sets the device or opens a profiler range (a grep of every `.cu`), every `-sys` crate's committed bindings are what the script generates now, no header the bindings read names a block, grid or warp, and every emitted unit equals its C++ twin byte for byte |
| `fb_kernel_abi` | nothing: what the kernel libraries share above their `-sys` crates | The one Rust copy of `device_headers/fb_launch.h`, generated by `devtools/rust-bindings.py` and re-exported by each `-sys` crate's bindings (`docs/gpu_layer.md`, "From bridge to C ABI"): `fb_launch_status` and its values, `FB_DEVICE_ALIGNMENT`, `fb_backend` and its values. `LaunchStatus`, the newtype a launch wrapper returns, and `Backend`, a library's `fb_backend` read by value (`TryFrom`, `Display`, the build's name through `const fn named`). `check`, the load check's comparison: a library's ABI version and `fb_backend` against the ABI version and `Backend` its `-sys` crate expects, a mismatch of either refused naming both values. Each `-sys` crate's `check_library()` calls it. And `launch_status`, the one conversion of a launch wrapper's `fb_launch_status` to a `Status` (`docs/gpu_layer.md`, "Statuses"): a refusal as `Unsupported` or `InvalidConfig`, a positive `wwr_status` handed to the caller's `gpu_status_at`, any other value `GpuRuntime`, each naming the wrapper and the status. Every safe layer calls it (each `-sys` crate's private `launched`, and `fb-core-device`'s). It links nothing, so its tests are tier 3 |
| `fb_io_xyz` | `src/io/xyz` | `parse_xyz` / `read_xyz`: a standard XYZ geometry into a `Molecule`, Angstrom to bohr by `BOHR_PER_ANGSTROM`, every refusal (too short, a bad count, a malformed line, an unknown symbol, a count the lines disagree with, a file that cannot be read) an `Io` error. A field must parse as a whole token, where the C++'s `operator>>` reads a prefix |
| `fb_io_basis` | `src/io/basis` | `parse_basis` / `read_basis`: a BSE JSON document into a `BasisStamp` (sp blocks and general contractions split, spherical shells only: one Cartesian or ambiguous shell refuses the document), over [`serde_json`](https://crates.io/crates/serde_json); `is_builtin_basis` and `resolve_basis`, a name (spelling folded) or else a path, over the eleven sets of `data/basis/` compiled in with `include_str!`, only that branch naming the stamp; `is_builtin_aux_basis` and `resolve_aux_basis`, the same over the thirteen density-fitting sets of `data/aux_basis/`, a table of its own disjoint from the orbital one (about 16 MB, compiled in for the reason the orbital sets are: nothing is installed, so a name cannot be misplaced) |
| `fb_sad` | `src/sad`, its serial form | `sad_density`, and `sad_density_into` into a caller's slice, the SAD guess: one spherically averaged, fractionally occupied atomic RHF per distinct (Z, shells) over closed-form one-center integrals, each density placed on its atoms' diagonal blocks, and `SadStampLookup`, the baked block that skips an element's SCF (checked by size). Every loop in the C++ order, so its stamps compare bitwise with the C++ generator's. `distinct_stamps` and `stamp_density` are public for the task form, which is `fb_pipeline_guess`'s |
| `fb_io_npy` | `src/io/npy` | `NpyArray` (a shape and its float64 elements, column-major) and `element_count`; `parse_npy` / `read_npy`, versions 1.0 to 3.0, `'<f8'` only, a C-order payload reordered, every refusal of a document (a bad magic, version, length or header field, another dtype, a payload the shape disagrees with, a file that cannot be read) an `Io` error and an overflowing shape `InvalidConfig`; `read_npy_into(path, shape, out)`, a file of a known shape streamed into a buffer already held (a step's pinned staging), C order placed element by element, another shape `InvalidConfig` naming the path; `npy_header`, `npy_to_bytes` and `write_npy(path, shape, data)`, the data borrowed so a CPU node writes its row in place, version 1.0 with `np.save`'s canonical header, byte for byte the C++ writer's (the `data/sad/STO-3G/` stamps written back are their files) |
| `fb_io_sad` | `src/io/sad` | `builtin_sad_density`, `fb-sad`'s `SadStampLookup`: every `data/sad/<basis>/<Z>.npy` compiled in with `include_bytes!` (the table written by `build.rs`), found by the directory name compared exactly with `BasisSet::name` and the atomic number, parsed by `fb-io-npy`; bytes that are not a square matrix abort. `carries_sad_stamp` is the same membership without the parse, which the guess plan asks at planning. A directory naming no `data/basis` set fails a unit test, a stem that is not a three-digit Z the build |
| `fb_tools_sad`, the `free-breakfast-sad-stamps` binary | `tools/sad` | The generator of `data/sad/`: `free-breakfast-sad-stamps <repo>/data (<basis>... \| --all)` resolves each named set through `fb-io-basis`'s compiled-in table (a set the binary does not carry refused `InvalidConfig`; `--all`, every `<data>/basis/*.json` by its document's BSE `name`), runs `fb-sad`'s atomic SCF for each element alone at the origin, each element's serial SCF on the next free one of its own scoped threads (as many as the CPU affinity allows, no more than the elements), with no old-layer crate, and writes `<data>/sad/<basis>/<Z>.npy` through `fb-io-npy` after removing each regenerated set's directory (every one under `--all`); an atom too large for the SCF (`Unsupported`) is skipped, any other failure exits with its kind's code before anything is removed. It links `fb-sad`, never `fb-io-sad`, so it is not linked against its own output. Its STO-3G output is the committed `data/sad/STO-3G/` byte for byte, `tests/stamps.rs` |
| `fb_localization_sys` | `src/localization/pipek_mezey`'s launchers (`pipek_mezey_bridge.h`, `pipek_mezey_kernels.cu`), forked to `localization/kernels/` (with the block and warp reductions it includes, forked into `device_headers/`) | The raw FFI over `libfb_localization_kernels.so`, built by `CMakeLists.txt` here: `fb_pm_populations`, `fb_pm_pair_angles`, `fb_pm_rotate_populations`, `fb_pm_rotate_columns`, `fb_pm_functional` and `fb_pm_identity`, the C++ launchers as `extern "C"` wrappers that only enqueue and refuse a count below 1, an `n_occ` past `FB_PM_MAX_OCCUPIED` (65535), an exponent other than 2 or 4 and a null pointer; and `check_library()`. Three of the launches trust indices in device memory (the atoms' AO offsets, a round's orbital pairs, the partners the pairs give), so each safe call uploads them itself or holds them in a type only it makes. The safe layer is the v0.0.4 layer's, over `fb-core-device`'s `Stream` and slices, for `fb_steps::LocalizationStep`: `pm_populations` (the offsets checked on the host and uploaded in the call that launches over them), `pm_schedule` (the tournament written into the host end and uploaded into a device slice the returned `Schedule` keeps by value), `pm_sweep` (every round over a `Schedule`, its `Angles` written by the pair kernel in the same call), `pm_functional`, `pm_identity`, `pm_schedule_len`, each length an always-on contract |
| `fb_integrals_df_metric` | nothing: rust/'s own host build (the v0.0.4 prototype's `experimental/rust/df_metric`); `boys` is the C++ tree's `eval_boys` with its coefficients (`gen_boys_series.py` copies them) | `metric(aux)`, and `metric_into(aux, m)` into a caller's slice: the spherical Coulomb metric (P\|Q) of a placed fitting set on the host, on rayon's threads, directly in the solid harmonics, far shell pairs from one table of `1/R` derivatives per center pair, the lower triangle packed at `fb_core_primitives::lower_index`; `metric_via_cartesian`, the route through the Cartesian integrals, kept to check and time against; `boys`, `hermite_expansion`, `far_pairs`; `HostPool`, the host integrals' rayon pool of a fixed size (`new(threads)`, `install(f)`), its threads joined when it drops, which `fb-steps` starts in setup (here because only the four host integral crates may name rayon). Device-free; its tests hold it to PySCF's `int2c2e` through i |
| `fb_integrals_s_overlap_host` | nothing: rust/'s own host build (the prototype's `experimental/rust/s_overlap_host`) | `overlap(basis)`, and `overlap_into(basis, s)` into a caller's slice: the spherical AO overlap S on the host, on rayon's threads, directly spherical by Hobson's theorem, negligible primitive pairs skipped, the lower triangle packed; `overlap_via_cartesian` (Obara-Saika and the sandwich), `negligible_pairs`. Device-free; held to PySCF through f |
| `fb_integrals_t_kinetic_host` | nothing: rust/'s own host build (the prototype's `experimental/rust/t_kinetic_host`) | `kinetic(basis)`, `kinetic_into(basis, t)` into a caller's slice, and `overlap_and_kinetic(basis)`, S and T from one pass: on the host, on rayon's threads, the lower triangle packed; `kinetic_via_cartesian`, `negligible_pairs`. Device-free; held to PySCF through f |
| `fb_integrals_v_potential_en_host` | nothing: rust/'s own host build (the prototype's `experimental/rust/v_potential_en_host`) | `potential_en(basis, charges)`, and `potential_en_into(basis, charges, v)` into a caller's slice: the spherical nuclear attraction V over `point_charges(molecule)` on the host, on rayon's threads, every nucleus summed into one Hermite table per primitive pair, the Boys function `fb_integrals_df_metric`'s; `potential_en_via_cartesian`, `negligible_pairs`, `PointCharge`. Device-free; held to PySCF through f |

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
`error_handling`; `recipes/sys` on `kernel_abi`, and `integrals/sys` and
`localization/sys` on `kernel_abi`, `core/device` and `error_handling`,
and `kernel_abi` on `error_handling`; `sad` on `basis`, `molecule`,
`atom`, `primitives` and `error_handling`; `io/npy` on `error_handling`;
`io/sad` on `io/npy` and `error_handling`; `tools/sad` on `sad`,
`io/basis`, `io/npy`, `basis`, `atom`, `types` and `error_handling`;
and the four host integral crates, `integrals/{s_overlap_host,t_kinetic_host}`,
`integrals/df_metric` and `integrals/v_potential_en_host`, on `basis`,
`core/primitives` and rayon, `v_potential_en_host` on `molecule` and
`df_metric` too.

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
not an `=` pin, holds the exact version. A git dependency (rustflow) is the
exception: it is pinned there alone by its full commit `rev`, and a
prototype under `experimental/rust/` names the same one
([`testing.md`](docs/testing.md), "What runs when").

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
  -p fb-io-npy -p fb-io-sad -p fb-tools-sad -p fb-core-pack \
  --no-default-features
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
by `fb-core-device` (each GPU's BLAS and dense-solver handles, the v0.0.4
layer's), and a workspace build unifies features, so every crate linking
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
`fb-kernel-abi`, `fb-device-check`, `fb-sad`, `fb-io-npy`, `fb-io-sad`, `fb-tools-sad`,
`fb-core-pack`, `fb-integrals-df-metric`, `fb-integrals-s-overlap-host`,
`fb-integrals-t-kinetic-host` and `fb-integrals-v-potential-en-host` do not link the library at all: they build and pass
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
`fb-integrals-sys`, `fb-localization-sys`, `fb-core-device`,
`fb-core-budget`, `fb-core-flow` and `fb-steps` always link (`fb-doc-check`'s `doc_lists.rs` holds these
three lists to the package graph). Use a
separate `CARGO_TARGET_DIR` per backend, and another for
`--no-default-features`: each `build.rs` bakes the library's path into the
rpath.

The v0.0.4 chain runs from a config through the `free-breakfast` binary,
`fb-steps`' own: setup, budget mode and execute mode, the SCF's energy and
its loop's times printed, and the run's record and products written into
`--output` (default `fbf_output`; `steps/tests/write_device.rs` reads them
back, and `steps/tests/examples.rs` runs every `examples/` config through
the binary):

```bash
cargo run -p fb-steps --release --bin free-breakfast -- examples/rhf.toml
cargo run -p fb-steps --release --bin free-breakfast -- steps/examples/gly10.toml \
  --devices 0,0 --mem-per-gpu-mib 512   # two logical GPUs on device 0, a smaller ERI buffer
cargo run -p fb-steps --bin free-breakfast -- steps/examples/gly10.toml --budget-only
```

`steps/tests/chain_device.rs` holds the chain to v0.0.2's recorded DF-RHF
energies (water, benzene, ethylene), at forced batches and rings of one, on
two logical GPUs, on one worker and on eight, through a replay, and water's
B to PySCF's (`steps/tests/data/b_reference.py`).
`steps/tests/recorded_device.rs` holds the steps to the old Rust path's
results, recorded once on an RTX 3080 and committed with their provenance
(`steps/tests/data/old_path.json`, water's S, H and metric as
`old_path_h2o_*.npy`), so `fb-steps` names no old-layer crate. Their writer,
the old `pipeline/scf`'s `tests/steps_reference.rs`, went with that crate
(#1873): `old_path.json` names it and the revision it ran at, and nothing
rewrites them now.

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
wrappers' refusals. None of those launchers has a safe call since the old
layer's tasks were deleted (#1874); the v0.0.4 layer's calls are
`integrals/sys/tests/{records_and_gather,record_bound,three_center_spherical}.rs`.
The FelizJueves units in
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
`integrals/kernels/eri/density_fitting/three_center/packed/kernels/`
(`speri3`, the bra Cartesian) and `.../three_center/spherical/kernels/`
(`ssperi3`, all three indices spherical).

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

2026-11-24, `recipes/trsm` (`fb_recipes_launch::trsm`, the tree's own blocked
triangular solve, its diagonal block forked as `fb_trsm_diagonal_f32` and
`_f64`, ABI version 7; `fb_core_wwr::blas` without its `trsm` binding and
with `trsv`, `scal` and `gemm_rows`; the Cholesky panel and the density
fit moved onto it), on the RTX 3080 (CUDA), the `default` preset:
`devtools/test.py pr --only cargo` passes at 9cb6a400, tier 3 396 tests;
tier 4 540 passed, the 13 tier-5 cases not run (one card visible). Every
side, fill, op and unit flag in `f32` and `f64` at orders 1 to 200 around
the 64-row block's edges and 1 to 97 right-hand sides, the unread triangle
and every padding row NaN, within `8 k ε` of the host's substitution; A and
B's padding bitwise as they went in; `alpha = 0` over an all-NaN A and B
exactly zero; `_C` the bits of `_T`. Compiled for HIP
(`devtools/cross-backend-check.sh --device-only`, PASS); not run on the
gfx1200. Not timed against the libraries here: the C++ recipe's README has
that table.
