# rust — free_breakfast's core, rebuilt in Rust

Can free_breakfast's C++ layers be rebuilt in Rust, one at a time, over
WarpWraps' C ABI (`wwr-sys`), keeping each one's contract? This Cargo
workspace is that rebuild. Its tree mirrors `src/`, one crate per module:
`core/log/` is the crate `fb-core-log` (`fb_core_log` in a path), and it is
`src/core/log`. It stands alone: nothing here uses the crates under
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
| `fb_core_device` | nothing in the C++ tree: the v0.0.4 prototype's `device.rs` but its kernel launches, the v0.0.4 layer (`docs/gpu_layer.md`, "The v0.0.4 runtime") | Over `wwr-sys`, `fb-recipes-sys`'s raw kernel bindings and the device-free crates: `ScopedDevice` (a device current on this thread, restored on drop, not `Send`), `device_count()`, `memory(physical)` (free and total bytes, for the budget's cap), `launch_status(what, status)` (a kernel library's launch status as a `Status`, the one bridge from `fb_kernel_abi::launch_status` to the runtime's statuses: a launch error under `wwr_status`'s name at the caller, out of memory naming the kernel stack and what to lower), `single_thread_host_blas()` (a linked host OpenBLAS, ROCm's on a HIP build, set to one thread and its load-time pool joined, once per process, by the steps' setup); `Stream` (non-blocking, on one device, `Send` and `Sync`) with `raw`, `physical`, `sync` (counted and timed, `sync_meter()`), `wait(event)`, `zero`, `fill`, and the asynchronous copies `upload`, `download` and `copy_device`, each making its stream's device current and each length, the device end's, an always-on contract; `upload` and `download` take the host end by value and return `Pending`, which holds it until a sync of the stream has covered the copy (`Pending::sync` hands it back, without blocking again after a covering sync; an upload's token, `Pending<'_, H>`, takes `and_upload(stream, dst)`, the same host end to another stream, joined through an event, so one token covers every GPU's copy; a download's, `Pending<'_, H, false>`, has none); `Event` (`record(stream)`, `sync`); `DeviceSlice` / `DeviceSliceMut` and the cuts `part`, `part_mut` into `Part` / `PartMut`, each cut an always-on contract; and `HostSlice` / `HostSliceMut`, the host end of a copy, an `unsafe` promise its owner makes that only the value reaches the bytes and a leaked one leaves them valid; `GpuResources::new(gpu, physical)`, a GPU's stream with a BLAS and a dense-solver handle made on it, inside a `Mutex`, its methods reached through the guard: `bind_workspace` (once, `HANDLE_WORKSPACE_BYTES`), the checked `gemm`, `gemm_strided_batched`, `gemv`, `geam`, `dgmm`, `syrk`, `syevd_lwork` and `syevd` (its work held to the repeated query at the caller's line), the recipes `trsm` and `cholesky` (`CHOLESKY_TILE`) and the recipes' `inverse_sqrt` (0 at or below a floor), `diis_solve` (the bordered DIIS system, the unit selector at `newest` when singular) and `mirror_triangle` (the strict triangle opposite a `fill` written from it, a `syrk`'s other half) on the guard's one stream, the kernels through `fb-recipes-sys`'s raw bindings; the BLAS mode enums those calls take (`wwr_blas_operation`, `wwr_blas_side_mode`, `wwr_blas_fill_mode`, `wwr_blas_diag_type`), re-exported. A create aborts; every other runtime failure is a `Status` |
| `fb_core_budget` | the v0.0.4 prototype's `budget.rs` and `host.rs`; the C++ tree's pre-flight and footprints, which it replaces by counting the allocations themselves (`docs/memory.md`, "The budget") | The v0.0.4 layer's memory budget, over `fb-core-device`, `fb-core-pack` and `wwr-sys`: `Ledger` (per-kind atomic counters, device and pinned per gpu and the heap for the host, the `over` flag, the per-step table, `finish`, the budget DAG's final node, which packs each gpu and kind's scratch by the closure, adds the packed peaks and the handle workspace (`HANDLE_WORKSPACE_BYTES`) per gpu, and refuses the plan with the whole `table` as an `Allocation` error or allocates the slabs and workspaces, `take_workspace` for a `GpuResources` to bind, and `lease`), `Budget` (`device`, `pinned`, `heap`, `scratch`, `locks`), the buffers `DeviceBuf`, `PinnedBuf` and `HeapSlab` (a `Box<[T]>`, huge pages asked for and every page touched), `PinnedBuf`, `HeapSlab` and `HostPart` each the host end of a copy (`HostSlice`); `Layout` and `Span`, `Lease` (`fill_host`, and `fill_device` on a stream, the wrapper's NaN fill) with its tenants `ScratchBuf` and `HostPart`, each carve recorded so no two meet; `PinnedBuf::tenant` and `DeviceBuf::tenant`, typed tenants of an `Output`'s own buffer by a `Span` (`PinnedTenant`, the host end of a copy; `DeviceTenant`), each carve recorded in the buffer and cleared by the tenant's drop, a leaked one's kept, each tenant holding the allocation alive, the whole buffer refusing to be reached while a carve is recorded (`docs/memory.md`, "Tenants of an Output's buffers"); `Cap` / `Caps` (`resolve`, `read`, every cap clamped to the reading, `device_margin`; the host's reading cgroup-aware), `Backing` (`Real`, `Host`, `Dry`), `Kind`, `Plain` (sealed), `LockId` / `ManagerId` (rank by derived `Ord`), `bytes`; `AllocationRecord`, the `#[global_allocator]` a test binary installs, `watch` around a `run` and `RECORD_THRESHOLD` (1 MiB), with `Ledger::measure_execute` / `heap_growth`, the table's unbudgeted row |
| `fb_core_flow` | nothing in the C++ tree: the v0.0.4 prototype's `flow.rs` and `flow_tuples.rs`, with the step wrapper that sat on the prototype's `MainBus` (`docs/programming_model.md`, "Steps on rustflow") | The v0.0.4 layer's step DAG on rustflow, over `fb-core-budget` and `fb-core-device`: `Step` (`NAME`, `Output`, `Reads`, `STREAMS`, `slot`, `placement`, `mem_budget`, `run`), `Slot` (the `Output` behind an `RwLock`, the step's `Lane` beside it), `View` (`shared`, `read`, which compiles only for a step in `Reads`: `ReadSet`, `Has`, `At`), `Read`, `Placement` (`Host`, `Gpu0`, `AllGpus`), `Bus` (`shared`, `ledger`, `locks`), `Locks` (each GPU's `GpuResources` in its `Mutex` and each manager's lock, no `Mutex` reachable, `query` given a `Budget`), `Held` (the guards the wrapper took in rank order: `gpu`, `all_gpus`, `taken`), `Lane` / `GpuLane` (a step's streams per GPU), `Dag` (`add`, `build`, refusing a read the config did not ask for, naming both, a step added twice, a cycle), `Plan` (`budget`, every `mem_budget` then `Ledger::finish`; `execute`, every `run` inside the wrapper, a failed flag per node, the first error in topological order, replayable; `output`, `edges`), `StepTime` (a step's times and its lock acquisitions: the locks, whether it waited, when it held and released them), `fan_out` (one scoped thread per GPU), rustflow's `Executor` re-exported, and under the `test-support` feature `controls` (`skip_step_end_sync`, `delay_host_steps`), the step-end sync's negative control for a dependent's tests |
| `fb_steps` | the v0.0.4 prototype's `bus.rs`, its seven steps (`overlap`, `core_hamiltonian`, `metric`, `guess`, `orthogonalizer`, `eri`, `scf`) and its `run`; the C++ tree's `src/pipeline` steps of the same names (`docs/programming_model.md`, "The MainBus and its steps") | The v0.0.4 layer's MainBus and steps, over `fb-core-flow`: `Config` (`parse`, `read`, a refusal naming the file and the line, an unknown key among them (the old binary's `[basis_set]`, `num_gpus` and `[memory]` too); `calculation` (`Calculation`: `"overlap"`, `"guess"`, `"rhf"`, `"localization"`, one row per calculation of the steps it adds, the tables it reads, filled at their defaults, and its products; a table it does not read refused at the table's line, `"rhf"` and `"localization"` with no `[eri]` at `calculation`'s; absent, the tables present pick the steps); `[molecule]` (`MoleculeTable`: `xyz`, `charge` and `multiplicity`, default 0 and 1, `occupations(electrons) -> (n_alpha, n_beta)`, an impossible multiplicity `InvalidConfig`), `[basis]` (`name`, a set the build carries, by name, or a BSE JSON path, a relative one joined to the config's directory; `[eri] aux_basis` the same), `[guess]` (`GuessTable`: `type` (`GuessType`, `"sad"`) or `from`, a float64 `.npy` D joined to the config's directory and loaded in place of the SAD density, its staging and budget the SAD guess's; refused in setup before anything is allocated, naming the file and the line: both keys together and a shape other than `n_ao × n_ao` `InvalidConfig`, an unreadable file `Io`, an open shell `Unsupported`, since `d.npy` is ROHF's spin densities summed; `tests/guess_device.rs` restarts water from the `d.npy` a run wrote, its first build within `energy_tolerance` of the converged energy), `[eri]`, `[scf]` (`ScfTable`: `max_iterations`, `energy_tolerance`, `commutator_tolerance`, `diis_history`, `stage_sync`, `check`), `[localization]` (`LocalizationTable`: `mode` (`LocalizationMode`, `"ibo"` refused `Unsupported`), `exponent` 2 or 4, `max_sweeps`, `functional_tolerance`, `rotation_gain_threshold`, `check`), `[resources]`), `MainBus::setup(config, backing)` (the geometry and bases on the host, `Dims::count` filling `Dims` (`n_alpha` and `n_beta`, the latter also the doubly occupied) and its `ClassDim`s by `for_each_pair`, a walk that builds no array the host integrals' `HostPool` of `[resources] host_threads`, each GPU's `GpuResources`, the caps), `Shared` (`host_pool`, `syevd_lwork`, 0 with no device), `plan(bus)` (`"overlap"` the overlap alone, `"guess"` the guess alone); the host steps `OverlapStep`, `CoreHamiltonianStep`, `MetricStep` (every GPU, one staging under one token) and `GuessStep` (the SAD density, or `[guess] from`'s D read into its staging by `fb_io_npy::read_npy_into`), each built inside the host pool through the host integral crates into its pinned staging and uploaded, no lock; `OrthogonalizerStep`, X = S^(-1/2) on gpu0 under its lock (`EIGENVALUE_FLOOR`); `EriStep` (every GPU, `[eri] mem_per_gpu` in bytes, `DEFAULT_MEM_PER_GPU`, `ring`, 2 or 1, and `schwarz_threshold`, default 1e-12, a shell pair under it skipped in B's build and its columns zero, `EriManager::dropped_pairs`; `EriStep::least_mem_per_gpu`, under which the plan is refused) and its `Output`, the `EriManager` (L on every GPU by the tree's Cholesky, B on the host heap built over pair batches through each GPU's ring of `ssperi3` launches and blocked triangular fits, each GPU scattering into its own contiguous columns of B, and `build_jk(shared, held, &[Density], j, k)`, a J/K pass from D or the occupied C, one density or an open shell's two in one pass, through each GPU's ring of aux batches on the manager's own streams, reduced onto gpu0 through the host, under the manager's lock and every GPU's, which the reader declares; `passes()`, the passes since B was last built); `ScfStep`, the DF-RHF loop with DIIS, DF-ROHF at a multiplicity above 1 (both spins' factors in one J/K pass, the Roothaan effective Fock matrix, the published open orbitals canonical in `(Fa + Fb)/2`), a plain loop in `run` on gpu0 in its scratch, every J/K pass from a factor, the first from the guess's eigen-factor above 1e-12 of its largest eigenvalue (`ScfOutput`: `energy`, `residual`, `iterations`, `converged`, `guess_rank`, `meter` (a `LoopMeter`), C, ε and D on the host, D on gpu0, and an open shell's effective Fock matrix `fock` on gpu0); `LocalizationStep`, Pipek-Mezey on gpu0 under its lock over each block of the SCF's occupied columns, a closed shell's doubly occupied one, an open shell's doubly then singly occupied ones each among itself, each block's arrays sized by its width after the previous block's (`Dims::n_atom` beside the occupations), a plain sweep loop in `run` whose turn read carries L, `C U` and the sort key `diag(Uᵀ diag(ε) U)`, each block's converged columns stably sorted by `<i|F|i>` (`LocalizationOutput`: `c` on the host, `n_cols`, `functional_before`, `functional_after`, `sweeps`, each summed over the blocks; `tests/open_localization_device.rs` holds H₂O⁺ and triplet CH₂ at exponents 2 and 4 to PySCF's DF-ROHF localization, `tests/data/localization_rohf.json`, at the first-order bars of `tests/data/open_localization_reference.py`); `run(config, backing)`, setup, budget mode and, unless dry, execute mode over the run's one executor, into a `Run` (the plan, the table, the times) whose `replay` executes again in the same buffers, and whose `write(dir)` puts the run on disk on the host after execute mode: the record, `budget.txt` (the table) and `timing.json` (the run's and each step's times), and with `[scf]` `scf.json` (`energy`, `residual`, `iterations`, `converged`, `n_alpha`, `n_beta`, ε, each double shortest round-trip) and `c.npy` and `d.npy` (float64, Fortran order, through `fb-io-npy`), and with `[localization]` `localization.json` (`mode`, `exponent`, L before and after, `sweeps`) and `c_local.npy` (`n_ao × n_alpha`); `overlap.npy` for `"overlap"` and `d.npy` for `"guess"`, from a host copy (`OverlapOutput::s_host`, `GuessOutput::d_host`) budgeted only for that calculation. The `free-breakfast` binary (`src/main.rs`): `free-breakfast <config.toml>` with `--output` (default `fbf_output`), `--budget-only`, the exact table with no device, `--devices`, `--threads`, `--mem-per-gpu-mib` and `--stage-sync`; its `main` the one place an error is logged, printing `completed: N errors, N warnings` and exiting with `ErrorKind::exit_code`; no old-layer crate in its graph. `rust/examples/` holds `overlap.toml`, `guess.toml`, `rhf.toml`, `rohf.toml`, `localization.toml`, `resources.toml` (the RHF with `[resources]`, `[eri] mem_per_gpu` and `ring` written) and `two_gpus.toml` (the RHF on devices 0 and 1, tier 5), every one run through the binary by `tests/examples.rs`, which also holds that every key of `Config` is set by an example |
| `fb_core_log` | `src/core/log` | The one backend behind the [`log`](https://crates.io/crates/log) facade: spdlog's line format (local time, OS thread id, `file:line`), the seven levels, `set_level` / `should_log` / `flush`, one file sink, the warning counter and the error counter (`error_count`, a failed destroy or free and the boundary's line, which `main` prints), and `log_at` for a location already held; the value checks `le!`, `lt!`, `eq!` and `fits_i32!` (`check`), which a contract's condition is written with so the panic hook ends its line with the operands' values |
| `fb_core_error_handling` | `src/core/error_handling` | `ErrorKind` (with the pinned exit codes as `exit_code`), `Error` and its `#[track_caller]` factories, `Result<T>` / `Status`, `check_or_abort` |
| `fb_core_types` | `src/core/types`, what the host modules name | `Double3`, `Vec<double, 3>`: `#[repr(C)]` with the backend's `double3` layout. The fixed-width integers and `make_vec` are the language's |
| `fb_core_primitives` | `src/core/primitives`, what the host modules name | `double_factorial_odd`, `factorial`, `binomial` as `const fn`, bitwise the C++ loops; `lower_index`, the host integral crates' packed lower triangle, rust/'s own. `idivup` / `align_up` are `div_ceil` / `next_multiple_of`, and the element-wise functions `f64`'s methods |
| `fb_core_constants` | `src/core/constants`, what the host modules name | `BOHR_PER_ANGSTROM`, `kBohrPerAngstrom<double>`, asserted against the C++ double's bit pattern |
| `fb_core_pack` | nothing: the C++ steps allocate as they go; the v0.0.4 prototype's `src/pack.rs` | Step scratch packed by the DAG (`docs/memory.md`, "Step scratch"): `closure` (the transitive closure, a `Reach` bitset of successors per step), `ordered`, `pack` (first fit, largest first and then by step name, each `Region` at the lowest offset clear of every unordered region placed) into a `Packed` with its peak, plain sum and `lower_bound`, the heaviest antichain and the regions that make it. Device-free |
| `fb_atom` | `src/atom` | `Element` (`#[repr(u8)]`, Z as the discriminant, `Sentinel` = 0), `z` / `from_z`, `symbol` / `from_symbol` (case-insensitive, `Sentinel` for none), and `Atom`, a `Double3` position in bohr and an `Element` |
| `fb_molecule` | `src/molecule`, its host half | `Molecule` as host data (atoms, charge, multiplicity; no atoms aborts), `distance`, `nuclear_charge`, `electron_count`, and the nuclear repulsion energy, gradient and Hessian. The nuclei on the device (`view`, `NucleiView`, `molecule_footprint`) are not ported: the v0.0.4 layer forms V on the host |
| `fb_basis` | `src/basis` | `Shell` and `Primitive`, the counts (`function_count_cart` / `function_count_sphe`) and `MAX_ANGULAR_MOMENTUM` (3, the C++ `kMaxAngularMomentum`), `cartesian_powers`, `solid_harmonic_block`, `BasisSet` (its Cartesian and spherical AO layouts, the per-atom shell grouping, the name) and `spherical_transform_matrix(basis, u)`, U written into a slice the caller hands it, `ShellStamp` with `primitive_normalized` and `place`, and `BasisStamp` with `place` onto a `Molecule`. Contractions are not renormalized, as in the C++ |
| `fb_recipes_sys` | `src/recipes/orthogonalize`'s `inverse_sqrt`, `src/recipes/symmetrize`'s two tile kernels, `src/recipes/cholesky`'s diagonal-tile kernel, `src/recipes/diis`'s bordered solve and `src/recipes/trsm`'s diagonal-block kernel, forked, and `src/recipes/trsm`'s blocked solve over it, ported to `recipes/kernels/` (with the device headers it includes, trimmed, in `device_headers/`) | The raw FFI over `libfb_recipes_kernels.so`, built by `CMakeLists.txt` here, each an `extern "C"` launch wrapper that only enqueues: `fb_inverse_sqrt` (the eigenvalue map `λ → λ^(-1/2)`, 0 at or below a floor passed in, optionally scaled by a value on the device), `fb_symmetrize` and `fb_mirror_triangle` (the exact average `(A + Aᵀ)/2` and one strict triangle copied onto the other, in place, refusing a negative order, an `lda` below it or a null matrix), `fb_cholesky_diagonal` (one diagonal tile of a blocked Cholesky in shared memory, the flag naming the first non-positive minor, the tile also written packed for the panel's trsm) and its tile order `FB_CHOLESKY_TILE` (64), `fb_diis_solve` (the bordered DIIS system solved by Gauss-Jordan in one block's shared memory, the unit selector at the newest column and a flag when a pivot falls below `1e-14`, refusing the C++ doc comment's bounds and null pointers) and its order `FB_DIIS_SOLVE_MAX_DIM` (65), `fb_trsm_diagonal_f64` (`M X = s B` in place for one triangular diagonal block of a blocked solve and every right-hand side, the triangle packed and 48 columns of B to a thread block in shared memory, either triangle, M or B read transposed, a unit diagonal unread, `s = 0` zeroing B with neither read; refusing a leading dimension under its column and null pointers beside the C++ launcher's arguments) and the block's order `FB_TRSM_TILE` (64), `fb_launch_status` and `FB_DEVICE_ALIGNMENT` (`docs/gpu_layer.md`, "Layering"), and `check_library()`, the load check of the library's ABI version and backend, compared by `fb-kernel-abi`. The bindings are generated from `recipes/kernels/fb_recipes.h` by `devtools/rust-bindings.py` and committed. The safe calls over the bindings are `fb-core-device`'s, which calls them directly (`docs/gpu_layer.md`, "The safe wrapper"). It declares the library's `links` key, so a change under `recipes/kernels/` selects it and every crate above it |
| `fb_integrals_sys` | `src/integrals/mmd`'s launchers (records, bound, dressed bound; not its order, the records left in enumeration order), `src/integrals/one_electron/{overlap,kinetic,potential_en}`'s and `src/integrals/eri/density_fitting/metric`'s and its three-center launcher's shape, forked to `integrals/kernels/`, with the FelizJueves units they launch (`pair_bound`, `record_bound`, the overlap's, kinetic energy's and nuclear attraction's ten classes each, and the Boys routine `eval_boys.h` they and the density-fitting units include, emitted there by `tools/emit/*.py --out-root`; the metric's 28 classes, `tools/emit/df_kernels.py`'s; the packed three-center integrals' 70, `speri3<la, lb, lP>`, the aux index in real solid harmonics, `tools/emit/packed_three_center_kernels.py`'s, rust/'s own in `integrals/kernels/eri/density_fitting/three_center/packed/`, and the fully spherical ones' 70, `ssperi3<la, lb, lP>` (FelizJueves #745), all three indices in real solid harmonics, from the same tool, in `.../three_center/spherical/`) | The raw FFI over `libfb_integrals_kernels.so`: `fb_mmd_records`, `fb_mmd_bound`, `fb_ordered_pair_bound`, `fb_overlap_class`, `fb_kinetic_class`, `fb_potential_en_class`, `fb_metric_class`, `fb_three_center_packed_class` (the only Cartesian-bra (mn\|P): the square tensor's kernels, launchers and symmetrization left in #1725), `fb_three_center_spherical_class` (the same over `ssperi3`) and `fb_packed_gather`, each an `extern "C"` launch wrapper that only enqueues, a class wrapper refusing a class outside its table as `FB_LAUNCH_UNSUPPORTED`; the tables' limits (`FB_*_MAX_MOMENTUM`, the aux basis's `FB_METRIC_MAX_MOMENTUM` and `FB_THREE_CENTER_AUX_MAX_MOMENTUM` among them); the density-fitting record, pair and cell field counts (`FB_METRIC_*_FIELDS`, `FB_THREE_CENTER_AUX_FIELDS` and `FB_THREE_CENTER_PACKED_CELL_FIELDS` among them), the three-center ones read by its safe layer; and the four structs that cross by pointer, `fb_primitive_pair_class_view` (the one-electron launchers', and the three-center one's for its bra records), `fb_primitive_pair_source`, `fb_primitive_pair_records` and `fb_nuclei_view` (`src/molecule`'s `NucleiView`), `#[repr(C)]` with their layout asserted on both sides; and `check_library()`, as in `fb_recipes_sys`. The bindings are generated from `integrals/kernels/fb_integrals.h` and committed. Beside them, the safe layer, the v0.0.4 layer's, over `fb-core-device`'s `Stream` and slices: `three_center_spherical_class`, one class of the fully spherical (μν\|P) into a batch of `n_col · naux` doubles (`n_col` spherical bra columns, aux fastest), every slice's length and `n_col · naux` within `i32` a `#[requires]` naming the kernel, and the cells handed in on the host by value, each one's record ranges and block checked inside the slices it indexes and then uploaded by the call itself, which returns the upload's token, so no device table a caller fills reaches the launch; held to the experiment `df_batched/single_gpu_spherical`'s output on water at 1e-12 of the largest element (`tests/three_center_spherical.rs`). Beside it, on the same terms, `mmd_records` (one class's records through `fb_mmd_records` into `RECORD_F64_FIELDS` and `RECORD_I32_FIELDS` fields of `n_rec`, field-major, the shell and pair tables handed in on the host, each shell's primitive range, each pair's shells and record prefix checked, and uploaded by the call) and `packed_gather` (the slab zeroed, then `fb_packed_gather`, each column's row and column handed in on the host, checked inside the matrix and uploaded by the call), each returning the upload's token and each refusing a short slice or a stray index before anything is enqueued (`tests/records_and_gather.rs`); and `record_bound` (a class's dressed Schwarz bounds through `fb_ordered_pair_bound` from those records, every length a contract; an s pair's against its closed form, `tests/record_bound.rs`), the eri step's screen. The other launchers have no safe call, their old-layer task builders deleted (#1874); `tests/overlap.rs`, `tests/core_hamiltonian.rs` and `tests/density_fitting.rs` hold the overlap, kinetic, nuclear-attraction, metric and packed three-center ones raw, one s class each against its closed form. The private `launched`, a launch status as a `Status` naming the wrapper and its class's momenta, is the one conversion every launcher here goes through |
| `fb_device_check` | nothing: a check on the kernel libraries | Tier-3 tests: no launch wrapper under `rust/` allocates, synchronizes, touches an event, copies, sets the device or opens a profiler range (a grep of every `.cu`), every `-sys` crate's committed bindings are what the script generates now, no header the bindings read names a block, grid or warp, and every emitted unit equals its C++ twin byte for byte |
| `fb_kernel_abi` | nothing: what the kernel libraries share above their `-sys` crates | The one Rust copy of `device_headers/fb_launch.h`, generated by `devtools/rust-bindings.py` and re-exported by each `-sys` crate's bindings (`docs/gpu_layer.md`, "From bridge to C ABI"): `fb_launch_status` and its values, `FB_DEVICE_ALIGNMENT`, `fb_backend` and its values. `LaunchStatus`, the newtype a launch wrapper returns, and `Backend`, a library's `fb_backend` read by value (`TryFrom`, `Display`, the build's name through `const fn named`). `check`, the load check's comparison: a library's ABI version and `fb_backend` against the ABI version and `Backend` its `-sys` crate expects, a mismatch of either refused naming both values. Each `-sys` crate's `check_library()` calls it. And `launch_status`, the one conversion of a launch wrapper's `fb_launch_status` to a `Status` (`docs/gpu_layer.md`, "Statuses"): a refusal as `Unsupported` or `InvalidConfig`, a positive `wwr_status` handed to the caller's bridge, the out-of-memory one (the code the caller names) also saying the kernel stack is the likely cause and what to lower, any other value `GpuRuntime`, each naming the wrapper and the status. Every safe layer calls it through `fb_core_device::launch_status`, the one bridge to the runtime's statuses. It links nothing, so its tests are tier 3 |
| `fb_io_xyz` | `src/io/xyz` | `parse_xyz` / `read_xyz`: a standard XYZ geometry into a `Molecule`, Angstrom to bohr by `BOHR_PER_ANGSTROM`, every refusal (too short, a bad count, a malformed line, an unknown symbol, a count the lines disagree with, a file that cannot be read) an `Io` error. A field must parse as a whole token, where the C++'s `operator>>` reads a prefix |
| `fb_io_basis` | `src/io/basis` | `parse_basis` / `read_basis`: a BSE JSON document into a `BasisStamp` (sp blocks and general contractions split, spherical shells only: one Cartesian or ambiguous shell refuses the document), over [`serde_json`](https://crates.io/crates/serde_json); `is_builtin_basis` and `resolve_basis`, a name (spelling folded) or else a path, over the eleven sets of `data/basis/` compiled in with `include_str!`, only that branch naming the stamp; `is_builtin_aux_basis` and `resolve_aux_basis`, the same over the thirteen density-fitting sets of `data/aux_basis/`, a table of its own disjoint from the orbital one (about 16 MB, compiled in for the reason the orbital sets are: nothing is installed, so a name cannot be misplaced) |
| `fb_sad` | `src/sad`, its serial form | `sad_density`, and `sad_density_into` into a caller's slice, the SAD guess: one spherically averaged, fractionally occupied atomic RHF per distinct (Z, shells) over closed-form one-center integrals, each density placed on its atoms' diagonal blocks, and `SadStampLookup`, the baked block that skips an element's SCF (checked by size). Every loop in the C++ order, so its stamps compare bitwise with the C++ generator's. |
| `fb_io_npy` | `src/io/npy` | `NpyArray` (a shape and its float64 elements, column-major) and `element_count`; `parse_npy` / `read_npy`, versions 1.0 to 3.0, `'<f8'` only, a C-order payload reordered, every refusal of a document (a bad magic, version, length or header field, another dtype, a payload the shape disagrees with, a file that cannot be read) an `Io` error and an overflowing shape `InvalidConfig`; `read_npy_into(path, shape, out)`, a file of a known shape streamed into a buffer already held (a step's pinned staging), C order placed element by element, another shape `InvalidConfig` naming the path; `npy_header`, `npy_to_bytes` and `write_npy(path, shape, data)`, the data borrowed so a CPU node writes its row in place, version 1.0 with `np.save`'s canonical header, byte for byte the C++ writer's (the `data/sad/STO-3G/` stamps written back are their files) |
| `fb_io_sad` | `src/io/sad` | `builtin_sad_density`, `fb-sad`'s `SadStampLookup`: every `data/sad/<basis>/<Z>.npy` compiled in with `include_bytes!` (the table written by `build.rs`), found by the directory name compared exactly with `BasisSet::name` and the atomic number, parsed by `fb-io-npy`; bytes that are not a square matrix abort. A directory naming no `data/basis` set fails a unit test, a stem that is not a three-digit Z the build |
| `fb_tools_sad`, the `free-breakfast-sad-stamps` binary | `tools/sad` | The generator of `data/sad/`: `free-breakfast-sad-stamps <repo>/data (<basis>... \| --all)` resolves each named set through `fb-io-basis`'s compiled-in table (a set the binary does not carry refused `InvalidConfig`; `--all`, every `<data>/basis/*.json` by its document's BSE `name`), runs `fb-sad`'s atomic SCF for each element alone at the origin, each element's serial SCF on the next free one of its own scoped threads (as many as the CPU affinity allows, no more than the elements), with no old-layer crate, and writes `<data>/sad/<basis>/<Z>.npy` through `fb-io-npy` after removing each regenerated set's directory (every one under `--all`); an atom too large for the SCF (`Unsupported`) is skipped, any other failure exits with its kind's code before anything is removed. It links `fb-sad`, never `fb-io-sad`, so it is not linked against its own output. Its STO-3G output is the committed `data/sad/STO-3G/` byte for byte, `tests/stamps.rs` |
| `fb_localization_sys` | `src/localization/pipek_mezey`'s launchers (`pipek_mezey_bridge.h`, `pipek_mezey_kernels.cu`), forked to `localization/kernels/` (with the block and warp reductions it includes, forked into `device_headers/`) | The raw FFI over `libfb_localization_kernels.so`, built by `CMakeLists.txt` here: `fb_pm_populations`, `fb_pm_pair_angles`, `fb_pm_rotate_populations`, `fb_pm_rotate_columns`, `fb_pm_functional` and `fb_pm_identity`, the C++ launchers as `extern "C"` wrappers that only enqueue and refuse a count below 1, an `n_occ` past `FB_PM_MAX_OCCUPIED` (65535), an exponent other than 2 or 4 and a null pointer; and `check_library()`. Three of the launches trust indices in device memory (the atoms' AO offsets, a round's orbital pairs, the partners the pairs give), so each safe call uploads them itself or holds them in a type only it makes. The safe layer is the v0.0.4 layer's, over `fb-core-device`'s `Stream` and slices, for `fb_steps::LocalizationStep`: `pm_populations` (the offsets checked on the host and uploaded in the call that launches over them), `pm_schedule` (the tournament written into the host end and uploaded into a device slice the returned `Schedule` keeps by value), `pm_sweep` (every round over a `Schedule`, its `Angles` written by the pair kernel in the same call), `pm_functional`, `pm_identity`, `pm_schedule_len`, each length an always-on contract |
| `fb_integrals_df_metric` | nothing: rust/'s own host build; `boys` is the C++ tree's `eval_boys` with its coefficients (`gen_boys_series.py` copies them) | `metric(aux)`, and `metric_into(aux, m)` into a caller's slice: the spherical Coulomb metric (P\|Q) of a placed fitting set on the host, on rayon's threads, directly in the solid harmonics, far shell pairs from one table of `1/R` derivatives per center pair, the lower triangle packed at `fb_core_primitives::lower_index`; `metric_via_cartesian`, the route through the Cartesian integrals, kept to check and time against; `boys`; `HostPool`, the host integrals' rayon pool of a fixed size (`new(threads)`, `install(f)`), its threads joined when it drops, which `fb-steps` starts in setup (here because only the four host integral crates may name rayon). Device-free; its tests hold it to PySCF's `int2c2e` through i |
| `fb_integrals_s_overlap_host` | nothing: rust/'s own host build | `overlap(basis)`, and `overlap_into(basis, s)` into a caller's slice: the spherical AO overlap S on the host, on rayon's threads, directly spherical by Hobson's theorem, negligible primitive pairs skipped, the lower triangle packed; `overlap_via_cartesian` (Obara-Saika and the sandwich). Device-free; held to PySCF through f |
| `fb_integrals_t_kinetic_host` | nothing: rust/'s own host build | `kinetic(basis)`, and `kinetic_into(basis, t)` into a caller's slice: the spherical AO kinetic energy T on the host, on rayon's threads, the lower triangle packed; `kinetic_via_cartesian`. Device-free; held to PySCF through f |
| `fb_integrals_v_potential_en_host` | nothing: rust/'s own host build | `potential_en(basis, charges)`, and `potential_en_into(basis, charges, v)` into a caller's slice: the spherical nuclear attraction V over `point_charges(molecule)` on the host, on rayon's threads, every nucleus summed into one Hermite table per primitive pair, the Boys function `fb_integrals_df_metric`'s; `potential_en_via_cartesian`, `PointCharge`. Device-free; held to PySCF through f |

One more crate holds no port:
`fb-doc-check` (`doc_check/`), the checks of the tree's text and package
graph, which its crate doc lists. There is no umbrella crate. The crate graph is the selection graph
([`testing.md`](docs/testing.md), "Selection: the workspace graph is the
selection graph"): `log` at the bottom, `error_handling` on it, `device`
on both (and on `kernel_abi` and `recipes/sys`), `budget` on `device` and
`pack`, and `flow` on `budget` and `device`; `log`'s macros are imported
from `log` directly and `critical!` lives in `fb-core-log`. Above `core`, `atom` stands on `types`, and `molecule` on
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
  (`fb_core_device`'s, at `error`).
- The move constructors, `release()`, the moved-from state and the
  null-handle sentinel. A Rust move leaves nothing behind to drop, and a
  failed create yields an `Err`, not an empty handle.
- `device_handle_pool` and `kit::DeviceHandle`, the rung a buffer allocates
  from a stream-ordered pool through. A `Budget` allocates each buffer
  once, in `mem_budget` ([`memory.md`](docs/memory.md), "The budget").
- Dangling views. A view borrows the handle it came from.

- The `FBF_LOG_*` macros, and the rule that formatting goes only through
  them. `log`'s macros check the level before they format.
- `LocatedFormat` and the `err_*` helpers. A `#[track_caller]` factory
  captures its caller even with a format string, so
  `Err(Error::io(format!(...)))` does what `err_io(...)` does.
- `FBF_TRY`. It is `?`, which moves the error's origin through unchanged.

One thing goes the other way, where Rust asks for more than the C++ does:

- wwr's borrow-safe operations (`sync`, `record`, `launch`, ...) are free
  functions on a raw handle. A destroyed handle there is a use-after-free in
  the runtime, so the raw forms are `unsafe fn`, and each owner has methods
  that call them safely.

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
cmake --build --preset cuda            # libwwr_capi_{runtime_api,blas,solver}.so and the kernel libraries
cargo test --workspace -- --include-ignored  # debug, device cases and contract tests
cargo test --workspace --release       # the debug contracts' tests are skipped
cargo test -p fb-core-budget           # one crate
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
CMake step at all. Its crates are not listed by hand: `devtools/test.py pr
--only cargo`, from the repository root, finds them in the graph and runs
them (below).

`CMakeLists.txt` here is a CMake project of its own, which the root
`CMakeLists.txt` does not add: it configures `deps/WarpWraps` alone (its
tests, examples and install rules off) and builds
`wwr.capi.runtime_api`, `wwr.capi.blas` and `wwr.capi.solver`,
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
built. Only the `runtime_api`, `blas` and `solver` surfaces of `wwr-sys`
are linked: `blas` and `solver` are asked for by `fb-core-device` (each
GPU's BLAS and dense-solver handles), and a workspace build unifies features, so every crate linking
`wwr-sys` in it links against `libwwr_capi_blas.so` and
`libwwr_capi_solver.so` (only a binary that calls one keeps it under
`NEEDED`).

The GPU tests (the ignored ones) use device 0 and need room on it for a
context; a tier-5 case (`tier 5: needs two GPUs`, named `tier5_*`) uses
devices 0 and 1, or every device it sees, and the gates run it only where
wwr sees two (`docs/testing.md`, "Selecting GPU tiers in cargo"). The host-only
tests (the fake handle kind, error codes and contracts) need no GPU, and
`fb-core-log`, `fb-core-error-handling`, `fb-core-types`,
`fb-core-primitives`, `fb-core-constants`, `fb-atom`, `fb-molecule`,
`fb-basis`, `fb-io-xyz`, `fb-io-basis`, `fb-doc-check`,
`fb-kernel-abi`, `fb-device-check`, `fb-sad`, `fb-io-npy`, `fb-io-sad`, `fb-tools-sad`,
`fb-core-pack`, `fb-integrals-df-metric`, `fb-integrals-s-overlap-host`,
`fb-integrals-t-kinetic-host` and `fb-integrals-v-potential-en-host` do not link the library at all: they build and pass
with it missing.
Only `fb-recipes-sys`, `fb-integrals-sys`, `fb-localization-sys`,
`fb-core-device`, `fb-core-budget`, `fb-core-flow` and `fb-steps` always
link (`fb-doc-check`'s `doc_lists.rs` holds these two lists to the package
graph). Use a separate `CARGO_TARGET_DIR` per backend: each `build.rs`
bakes the library's path into the rpath.

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

`steps/tests/chain_device.rs` holds the chain to the v0.0.2 prototype's recorded DF-RHF
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

The logger is process-global, so `core/error_handling/tests/log.rs` is a
binary of its own too,
and its in-process checks are one test. Its abort and exit cases re-run
that binary as a child and read the child's stdout: what they check (the
critical line flushed before `abort`, the buffered tail flushed at exit) is
what survives the process ending.

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
wrappers' refusals. None of those launchers has a safe call; the steps'
calls are
`integrals/sys/tests/{records_and_gather,record_bound,three_center_spherical}.rs`.
The FelizJueves units in
`integrals/kernels/` are emitted, not written: after bumping
`deps/FelizJueves`, emit the tools' whole trees into a scratch directory
and copy the six directories this library carries from there, byte for
byte what the defaults write under `src/integrals/`.
`tools/emit/df_kernels.py` writes nothing else, so its metric family, the
one rust/ carries (`--family metric`), is emitted straight into
`integrals/kernels/`:

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
tools/emit/packed_three_center_kernels.py
```

`tools/emit/packed_three_center_kernels.py` writes the packed
three-center units, which the C++ tree does not have, straight into
`integrals/kernels/eri/density_fitting/three_center/packed/kernels/`
(`speri3`, the bra Cartesian) and `.../three_center/spherical/kernels/`
(`ssperi3`, all three indices spherical).

`fb-doc-check`'s checks are listed in one place, its crate doc
(`doc_check/src/lib.rs`), a line per test file, and each rule they hold is
a row of [`rules.md`](docs/rules.md) naming its test. Its crate links
nothing, so `cargo test -p fb-doc-check` needs neither card nor library.

Tests run in parallel, so a capture test must not use global capture mode.
A global capture forbids other threads' unsafe calls, and on the 3080 it
made a neighbouring pool's create abort or its destroy segfault.

Nothing in ctest runs cargo. Both gates do, for a diff under `rust/`:
`devtools/test.py push` and `pr` run tiers 1 to 3, `cargo test` over every
member whose graph declares no `links` (found from `cargo tree`, so a new
link-free crate is gated without being named), then `RUSTDOCFLAGS='-D
warnings' cargo doc --no-deps` over the same members, in `target/link-free`
([`docs/testing.md`](../docs/testing.md), "Gates", at the root). One
feature set serves: no member links a library only through a default
feature. A commit touching `rust/` runs `devtools/test.py cargo-lint`,
`cargo fmt --check` and clippy with warnings denied, over the link-free
members, and so does CI's `lint` job. Tier 4 runs in both
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
The programming model and memory are also rendered as pages on the Pages
site, under `/design/` beside the API at `/api/`
([`documentation.md`](docs/documentation.md), "Publishing"); the `.md` is
the source, and nothing rendered is tracked.
[`rules.md`](docs/rules.md) is the register of the rules they state, each
with what checks it. **A new rule is added to the register in the PR that
states it, with its check or an *unenforced* entry.**
They describe the finished shape. A module is ported bottom-up from `core`,
one module per crate, its device code forked with it
([`gpu_layer.md`](docs/gpu_layer.md), "What the port changes").
