# `free_breakfast.core`

The starting set — one module library, here so a fresh clone has something that
exercises the full chain rather than a tree that configures and builds nothing.

| File | What it is |
| ---- | ---------- |
| `core.cppm` | The primary interface. Re-exports `:version` and the modules below, except `resources/`. |
| `version.cppm` | A partition holding the project version and `version_string()`. |
| `types/` | `free_breakfast.core.types` — fixed-width integers in the global namespace, `Vec<T, N>` / `make_vec` over the builtin GPU vector types, and wwr's `real_fp` / `int_type` concepts re-exported into `free_breakfast`. |
| `constants/` | `free_breakfast.core.constants` — typed `zero`/`one`/…, physical constants, bohr ↔ Angstrom, SI prefixes. |
| `primitives/` | `free_breakfast.core.primitives` — `math::sqrt`, `math::log`, `math::sin`, … templated on float/double, and `math::factorial` / `double_factorial_odd` / `binomial`, `math::idivup` / `align_up`; `primitives.h` gives a device `.cu` the same names (target `free_breakfast::core::primitives_header`). |
| `kernel/` | `free_breakfast.core.kernel` — `kWarpSize` and the `NumWarps` launch-shape concept; device-only headers beside it (target `free_breakfast::core::kernel_headers`) give a `.cu` the `DeviceFunctor` concept and `device::warp_reduce` / `block_reduce`. |
| `nontransferrable/` | `free_breakfast.core.nontransferrable` — the `NonCopyable` / `NonMovable` / `NonTransferable` mixins. |
| `error_handling/` | `free_breakfast.core.error_handling` — `Error`, `ErrorKind`, `Result<T>`, `Status`, `err_*`, `to_exit_code`; `check_or_abort`. Names no wwr. |
| `log/` | `free_breakfast.core.log` — the spdlog-backed logging facade. |
| `diagnostics/` | `diagnostics.h` only (`FBF_TRY`, `FBF_GPU_MUST`, `FBF_LOG_*`), target `free_breakfast::core::diagnostics_macros`. |
| `wwr/` | `free_breakfast.core.wwr` — the interface to WarpWraps: re-exports every wwr module the tree uses, `gpu_status`, the GPU error policies for the wwr wrappers (abort on a failed create, log on a failed destroy), and the wrappers bound to them: `Device`, `Buffers`, `BlasHandle`, `SolverHandle`, …; plus `ForkStream`, a `Stream` with fork / join `Event`s for fanning work out inside a graph capture, and `Device`, whose work stream is one. Import this rather than `wwr.*`; `wwr/README.md` lists the exceptions. |
| `resources/` | `free_breakfast.core.resources` — `NodeResources`: a `DeviceResources` (a `Device` plus BLAS and solver handles bound to its stream) per GPU, and a taskflow executor; each `DeviceResources` is driven by one thread at a time. **Not** re-exported by `core.cppm`, so taskflow stays out of every `import free_breakfast.core`; import it explicitly. |

Each subdirectory is its own library with its own README — a module library,
except `diagnostics/`, which is a macro header — and can be linked alone
(`free_breakfast::core::log`, ...) by a consumer that does not want the rest.

```cpp
import free_breakfast.core;

auto v = free_breakfast::core::version_string();   // "0.1.0"
```

The GPU layer comes along with that import: `free_breakfast.core.wwr`
re-exports the `wwr.*` modules the tree uses (used in place from the
submodule, not copied), and its aliases already carry this build's error
policy — see `wwr/README.md`.

Link it as `free_breakfast::core`.

**What it proves when it builds.** A green build of this target means clang
resolved `import std` against libc++'s module manifest, `clang-scan-deps`
ordered a partition before its primary interface, and the WarpWraps submodule
configured and built for the selected backend. That is most of what can go
wrong in the toolchain, which is why this stays in the tree until real modules
replace it.
