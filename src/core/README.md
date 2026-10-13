# `free_breakfast.core`

The starting set — one module library, here so a fresh clone has something that
exercises the full chain rather than a tree that configures and builds nothing.

| File | What it is |
| ---- | ---------- |
| `core.cppm` | The primary interface. Re-exports `:version` and the modules below. |
| `version.cppm` | A partition holding the project version and `version_string()`. |
| `types/` | `free_breakfast.core.types` — fixed-width integers in the global namespace, and `Vec<T, N>` / `make_vec` over the builtin GPU vector types. |
| `concepts/` | `free_breakfast.core.concepts` — wwr's `real_fp` and `int_type`, re-exported into `free_breakfast`. |
| `constants/` | `free_breakfast.core.constants` — typed `zero`/`one`/…, physical constants, bohr ↔ Angstrom, SI prefixes. |
| `nontransferrable/` | `free_breakfast.core.nontransferrable` — the `NonCopyable` / `NonMovable` / `NonTransferable` mixins. |
| `error_handling/` | `free_breakfast.core.error_handling` — `Error`, `ErrorKind`, `Result<T>`, `Status`, `err_*`, `to_exit_code`; `gpu_status`, `check_or_abort`; and the ABORT/RECORD GPU error policies chosen by `FREE_BREAKFAST_GPU_ERROR_POLICY`, with `capture`. |
| `log/` | `free_breakfast.core.log` — the spdlog-backed logging facade. |
| `diagnostics/` | `diagnostics.h` only (`FBF_TRY`, `FBF_GPU_MUST`, `FBF_LOG_*`), target `free_breakfast::core::diagnostics_macros`. |
| `gpu/` | `free_breakfast.core.gpu` — wwr's RAII wrappers bound to those policies: `Device`, `Buffers`, `BlasHandle`, `SolverHandle`, … |

Each subdirectory is its own library with its own README — a module library,
except `diagnostics/`, which is a macro header — and can be linked alone
(`free_breakfast::core::log`, ...) by a consumer that does not want the rest.

```cpp
import free_breakfast.core;

auto v = free_breakfast::core::version_string();   // "0.1.0"
```

The GPU layer comes along with that import: `free_breakfast.core.gpu`
re-exports the `wwr.extension.*` modules it binds (used in place from the
submodule, not copied), and its aliases already carry this build's error
policy — see `error_handling/README.md` for ABORT vs RECORD and `capture`.

Link it as `free_breakfast::core`.

**What it proves when it builds.** A green build of this target means clang
resolved `import std` against libc++'s module manifest, `clang-scan-deps`
ordered a partition before its primary interface, and the WarpWraps submodule
configured and built for the selected backend. That is most of what can go
wrong in the toolchain, which is why this stays in the tree until real modules
replace it.
