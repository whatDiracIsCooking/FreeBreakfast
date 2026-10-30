# `cmake/`

Three macros define every target in this project, plus the internal helpers
they share. They exist so a new module is four lines rather than twenty, and so
the C++23-module and GPU-backend boilerplate is written once.

| File | Macro | For |
| ---- | ----- | --- |
| `free_breakfast_add_cxx_module_library.cmake` | `free_breakfast_add_cxx_module_library` | A C++23 named-module library (`.cppm`) |
| `free_breakfast_add_gpu_device_library.cmake` | `free_breakfast_add_gpu_device_library` | A STATIC library of device-compiled `.cu` sources |
| `free_breakfast_add_interface_library.cmake` | `free_breakfast_add_interface_library` | Header-only code (`.h` / `.cuh`) |
| `free_breakfast_internal_helpers.cmake` | — | `_free_breakfast_*` helpers; not public API |

All three are `include()`d from the root `CMakeLists.txt`, in that order —
helpers first, because the others call them.

## Target naming

Target names use **dots**; each macro also creates a `::` **alias**, and the
alias is what you link:

```cmake
free_breakfast_add_cxx_module_library(NAME free_breakfast.core.widget ...)
# defines  free_breakfast.core.widget
# aliases  free_breakfast::core::widget
```

Link the alias (`free_breakfast::core::widget`), never the dotted name. The
alias is what makes a typo a configure error instead of CMake inventing an
unbuilt target name later.

## A module library

```cmake
free_breakfast_add_cxx_module_library(
  NAME free_breakfast.core.widget
  PRIMARY_INTERFACE widget.cppm          # exactly one: `export module M;`
  PARTITIONS widget_impl.cppm            # `export module M:P;`
  IMPLEMENTATION widget.cpp              # `module M;`
  IMPORT_STD                             # sets CXX_MODULE_STD 1
  LINK_PUBLIC free_breakfast::core
)
```

| Argument | Meaning |
| --- | --- |
| `NAME` | Target name. Required. |
| `PRIMARY_INTERFACE` | The single primary module interface unit. Required. |
| `PARTITIONS` | Partition interface units. |
| `IMPLEMENTATION` | Module implementation units (plain `.cpp`). |
| `IMPORT_STD` | Set this whenever the module says `import std;`. |
| `LINK_PUBLIC` / `LINK_PRIVATE` | Link dependencies. |
| `INCLUDE_DIRS_PUBLIC` / `INCLUDE_DIRS_PRIVATE` | Extra include directories. |
| `NO_GPU_DEVICE_LINKING` | Separable compilation + device-symbol resolution OFF. CUDA only; inert under HIP. |
| `WHOLE_ARCHIVE` | `--whole-archive` the named target into consumers (for self-registering symbols a linker would otherwise drop). |

**`IMPORT_STD` is not optional bookkeeping.** Without it the target is not
built with `CXX_MODULE_STD`, and `import std;` fails to resolve — with an error
that names the standard library, not this flag.

**There is no option to add the vendor toolkit's include directories.** A
module reaches the vendor headers only through a WarpWraps target
(`free_breakfast::core::wwr`, or `wwr::vector_types::host` / `wwr::device` for
the exceptions `src/core/wwr/README.md` lists), so there is one route to them
and it is wwr's.

## A device library

```cmake
free_breakfast_add_gpu_device_library(
  NAME free_breakfast.core.widget.device
  SOURCES widget_kernels.cu
  LINK_PRIVATE free_breakfast::core::widget
)
```

`.cu` here means **device pass**, not **nvcc**. Under CUDA the file is nvcc's;
under HIP it goes through clang's `-x hip` front end, which is why the macro
forces the sources back to the `CXX` language there — a HIP build enables no
CUDA language at all, so CMake's extension-based mapping of `.cu` would find no
compiler for them.

**Keep device sources in their own library.** That is what keeps their language
and `-x hip` flags off the host module compiles, which must not see either. It
is also why a `.cu` never goes into a test executable directly: those are built
with `CXX_MODULE_STD` and `CXX_SCAN_FOR_MODULES`, and nvcc neither understands
`import std` nor participates in module dependency scanning.

Every device library also joins the `free_breakfast_device_libraries` umbrella
target, which `devtools/cross-backend-check.sh --device-only` builds — so a new
`.cu` library is cross-checked for the other backend with no list to update.

`GPU_OPTIONS` passes compile options to the device pass only. By default
separable compilation and device-symbol resolution are **ON** under CUDA; pass
`NO_DEVICE_LINKING` to turn both off.

Note the consequence of device-symbol resolution being on: each library
device-links only its **own** objects — `nvlink` does not reach into a linked
static library's device code. A `__device__` symbol defined in a header and used
by two libraries is device-linked into each of them, and if both land in one
executable under the same translation-unit identity the host linker rejects the
duplicate CUDA registration symbol. Compile such a shim through a
differently-named wrapper TU per library when that happens.

## An interface library

```cmake
free_breakfast_add_interface_library(
  NAME free_breakfast.core.bitutils
  LINK_PUBLIC free_breakfast::core::types
)
```

Adds the current source directory as the public include path, so a consumer
`#include "bitutils.h"` resolves. Nothing is compiled.

## What these deliberately do not do

- **No install rules.** Nothing here is packaged for external consumption yet;
  the `$<INSTALL_INTERFACE:include>` in the include directories is there so
  adding them later does not mean editing every call site. (For comparison,
  `deps/WarpWraps` *is* installable and has a whole `wwr_install.cmake` for it.)
- **No automatic test registration.** `test/core/`, `test/io/`, `test/pipeline/` and `test/gtest/` call
  `gtest_discover_tests` themselves, because the discovery mode and the per-binary
  labels are decisions worth seeing at the call site.
- **No GPU-architecture logic.** That is `CMakePresets.json`'s job, and the
  sm_80 floor is checked once in the root `CMakeLists.txt`.
