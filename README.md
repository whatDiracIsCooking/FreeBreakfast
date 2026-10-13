# free-breakfast

A GPU-first scientific computing project in C++23 named modules, targeting
**either CUDA or HIP** from one source tree.

This is a **starter**: the build system, containers, dev tooling and CI are
complete and exercised; `src/` holds one module whose job is to prove the
toolchain works. Add your domain code as sibling directories under `src/` and
delete what you do not use.

The GPU API layer is not written here. [WarpWraps](https://github.com/whatDiracIsCooking/WarpWraps)
is a submodule at `deps/WarpWraps` and supplies backend-neutral `wwr*` modules
over the CUDA and ROCm APIs, so code under `src/` is written once and builds for
both backends.

## Quick start

```sh
git clone --recurse-submodules git@github.com:whatDiracIsCooking/FreeBreakfast.git
```

Already cloned without `--recurse-submodules`? `git submodule update --init
--recursive`. Nothing configures without it — `deps/CMakeLists.txt` fails with
a `FATAL_ERROR` naming the submodule.

Interactive, and the one to develop in:

```sh
devtools/devcontainer.sh rebuild
devtools/devcontainer.sh shell -c devtools/cpp-tier.sh   # configure + build + ctest
devtools/devcontainer.sh shell                           # or just live in it
```

Batch, onto the same image:

```sh
docker/build.sh cuda                     # walks Dockerfile.base -> .cuda
export HOST_UID=$(id -u) HOST_GID=$(id -g)
docker compose -f docker/compose.yaml run --rm build
docker compose -f docker/compose.yaml run --rm test
```

Host-side, for editing and the Python suite:

```sh
uv sync && pre-commit install
pytest -m "not slow" -n auto -rs
devtools/doctor.sh
```

## Requirements

Everything is pinned inside the container, which is the supported path:

- Docker with BuildKit, and the **NVIDIA Container Toolkit** for CUDA device
  access (or a ROCm-capable host with `/dev/kfd` for the HIP backend)
- A GPU. Not only for the test services: `CMAKE_CUDA_ARCHITECTURES=native`
  queries a live device at **configure** time, so even a CPU-only change needs
  one unless you use a pinned preset (`ci-cuda`).
- **An Ampere (sm_80) or newer NVIDIA card** for the CUDA backend. WarpWraps
  wraps bf16 tensor-core WMMA and rejects anything below sm_80 at configure;
  that floor binds before CUDA 13's own `compute_75` one.
- `uv` on the host — the only assumed host tool, and the only thing the Python
  half needs.

Building on a bare host additionally needs Clang 20 with libc++ (including
`libc++.modules.json` and `clang-scan-deps`), CMake 4.2+, Ninja, and the CUDA
Toolkit or ROCm, plus the `workstation` preset — `default` expects the prebuilt
`/opt` trees only the image has.

`devtools/doctor.sh` tells you which of these you have. Run it on the host and
again inside the container — the difference between the two lists is the point.
On a bare host, ~11 warnings (the C++ toolchain and the three `/opt` trees) is
the expected healthy state, not a broken machine.

---

## Layout

```
CMakeLists.txt        Toolchain discovery, backend switch, project(), options
CMakePresets.json     Configure/build/test presets
cmake/                The three macros that define every target
deps/
  WarpWraps/          SUBMODULE: the CUDA/HIP layer, and the GoogleTest the tests link
  spdlog/ taskflow/   `.cppm` module shims over the prebuilt /opt libraries
  tomlplusplus/
  nlohmann/           ...and one FetchContent header-only library
src/core/             The starting set: one module over wwr and `import std`
test/gtest/           C++ tests, two binaries split by device requirement
test/pytest/          Python tests over the built binaries
docker/
  Dockerfile.base       Vendor-neutral toolchain + the three /opt deps
  Dockerfile.cuda       base + CUDA. The default backend
  Dockerfile.hip        base + ROCm, no CUDA
  Dockerfile.combined   cuda + ROCm (~40GB)
  build.sh              Builds one of them, and its ancestors, in order
  compose.yaml          The batch path
.devcontainer/{cuda,hip,combined}/   The interactive path, one per backend
devtools/             Container, worktree, test-tier, coverage and doctor scripts
tools/                Standalone Python: BSE basis download, aCD/acCD atomic Cholesky
```

---

## The backend switch

A build targets **exactly one** backend, chosen by `FREE_BREAKFAST_GPU_BACKEND`
(`CUDA` or `HIP`). It is read **before `project()`**, because it decides whether
the CUDA language is enabled at all — which is also why it has no default: a
silent one would quietly require a CUDA toolkit on a HIP-only host.

Every preset carries it, so only a raw preset-less `cmake` has to pass it. The
root `CMakeLists.txt` mirrors the choice onto `WWR_GPU_BACKEND` so the submodule
can never disagree — a tree built CUDA against a HIP wwr fails deep inside wwr's
own sources with nothing naming the mismatch.

Import the backend-neutral modules and link the matching target:

```cpp
import wwr.extension.memory_buffer;   // DeviceBuffer, PinnedBuffer, ...
import wwr.wrappers.blas;             // gemm<float>(...), either backend
```

A vendor-only entry point goes through the low-level module
(`import wwr.cuda.cublas_v2;`), and that file is then CUDA-only by construction.
WarpWraps' own README lists which functions those are and why.

---

## Presets

```sh
cmake --preset default && cmake --build --preset default && ctest --preset default
```

or all three with a log and a PASS/FAIL line, which is what the gates run:

```sh
devtools/cpp-tier.sh [--preset NAME] [--fresh] [--clean] [--no-test] [--tidy] [-j N]
```

| Preset | What it is for |
| ------ | -------------- |
| `default` | Release, CUDA, in-container `/opt` paths, `CUDA_ARCHITECTURES=native` |
| `workstation` | Same, for a bare host — builds into `build-workstation/`, no `/opt` |
| `debug` / `asan` | Debug, and Debug + AddressSanitizer on host code |
| `compute-sanitizer` | Debug, for the NVIDIA memcheck/racecheck run. Needs `-DFREE_BREAKFAST_COMPUTE_SANITIZER_TOOL=memcheck` (or `racecheck`/`initcheck`/`synccheck`) to actually wrap the tests — see below |
| `coverage` | Debug + `FREE_BREAKFAST_COVERAGE=ON`; driven by `devtools/coverage.sh` |
| `hip` / `hip-asan` | The ROCm backend, into `build-hip/` — needs ROCm, not CUDA |
| `experimental` | `default` plus `FREE_BREAKFAST_BUILD_EXPERIMENTAL=ON` |
| `ci-cuda` / `ci-hip` | What CI builds: same shape as each other, architecture pinned, `-LE gpu`. **Neither needs a device.** |

Every configure preset has its own `binaryDir`, and it needs to stay that way —
two presets sharing one build directory silently reconfigure it back and forth,
a full rebuild each way.

**There are no per-architecture presets.** `base` uses `native` and `ci-cuda`
pins `86`. Target something else with a command-line `-D` and a build directory
of its own — it has to be `-D`, which overrides a preset's `cacheVariables`
where an environment variable does not:

```sh
cmake --preset default -B build-h100 -DCMAKE_CUDA_ARCHITECTURES="90"
```

Anything machine-specific belongs in `CMakeUserPresets.json`, which is
gitignored.

### Two options the sanitizer runs need

```sh
# every ctest entry under compute-sanitizer, with a 10x timeout headroom
cmake --preset compute-sanitizer \
  -DFREE_BREAKFAST_COMPUTE_SANITIZER_TOOL=memcheck \
  -DFREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER=10
```

`FREE_BREAKFAST_COMPUTE_SANITIZER_TOOL` is empty by default, and the
`compute-sanitizer` preset is **only a Debug build** without it — the option is
what sets `CMAKE_TEST_LAUNCHER`, so it covers every test in the tree (the
submodule's included) without any of them knowing. It passes
`--error-exitcode=1`, because compute-sanitizer otherwise exits 0 even when it
reports errors and a finding would print while the test still passed.

`FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER` scales every `TIMEOUT` this project
sets. Under compute-sanitizer a GPU test is an order of magnitude slower, so the
120s backstop becomes the thing that fails the suite.
`docker/compose.yaml`'s `compute-sanitizer` service passes both (its multiplier
default is 10).

**ASan needs one runtime setting, and it is not optional.** The `asan` and
`hip-asan` *test* presets set `ASAN_OPTIONS=protect_shadow_gap=0`; ASan's
reserved shadow gap collides with the GPU driver's own large virtual-address
reservation, and without it the first device allocation fails as
`cudaErrorMemoryAllocation` on a card with plenty of free memory. Running those
binaries by hand needs the same variable.

---

## Tests

Two suites, and neither covers the other.

**C++ — `test/gtest/`, run by ctest.** Two binaries split by device
requirement: `free_breakfast_tests_host` (the `*_host.cpp` TUs, which need no
device and all run everywhere) and `free_breakfast_tests_gpu` (the `*_gpu.cpp`
TUs, which drive the device). The GPU cases call `FREE_BREAKFAST_REQUIRE_GPU()`
and **skip with a reason** rather than failing, so the suite stays usable on a
device-less build host — read the skip list before calling a run green.

Tests are registered one ctest entry per gtest case via
`gtest_discover_tests(… DISCOVERY_MODE PRE_TEST)`. `PRE_TEST` defers enumeration
to the start of the ctest run rather than to build time, so a GPU-less build
host still configures, and `--gtest_list_tests` only walks the registry so it
never initialises the GPU runtime. The payoff is that `ctest -R <Suite>` selects
properly and every skip is named:

```sh
devtools/cpp-tier.sh -- -R Version
```

**`ctest` runs this project's suites only.** The WarpWraps submodule is nested
via `add_subdirectory` with `WWR_BUILD_TESTS` and `WWR_BUILD_EXAMPLES` off (set
in `deps/CMakeLists.txt`), so its own suites and examples are neither built nor
registered here; they are verified by WarpWraps' own CI.

**Python — `test/pytest/`, run by pytest** over the binaries the C++ side
produces:

```sh
pytest -m "not slow" -n auto -rs   # the iteration loop
devtools/slow-tier.sh              # the expensive tier, logged
```

Three markers, declared in `pyproject.toml`: `slow` is the tier boundary, `gpu`
skips without a device, and `no_sanitizer` is deselected by the
`compute-sanitizer` compose service. **Always pass `-rs` and read the skip
list** — a green pytest run on a laptop mostly means the GPU cases skipped.

### One thing that is not a tier

```sh
devtools/cross-backend-check.sh   # does the OTHER backend still compile?
```

Every tier above builds exactly one backend, so none of them can see a
portability break. A `.cu` goes through nvcc under CUDA and clang under HIP, and
nvcc is the more permissive of the two — a functor holding a `const` member of
class type, for instance, passes the whole CUDA tier and only breaks on a ROCm
build. Run this whenever a change touches a `.cu`. It exits 2, not 0, when it
cannot run, so a skip is never mistaken for a pass.

### What is gated where

`git push` runs the fast pytest tier, and only when the push touches a `.py`;
`devtools/cpp-smoke.sh` runs the C++ smoke gate when it touches C++. CI runs
lint + the fast pytest tier per PR.

**The two `cpp` jobs in `.github/workflows/ci.yml` are `if: false`.** Until one
of the two routes in that file's comment is set up, a PR touching only `src/` is
checked by nothing on the server, and `devtools/cpp-tier.sh` before opening it
is the whole gate. Note the `ci-cuda`/`ci-hip` presets were built for that job
and need no GPU — the only obstacle is getting the toolchain image onto a
runner.

---

## Containers

Four files under `docker/`, in a diamond:

| File | What it adds |
| ---- | ------------ |
| `Dockerfile.base` | clang-20 + libc++, CMake 4.2, Ninja, sccache, uv/Python, and spdlog/taskflow/toml++ compiled into `/opt`. No GPU SDK. |
| `Dockerfile.cuda` | `base` + the CUDA toolkit. The default, and what the devcontainer and compose build. |
| `Dockerfile.hip` | `base` + ROCm/HIP. No CUDA at all. |
| `Dockerfile.combined` | `cuda` + ROCm, for switching backends without switching containers (~40GB). |

```sh
docker/build.sh cuda        # base, then cuda -> free-breakfast:cuda and :latest
docker/build.sh hip
docker/build.sh combined
```

**They chain by tag, not by stage.** Each child opens with
`FROM ${PARENT_IMAGE}`, so the parent must be built and tagged first —
`build.sh` is what walks the chain, and building a child by hand with no parent
tagged fails with `pull access denied for free-breakfast` rather than building
one.

Two front ends onto the same image:

| | `docker/compose.yaml` | `.devcontainer/` + `devtools/` |
|---|---|---|
| Shape | one-shot: configure, build, test, exit | long-lived container you stay in |
| Logs | tees everything to `.log/` | your terminal; `cpp-tier.sh` also logs |
| Good for | CI, nightlies, sanitizer sweeps | iterating, agents, debugging |
| Per worktree? | no | yes — container and volumes keyed on the workspace path |
| CPU bounds | none | `CPUSET`/`CPUS` in `devtools/config.sh` |
| Entry | `docker compose … run --rm test` | `devtools/devcontainer.sh shell` |

**They do not share a build directory, and cannot.** The devcontainer mounts the
workspace at its *host* path; compose mounts it at `/workspace`. A CMake cache
records absolute source and binary directories, so a tree configured by one is
rejected by the other. Compose configures into its own `build-compose*/` for
exactly that reason.

Use **`rebuild`**, not `up`, after editing a `Dockerfile` or `devcontainer.json`:
`up` reuses the running container, applies none of the change, and reports
success.

See [`docker/README.md`](docker/README.md) and
[`.devcontainer/README.md`](.devcontainer/README.md) for the variables each
reads, GPU selection, and `HOST_UID`/`HOST_GID`.

---

## Adding a module

Targets are declared with three macros from `cmake/`, documented in
[`cmake/README.md`](cmake/README.md). Names use dots and are aliased to `::` —
`free_breakfast.core.widget` is consumable as `free_breakfast::core::widget`.

```cmake
free_breakfast_add_cxx_module_library(
  NAME free_breakfast.core.widget
  PRIMARY_INTERFACE widget.cppm
  IMPORT_STD
  LINK_PUBLIC free_breakfast::core
)
```

Then `add_subdirectory(widget)` in the parent, add it to that parent's
`LINK_PUBLIC`, and `export import` it from the aggregating module if it is part
of the public surface. For `.cu` device code use
`free_breakfast_add_gpu_device_library`; for header-only code use
`free_breakfast_add_interface_library`.

---

## Tooling

Everything project-specific lives in **`devtools/config.sh`** — the project
name, which devcontainer variant, which presets, the test command, worker
counts, CPU bounds, marker expressions, the doctor lists. Edit that file, not
the scripts.

| Script | What it does |
| ------ | ------------ |
| `doctor.sh` | Report what is degraded here. Non-zero only on a FAIL. |
| `devcontainer.sh` | `up` / `rebuild` / `shell` / `test` / `down`, per worktree. Takes `--cuda`/`--hip`/`--combined`. |
| `worktree.sh` | `add` / `rm` / `sync` / `gc` / `list` for container-backed worktrees. |
| `cpp-tier.sh` | Configure + build + ctest, logged. The C++ tier. |
| `cpp-slow-tier.sh` | The same with nothing skipped — the deep-validation sweep. |
| `cpp-smoke.sh` | The C++ half of the pre-push gate. |
| `cross-backend-check.sh` | Compile the tree for the *other* GPU backend. |
| `coverage.sh` | Build instrumented, run ctest, `llvm-cov` summary of `src/`. |
| `claude-version.sh` | Read/compare/bump the Claude Code pin across the three GPU Dockerfiles. |
| `prepush-tests.sh` | The fast Python gate that `git push` runs. |

The coding style is in [`docs/style.md`](docs/style.md). The mechanical half
is enforced:

```sh
devtools/format.sh            # clang-format + ruff format, in place
devtools/format.sh --check    # what the commit hook and CI run
```

**No hook rewrites a file** — a formatter run as a hook buries the real diff
in restyling — so the commit hook and CI's `lint` job *check* the format and
fail on drift, and you run `devtools/format.sh` to fix it. `ruff check` and
`cmake-lint` (80 columns) are enforced the same way. clang-tidy is advisory and
off by default (`cpp-tier.sh --tidy`); its handling of C++23 named modules is
still incomplete, so expect noise on `.cppm` files.

`BUILD_JOBS` in `devtools/config.sh` is the one to set conservatively — a C++23
module build is memory-hungry per job, and letting Ninja pick (cores + 2) is
what OOM-kills a 16-core box. An OOM-killed compiler surfaces as a bare
`ninja: build stopped` with nothing about memory in it.

---

## Notes on the toolchain

- **`import std` is experimental, and its gate UUID is tied to the CMake
  version.** The root `CMakeLists.txt` pins the value for 4.2.x and
  **hard-errors** on any version it has not been told about, rather than let a
  stale UUID be ignored silently and break every `import std`.
- **libc++ and libstdc++ coexist.** The C++ side builds against libc++, but
  nvcc links libstdc++ internally, so the CUDA branch locates `libstdc++.so`
  via `gcc --print-file-name` for the targets that need both.
- **GoogleTest is unpatched.** It comes from WarpWraps' `deps/`, built as plain
  C++ with `CXX_SCAN_FOR_MODULES OFF`. The older approach — a ~24KB patch
  making GoogleTest's own sources `import std;` — is unnecessary: that define
  was always `PRIVATE` to GoogleTest, so consuming TUs got the header build
  either way, and this leaves no patch to re-roll on every upgrade.
- **`CMAKE_CUDA_ARCHITECTURES=native` queries a live device at configure time.**
  No GPU, no configure. Use `ci-cuda` (pinned `86`) on a card-less box.
- **sccache is wired in by `find_program`, not by a preset**, so an image or
  host without it still builds — it just caches nothing.

`CLAUDE.md` is the fuller picture: which duplications are load-bearing, what
verifies what, and how to read a test run.
