# CLAUDE.md

## What this is

free-breakfast is a GPU-first scientific computing project in C++23 named
modules, targeting **either CUDA or HIP** from one source tree. `test/pytest/`
and `devtools/` are a Python tooling layer around it, not a shippable package.

It is currently a **starter**: the build system, containers, tooling and CI are
real and exercised; `src/core` is one module whose job is to prove the toolchain
works. Say so rather than describing the tree as if it had a domain.

## Setup

```bash
git submodule update --init --recursive   # REQUIRED -- nothing configures without it
uv sync                                   # creates .venv from uv.lock
pre-commit install                        # commit-time lint + pre-push gate
devtools/devcontainer.sh rebuild          # the C++ toolchain lives in here
```

`uv` is the only assumed host tool. **`uv.lock` is tracked, and it is the only
place Python dependency versions are written down:** `pyproject.toml` says what
is wanted, the lock says which exact versions satisfy it, and
`Dockerfile.base`'s `uv sync --frozen` installs that same lock — so the
container and the host cannot drift. After editing `pyproject.toml`, run
`uv lock` and commit the result. Do not hand-install with `uv pip install`.

`devtools/doctor.sh` reports what is degraded. Run it first when something
behaves oddly — **and run it on both sides**, because this repo has a host half
and a container half and doctor is what tells you which one you are on. On a
bare host, ~11 warnings (the whole C++ toolchain and the three `/opt` trees) is
the expected healthy state, not a broken machine.

Doctor resolves the tools it reports through `VENV_PATHS`, not `PATH` — the
cmakelang pair lives in `.venv/bin` on the host and `/opt/venv/bin` in the
image, and a bare `command -v` called them missing on the host while passing in
the container.

## The three dependency worlds

They are wired differently and the difference matters.

| World | Where | How it arrives |
| --- | --- | --- |
| **WarpWraps** | `deps/WarpWraps` | git submodule, built from source as part of this build |
| **Prebuilt C++** | `/opt/{spdlog,taskflow,tomlplusplus}` | compiled into the image by `docker/Dockerfile.base`; `find_package` |
| **Python** | `.venv` / `/opt/venv` | `uv sync --frozen` from `uv.lock` |

Plus `nlohmann_json`, which is FetchContent because it is header-only and small.

**WarpWraps is the GPU API layer, and this project does not duplicate it.**
There is no `deps/cuda/` of hand-written `cuda_runtime_api.cppm` /
`cublas_v2.cppm` shims here; that is exactly what wwr's `src/cuda` (and its
`src/hip` mirror) is, and an earlier ancestor of this tree carried a narrower
CUDA-only copy of it. Import `wwr.*` and link `wwr::*`.

**GoogleTest comes from WarpWraps too**, not from this tree. wwr's own `deps/`
declares it, and FetchContent is first-declaration-wins, so declaring a second
one here would not give this project a different GoogleTest — it would only make
the pin look like it lived here.

### wwr's own tests are switched off

`deps/CMakeLists.txt` adds it with `add_subdirectory` and sets
`WWR_BUILD_TESTS` and `WWR_BUILD_EXAMPLES` **OFF** (both already default OFF
for a nested build, like `WWR_INSTALL`; they are set explicitly so this does
not depend on a default staying put). So wwr calls no `enable_testing()`, adds
no `test/` or `example/` directory, and a top-level `ctest` lists only this
project's entries. wwr's suites are wwr's CI's job, not this tree's.

GoogleTest is **not** behind those switches — wwr's `deps/` is gated only by
`WWR_COMPILE_TIME_ONLY` — which is what keeps the `GTest::*` targets this
project's tests link.

Coverage still excludes `/deps/` (`COVERAGE_IGNORE_REGEX` in
`devtools/config.sh`): wwr's libraries and GoogleTest are still compiled into
this build.

`enable_testing()` is still called before `add_subdirectory(deps)`. CMake
writes a `subdirs(...)` line into a directory's `CTestTestfile.cmake` only if
testing was enabled when that directory was *processed*, so a directory added
above the call has its tests silently dropped — the same trap the
`experimental/` guard at the bottom of the root `CMakeLists.txt` documents.
Keeping the call first means every directory gets the same treatment.

## The backend switch, and why it is before `project()`

`FREE_BREAKFAST_GPU_BACKEND` is `CUDA` or `HIP`, and a build is exactly one of
them. It is read **before `project()`** because it decides whether the CUDA
language is enabled at all — and it has **no default** for the same reason: a
silent one would quietly require a CUDA toolkit on a HIP-only host, or ROCm on a
CUDA-only one. Every preset carries it.

The root `CMakeLists.txt` mirrors it onto `WWR_GPU_BACKEND` (and
`FREE_BREAKFAST_COMPILE_TIME_ONLY` onto `WWR_COMPILE_TIME_ONLY`) before the
subdirectory is added. **Do not set either `WWR_*` variable anywhere else.** A
tree built CUDA against a HIP wwr fails deep inside wwr's own sources with
nothing naming the mismatch.

**The architecture floor is sm_80, and it is WarpWraps' floor, not the
toolkit's.** wwr wraps bf16 tensor-core WMMA (Ampere), so it rejects a sub-80
`CMAKE_CUDA_ARCHITECTURES` at configure. CUDA 13's own nvcc floor is lower
(`compute_75`) but never binds. The root CMakeLists checks it too, so the error
names this project. Keep `CUDA_ARCH` in `docker/Dockerfile.cuda` at 80+.

## Sanitizers: two options that are easy to get silently wrong

**`compute-sanitizer` the preset is only a Debug build.** What wraps the tests
is `FREE_BREAKFAST_COMPUTE_SANITIZER_TOOL` (empty by default), which sets
`CMAKE_TEST_LAUNCHER` so it covers the whole tree including the submodule. It
must be set before the test targets are created — CMake snapshots it onto each
target as `TEST_LAUNCHER` at creation time — which is why it lives near the top
of the root `CMakeLists.txt` and not beside `add_subdirectory(test)`.

`docker/compose.yaml`'s service passes it, and before the option existed that
`-D` was accepted and ignored: the service built, ran plain ctest, and reported
success having checked nothing. If you add another `-D` that compose passes,
check the option actually exists.

`--error-exitcode=1` is load-bearing: compute-sanitizer exits 0 by default even
when it reports errors.

**ASan needs `ASAN_OPTIONS=protect_shadow_gap=0`.** Not tuning — ASan's reserved
shadow gap collides with the GPU driver's virtual-address reservation, and the
first device allocation then fails as `cudaErrorMemoryAllocation` on a card with
plenty of free memory. It is set in the `asan`/`hip-asan` *test* presets (so
`cpp-tier.sh --preset asan` works) and separately in compose's `asan` service,
because compose runs bare `ctest --test-dir` and never sees a test preset. Both
copies are deliberate; keep them in step.

## The two languages, and what verifies which

| | C++ | Python |
|---|---|---|
| Source | `src/`, `cmake/`, `deps/` | `test/pytest/` |
| Tests | `test/gtest/`, run by ctest | `test/pytest/`, run by pytest |
| Driver | `devtools/cpp-tier.sh`, `cpp-slow-tier.sh` | `pytest`, `devtools/slow-tier.sh` |
| Lint | `cmake-lint` (commit hook + CI); `clang-tidy` via `--tidy` | `ruff` |
| Gated by | `cmake-lint` at commit, `cpp-smoke.sh` at push; **nothing on the server** | pre-push hook + CI |

**A change under `src/` is verified by the C++ tier and by nothing else.** A
green pytest run on it means the Python that would have exercised the *old*
binary still imports. Say which suite you ran.

`clang-tidy` is configured and runnable (`devtools/cpp-tier.sh --tidy`) but is
**advisory, not a gate**, and off by default: its C++23 named-module support is
still incomplete enough to emit bogus diagnostics on a `.cppm`. Do not bend the
module hierarchy to satisfy it.

### What is gated, and what is not

`git push` runs the fast pytest tier when the push touches a `.py`, and
`devtools/cpp-smoke.sh` when it touches C++. CI runs lint + the fast pytest
tier per PR.

**The two `cpp` jobs in `.github/workflows/ci.yml` are `if: false`.** A PR
touching only `src/` is checked by nothing on the server; `devtools/cpp-tier.sh`
before opening it is the whole gate, and it is a human one. The `ci-cuda` and
`ci-hip` presets were built for that job and need **no GPU** — the only obstacle
is getting the toolchain image onto a runner, and that file's comment has the
two routes.

## Containers: four files, two front ends

`docker/Dockerfile.{base,cuda,hip,combined}` is a diamond, and the four
**chain by tag, not by stage**: each child opens `FROM ${PARENT_IMAGE}`, so the
parent must be built and tagged first. `docker/build.sh <target>` walks the
chain. Building a child by hand with no parent tagged does not fall back to
building one — it tries to *pull*, and fails with `pull access denied for
free-breakfast`, which reads like a registry problem.

`combined` is the bottom of the diamond in intent only: docker has no multiple
inheritance, so it takes `cuda` as its parent and **re-runs**
`docker/install-rocm.sh`. That is why that block is a script rather than an
inline `RUN`.

Two front ends onto the same image:

- **`docker/compose.yaml`** — the batch path. `build` / `test` / `asan` /
  `compute-sanitizer`, each one-shot, teeing to `.log/`. Requires
  `HOST_UID`/`HOST_GID` in the environment (`:?`, not a default, so a forgotten
  export fails loudly instead of writing root-owned files).
- **`.devcontainer/{cuda,hip,combined}/`** driven by
  `devtools/devcontainer.sh` — the interactive path, per worktree, with its own
  container, volumes and CPU bounds. Pick a variant with `--cuda`/`--hip`/
  `--combined`, or `DEVCONTAINER_CONFIG`.

**They do not share a build directory, and cannot.** The devcontainer mounts the
workspace at its *host* path; compose mounts it at `/workspace`. A CMake cache
records absolute source and binary directories, so a tree configured by one
front end is rejected by the other with `The current CMakeCache.txt directory
... is different than the directory ... where CMakeCache.txt was created`.
Caches are not relocatable, so compose configures with `-B $BUILD_DIR` into its
own `build-compose*/` rather than the preset's default.

**Use `rebuild`, not `up`, after editing a `Dockerfile` or `devcontainer.json`.**
`up` reuses the running container, applies none of the change, and reports
success with the same container id.

### The duplication that is load-bearing

`PROJECT_NAME` in `devtools/config.sh` and the volume/container names in the
three `.devcontainer/*/devcontainer.json` files **must agree**, because JSON
cannot source shell. It is `free-breakfast` — the hyphenated spelling — in both
places, while the C++ namespace and CMake targets use `free_breakfast`. Get this
wrong and `worktree.sh rm`/`gc` stop recognising this project's volumes as its
own. `doctor.sh` checks the two agree and warns when they have drifted; it
prints one `ok` line per variant.

## Reading a test run

**A green run means nothing until you know what skipped.**

- C++ GPU cases call `FREE_BREAKFAST_REQUIRE_GPU()` (see
  `test/gtest/gpu_requirement.h`) and `GTEST_SKIP` with a reason rather than
  failing, so the suite stays usable on a device-less host. Read the
  `***Skipped` list.
- Python: always pass `-rs`. Without it the suite looks identical on a GPU box
  and a laptop.
- `FREE_BREAKFAST_FAST_TESTS` makes `slow_test.h`'s macro skip the heavy
  validation cases. `cpp-tier.sh` and `cpp-smoke.sh` set it;
  `cpp-slow-tier.sh` is the un-skipped run.

Tests are registered **one ctest entry per gtest case** via
`gtest_discover_tests(… DISCOVERY_MODE PRE_TEST)`. `PRE_TEST` defers enumeration
to the start of the ctest run on the machine that will run the tests, and
`--gtest_list_tests` only walks the registry so it never initialises the GPU
runtime. A single `add_test` entry instead would make `ctest -R` select nothing
while still reporting success, and `1 of 1 passed` would read identically
whether every case ran or most of them skipped.

### The one portability hole no tier sees

Every tier builds exactly one backend. A `.cu` goes through **nvcc** under CUDA
and **clang `-x hip`** under HIP, and nvcc is the more permissive of the two — a
functor holding a `const` member of class type passes the entire CUDA tier and
only breaks on a ROCm build. `devtools/cross-backend-check.sh` is the
seconds-long answer; run it whenever a change touches a `.cu`. It exits **2**,
not 0, when it cannot run, so a skip is never mistaken for a pass.

## Conventions worth not rediscovering

- **`IMPORT_STD` is not bookkeeping.** A module library that says `import std;`
  without it is not built with `CXX_MODULE_STD`, and the failure names the
  standard library rather than the flag.
- **A `.cu` never goes into a test executable or a module library.** Those are
  built with `CXX_MODULE_STD` and `CXX_SCAN_FOR_MODULES`, and nvcc neither
  understands `import std` nor participates in module dependency scanning. Put
  device code in its own `free_breakfast_add_gpu_device_library` target.
- **`hip::device` is always PRIVATE**, never PUBLIC or INTERFACE: its `-x hip`
  must reach one library's own sources and nothing else. A consumer picking it
  up transitively starts compiling its plain C++ as HIP.
- **Link the `::` alias, not the dotted target name.** The alias makes a typo a
  configure error.
- **No formatter is a git hook, deliberately.** A formatter rewrites whole files
  and buries a real diff in restyling. Run `clang-format`/`cmake-format` by
  hand. `ruff` and `cmake-lint` *are* enforced.
- **`BUILD_JOBS` is lower than the core count on purpose.** A module build is
  memory-hungry per job; an OOM-killed compiler surfaces as a bare
  `ninja: build stopped` with nothing about memory in it.

## Agent-specific

`.claude/settings.json` wires `devtools/hooks/protect-main.py` as a PreToolUse
hook on `Write|Edit|NotebookEdit`: it **blocks edits to the primary checkout**,
so work belongs in a `.claude/worktrees/<name>` worktree (the `worktree` skill)
or a container-backed sibling worktree (`devtools/worktree.sh`). The nested
`.claude/worktrees/` subtree is itself allowed. Escape hatch, for deliberate
work on main: `CLAUDE_ALLOW_MAIN_EDITS=1`. The guard fails **open** on any
error — a guard bug must never brick editing.

`devtools/hooks/session-env.py` runs at SessionStart and reports whether you are
on the host or in the container.

The seven skills in `.claude/skills/` are the tooling ones — `audit`, `devbox`,
`doctor`, `milestone`, `pr`, `test`, `worktree` — and they apply to this tree as
written.

Three more came across in the hoist and were **removed**: `constants`,
`error-handling` and `memory-buffer` documented
`src/core/{constants,error_handling,memory_buffer}` APIs that do not exist here,
and described the ancestor project's module tree in detail — which has no place
in a public repo. Their equivalents live in the WarpWraps submodule
(`wwr.extension.common`, `wwr.extension.memory_buffer`), which has its own docs.
If you want skills for those, write them against the `wwr.*` surface rather than
restoring the old ones.
