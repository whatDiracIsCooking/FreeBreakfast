---
name: test
description: >-
  Run this repo's two suites the right way and interpret the result honestly —
  the C++ tier (devtools/cpp-tier.sh, configure + build + ctest) and the Python
  tier (pytest, fast/slow split) — pick the tier, use the right worker count for
  host vs container, and ALWAYS audit what skipped before calling a run green.
  Use when the user asks to run tests, check a change, or verify an edit.
---

# Running tests, and trusting the result

The one rule that matters: **a green run is meaningless until you read the skip
list.** Tests that need a tool the environment lacks skip *silently*, so "all
passed" can mean "the tests that would have caught this never ran." Always pass
`-rs` and read the reasons. On a bare host that is the *normal* case here — the
whole C++ toolchain lives in the CUDA image.

## Two suites, and neither covers the other

| | `test/gtest/` | `test/pytest/` |
|---|---|---|
| Language | C++23, googletest | Python, pytest |
| Runner | `ctest` via `devtools/cpp-tier.sh` | `pytest` |
| Tests | the library's own units, including CUDA kernels | the Python-side suite — `test/pytest/`, marker and harness coverage plus the live BSE download test |
| Needs | the CUDA image (clang-20, CMake 4.2, nvcc) | a venv |
| Gated by | nothing automatic — see below | the pre-push hook and CI |

**Which one a change needs is not negotiable by convenience.** A change under
`src/`, `deps/`, `cmake/` or `CMakeLists.txt` is verified by the C++ tier and by
nothing else; a green `pytest` on it means only that the Python that would have
exercised the old binary still imports. Say which suite you ran.

## The C++ tier

```bash
devtools/cpp-tier.sh                       # configure + build + ctest, default preset
devtools/cpp-tier.sh --preset debug        # switches the ctest preset too
devtools/cpp-tier.sh --preset asan
devtools/cpp-tier.sh --fresh               # wipe the CMake cache and reconfigure
devtools/cpp-tier.sh --no-test             # configure + build only
devtools/cpp-tier.sh --clean               # rebuild every object, keep the cache
devtools/cpp-tier.sh --tidy                # + clang-tidy over src/*.cpp, ADVISORY
devtools/cpp-tier.sh -j 8                  # parallel compile jobs (BUILD_JOBS)
devtools/cpp-tier.sh -- -R Constants       # everything after -- goes to ctest
devtools/cpp-tier.sh -- -R MemoryBuffer    # ctest -R matches gtest suite names
```

**`cpp-tier.sh` always skips the reference-grade/end-to-end validation cases**
(it exports `FREE_BREAKFAST_FAST_TESTS=1` for the ctest phase, which makes
every case guarded by `FREE_BREAKFAST_SKIP_IF_FAST_TESTS()` `GTEST_SKIP` — see
`test/gtest/slow_test.h`). Those are the cases that check a result against an
external oracle to tight tolerance, or re-run a production-scale solve to
compare timing or convergence. There is no flag on this script that turns them
back on, and in a mature tree they are a handful of entries and the majority of
a full run's wall time.

```bash
devtools/cpp-slow-tier.sh                  # same flags, nothing skipped
```

is the un-skipped counterpart — the C++ analogue of `devtools/slow-tier.sh`.
It is not gated anywhere and is not a per-PR requirement; reach for it the way
you'd reach for `asan`/`compute-sanitizer` (an occasional deep-validation
sweep, or when a change has real risk of reaching code you did not directly
touch), not routinely before opening a PR.

`--tidy` runs `clang-tidy -p <the preset's own build dir>` over the tracked
`src/*.cpp` after a successful build. It is **advisory and never changes the
exit status** — the `.cppm` module interfaces are not clean, and what remains
there is the CRTP design rather than defects. It prints a skip line instead of
failing when clang-tidy is absent.

Run it **inside the container**, which is where the toolchain is:

```bash
devtools/devcontainer.sh shell -c devtools/cpp-tier.sh
```

On a host with no `cmake`/`ninja` the script refuses with exactly that command
rather than half-running — take the refusal at face value; it is not a bug to
work around.

Things that will bite:

- Only `default`, `workstation`, `debug` and `asan` have **test** presets. With
  any other configure preset the script builds and then reports that there is
  nothing to ctest — which is not the same as passing.
- The `default` preset wants the prebuilt `/opt/{spdlog,taskflow,tomlplusplus,googletest-src}`
  trees, which exist only in the image. On a bare host use `--preset
  workstation`, which fetches them instead; a `default` configure there fails at
  `find_package(spdlog)` with a message that never mentions containers.
- `CMAKE_CUDA_ARCHITECTURES=native` queries a live device **at configure time**.
  No GPU, no configure — even for a CPU-only change.
- Every run tees to `.slow-tier-reports/cpp-<stamp>-<rev>-<preset>.log`
  (`cpp-slow-<stamp>-...` for `cpp-slow-tier.sh`) and appends a PASS/FAIL line
  to `summary.log`. Quote the log path when reporting a failure; it has the
  compiler diagnostics that the summary does not.
- **ctest reports one entry per gtest case**, discovered at test time
  (`DISCOVERY_MODE PRE_TEST`), so `-R` selects by gtest suite name and a skip
  is named rather than hidden. A GPU-less box reports a large `***Skipped`
  count and is still green — read the skip list before calling it verified,
  exactly as with `pytest -rs`. Deliberately no case counts quoted here: the
  suite grows, and a stale number in a skill is worse than none.

## The Python tier: pick the tier

The fast/slow split is the spine of this repo's tooling — the marker
expressions live in `devtools/config.sh` (`FAST_TEST_ARGS`, `SLOW_TEST_ARGS`),
not in `pyproject.toml`, so an explicit `pytest -m slow` still works.

```bash
pytest -m "not slow" -n auto -rs          # the iteration loop — default to this
pytest -m "not slow" -n auto -rs -k name  # narrower
devtools/slow-tier.sh                     # the expensive tier, with a log
pytest -n auto -rs                        # everything, however long that takes
```

**Default to the fast tier** unless the user specifically asked for the
expensive one. Before running the whole suite, check what it costs — a slow
tier is slow by construction, and `devtools/slow-tier.sh --skip-huge` exists
for the cases (listed as `SLOW_HUGE_CASES` in the config) that need more RAM
than a small box has and OOM rather than failing cleanly.

## Host vs container — the worker count differs

- **Host:** `pytest -m "not slow" -n auto -rs` from the repo root.
- **Container:** `devtools/devcontainer.sh test` runs the config's `TEST_CMD`
  with `-n $JOBS -rs` and the fast marker inside. `JOBS` defaults to 8 rather
  than `auto` on purpose — several worktree containers each claiming every core
  is how a box ends up thrashing — and every config value honours an
  environment override, so give a concurrent run its own count on the *host*
  side of the call:

  ```bash
  JOBS=4 devtools/devcontainer.sh test        # -> pytest -n 4 -rs -m "not slow"
  devtools/devcontainer.sh test -k some_test  # extra pytest args pass through
  ```

  Passing your own `-m` suppresses the fast marker (which tier runs is then
  your call); anything else is appended after the defaults.

  `JOBS` bounds pytest and nothing else. To bound the *container* — so a
  concurrent run in another worktree cannot steal the cores, and a timing
  number means something — pin it with `CPUSET`/`CPUS` on `up`; see the
  `devbox` skill.

The container runs *more* tests than the host, and that is the whole point
here. `test/pytest/` exercises built binaries and GPU behaviour; on a laptop those
cases skip, so a clean host run routinely has the real coverage missing rather
than passing.

## The third marker: `gpu`, and `no_sanitizer`

`pyproject.toml` declares three markers, and only one of them is part of the
tier split:

- `slow` — the tier boundary. Minutes-long, big machine, compiler, GPU or
  network.
- `gpu` — needs a device. These skip on any host without `nvidia-smi`, silently
  unless you passed `-rs`. **This is the marker that makes a laptop run look
  green.**
- `no_sanitizer` — deselected by `docker compose run --rm compute-sanitizer`,
  because instrumentation turns a merely-slow case into an hours-long one. It
  changes nothing about a normal run.

## The batch path

`docker/compose.yaml` runs both suites in one shot, gtest then pytest, logging
to `.log/`:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
docker compose -f docker/compose.yaml run --rm test
docker compose -f docker/compose.yaml run --rm asan
docker compose -f docker/compose.yaml run --rm compute-sanitizer
```

`SKIP_GTEST=1` / `SKIP_PYTEST=1` narrow it to one suite, `TEST_FILTER` is
`--gtest_filter` and `PYTEST_ARGS` is appended to pytest. Prefer this when you
want a clean throwaway run; prefer `cpp-tier.sh` inside the devcontainer when
you are iterating. See the `devbox` skill for the difference.

## After the run: audit the skips, then the workers

1. **Read the `-rs` skip reasons.** Separate the expected ones (a test
   reporting a measurement, a case that does not apply here) from "that
   toolchain is absent and its coverage did not run" — and say the latter
   explicitly rather than claiming the change is verified.
   `devtools/doctor.sh` reports the whole tool list in one command (the
   `doctor` skill turns each of its findings into a fix).

2. **If you interrupted a container run, check for orphaned workers.** Killing
   `devtools/devcontainer.sh test` on the host kills only npx — the pytest
   master and its xdist workers keep running inside at 100% CPU, and nothing
   tells you; the next run just comes out slower and you measure *that*:

   ```bash
   devtools/devcontainer.sh shell -c "ps -eo pid,pcpu,cmd --sort=-pcpu | head"
   ```

   `pkill -f pytest` reaches only the master (a worker's command line is a bare
   `python3 -u -c import sys;...`), so kill the workers **by PID**.

## What to report

Say **which suite** ran, which tier of it, how many passed, and what skipped and
why. "N passed, M skipped — all M because <tool> is absent, so <what that
covers> was not verified here" is an honest result; "all green" on the same run
is not.

And say what did *not* run at all. A pytest-only result on a change under `src/`
should be reported as exactly that: "the Python tier passes; the C++ tier did
not run here (no GPU on this host), so the change is unverified." That sentence
is the difference between a useful report and a misleading one.
