---
name: test
description: >-
  Run this repo's two suites the right way and interpret the result honestly —
  the PR gate (devtools/test.py pr: configure + build + each C++ family's
  representative, then the fast pytest set) and the Python tier (pytest, fast/slow split) — pick the tier, use the right worker count for
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

| | `test/core/`, `test/atom/`, `test/io/`, `test/pipeline/`, `test/gtest/` | `test/pytest/` |
|---|---|---|
| Language | C++23, googletest | Python, pytest |
| Runner | `ctest` via `devtools/test.py pr` | `pytest` |
| Tests | the library's own units, including CUDA kernels | the Python-side suite — `test/pytest/`, marker and harness coverage plus the live BSE download test |
| Needs | the CUDA image (clang-20, CMake 4.2, nvcc) | a venv |
| Gated by | the push gate's smoke (`devtools/test.py push`); `test.py pr` by hand | the push gate and CI |

**Which one a change needs is not negotiable by convenience.** A change under
`src/`, `deps/`, `cmake/` or `CMakeLists.txt` is verified by the C++ tier and by
nothing else; a green `pytest` on it means only that the Python that would have
exercised the old binary still imports. Say which suite you ran.

## The C++ tier: `devtools/test.py pr`

`devtools/test.py` is the one entry point, a subcommand per gate
(`docs/testing.md` "Gates"): `push` is what `git push` runs, `pr` is the gate
before a PR. `pr` is the C++ tier followed by the fast pytest set.

**`pr` runs each C++ family's representative and nothing else**
(`docs/testing.md` "Test selection"): the set `devtools/test_families.py`
lists, an exempt family left out and a flagged one run whole. The nightly
runs the rest. So `-- -R Foo` runs only `Foo`'s representatives; add `--all`
for its whole fast set, which is what a targeted rerun wants. Read the
`pr: representatives only: N of M ctest entries selected, against the
lister's K representatives` line, and the `skipped representative:` lines:
a family whose representative skipped ran nothing.

**`push` runs the same set on its GPU binaries** (#1126): the
`push: representatives only: N of M GPU-binary cases selected` line, then
`push: GPU smoke took Ns`. The host binaries still run whole. Where the
lister cannot list (a wrong marker, no `metrics` branch), push says
`push: the GPU smoke runs the whole fast set` and goes on rather than fail;
`push --all` is that smoke on purpose.

```bash
devtools/test.py pr                        # configure + build + ctest + pytest, `auto` preset
devtools/test.py pr --preset default       # pin the CUDA card
devtools/test.py pr --preset hip           # pin the HIP card
devtools/test.py pr --preset debug         # switches the ctest preset too
devtools/test.py pr --preset asan
devtools/test.py pr --only cpp             # the C++ half alone (or --only python)
devtools/test.py pr --fresh                # wipe the CMake cache and reconfigure
devtools/test.py pr --no-test              # configure + build only
devtools/test.py pr --clean                # rebuild every object, keep the cache
devtools/test.py pr --tidy                 # + clang-tidy over src/*.cpp, ADVISORY
devtools/test.py pr -j 8                   # parallel compile jobs (BUILD_JOBS)
devtools/test.py pr --all                  # the whole fast set, not only representatives
devtools/test.py pr -- -R Constants        # everything after -- goes to ctest
devtools/test.py pr --all -- -R MemoryBuffer  # a suite's whole fast set
devtools/test.py push --dry-run            # which halves a push would run
devtools/test.py push --all                # the GPU smoke's whole fast set
```

**`test.py` always skips the reference-grade/end-to-end validation cases**
(it exports `FREE_BREAKFAST_FAST_TESTS=1` for its C++ tests, which makes
every case guarded by `FBF_SKIP_IF_FAST_TESTS()` `GTEST_SKIP` — see
`test/support/slow_test.h`). Those are the cases that check a result against an
external oracle to tight tolerance, or re-run a production-scale solve to
compare timing or convergence. There is no flag on `test.py` that turns them
back on, and in a mature tree they are a handful of entries and the majority of
a full run's wall time.

```bash
devtools/test.py nightly                   # origin/main, nothing skipped
ctest --preset default -R <Suite>          # one family, unskipped, by hand
```

The **nightly** is the un-skipped run, of both languages: a timer on this box
runs it on a clean checkout of `origin/main` and records it on `metrics`
(`docs/testing.md` "The nightly"). By hand it takes the better part of an hour
or more and tests `origin/main`, not your branch (`--rev` names another
commit). For a change with real risk to the cases the gates skip, a bare
`ctest` in the build directory, which does not set
`FREE_BREAKFAST_FAST_TESTS`, runs them unskipped; scope it with `-R`.

`--tidy` runs `clang-tidy -p <the preset's own build dir>` over the tracked
`src/*.cpp` after a successful build. It is **advisory and never changes the
exit status** — the `.cppm` module interfaces are not clean, and what remains
there is the CRTP design rather than defects. It prints a skip line instead of
failing when clang-tidy is absent.

### Which card: `auto`

The default preset is **`auto`** (`CMAKE_PRESET` in `devtools/config.sh`).
Several sessions usually test at once and one backend passing is normally
enough, so before building, `auto` picks the live card no other run holds,
then the lowest load (max of compute % and VRAM %), and prints why:

```
gpu: auto -> hip (hip) [ cuda=load:97%,vram:31%,held hip=load:0%,vram:2%,free ]
```

A card counts as live only if its device nodes and compiler both exist, so in
the single-backend `cuda`/`hip` containers `auto` is just that backend. The
ctest phase (and the push gate's GPU smoke) then holds the WHOLE card
(the `GPU_LOCK_DIR` locks `with_gpu_card` in lib.sh takes; `test.py` takes the
same files with `fcntl.flock`): **one run per card**, because two runs sharing the 10 GB
3080 ran it out of memory and bought no speed. It waits, printing why, while
another run holds the card, and then while the card's VRAM is at or above
`GPU_FOREIGN_VRAM_PCT` (20%) -- memory held by something that takes no lock,
such as a binary run by hand. That second wait has no timeout. Compute load
never makes a run wait. The two cards never block each other. Inside its card
`pr` runs ctest at `-j CTEST_JOBS` (3) and then the GPU batch as one gtest
process per GPU binary; `push` runs only the batches, each cut to the representatives. The batch's summary lines are tagged
`[gpu]`, and a crash prints the case it died in. One process, not shards:
processes on one card time-slice it and gained nothing (`docs/testing.md`, "One
process, not shards"), so a `[gpu]` log's per-case times are real. A silent or
overlong batch is killed by the watchdog, which names it on a
`[gpu] hung in <case>` line and writes `<log>.gpu-wedge.txt` first. Knobs are
the `GPU_*` variables and `CTEST_JOBS` in `config.sh`.

**Report the preset the run resolved to**, not `auto` — the
`test.py pr PASS  rev=…  preset=…` line and the log name carry it. Pin a backend with `--preset` when it matters: a change
touching a `.cu` needs **both** `--preset default` and `--preset hip` (or
`devtools/cross-backend-check.sh` for the side you did not run), because nvcc
passes code that clang rejects.

Run it **inside the container**, which is where the toolchain is:

```bash
devtools/devcontainer.sh shell -c 'devtools/test.py pr'
```

On a host with no `cmake`/`ninja` `test.py pr` refuses with exactly that command
rather than half-running — take the refusal at face value; it is not a bug to
work around.

Things that will bite:

- Only `default`, `workstation`, `debug` and `asan` have **test** presets. With
  any other configure preset `test.py pr` builds and then reports that there is
  nothing to ctest — which is not the same as passing.
- The `default` preset wants the prebuilt `/opt/{spdlog,taskflow,tomlplusplus,googletest-src}`
  trees, which exist only in the image. On a bare host use `--preset
  workstation`, which fetches them instead; a `default` configure there fails at
  `find_package(spdlog)` with a message that never mentions containers.
- `CMAKE_CUDA_ARCHITECTURES=native` queries a live device **at configure time**.
  No GPU, no configure — even for a CPU-only change.
- Every `pr` run tees to `.slow-tier-reports/pr-<stamp>-<rev>-<preset>.log`
  (`push-…` for the push gate's C++ half, `nightly-…`/`weekly-…` for the
  nightly) and `pr` appends its PASS/FAIL line to `summary.log`. Quote the log path when reporting a failure; it has the
  compiler diagnostics that the summary does not.
- **ctest reports one entry per gtest case**, discovered at test time
  (`DISCOVERY_MODE PRE_TEST`), so `-R` selects by gtest suite name and a skip
  is named rather than hidden. A GPU-less box reports a large `***Skipped`
  count and is still green — read the skip list before calling it verified,
  exactly as with `pytest -rs`. `test.py pr` hands most GPU entries to
  one gtest process rather than one process each (the CUDA context costs
  ~0.45s per process). ctest's summary then counts only the host,
  `RUN_SERIAL` and launcher-wrapped entries, and the rest report in the
  `[gpu]` lines after it, with their own `[  SKIPPED ]` list. `-- -R` still selects both. Deliberately no case counts quoted here: the
  suite grows, and a stale number in a skill is worse than none.

## The Python tier: pick the tier

The fast/slow split is the spine of this repo's tooling — the fast marker
expression lives in `devtools/config.sh` (`FAST_TEST_ARGS`), not in
`pyproject.toml`, so an explicit `pytest -m slow` still works.

```bash
pytest -m "not slow" -n auto -rs          # the iteration loop — default to this
pytest -m "not slow" -n auto -rs -k name  # narrower
pytest -m slow -n auto -rs                # the expensive tier alone
pytest -n auto -rs                        # everything: what the nightly runs
```

**Default to the fast tier** unless the user specifically asked for the
expensive one. Before running the whole suite, check what it costs — a slow
tier is slow by construction.

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
want a clean throwaway run; prefer `test.py pr` inside the devcontainer when
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
