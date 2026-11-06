# Testing strategy

This is where the test infrastructure is heading, and the contract each piece of
it is built against. The work is the
[Automated test infrastructure](https://github.com/whatDiracIsCooking/FreeBreakfast/milestone/28)
milestone. Each section says what exists **today** and what is still
**planned**, and names the issue that delivers it. Until those land, CLAUDE.md's
"The two languages, and what verifies which" and "Reading a test run" describe
the scripts as they actually behave.

## The problem

Every case in the fast set runs before every PR, and the fast set grows with
the tree. Both PR gates have a budget, **under five minutes on a busy card**,
and `test/support/slow_test.h`'s rule keeps the fast set inside it by moving
anything past about ten seconds into the slow set. That rule only works if
something else runs the slow set, and today nothing does on a schedule:
`cpp-slow-tier.sh` is run by hand, when someone remembers.

The plan has three parts, and each leans on the one before:

1. **A nightly full run** on this box. Once everything runs every night, the PR
   gate no longer has to be the only thing that would catch a regression.
2. **Recorded build and test times**, so the cost of a test and the trend of a
   build are data rather than memory.
3. **Test selection** for PRs: run the tests that the diff can affect, using
   the build's own dependency graph, and leave the rest to the nightly.

## Gates

A gate is named by **when it runs**, not by what it contains.

| Gate | When | Runs | Budget |
| --- | --- | --- | --- |
| `push` | `git push`, by pre-commit's pre-push stage | the C++ half when the push touches C++, the Python half when it touches `.py`; the fast set of each | under 5 min on a busy card |
| `pr` | by hand before opening a PR; `milestone-run` requires it | the fast set, both languages; later, only the selected part of it | under 5 min on a busy card |
| `nightly` | a timer on this box | everything, unskipped, both languages, plus the cross-backend compile check | none; it has the night |

**The single-backend rule holds for all three.** A gate runs on one card, and
which one is printed in its result line. No gate runs both backends unless asked.
The nightly is **CUDA**, because production numbers are CUDA and because an
unattended HIP run on the gfx1200 can wedge the card (#477). It covers HIP with
`devtools/cross-backend-check.sh`: a seconds-long compile of every `.cu` through
clang `-x hip`, which is the one portability hole a single-backend run cannot see.

**The push gate's test set does not change in this milestone.** It is already a
fixed set, and its job is "is this obviously broken". Selection applies to `pr`.

## The entry point: `devtools/test.py`

**Today:** five scripts, named three ways. `cpp-smoke.sh` and
`prepush-tests.sh` are the push gate's two halves, `cpp-tier.sh` is the PR
gate, and `cpp-slow-tier.sh` and `slow-tier.sh` are the full C++ and Python
runs.

**Planned:** one entry point with a subcommand per gate.

```bash
devtools/test.py push       # what the pre-push hook calls (#981)
devtools/test.py pr         # before a PR (#981); --affected arrives in #989
devtools/test.py nightly    # what the timer calls (#985)
```

- **Standard library only.** It is the push hook on the host and in the
  container, so it must run under a bare `python3`, with neither `.venv` nor
  `/opt/venv` present. It still finds pytest through `VENV_PATHS` to run the
  Python half, exactly as `prepush-tests.sh` does now.
- **No shims.** The old scripts are deleted, not wrapped, and every reference to
  them is updated in the same PR. #981 removes the three gate scripts, and #985
  removes the two full-run scripts.
- **`devtools/config.sh` stays the one home of its values.** `test.py` reads it
  by evaluating it in bash and reading back the variables it needs, so a value
  is never written down twice, and an environment override (`CTEST_JOBS=1`)
  still wins as it does for every shell tool. The rest of `devtools/` is shell,
  and it keeps reading the same file.
- **The card lock is shared with the shell tools.** `with_gpu_card` in
  `devtools/lib.sh` holds `flock(1)` locks under `GPU_LOCK_DIR`. `test.py`
  takes the same files with `fcntl.flock`, which is the same kernel lock, so a
  `test.py` run and a `kernel-compile-watchdog.sh` run still exclude each other.
- **What it keeps from the scripts:** preset auto-pick, `FREE_BREAKFAST_FAST_TESTS`,
  the GPU batch as one gtest process per GPU binary (`split-gpu-entries.py`'s
  logic), the stall watchdog with its wedge-state dump, and the skip lists
  printed after every run.

## The nightly

**Planned in #985.** It runs on this box, from a systemd user timer.

1. Check out `origin/main` clean in `.claude/worktrees/nightly`. It never runs
   in the primary checkout, which may be mid-edit.
2. Build. On six nights out of seven that is the usual sccache-warm build, and
   its time is recorded as `warm` and kept out of the build-time trend. On the
   seventh it is a **clean build with the compiler launcher off**, the way
   `kernel-compile-watchdog.sh` compiles, so its times are real.
3. Run everything, unskipped, under `with_gpu_card`: the C++ slow tier on CUDA
   and the slow pytest tier. If the card is held, the run waits for it rather
   than sharing it.
4. Run `devtools/cross-backend-check.sh`.
5. Record the run (see "Metrics").

The build and the GPU run are **strictly sequential**. Many GPU cases are bound
by launch latency, so a busy host CPU would inflate their times and make the
series noisy.

**Failures open an issue (#986).** A failed nightly opens a GitHub issue
labelled `nightly`, or comments on the one already open with the same title, so
a failure that persists stays one issue. The body carries the failing cases, a
log excerpt and the metrics commit. The next green nightly comments that it
recovered, and closes the issue.

**A nightly failure is a blocker,** like any other test failure on main. It gets
fixed on main first, and any work it paused is rebased onto the fix.

**The nightly also enforces the slow-test rule (#987).** It flags any fast-set
case over about ten seconds, and any case whose time stepped up against its own
recent history, so the rule no longer depends on someone reading a log.

## Metrics

**Planned in #982.** Every recorded run appends one JSON record, on one line,
to the orphan branch `metrics`. The branch shares no history with `main`, never
appears in a diff, and is never merged. It is written through its own worktree
by `test.py`, never by hand. Records go into one file per month,
`runs/YYYY-MM.jsonl`, so no single file grows without bound.

A record, schema version 1:

| Field | Meaning |
| --- | --- |
| `schema` | `1` |
| `kind` | `nightly`, `weekly` (the clean-build night), `manual`, or `shadow` (see "Test selection") |
| `started`, `finished` | UTC, ISO 8601 |
| `host` | the machine's hostname; comparisons only ever use one host's records |
| `commit` | the full SHA that was tested |
| `backend`, `preset` | as the gate printed them |
| `gpu` | name, driver version, VRAM |
| `toolchain` | clang, CMake and nvcc or hipcc versions |
| `build` | `mode` (`warm` or `clean`), `launcher` (on or off), `jobs` (`BUILD_JOBS`), wall time, and for a clean build the per-file times from `.ninja_log` |
| `cases` | per test case: name, binary or pytest node, status (`pass`, `fail`, `skip`), seconds. From gtest's `--gtest_output=json`, ctest's `--output-junit` and pytest's JUnit XML |
| `selection` | `shadow` records only: what was selected and any misses |
| `result` | `pass` or `fail` |

**Only this box's nightly and weekly records form the timing series.** Times
from different hardware are not comparable, so every record carries its host,
and anything that reads the series filters on it. A VM may write `shadow`
records, which count PRs and misses, not seconds.

Recording the toolchain, the driver and `BUILD_JOBS` means a step in a series is
either a code change or a recorded environment change, never a mystery.

## Families

`slow_test.h`'s rule says a slow case leaves the fast set "unless it is the
cheapest of its family: one parameter, one molecule or one mode stays in the
fast set to stand for the rest". Today a family is judged by hand when the test
is written. Test selection needs it to be readable by a machine, so (#984):

- **A family is a gtest suite by default:** the suite name with any
  `INSTANTIATE_TEST_SUITE_P` prefix removed, so every parameter of one
  `TEST_P` is one family.
- **A marker overrides it.** A macro in `test/support/` marks a case as its
  family's representative, where the suite is the wrong boundary. #984 names it.
- **Each family has one representative:** the marked case, or else the cheapest
  case by the metrics series. A family with neither is flagged.
- **The slow-test rule, restated:** a case past about ten seconds goes under
  `FBF_SKIP_IF_FAST_TESTS()` unless it is its family's representative.

## Test selection

### The contract

**Planned in #983 (the map) and #989 (the selector).** This is the draft
contract. #989 updates it to what was actually built.

**Inputs:** the diff from the merge base with `origin/main`, and the gate's own
build directory once it is built and scanned. The graph is read from the build
being gated, so it is per backend and per preset by construction.

**The graph.** Nothing in it is written by hand; the compiler and CMake are the
sources of truth.

- **Module edges** come from each translation unit's `.ddi` file, the
  dependency scanner's output, whose `requires` lists the modules that unit
  imports directly. Every unit counts, **implementation units included**, and
  its edges belong to its module. An implementation unit's imports never reach
  an importer's interface, but they run when the importer calls in.
- **Header edges** come from each unit's `.ddi.d` file, a make-style list of
  every header it includes. A changed header affects every unit that includes
  it.
- **Target and link edges** come from the CMake File API (`codemodel-v2`): which
  target compiles each source, and what each target links. That is the only
  route by which a `.cu` in a device library reaches a test, because device
  code is linked, never imported.
- **Not** the `.modmap` named in `compile_commands.json`. It lists every
  transitively visible module, which clang needs, so it is a closure rather than
  an edge list. Selecting from it would select nearly everything.

**Selection:**

1. Map each changed file to its owner: its module for a module unit, its target
   otherwise.
2. Take every module or target that depends on an owner, directly or
   transitively, with its distance from the nearest owner.
3. A **test file** is selected through its own direct `requires` and link edges:
   - a test file that is itself changed, or that directly requires an owner or
     a direct dependent of one (distance 0 or 1), runs **its whole fast set**;
   - a test file whose nearest affected requirement is further away (distance 2
     or more) runs **only its families' representatives**;
   - any other test file does not run.
4. **Host binaries always run in full.** They take seconds, and selecting among
   them would buy nothing but risk. Selection applies to the GPU batches.
5. **The Python fast tier is not selected.** It runs whole when a `.py` changed,
   as it does now.

**These changes run everything:**

- anything under `cmake/`, `deps/` or `test/support/`;
- any `CMakeLists.txt` or `CMakePresets.json`;
- anything under `data/` (basis sets and SAD stamps are compiled into the binary);
- anything under `tools/emit/`;
- a file under `test/*/data/` that no test can be shown to read;
- any file the map cannot place. An unmapped file is a reason to run more,
  never fewer.

**The selector prints what it selected and why,** file by file, the way the gates
print their skip lists now. A selection nobody can read is one nobody can trust.

### Shadow mode, and when it becomes the gate

**#989** adds `test.py pr --affected` in **shadow mode**: the full fast set still
runs, and the run also computes the selection and records it as a `shadow`
metrics record. A **miss** is a failing entry that the selection did not select.

**#990 makes the selection the PR gate,** with `--all` as the escape hatch, once
the shadow records show:

> **2 weeks or 20 PRs, whichever is later, with zero misses.**

A miss resets the count. Its cause is fixed in the contract or the map first.

### The replay, and its go or no-go

Before any of the selector is built, **#988** replays the last ~50 merged PRs:
for each, what the contract would have selected, and its share of the fast GPU
batch's time by the nightly's per-case times. Selection saves little if most
PRs land in modules that nearly every test imports (`core`, `recipes`, the base
of `integrals`). The replay decides with a fixed bar:

- **Go:** the median PR's selection costs **at most half** the full fast GPU
  batch. #989 and #990 proceed as written.
- **No-go:** otherwise. #989 becomes "`test.py pr` runs every family's
  representative and nothing else, and the nightly carries the rest", which
  needs no graph at all, and #990 closes. #983's map stays as an analysis tool.

Either answer shortens the PR gate. The nightly is what makes the no-go version
safe: it runs everything the gate leaves out, within a day.

## What stays out of scope

- **`.github/workflows/ci.yml`.** The server-side C++ job stays `if: false`.
  Getting the toolchain image onto a runner is its own project.
- **A HIP nightly**, and the sanitizers on a schedule. The cross-backend compile
  check is the nightly's HIP cover for now.
- **Moving CLAUDE.md's testing sections here.** Those sections describe the
  scripts as they run today, and #981 and #985 rewrite those scripts. Each of
  those PRs moves the matching text into this file and leaves CLAUDE.md a
  summary and a link, so the prose moves once, with the code it describes.
