# Testing strategy

This is where the test infrastructure is heading, and the contract each piece of
it is built against. The work is the
[Automated test infrastructure](https://github.com/whatDiracIsCooking/FreeBreakfast/milestone/28)
milestone. Each section says what exists **today** and what is still
**planned**, and names the issue that delivers it. The `push` and `pr` gates
exist (#981); "Reading a gate's run" below is how to read them.

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

## Acceptance criteria: one run, a bar from analysis

This applies to every issue's `## Acceptance`, every test's assertion, and
every flake fix. Comp-chem packages validate the same way.

- **One run against a fixed bar set by analysis.** The algorithms are
  deterministic. The only run-to-run variation on a GPU is the order of atomic
  sums, which is bounded, at about √N·ε. So a bar comes from that bound, a
  solver tolerance, or a conditioning estimate, written down where the bar is
  set. It never comes from a spread measured over repeats.
- **No repeat counts, ever, as acceptance.** That rules out "N consecutive
  passes", "passes 20 times" and "replay 50×". A pass streak is neither
  necessary nor sufficient: a race that fails 1 run in 30 passes 20 runs in a
  row about half the time, and a correct fix needs no streak to show it.
- **Repeated runs are for diagnosis only.** Rerunning a flaky case until it
  fails, to capture a failing instance or its input, is legitimate. It
  belongs in the investigation, and its numbers go in the PR as evidence. It
  never goes in Acceptance.
- **A flake fix is accepted like any other fix.** Find the cause by analysis,
  usually of a captured input. Then assert the fix in one run on that input,
  for example a unit test feeding the captured matrix, plus the suite that
  flaked, plus unchanged goldens.
- **Criteria must be necessary and sufficient.** Drop a check that another
  check implies, and list what was dropped under "Not gated (implied: …)", so
  nobody re-adds it. Close any gap where every check could pass on a wrong
  result.
- **If the bar can't be met, report the numbers. Never move the bar.**

## The entry point: `devtools/test.py`

**Today:** `devtools/test.py push` and `devtools/test.py pr` (#981), which
replaced the three gate scripts. `cpp-slow-tier.sh` and `slow-tier.sh`, the
full C++ and Python runs, are still scripts of their own.

**Planned:** the `nightly` subcommand, and selection for `pr`.

```bash
devtools/test.py push       # what the pre-push hook calls
devtools/test.py pr         # before a PR; --affected arrives in #989
devtools/test.py nightly    # what the timer calls (#985)
```

`devtools/test.py --help` lists every flag. The ones worth knowing:

```bash
devtools/test.py push --dry-run              # which halves this push would run
devtools/test.py pr --preset hip             # pin a card; the test preset follows
devtools/test.py pr --only cpp -- -R Exch    # one half; after -- goes to ctest
devtools/test.py pr --no-test                # configure + build only
devtools/test.py pr --record                 # and append it to `metrics`
```

- **`push` dispatches on the diff.** It reads the pushed range from
  pre-commit's `PRE_COMMIT_FROM_REF...PRE_COMMIT_TO_REF` (by hand,
  `origin/main...HEAD`), runs the Python half when a `.py` changed and the C++
  half when a source, a header, a `CMakeLists.txt`, `CMakePresets.json` or a
  `.cmake` did, and runs nothing for a docs-only push. The hook in
  `.pre-commit-config.yaml` is `always_run` with no `files:` filter, so the
  dispatch is written down once, in `test.py`. A push of a whole history
  carries no range, and runs both; so does a range git cannot read under the
  hook (a shallow clone), with a line saying so. By hand, a range it cannot
  read (no `origin/main`) exits 2 and names the fetch that fixes it.
- **The C++ half of `push` skips itself**, exit 0, where `cmake` or `ninja` is
  missing, `cmake` is below the tree's floor, or the chosen backend has no
  device node: the toolchain lives in the GPU container, and a host-side push
  should not be forced onto `--no-verify`.
- **`pr` runs both languages:** configure, build, ctest and the GPU batches,
  then the fast pytest set (`FAST_TEST_ARGS` over `PREPUSH_PATHS`, the same set
  `push` runs). It logs to `.slow-tier-reports/pr-<stamp>-<rev>-<preset>.log`
  and appends its result line, `test.py pr PASS  rev=…  preset=…  stamp=…`, to
  `summary.log`. `push`'s C++ half logs to `push-<stamp>-<rev>-<preset>.log`.
  `rev` is `+dirty` when tracked files differ from HEAD (lib.sh `tree_rev`).
- **Report the preset the run resolved to**, not `auto`. `auto` picks the
  least-loaded live card before the build and prints
  `gpu: auto -> <preset> (<backend>) [ <each card's load> ]`.

- **Standard library only.** It is the push hook on the host and in the
  container, so it must run under a bare `python3`, with neither `.venv` nor
  `/opt/venv` present. It still finds pytest through `VENV_PATHS` to run the
  Python half, as the shell tools' `find_tool` does.
- **No shims.** The old scripts are deleted, not wrapped, and every reference to
  them is updated in the same PR. #981 removes the three gate scripts, and #985
  removes the two full-run scripts.
- **`devtools/config.sh` stays the one home of its values.** `test.py` reads it
  by evaluating it in bash and reading back the variables it needs, so a value
  is never written down twice, and an environment override (`CTEST_JOBS=1`)
  still wins as it does for every shell tool. The rest of `devtools/` is shell,
  and it keeps reading the same file.
- **lib.sh's functions are called, not copied:** `gpu_auto_preset`,
  `cpp_build` (the one copy of the configure and build phases), `tree_rev` and
  `gpu_card_mem` run in bash. What only the gates used, the watched GPU batch
  and the batch split, moved into `test.py`.
- **The card lock is shared with the shell tools.** `with_gpu_card` in
  `devtools/lib.sh` holds `flock(1)` locks under `GPU_LOCK_DIR`. `test.py`
  takes the same files with `fcntl.flock`, which is the same kernel lock, so a
  `test.py` run and a `kernel-compile-watchdog.sh` run still exclude each other.
- **What it keeps from the scripts:** preset auto-pick, `FREE_BREAKFAST_FAST_TESTS`,
  the GPU batch as one gtest process per GPU binary (`batchable` and
  `split_gpu_entries`), the stall watchdog with its wedge-state dump, and the skip lists
  printed after every run.

## Reading a gate's run

**A green run means nothing until you know what skipped.**

- C++ GPU cases call `FBF_REQUIRE_GPU()` (see
  `test/support/gpu_requirement.h`) and `GTEST_SKIP` with a reason rather than
  failing, so the suite stays usable on a device-less host. Read the skip
  lists: ctest's `***Skipped`, and gtest's `[  SKIPPED ]` in the `[gpu]`
  summary lines both gates print.
- Python: always pass `-rs`. Without it the suite looks identical on a GPU box
  and a laptop. Both gates pass it.
- `FREE_BREAKFAST_FAST_TESTS` makes `slow_test.h`'s macro skip the heavy
  validation cases. Both gates set it, and no flag turns it off;
  `cpp-slow-tier.sh` is the un-skipped run.
- **A card can be enumerated and still run nothing**, in which case every GPU
  case skips and ctest prints `100% passed`. The AMD one wedges with `amdgpu`
  still bound and the KFD node still listing it, while HIP returns
  `hipErrorNoDevice`; `devtools/doctor.sh`'s `gpu devices` section reads the
  PCI device's runtime-PM state and warns rather than reporting the card `ok`,
  on both sides of the host/container split. The cause is host-side (a reboot
  fixes it), so believe doctor over the enumeration.
- **A card can also wedge mid-run** (gfx1200, #477): every process on it
  spins in a HIP sync and prints nothing. Every GPU batch of both gates is
  watched for that (`GPU_BATCH_STALL_SECONDS` and `GPU_BATCH_BUDGET_SECONDS`
  in `devtools/config.sh`): a silent batch is killed and the line
  `[gpu] hung in <case>` names where. When the same gate hangs in a different
  case each time, suspect the card before the case, and rerun the gate on the
  other backend. Before the kill the watchdog writes the batch's state (vendor
  tools, a rocgdb backtrace of every thread) beside the gate's log, as
  `<log>.gpu-wedge.txt`, and the `state before the kill:` line names it.
  Attach it to #477, with the host's `dmesg` it ends by asking for. The batch
  is started with `PR_SET_PTRACER` naming `test.py`, which is what lets the
  debugger attach under Yama's `ptrace_scope=1`.

Tests are registered **one ctest entry per gtest case** via
`gtest_discover_tests(… DISCOVERY_MODE PRE_TEST)`. `PRE_TEST` defers enumeration
to the start of the ctest run on the machine that will run the tests, and
`--gtest_list_tests` only walks the registry so it never initialises the GPU
runtime. A single `add_test` entry instead would make `ctest -R` select nothing
while still reporting success, and `1 of 1 passed` would read identically
whether every case ran or most of them skipped.

**`test.py pr` does not run every entry as its own process, though.** A GPU
process spends about 0.45s creating the runtime's context before its case
starts, which was most of the gate's wall clock. So it lists the entries
(`ctest --show-only=json-v1`, so the preset's filters and `-- -R …` still
select), and takes out the GPU ones that can share a process: exactly one GPU
binary with one `--gtest_filter`, labelled `gpu`, not `RUN_SERIAL`, and no
launcher in front (`batchable` in `test.py`). Those run as one gtest process
per GPU binary (`free_breakfast_core_tests_gpu`,
`free_breakfast_recipes_tests_gpu`, then `free_breakfast_tests_gpu`), one after
the other and never at once, as in `push`. ctest runs the rest, `CTEST_JOBS` at
once: the host entries, the `RUN_SERIAL` cases it keeps alone on the card, and
every entry when a launcher such as compute-sanitizer wraps them. So its ctest
summary counts only those, and each batch reports in its own `--- gpu batch`
and `[gpu]` lines below it: read both. `cpp-slow-tier.sh` still runs plain
ctest.

**The card is held for the whole test phase**, by the lock files under
`GPU_LOCK_DIR` (`with_gpu_card` in `devtools/lib.sh` for the shell tools,
`fcntl.flock` on the same files in `test.py`). A gate waits, saying so, while
another run holds the card, and then while the card's VRAM is at or above
`GPU_FOREIGN_VRAM_PCT`, which is memory held by something that takes no lock.
Compute load never makes it wait. The two cards never block each other.

### One process, not shards

Processes on one card time-slice it, and a latency-bound case (a CASSCF's
thousands of tiny synchronised launches) waits behind its neighbours' long
kernels: `PipelineNevpt2.PcRefusalKeepsSc` is 2.3s alone and 158s beside two
`F/ExchangeGolden` processes. Shards bought no wall clock on either card
(`devtools/config.sh` has the numbers) and made every per-case time in a log
unreadable, so the times in a `[gpu]` log are what `slow_test.h`'s rule is
applied to.

**Both gates have a budget: under five minutes on a busy card.**
`test/support/slow_test.h` has the rule that keeps them inside it: a case past
about ten seconds goes under `FBF_SKIP_IF_FAST_TESTS()` unless it is its
family's cheapest.

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

**Today:** `devtools/metrics.py` (#982), the recorder, and
`devtools/test.py pr --record`, which records a run by hand as `manual`.
**Planned:** the nightly's records (#985).

Every recorded run appends one JSON record, on one line, to the orphan branch
`metrics`. The branch shares no history with `main`, never appears in a diff,
and is never merged. It is written through its own worktree by `test.py`,
never by hand. Records go into one file per month, `runs/YYYY-MM.jsonl`
(the month the run started), so no single file grows without bound.

- **The worktree** is `METRICS_WORKTREE` in `devtools/config.sh`,
  `.claude/worktrees/metrics` under the primary checkout, so every worktree
  records through the one copy. The first record makes it: on the local
  `metrics` branch, else one tracking `origin/metrics`, else a new orphan.
  `git worktree add` writes under the common git directory and into the new
  worktree, and never into `main`'s tree or index.
- **The branch is local to this box.** Nothing pushes it yet.
- **A record is committed with `git commit-tree` and `git update-ref`**, not
  `git commit`: the hooks in the common git directory are `main`'s
  pre-commit hooks, which have no config on a data branch. Two runs
  appending at once are serialised by `metrics.lock` in the common git
  directory.
- **`--record` refuses a dirty tree** (exit 2), because the record's commit
  would not be what was tested.
- **Where the cases come from:** `test.py pr --record` has ctest write
  `--output-junit`, each GPU batch write `GTEST_OUTPUT=json:…`, and pytest
  write `--junitxml` (`junit_family=xunit1`, which carries each case's file),
  all into `<log>.results/` beside the gate's log, with ctest's listing,
  which names each entry's binary. A batch that crashed writes no report, so
  its cases are absent; the record's `result` still says `fail`.

A record, schema version 1:

| Field | Meaning |
| --- | --- |
| `schema` | `1` |
| `kind` | `nightly`, `weekly` (the clean-build night), `manual`, or `shadow` (see "Test selection") |
| `started`, `finished` | UTC, ISO 8601 |
| `host` | the machine's hostname; comparisons only ever use one host's records |
| `commit` | the full SHA that was tested |
| `backend`, `preset` | as the gate printed them |
| `gpu` | `name`, `driver` (version), `vram_mib`; `null` where the vendor tool cannot say |
| `toolchain` | `clang`, `cmake`, and `nvcc` or `hipcc`: versions, the compilers' read from the build directory's own configure |
| `build` | `mode` (`warm` or `clean`), `launcher` (`on` or `off`: whether the configure found sccache), `jobs` (`BUILD_JOBS`), `seconds` (configure and build wall time), and for a clean build `files`, each output's `file` and `seconds` from `.ninja_log`; `null` when the run built nothing |
| `cases` | per test case: `name` (`Suite.Case`, or the pytest node id), `binary` (the gtest binary's name, or `pytest`), `status` (`pass`, `fail`, `skip`), `seconds`. From gtest's `--gtest_output=json`, ctest's `--output-junit` and pytest's JUnit XML |
| `selection` | `shadow` records only, and absent from the rest: what was selected and any misses |
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

**The map exists (#983, "The map" below); the selector is planned in #989.**
This is the draft contract. #989 updates it to what was actually built.

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

### The map: `devtools/test_map.py`

**Today (#983):** the graph above, built and written as one JSON map. It is
an analysis tool, not wired into any gate; #989's selector reads it.

```bash
python3 devtools/test_map.py build-hip -o map.json   # the map; report on stderr
python3 devtools/test_map.py build-hip --file src/integrals/mmd/mmd_launch.cu
```

It needs a configured and built build dir (the scan is part of the build).
It writes its own File API query, and when the dir has no codemodel reply yet
it re-runs `cmake` on the dir once to get one. It runs `ctest --show-only=json-v1`
and each gtest binary's `--gtest_list_tests --gtest_output=json`, which lists
every case with its file and never initialises the GPU runtime. `--ctest-json`
takes a saved listing instead. On `build-hip` it takes about 20 s.

What it reads, beyond the graph above:

- **Headers of units the scanner never sees**, `.cu` device code, come from
  `ninja -t deps`; ninja deletes their depfiles once it has read them. Every
  source the codemodel names is placed even when no depfile is found.
- **An owner is a module or a target.** A module's interface, partitions and
  implementation units are one owner, its target. A link edge is kept only to
  a non-module target (a device library, gtest): between module targets the
  import edges are the finer record. An `INTERFACE` or imported target
  compiles nothing, so the map looks through it to what it links.
- **A test file** is a source that defines a gtest case some ctest entry
  runs. Its direct dependencies, all at distance 0, are its own `requires`,
  its binary (the binary's other units, such as a `main.cpp`, are that
  binary's own owner), and the non-module targets the binary links. A header
  a test file includes is owned by that test file alone.

The map, schema 1:

| Key | Holds |
| --- | --- |
| `files` | every placed file under the source dir: its `kinds` (`module-interface`, `module-partition`, `module-implementation`, `source`, `header`, `test`), its `owners`, and `tests`, each test file it reaches with its distance |
| `owners` | each module or target: `kind`, its `target`, and `depends_on`, its direct edges |
| `tests` | each test file: `targets`, its direct `requires` as the scanner wrote them, `links`, and its ctest `entries` |
| `unmapped` | `files` under `src/` and `test/` it could not place, each with a `category`; ctest `entries` that are not one gtest case; `modules` required but provided by no unit |

`category` says why a file is unplaced. `cmake` and `data` run everything, per
the list above. `doc`, `config` (`.clang-format`), `python` and `unconfigured`
(the nearest `CMakeLists.txt` is not configured in this preset: `test/experimental/`
outside the experimental presets) have no effect on this build. `other` is
anything else, a file the map cannot account for, and the report lists each
one. On `build-hip` at #983, `other` is empty, and the unmapped entries are
`example_rhf` and `example_hessian`, which run `free-breakfast` through a cmake
script rather than a gtest case.

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
- **Moving CLAUDE.md's testing sections here** happens with the code each
  describes. #981 moved "Reading a test run" (now "Reading a gate's run"
  above) and left CLAUDE.md a summary and a link; #985 does the same for the
  full-run scripts.
