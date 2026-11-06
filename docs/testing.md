# Testing strategy

This is where the test infrastructure is heading, and the contract each piece of
it is built against. The work is the
[Automated test infrastructure](https://github.com/whatDiracIsCooking/FreeBreakfast/milestone/28)
milestone. Each section says what exists **today** and what is still
**planned**, and names the issue that delivers it. The `push` and `pr` gates
exist (#981), and so does the nightly (#985); "Reading a gate's run" below is
how to read them.

## The problem

Every case in the fast set runs before every PR, and the fast set grows with
the tree. Both PR gates have a budget, **under five minutes on a busy card**,
and `test/support/slow_test.h`'s rule keeps the fast set inside it by moving
anything past about 1.5 s on an idle RTX 3080 into the slow set, unless it is
its family's representative. That rule only works if something else runs the
slow set on a schedule. Until #985 nothing did: the
full runs were scripts run by hand, when someone remembered.

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
replaced the three gate scripts, and `devtools/test.py nightly` (#985), which
replaced the two full-run scripts.

**Planned:** selection for `pr`.

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
  `test.py nightly` is the un-skipped run, and it unsets it.
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
and `[gpu]` lines below it: read both. The nightly splits its run the same
way.

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
about 1.5 s on an idle RTX 3080 goes under `FBF_SKIP_IF_FAST_TESTS()` unless it
is its family's representative ("Families" below). The nightly flags every case
that breaks it ("The slow-test rule, enforced").

## The nightly

**Today (#985):** `devtools/test.py nightly`, run on this box by a systemd user
timer, `devtools/systemd/free-breakfast-nightly@.timer`.

1. **Check out `origin/main` clean** in `NIGHTLY_WORKTREE`
   (`.claude/worktrees/nightly` under the primary checkout), after a
   `git fetch origin`. It is a detached worktree, made on the first night. Each
   later night resets it: tracked changes and untracked files go, in the
   submodules too, and ignored files stay, so the build directory is still
   warm. It never runs in the primary checkout, which may be mid-edit.
   `--rev` tests another commit.
2. **Run that checkout's own `test.py`**, so what runs is the code under test,
   not whatever the primary checkout has. Everything below is that run.
3. **Build** with `--preset default`, the CUDA card. On six nights out of seven
   that is the usual sccache-warm build, recorded as `nightly` with
   `build.mode` `warm` and kept out of the build-time trend. On
   `NIGHTLY_WEEKLY_DAY` (Sunday; `--mode weekly` forces it) it is a **clean
   build with the compiler launcher off**, recorded as `weekly`, so its times,
   per file from `.ninja_log`, are real. The launcher goes off by configuring
   with `-DFREE_BREAKFAST_SCCACHE=OFF`, which `find_program` leaves alone, so
   no launcher is installed, sccache or the GPU object cache. The next warm
   night configures with `-UFREE_BREAKFAST_SCCACHE`, which finds it again.
4. **Run everything, unskipped, under the card lock** (`Card` in `test.py`, the
   same locks as `with_gpu_card`): every C++ case with no
   `FREE_BREAKFAST_FAST_TESTS`, split as `pr` splits it (ctest for the host and
   `RUN_SERIAL` entries, one watched process per GPU binary for the rest), then
   the whole pytest suite, `slow` included. If the card is held, the run waits
   for it rather than sharing it. The GPU batches' budget is
   `NIGHTLY_GPU_BATCH_BUDGET_SECONDS` (4 h), since unskipped they run far past
   the gates' 45 min; the stall watchdog is the gates' own.
5. **Run `devtools/cross-backend-check.sh`**, outside the card, after the
   tests. Its exit 2 (it could not run) fails the night, with a line saying so:
   it is never a pass.
6. **Record the run** (see "Metrics"), pass or fail, even when the build
   failed and nothing ran.
7. **Report the slow-test rule** over that record and the series before it
   ("The slow-test rule, enforced" below), when any case ran.

The build and the GPU run are **strictly sequential**. Many GPU cases are bound
by launch latency, so a busy host CPU would inflate their times and make the
series noisy.

It logs to `<primary>/.slow-tier-reports/<mode>-<stamp>-<rev>-default.log`,
with the per-case reports in `<log>.results/`, and appends
`test.py nightly PASS  rev=…  preset=default  mode=nightly  stamp=…  metrics=…  slow-rule=N/M`
to `summary.log`. The exit status is the first failing phase's.

### Installing the timer (host only)

The timer is a systemd **user** unit on the **host**: the container has no
systemd user session. The service enters this checkout's **combined**
devcontainer with `devtools/devcontainer.sh --combined shell -c`, because the
C++ toolchain lives there, and the cross-backend check needs `hipconfig` as
well as `nvcc`. It never starts or rebuilds the container: if it is not
running, `shell` refuses, naming `up`, and the night fails loudly. Both units
are templates whose instance is the primary checkout's path, so nothing in them
names a machine or a user. From the primary checkout, on the host:

```bash
systemctl --user link "$PWD/devtools/systemd/free-breakfast-nightly@.service"
systemctl --user link "$PWD/devtools/systemd/free-breakfast-nightly@.timer"
unit="free-breakfast-nightly@$(systemd-escape --path "$PWD")"
systemctl --user enable --now "$unit.timer"
loginctl enable-linger "$USER"     # run with nobody logged in
systemctl --user list-timers "$unit.timer"
```

- **A first run by hand**, to see it work before it runs unattended:
  `systemctl --user start "$unit.service"`, then
  `journalctl --user -u "$unit.service" -f`.
- **It fires at 02:30** local time (`OnCalendar`). A night missed while the box
  was off is not caught up at boot (`Persistent=false`).
- **The service runs a login shell** (`bash -l`), for the `PATH` that finds
  `npx`, the devcontainer CLI. `TimeoutStartSec=16h` is a backstop only.
- **Stopping the unit stops the CLI, not the run:** `devcontainer exec` is an
  attach, and the run carries on in the container.
- **After editing a unit**, `systemctl --user daemon-reload`. To remove it,
  `systemctl --user disable --now "$unit.timer"`.

**Failures open an issue (#986).** A failed nightly opens a GitHub issue
labelled `nightly`, or comments on the one already open with the same title, so
a failure that persists stays one issue. The body carries the failing cases, a
log excerpt and the metrics commit. The next green nightly comments that it
recovered, and closes the issue.

**A nightly failure is a blocker,** like any other test failure on main. It gets
fixed on main first, and any work it paused is rebased onto the fix.

### The slow-test rule, enforced

**Today (#987):** after its record is on the `metrics` branch, the nightly runs
`devtools/slow_rule.py` over it, so the rule no longer depends on someone
reading a log. It reports two things, and neither fails the night:

- **Over the line:** a fast-set gtest case that took more than **1.5 s**
  (`LINE_SECONDS`, `slow_test.h`'s line, which a pytest holds it to) and is
  not its family's representative, nor in an exempt family. The families and
  representatives are `devtools/test_families.py`'s, read from the night's
  build directory, the cheapest by the same records' times. pytest cases and
  the `example_*` entries are outside `slow_test.h`'s rule.
- **A step:** a case, in either set and either language, whose time stepped up
  against its own recent history past the bar below.

**Which set a case is in comes from the run itself.** `FBF_SKIP_IF_FAST_TESTS()`
records the gtest property `fbf_slow` in every mode, before it decides whether
to skip, and `metrics.gtest_cases` carries it as each case's `slow`. A
nightly runs everything, so its record says which cases a fast run would
skip, including a `TEST_P` that reaches the macro only for some params. Only
a gtest JSON report carries properties, so a case ctest runs (the host
binaries, and the `RUN_SERIAL` cases) has no `slow`; one of those over the
line is listed apart, as membership not recorded, never dropped.

**The step bar comes from the series' own spread**, by analysis, not tuning
(see "Acceptance criteria"):

- **The window** is the 14 records before tonight's (`WINDOW`), this host's
  `nightly` and `weekly` only: two weeks of nights, the clean-build weekly
  twice among them. Its median and median absolute deviation (MAD) have a
  breakdown point of 50%, so up to six bad nights (a busy host, a wedged
  card) move neither. A deliberate change stops being reported once it holds
  half the window, about a week, and the PR that made it says why.
- **A case is judged** only with at least 7 passes in the window
  (`MIN_HISTORY`, half of it); a new case, or a series younger than a week,
  is counted as too short instead.
- **The spread** σ is the largest of three. The case's own: 1.4826 × its MAD
  (`MAD_TO_SIGMA`, 1/Φ⁻¹(3/4), which makes the MAD of normal noise its
  standard deviation, and an outlier night cannot inflate it). The box's: the
  median over every case with 7 passes in the window of that, divided by its
  median, times this
  case's median; only cases at 0.1 s and over (`POOL_MIN_SECONDS`) enter it,
  where the 1 ms resolution is under 1% of the time. And the resolution
  itself, 1 ms (`RESOLUTION`; gtest's JSON times are whole milliseconds). The
  box's floor is needed: 14 points estimate a MAD poorly, and a sample whose
  MAD came out small would make noise look like a step. Under normal noise,
  a bar of 5 × the case's own MAD alone is crossed by about 2·10⁻³ of cases a
  night at n = 14, and 1.6·10⁻² at n = 7 (a Monte Carlo of that null, not of
  this tree's times), which is about four false steps a night (n = 14) to
  thirty (n = 7) across 2,100 cases. The box's spread is a median over
  hundreds of cases, so it is not small by chance.
- **The bar** is the median + K·σ, K = 5 (`K`). With σ at least the true
  spread, a new time minus the window's median has a standard deviation of
  √(1 + π/2n)·σ (the median's own variance is πσ²/2n), so 5σ is 4.74 of them
  at n = 14 and 4.52 at n = 7: a one-sided tail of 1.1·10⁻⁶ to 3.1·10⁻⁶ a case,
  about 2·10⁻³ to 7·10⁻³ false steps a night across 2,100 cases, or one in five
  months at worst. A case noisier than the box gets its own, larger σ.

**Where it reports:** each finding as a `<mode>: slow-rule: …` line in the
nightly's log and output; `slow-rule=<over>/<steps>` on its `summary.log`
line (`slow-rule=none` when the series has no record of this host, and
`slow-rule=FAILED`, which fails the night, when the report could not run); and
the findings as JSON in `<log>.results/slow-rule.json`. That file is the hook
for the nightly issue (#986): it carries `over_line`, `over_line_unrecorded`
and `steps`, each case with its numbers, and the constants they were judged
by.

```bash
python3 devtools/slow_rule.py build                # the latest nightly's report
python3 devtools/slow_rule.py build --json         # its findings
python3 devtools/slow_rule.py build --metrics runs.jsonl --host NAME
```

It reads the `metrics` branch by default and this host by `metrics.host_name`,
as `test_families.py` does, and exits 2 when there is no record or no build to
read.

## Metrics

**Today:** `devtools/metrics.py` (#982), the recorder;
`devtools/test.py pr --record`, which records a run by hand as `manual`; and
`devtools/test.py nightly` (#985), which records every night as `nightly` or
`weekly`.

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
- **`host` is the machine, not the hostname** (`host_name` in `metrics.py`):
  `METRICS_HOST` in `devtools/config.sh` when set, else the board's vendor and
  name from `/sys/class/dmi/id` (`board_vendor`, `board_name`; else
  `sys_vendor`, `product_name`), else the hostname where there is no DMI. The
  hostname inside the devcontainer is the container's id, which changes at
  every `rebuild` (#982's first record carries one), and `/etc/machine-id`
  there is the image's. DMI is world-readable in the container and reads the
  same as on the host, so every record from this box carries one name, through
  rebuilds and from either side, with nothing passed in. Two machines with the
  same board would share it, which is why `METRICS_HOST` exists; set it on a
  cloud VM, whose DMI names only its instance type.
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
| `host` | the machine (`METRICS_HOST`, else its board from DMI); comparisons only ever use one host's records |
| `commit` | the full SHA that was tested |
| `backend`, `preset` | as the gate printed them |
| `gpu` | `name`, `driver` (version), `vram_mib`; `null` where the vendor tool cannot say |
| `toolchain` | `clang`, `cmake`, and `nvcc` or `hipcc`: versions, the compilers' read from the build directory's own configure |
| `build` | `mode` (`warm` or `clean`), `launcher` (`on` or `off`: whether the configure found sccache), `jobs` (`BUILD_JOBS`), `seconds` (configure and build wall time), and for a clean build `files`, each output's `file` and `seconds` from `.ninja_log`; `null` when the run built nothing |
| `cases` | per test case: `name` (`Suite.Case`, or the pytest node id), `binary` (the gtest binary's name, or `pytest`), `status` (`pass`, `fail`, `skip`), `seconds`, and for a case from a gtest JSON report `slow` (whether it reached `FBF_SKIP_IF_FAST_TESTS()`, which a fast run skips; #987, absent before it). From gtest's `--gtest_output=json`, ctest's `--output-junit` and pytest's JUnit XML |
| `selection` | `shadow` records only, and absent from the rest: what was selected and any misses |
| `result` | `pass` or `fail` |

**Only this box's nightly and weekly records form the timing series.** Times
from different hardware are not comparable, so every record carries its host,
and anything that reads the series filters on it. A VM may write `shadow`
records, which count PRs and misses, not seconds.

Recording the toolchain, the driver and `BUILD_JOBS` means a step in a series is
either a code change or a recorded environment change, never a mystery.

## Families

**Today (#984):** a family is readable by a machine. `slow_test.h` states the
slow-test rule in these terms, the marker lives beside
`FBF_SKIP_IF_FAST_TESTS()` in it, and `devtools/test_families.py` lists every
family with its representative.

- **A family is a gtest suite by default** (`TEST`, `TEST_F`). A `TEST_P` is a
  family of its own, across its params and its `INSTANTIATE_TEST_SUITE_P`
  prefixes: `Cases/Foo.Bar/x` and `More/Foo.Bar/y` are both `Foo.Bar`. A
  `TYPED_TEST` is the same across its types. This is the boundary #993 applied.
- **A marker overrides the cheapest**, where it is the wrong stand-in:
  `FBF_FAMILY_REPRESENTATIVE`, at the top of the test body. It compiles to
  `static_assert(true)`, nothing at run time; the lister reads it from the
  source.

  ```cpp
  FBF_FAMILY_REPRESENTATIVE();                 // a TEST or TEST_F: this case
  FBF_FAMILY_REPRESENTATIVE("p");              // a TEST_P: the instance p
  FBF_FAMILY_REPRESENTATIVE("p", "regex");     // ... of a family of its own
  ```

  An instance is named by its param, or by `Prefix/param` where two
  instantiations share a param. The two-argument form splits a mode off a
  `TEST_P`: the instances whose param matches the regex (Python `re.search`)
  are the family `Foo.Bar[regex]`, and `p` represents it. The arguments are
  string literals, and nothing but the lister checks them.
- **Each family has one representative:** the marked case, or else the cheapest
  member by the metrics series (the median of its passes, this host's
  `nightly`, `weekly` and `manual` records). A family with neither is flagged,
  unless it is exempt.
- **An exempt family has no fast case, on purpose** (#1008):
  `FBF_FAMILY_EXEMPT("why")` in any one of its cases, the reason a single
  string literal on one line. The lister reports it as `exempt: why`, not
  flagged. Exempt only where no smaller input keeps the family's check: a
  measurement that asserts nothing, or a run whose size is the point. A
  family takes a representative or an exemption, never both. The one at
  #1008 is `McscfIntersectionTiming`, a timing; `Nevpt2QdIntersection.Timing`
  needs none, being one case of a family with fast ones.
- **Where to mark.** Where a family's cheapest case is a refusal, mark its
  cheapest validation case, which stays in the fast set too. Where one mode of
  a `TEST_P` needs a fast case of its own, split it off. The markers at #984 are
  exactly #993's exceptions: eight refusal-led families' validation cases, and
  `Nevpt2EnergyDeviceRdms`'s `n2_ccpvdz_cas1010_df_batch0` for the tiled rdms.
  #1008 added a smaller input as the representative of each family that had
  none, where its every case was past the line: STO-3G or 6-31G water in
  place of cc-pVDZ (`h2o_631g` in `data/casscf/`, PySCF over the same basis),
  and fewer coordinates of a ladder; `E2eCis`'s one case, at 1.5 s on the
  gfx1200, was only ungated.
- **The slow-test rule, restated:** a case past about 1.5 s on an idle RTX 3080
  goes under `FBF_SKIP_IF_FAST_TESTS()` unless it is its family's
  representative (`slow_test.h` has where the line comes from).

```bash
python3 devtools/test_families.py build-hip            # family [members] -> representative
python3 devtools/test_families.py build-hip --flagged  # only the flagged
python3 devtools/test_families.py build-hip --json     # for #989
python3 devtools/test_families.py build-hip --log .slow-tier-reports/pr-….log
```

It lists the ctest entries that are one gtest case, and each gtest binary's
`--gtest_list_tests --gtest_output=json`, which gives each case's file and
line and never initialises the GPU runtime. A marker belongs to the case
whose definition line is the nearest one above it. Times come from
`--metrics FILE`, or else the checkout's `metrics` branch when it has one
(`--no-metrics` turns that off); `--log` adds a gate's log, its gtest
`[       OK ] … (N ms)` and ctest `Passed … sec` lines. With no times, every
unmarked family is flagged, which is the state until #982's records exist.
It exits 1 when a marker cannot be placed (outside every test, an instance
that matches nothing or more than one, a second marker for one family, an
exemption without one reason, or a family both marked and exempt), and 2
when there is nothing to list.

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
  above) and left CLAUDE.md a summary and a link; #985 left CLAUDE.md one line
  on the nightly and a link to "The nightly" above.
