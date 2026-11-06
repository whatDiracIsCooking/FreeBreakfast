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
| `push` | `git push`, by pre-commit's pre-push stage | the C++ half when the push touches C++ outside `experimental/` and `test/experimental/`, the Python half when it touches `.py`; the fast set of each | under 5 min on a busy card |
| `pr` | by hand before opening a PR; `milestone-run` requires it | C++: each family's representative and nothing else (#989; `--all`, the whole fast set). Python: the fast set | under 5 min on a busy card |
| `nightly` | a timer on this box | everything, unskipped, both languages, plus the cross-backend compile check | none; it has the night |

**The single-backend rule holds for all three.** A gate runs on one card, and
which one is printed in its result line. No gate runs both backends unless asked.
The nightly is **CUDA**, because production numbers are CUDA and because an
unattended HIP run on the gfx1200 can wedge the card (#477). It covers HIP with
`devtools/cross-backend-check.sh`: a seconds-long compile of every `.cu` through
clang `-x hip`, which is the one portability hole a single-backend run cannot see.

**C++ under `experimental/` or `test/experimental/` fires no C++ half.** The
smoke builds the `default` preset, which never configures either one: they
sit behind `FREE_BREAKFAST_BUILD_EXPERIMENTAL`. A cold build there would check
nothing in the push. Build the `experimental` preset by hand instead. The root
`CMakeLists.txt` line that adds them is outside both, so it still fires the
half (`NOT_IN_CPP_SMOKE` in `devtools/test.py`).

**The push gate's test set does not change in this milestone.** It is already a
fixed set, and its job is "is this obviously broken". Selection applies to `pr`,
and only to its C++ half ("Test selection" below). **The nightly is what makes
that safe:** it runs every case the `pr` gate leaves out, within a day.

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

## Which tier a check goes in

A C++ check runs in one of three tiers, from cheapest to dearest:

| Tier | How it fails | Where it lives |
| --- | --- | --- |
| **compile time** | the build | `static_assert`s; a TU of them alone is `test/compile/<tree>_compile.cpp` |
| **host** | a host binary's case, on a box with no card | `*_host.cpp`, in a tree's `*_tests_host` binary |
| **GPU** | a GPU binary's case, which skips without a card | `*_gpu.cpp`, in a tree's `*_tests_gpu` binary, guarded by `FBF_REQUIRE_GPU()` |

**The rule: each claim goes in the cheapest tier that can check it fully, and
is checked there only.** A claim the compiler can settle is a `static_assert`,
not a case. A claim a host case can settle without a shim is a host case, not
a GPU one. Only what needs the device goes in the GPU tier.

- **The unit is a claim, not a feature.** A feature usually spans tiers.
  `einsum`'s refusals return before any launch, so they are host cases
  (`test/recipes/einsum_host.cpp`), and its numbers are GPU cases
  (`einsum_gpu.cpp`). Placing the refusals low does not excuse the numbers
  from the GPU tier, and a claim checked in a lower tier is not checked again
  above it.
- **Compile time owns logic, never numbers.** It owns concepts, types,
  layouts, launch shapes, integer arithmetic and refusals of the wrong type.
  A `static_assert` on a `constexpr` function checks it on the compiler's
  evaluator, and the device's floating point can differ from that (FMA
  contraction, intrinsics, flags). So a floating-point result a kernel
  produces is a GPU claim, even where the same `constexpr` source runs on
  both.
- **A shim never buys a lower tier.** A shim is a fake device layer, a mock, or
  a host-only branch of production code added so a test can run without a
  card. A claim that needs one is a GPU claim. Never restructure `src/` to make
  something host-testable: the tree is GPU-first, with no host-only splits.
- **An oracle is not a shim.** `test/ci_oracles/` and the other host
  references a GPU case compares its result against are a reference, not a
  stand-in for the device, and they stay.
- **The tier decides the binary, and so the label.** A host case in a GPU
  binary carries that binary's `gpu` label, which the `ci-cuda` and `ci-hip`
  test presets exclude, and runs in `test.py`'s GPU batch on the card. It
  belongs in its tree's host binary, made for it if the tree has none. A host
  binary may link the device runtime through the modules it imports: what
  makes it host is that no case initialises a device, which the device-less
  presets then hold it to.
- **A compile-time TU builds without GoogleTest.** `test/compile/` is added
  from the root `CMakeLists.txt`, outside the `FREE_BREAKFAST_COMPILE_TIME_ONLY`
  gate that skips `test/`, so the compile-only configure builds it and
  `devtools/cross-backend-check.sh` checks it for the other backend. It
  declares no case, so ctest never lists it. A `static_assert` that guards a
  runtime case's own setup (a table of what a call takes, which the case then
  loops over, as `gemm_gpu.cpp`'s `kGemmTakes`) stays beside that case; only a
  TU with no case moves.

**Today:** `test/compile/recipes_compile.cpp` (the concepts `src/recipes`
exports) is the compile-time tier, and `test/recipes/` has a host binary for
its two host TUs, which rode in its GPU binary before.

**A GPU-binary case that never reaches `FBF_REQUIRE_GPU()` is flagged**
(#1037), as the mark of a host claim in the wrong binary:
`devtools/test_families.py BUILD --unguarded` lists each one, by file and line,
and exits 1 when there are any. A case reaches the guard when its body calls
it, or calls a function or lambda that does, at any depth, or when its
fixture's class or a base of it does. The analysis is by name, over the case's
file and the headers beside it it includes by quotes, with comments and
strings blanked. At #1037 it found nine on main, and each moved to its
tree's host binary: `analysis_host.cpp`, `df_high_aux_host.cpp`,
`infrared_host.cpp`, `nevpt2_density_fitting_host.cpp` and
`stable_part_host.cpp` in `test/gtest/`, and the NEVPT2 Hessian's config
refusals in `test/pipeline/hessian_host.cpp`. Where a host case shares a
helper with GPU cases, the helper is a header beside them (`stable_part.h`,
`df_high_aux_fixture.h`), never a second copy. The two copies of
`stable_part`'s case became one. The check is not yet wired into a gate.

**Planned:** a home for a "must not compile" claim that no concept can
express, which today needs a try-compile entry the tree does not have (one a
concept expresses is `static_assert(!Concept<…>)`).

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
- **`push` prefers a built tree.** Its C++ half is an incremental build, so
  under `auto`, where some presets' trees are built (every binary the push
  runs) and others are not, it picks among the built ones and says
  `push: auto -> <preset>, the built tree`. A fresh `.claude/worktrees/`
  checkout whose `pr --preset default` built `build/` therefore pushes on
  CUDA, rather than building `build-hip/` from cold (about four minutes even
  from sccache) and then running the GPU binaries on the slower card. With every
  tree built, or none, or no built tree's card live, it is the plain card
  choice. `CMAKE_PRESET=<preset> git push` still names one.

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
8. **File the night's issue** ("Failures open an issue" below): a failure
   opens or comments on it, a pass closes it.

The build and the GPU run are **strictly sequential**. Many GPU cases are bound
by launch latency, so a busy host CPU would inflate their times and make the
series noisy.

It logs to `<primary>/.slow-tier-reports/<mode>-<stamp>-<rev>-default.log`,
with the per-case reports in `<log>.results/`, and appends
`test.py nightly PASS  rev=…  preset=default  mode=nightly  stamp=…  metrics=…  slow-rule=N/M`
to `summary.log`, with `issue=…` after it when the night filed one. The exit
status is the first failing phase's.

**A reduced run** takes ctest's arguments after `--`, as `pr` does:
`devtools/test.py nightly --rev REF -- -R Name`. They select the ctest
entries and the GPU batches; pytest and the slow-test report do not run, and
the record is `manual`, so it never enters the timing series. It is how a
change to the nightly itself is tried end to end without an hour's run (#986's
acceptance was three of them). Its summary line carries `run=reduced`, and a
`--rev` run's carries `ref=REF`, so neither is mistaken for the timer's night.

### How to tell it ran

A passing night makes no noise, and neither does a night that never ran: the
box off at 02:30, the combined devcontainer down, a broken unit (#1120's exited
127 in 6 ms). The signals, cheapest first:

- **The summary line**, the newest `test.py nightly` line in
  `<primary>/.slow-tier-reports/summary.log`.
- **The `metrics` record**, its `metrics=` commit on the box's local `metrics`
  branch ("Metrics" below).
- **systemd, on the host:** `systemctl --user list-timers "$unit.timer"` for
  the last and next fire, and
  `journalctl --user -u "$unit.service" --since yesterday` for what it did.
- **The `nightly` issue**, which is opened only by a night that ran and failed.

**The staleness check** (#1127, `devtools/nightly_stale.py`) reads the first of
these for you. `devtools/doctor.sh` reports it under `nightly`, and
`test.py push` and `pr` print it as one `WARNING nightly:` line at the end, and
never fail on it. It warns when the newest timer night, any result, is older
than `NIGHTLY_STALE_HOURS` (36, `devtools/config.sh`), or is a FAIL, and says
"no nightly has run here" when there is none but the timer is installed. Lines
with `ref=` or `run=reduced` do not count. With no line and no timer unit, a
VM or another box not meant to run the nightly, it is a note, not a warning.
On the host, doctor also shows the timer's last and next trigger, and the
service's last result when it is not `success`; in the container there is no
systemd user manager, and it says so.

- **It reads the primary checkout's** `summary.log`, from any worktree, since
  that is where the nightly writes it. A worktree's own reports directory holds
  only its `pr` runs.
- **A night's time is its log's mtime**, the end of the run, where the log is
  still there, else its `stamp=` as local time. The host and the container keep
  different time zones, and the mtime means the same in both.

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

### Failures open an issue

**Today (#986):** the nightly's last step, `devtools/nightly_issue.py`, run
through `gh`.

- **A failed night opens an issue** labelled `nightly`, titled
  `Nightly fails on origin/main`, and creates the label first where the
  repository has none. When an open `nightly` issue already has that exact
  title, the night **comments on it instead**, so a failure that persists
  stays one issue. Open issues are *listed*, not searched: GitHub's search
  index lags, and the next night must find this one's issue.
- **The body** carries the summary line, the commit, the host, the **metrics
  commit** (the record's, on the box's local `metrics` branch), the log's path,
  the phases that failed, the **failing cases** (the record's `fail` cases),
  the **slow-test findings** when the night wrote `slow-rule.json` (its
  over-the-line cases and steps, as `slow_rule.py` prints them), and a **log
  excerpt**: the failed cases' output, `[ RUN ]` to verdict, else the log's
  tail, where a build or a phase says why it stopped. GitHub caps a body at
  65,536 characters; the excerpt gives way first.
- **The next green night comments that it recovered, and closes the issue.**
  The failure after that opens a new one.
- **Only a night of `origin/main` files**, by default. A run of another rev
  files nothing unless given `--issue-title`, so a by-hand `--rev` run never
  writes to the repository by surprise. A checkout that cannot be made (the
  fetch, a bad rev) files from the outer half, since no inner half runs.
- **An issue that cannot be filed fails the night**, as `issue=FAILED` on the
  summary line, like a record that could not be written: `gh` missing,
  unauthenticated (it reads `GH_TOKEN`), or refused.

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
skip, including a `TEST_P` that reaches the macro only for some params.
ctest's JUnit carries no gtest property, so a case ctest runs (the host
binaries, and the `RUN_SERIAL` cases) gets its `slow` from its own gtest XML
report instead (#1016): every host and `RUN_SERIAL` registration passes
`XML_OUTPUT_DIR`, one report per entry in the build directory's `gtest-xml/`
(not the batched GPU ones, which have their batch's JSON, and whose command
must stay bare for `batchable()`),
and a recorded run moves that directory into its reports
(`metrics.gtest_xml_slow`). A case with no report there (one that crashed, a
record from before #1016) has no `slow`; one of those over the line is listed
apart, as membership not recorded, never dropped.

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
the findings as JSON in `<log>.results/slow-rule.json`, which a failed night's
issue (#986) carries in its body. It holds `over_line`, `over_line_unrecorded`
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
  which names each entry's binary, and `gtest-xml/`, each ctest entry's own
  gtest XML (moved from the build directory, which a recorded run clears
  first), which gives a ctest case its `slow`. A batch that crashed writes no report, so
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
| `cases` | per test case: `name` (`Suite.Case`, or the pytest node id), `binary` (the gtest binary's name, or `pytest`), `status` (`pass`, `fail`, `skip`), `seconds`, and for a gtest case `slow` (whether it reached `FBF_SKIP_IF_FAST_TESTS()`, which a fast run skips; #987, absent before it, and for a ctest-run case before #1016). From gtest's `--gtest_output=json`, ctest's `--output-junit` with each entry's gtest XML, and pytest's JUnit XML |
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
python3 devtools/test_families.py build-hip --unguarded  # GPU cases with no FBF_REQUIRE_GPU()
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

**Today (#989):** `test.py pr` runs **each family's representative and nothing
else** of the C++ cases, and the nightly runs the rest. The selection needs no
diff and no graph: it is the same set for every PR. It came out of #988's
replay, below, which found diff-based selection too expensive to be worth it.

- **The set is the lister's.** After the build, `pr` runs
  `devtools/test_families.py <build dir> --json` ("Families" above) and takes
  each family's `representative`, the marked case or else the cheapest by the
  metrics series. Nothing in `test.py` re-derives it.
- **An exempt family runs nothing**; that is what `FBF_FAMILY_EXEMPT` says.
- **A flagged family runs whole**, every member of it in the fast set: with no
  representative to stand in for it, selection runs more, never fewer. Each
  one is named on a `pr:   flagged, run whole:` line.
- **A ctest entry that is not one gtest case** (the examples, which run
  `free-breakfast` through a cmake script) is in no family, and runs.
- **A wrong marker fails the gate** (the lister's exit 1), before the card is
  taken: the set would be missing a family. So does a lister that cannot list.
  Fix the marker, or run the whole fast set with `--all`.
- **The join is on gtest's names.** The lister names a case as gtest's own
  `--gtest_list_tests` does, and so does each ctest entry's `--gtest_filter`,
  so `pr` reads the case from the filter and never from ctest's entry name,
  which prints a value param differently (`/"Singlet"` against `/Singlet`,
  `<double>` against `/1`; #988).
- **Then the split is the usual one** ("Reading a gate's run"): the unselected
  entries go into ctest's exclude file beside the batched ones, and the
  batches carry only selected cases. The preset's filters and a `-- -R …`
  still narrow the listing first, so `-- -R Foo` runs `Foo`'s representatives;
  `--all -- -R Foo` runs its whole fast set.
- **It prints the count against the lister's:**
  `pr: representatives only: N of M ctest entries selected, against the
  lister's K representatives (…)`. Without a `--` filter, `N` is `K` plus the
  flagged families' members and the entries that are not gtest cases.
- **And the representatives that skipped:** `pr: N representative(s)
  skipped`, each named on a `skipped representative:` line. A family whose
  representative skipped ran nothing in this gate; under
  `FREE_BREAKFAST_FAST_TESTS` that is a case that must be ungated, or a
  different representative marked. #989 found nine families whose cheapest
  case by the nightly's (unskipped) times is one `FBF_SKIP_IF_FAST_TESTS()`
  gates, and marked each one's single fast instance: cheapest by time is not
  cheapest in the fast set.
- **Python is not selected.** `pr`'s Python half is the fast pytest set, as
  before.
- **What the gate gives up** is a regression that only a non-representative
  case would catch: it is caught by the next nightly, which opens an issue
  (#986), rather than before the merge.

### The map: `devtools/test_map.py`

**Today (#983):** the build's dependency graph, written as one JSON map. It is
an analysis tool, not wired into any gate. #988's replay read it, and it stays
for that.

```bash
python3 devtools/test_map.py build-hip -o map.json   # the map; report on stderr
python3 devtools/test_map.py build-hip --file src/integrals/mmd/mmd_launch.cu
```

Nothing in it is written by hand; the compiler and CMake are the sources of
truth.

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

`category` says why a file is unplaced. `cmake` and `data` would run
everything under a diff-based selection. `doc`, `config` (`.clang-format`),
`python` and `unconfigured` (the nearest `CMakeLists.txt` is not configured in
this preset: `test/experimental/` outside the experimental presets) have no
effect on this build. `other` is anything else, a file the map cannot account
for, and the report lists each one. On `build-hip` at #983, `other` is empty,
and the unmapped entries are `example_rhf` and `example_hessian`, which run
`free-breakfast` through a cmake script rather than a gtest case.

### The replay, and why selection is not diff-based

**#988** replayed the last 50 merged PRs (#911 to #1013) against the drafted
contract: a test file at distance 0 or 1 from a changed file's owner ran its
whole fast set, one at distance 2 or more ran its families' representatives,
and a `CMakeLists.txt`, `cmake/`, `deps/`, `data/`, `test/support/`,
`tools/emit/`, `test/*/data/` or an unplaceable file ran everything. Each PR's
selection was costed by the nightly's per-case times, against a fixed bar:
**go** if the median PR's selection cost at most half the fast GPU batch.

It was a **no-go**: the median was **73%** (66% even with a `CMakeLists.txt`
beside a new source counting only as that source), and no PR touching a
`src/` module selected less than 56%. Two causes, both structural:

- **A `CMakeLists.txt` runs everything, and adding a file means touching one.**
  17 of the 23 run-everything PRs were there only for that.
- **Umbrella imports flatten the graph.** `free_breakfast.pipeline.config`
  imports every pipeline step and is in 48 of the 157 GPU test files;
  `free_breakfast.core` is in 154.

The representatives alone cost **53%** of that batch, for every PR, which is
less than the contract selected for nearly every PR touching `src/`. So the
gate is the representatives, and **#990**, which would have made a diff-based
selection the gate, was closed. The replay's analysis is in #988's comments.

## What stays out of scope

- **`.github/workflows/ci.yml`.** The server-side C++ job stays `if: false`.
  Getting the toolchain image onto a runner is its own project.
- **A HIP nightly**, and the sanitizers on a schedule. The cross-backend compile
  check is the nightly's HIP cover for now.
- **Moving CLAUDE.md's testing sections here** happens with the code each
  describes. #981 moved "Reading a test run" (now "Reading a gate's run"
  above) and left CLAUDE.md a summary and a link; #985 left CLAUDE.md one line
  on the nightly and a link to "The nightly" above.
