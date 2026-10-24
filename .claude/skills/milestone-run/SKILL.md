---
name: milestone-run
description: >-
  Complete a GitHub milestone end to end by orchestrating subagents: read the
  milestone's issue DAG, fan out one subagent per ready issue (each in its own
  .claude/worktrees/ worktree, building and testing on a GPU card), merge
  their PRs one at a time, and launch newly unblocked issues until the
  milestone is done. Use when the user invokes `/milestone-run <milestone
  URL|number> [max=N] [both] [dry-run]` or asks to "complete / execute / ship
  this milestone with subagents". Invoking it is the user's merge grant for
  this milestone's PRs. Requires gh + GH_TOKEN. Not for creating a milestone —
  that is `milestone`.
---

# `/milestone-run <milestone> [max=N] [both] [dry-run]` — milestone DAG → merged PRs

`milestone` turns a plan into a DAG of PR-sized issues; this skill **executes**
one. You are the **orchestrator**: you schedule, create worktrees, review, merge
and tear down. Subagents implement one issue each, ship a PR with their own
test run reported in it, and **hand it back without merging**.

**Invoking `/milestone-run` is the merge grant** — for PRs that close an issue
in *this* milestone, opened by *this* run's subagents, once the gates in step 5
are green. It does not cover any other PR, `--admin`, or merging red. If the
auto-mode classifier still blocks a merge, stop merging and report the green PR
links; do not work around the block.

Run it **inside the devcontainer** — the C++ toolchain only exists there — and
from the `combined` variant if you want both cards (`cuda` and `hip` see one
each). Everything below runs from the main checkout root.

## Arguments

- **Milestone** — a URL (`…/milestone/<N>`), `#N`, or a bare number. The
  number is all you need.
- **`max=N`** — the most subagents in flight at once. Default **3**. Every
  agent builds the whole module tree (plus WarpWraps) in its own worktree, and
  a module build is memory-hungry per job, so the box, not GitHub, is the
  limit: past ~4, warn the user that builds will thrash and GPU runs will
  queue, then honour the number they gave.
- **`both`** — require every PR that touches C++ to pass `cpp-tier.sh` on
  **both** cards (`--preset default` = CUDA, `--preset hip` = HIP). Without
  it, **one backend suffices**: each agent runs the tier with the default
  `auto` preset, which takes the least-loaded live card, and the other backend
  gets only `cross-backend-check.sh`'s
  compile check (when a `.cu` is touched) — CI's `cpp` job is `if: false`, so
  **nothing on the server builds or runs C++ at all**. Prefer `both` for
  anything touching a `.cu`, a kernel under `src/integrals/`, or `src/recipes`;
  one backend is for work whose device code is unchanged.
- **`dry-run`** — do steps 1–2, print the schedule, and stop. No worktree, no
  agent, no `gh` write.

## 1. Read the milestone and rebuild the DAG

```bash
gh api repos/{owner}/{repo}/milestones/<N> --jq '{title,state,open_issues,closed_issues}'
gh issue list --milestone "<title>" --state all --limit 200 \
  --json number,title,state,body
```

`--milestone` wants the **title**, not the number. For each issue, take its
blockers from the `## Depends on` lines that `milestone` writes into every
body — that is this repo's source of truth — plus any native dependencies
someone added on top:

```bash
gh api repos/{owner}/{repo}/issues/<n>/dependencies/blocked_by --jq '.[].number'
```

Then classify every issue:

- **done** — closed.
- **ready** — open, every blocker closed, and no open PR already references it
  (`gh pr list --search "<n> in:body" --state open --json number,headRefName`).
- **resumable** — open, with an open PR whose head branch is `issue-<n>-*`
  *and* a local `.claude/worktrees/issue-<n>-*` checkout. An earlier run of
  this skill died or was compacted mid-flight: adopt it. A PR whose body
  reports a passing run goes straight to step 5. Otherwise give a fresh agent
  the existing worktree and PR ("finish PR #M") instead of a new branch. An
  open PR that fails either test is someone else's: leave it alone and report
  it.
- **blocked** — some blocker still open. A blocker *outside* the milestone that
  is open is a hard stop for that branch: report it, don't work around it.

A cycle, or an issue with no `Acceptance` section, means the milestone is
malformed: stop and say so rather than guess an order.

## 2. Show the schedule, then start

Print the wave table (issue, title, blockers, state), the ready set, `max`, and
whether `both` is on. On `dry-run`, stop here. Otherwise start without waiting
for a reply — the invocation was the go-ahead. Stop and ask instead only when
step 1 found something odd (cycle, external open blocker, an issue that looks
bigger than one PR).

## 3. Pre-flight, once

```bash
git fetch origin main --quiet && devtools/worktree.sh sync
devtools/doctor.sh                       # container side? flock present?
nvidia-smi -L                            # CUDA card alive?
rocminfo | grep -m1 gfx                  # HIP card alive? (gfx1200 here)
```

GPU commands from Bash need `dangerouslyDisableSandbox` to see either device —
without it both cards look dead. A `nvidia-smi` that says `Failed to
initialize NVML` means the CUDA card dropped out of the container; `rocminfo`
listing only the CPU agent means the amdgpu DKMS module is not loaded on the
host. Neither is fixable from here.

Then warm the shared `.sccache` from the main checkout, one build per backend
you will use, so `max` cold module builds do not all start from nothing:

```bash
devtools/cpp-tier.sh --no-test                   # CUDA (build/)
devtools/cpp-tier.sh --preset hip --no-test      # HIP  (build-hip/)
```

The live cards decide what can run: `auto` already skips a card with no
device node (or no compiler), so one dead card needs nothing from you beyond
saying so; both dead, or one dead under `both` → no C++ agent can pass its
gate, so don't launch one. Report it and stop (a milestone of pure
Python/tooling issues can still run).

## 4. Dispatch loop

While any issue is ready or in flight:

1. **Launch** `min(max − in_flight, |ready|)` ready issues. For each, **you**
   create the worktree, one at a time, so parallel agents never race on
   `git fetch` or the ref lock:

   ```bash
   git fetch origin main --quiet
   git worktree add .claude/worktrees/<wt> -b <wt> origin/main
   git branch --unset-upstream <wt>
   ```

   `<wt>` is `issue-<n>-<slug>`. Don't use the Agent tool's
   `isolation: "worktree"`: it lives elsewhere and tears itself down, which
   removes the checkout you need for a rebase. Also hand each agent:

   - **no backend** (unless `both`): agents run the default `auto` preset,
     which picks the least-loaded live card at the start of each run
     (`gpu_auto_preset` in `devtools/lib.sh`) and prints the choice. Each
     card has `GPU_SLOTS` (default 2) slot locks (`with_gpu_slot`, lock files
     under `GPU_LOCK_DIR` in `devtools/config.sh`), so the two cards never
     block each other and up to two siblings share one card; a run waits only
     when its card's slots are all held or its load is at `GPU_BUSY_PCT`
     (80%). Compiles hold no slot. Both compilers seeing the milestone is
     then a matter of load, not schedule — `both`, or the `.cu` rule below,
     is what guarantees it.
   - **the files it owns.** Two agents editing the same module, or both
     touching the generated kernels under `src/integrals/`, conflict on every
     merge — and an emitted kernel file is a whole-file diff with no clean
     3-way resolution. Partition the work so each in-flight agent owns
     disjoint files — different modules, different kernel classes — and say
     which explicitly in its prompt. If two ready issues cannot be
     partitioned, launch them in sequence, not together.
   - **a job count** `J = BUILD_JOBS / max` (default 12, `max=3` → `-j 4`).
     Not `nproc / max`: `BUILD_JOBS` is already capped below the core count
     because module builds OOM, and an OOM-killed compiler surfaces as a bare
     `ninja: build stopped`.

   Then spawn a background `general-purpose` subagent with the prompt in
   "Subagent prompt" below. Record `issue → {wt, agent id, files}`, and the backend once the agent reports it.

2. **Wait for notifications**; never poll on a short timer. When an agent
   reports:
   - **"PR #M ready"** → step 5. Merges are serial: if one is in progress,
     queue this one.
   - **stuck** (ambiguous issue, a suite failed, would need a tolerance
     loosened, scope bigger than one PR) → don't merge. Mark the issue
     **stuck**. Its dependents stay blocked. Keep its worktree. Carry on with
     every other branch of the DAG.

3. **Watch for stragglers.** If an agent runs long, check `gh pr checks <M>`
   yourself. A PR whose checks are done but whose agent is still waiting is
   usually waiting on the lock behind a sibling's ctest, or on a check that
   never runs on that PR — nudge it with `SendMessage`. Use `TaskStop` on one
   that has already handed back.

**Status line.** After every launch, merge, or stuck report, print one line —
`merged #301 → PR #310 (3/9 done) · launched #305 [hip] · in flight: #302 #304
· stuck: none` — so the user can follow a long run without reading the
transcript.

## 5. Review and merge — one PR at a time

Merge **sequentially**, never in parallel and never with `--auto` — you need
to see `main` move before deciding the next merge. Before each merge, check:

```bash
gh pr view <M> --json mergeable,body,files,commits
gh pr checks <M> --required --watch          # lint, test (3.11), test (3.13)
```

- `mergeable` is `MERGEABLE` and the required checks are green. Those are
  `ci.yml`'s lint + fast pytest jobs only — they say nothing about C++
  beyond formatting and `cmake-lint`;
- the body has `Closes #<n>` and, under *What was run*, the preset(s), card,
  and ctest pass **and skip** counts — at least one card, or **both** under
  `both`. A `--no-test` build is not a test; a run whose gpu cases all
  `GTEST_SKIP`ped ("no GPU device visible") is not a GPU test, however green;
  `cross-backend-check.sh` exit 2 is not a pass;
- a `.cu` in `files` comes with a `cross-backend-check.sh` result (or both
  presets run);
- no tolerance was loosened and no ratchet moved to get green.

**Rebuild on the fresh main before merging any PR that touches C++.** Text
mergeability is not enough: a symbol another PR moved or renamed does not show
up as a conflict, and with the `cpp` CI job off GitHub will squash-merge a PR
that no longer builds. So if `origin/main` has moved past the PR's base (it
will have, after any earlier merge in this run):

```bash
git -C <wt> fetch origin main && git -C <wt> rebase origin/main
devtools/cpp-tier.sh --preset <the preset its run used> -j <J>   # inside <wt>; that build dir is warm
git -C <wt> push --force-with-lease
```

A Python- or docs-only PR skips the rebuild; the required checks cover it.

Then `gh pr merge <M> --squash` — **never `--delete-branch`**: it tries a
local `git checkout main`, which fails because `main` is held by the primary
checkout. The server deletes the remote branch on merge.

**Conflicts.** Each merge moves `main`, so a later sibling can still conflict
despite the partition. Two cases:

- **A pure list or registry conflict** (both sides appended an entry — a
  `CMakeLists.txt` source list, a README table) → resolve it yourself in the
  PR's worktree during the rebase above, keeping both sides, then
  `devtools/format.sh --check` and the rebuild.
- **Anything else** — module code, a test, or any **generated kernel** under
  `src/integrals/` → send it back to the agent that wrote the PR
  (`SendMessage` resumes it with its context): rebase, resolve, **regenerate**
  kernels with the FelizJueves emitters rather than hand-merging emitted text,
  re-run its tier, push, re-report.

**After each merge:**

- confirm it merged (`gh pr view <M> --json state` reads `MERGED`) and the
  issue closed (`gh issue view <n> --json state`);
- tear the worktree down. An agent that built has initialised
  `deps/WarpWraps`, so plain `remove` refuses — check it is clean first, then
  force (see the `worktree` skill):

  ```bash
  git -C .claude/worktrees/<wt> status --porcelain --ignore-submodules=none
  git worktree remove --force .claude/worktrees/<wt> && git branch -D <wt>
  ```

  `-D`, because the squash merge gave `main` a different SHA;
- `devtools/worktree.sh sync` so local `main` follows;
- print the status line, recompute **ready**, and go back to step 4.1.

## 6. Finish

When nothing is ready or in flight:

- if every issue is closed, close the milestone
  (`gh api -X PATCH repos/{owner}/{repo}/milestones/<N> -f state=closed`);
- report one line per issue: `#n → PR #M merged [cuda|hip|both]`, or
  **stuck** with its reason, or **blocked** naming its blocker. Spend words
  only on what deviated: blocked merges, rebases, stuck issues, card
  fallbacks, the backend no run covered, decisions made for the user.

Leave the worktrees of stuck issues standing, and say where they are.

## Subagent prompt

Fill in `<…>`. Keep it self-contained — the agent has none of your context.
`<preset>` is `auto` — the least-loaded live card; the run prints which. Under
`both`, give the two runs `--preset default` (CUDA) and `--preset hip` (HIP)
and drop the fallback line.

```
Resolve GitHub issue #<n> (<title>) in this repo. Read it first:
`gh issue view <n>` — its Scope, Acceptance and Files / area sections are your
spec. Your worktree already exists at <abs path to .claude/worktrees/<wt>>, on
branch <wt>, forked from origin/main — EnterWorktree {"path": …} into it and
do not create another.

Read CLAUDE.md, and docs/style.md before writing C++. Keep the public surface
small: no helper type, alias or template just to remove repetition.

Rules:
- Do NOT merge and do NOT tear down the worktree. Your job ends at an open PR
  with your own run reported in it. Then reply exactly: "PR #<M> ready" + one
  line naming the preset, card, and ctest pass/skip counts.
- CI runs only lint + the fast pytest tier; nothing on the server builds C++.
  Your reported run is the only evidence the PR's C++ works. Never claim a
  check passed that you did not see.
- C++ changes: run the fast tier on your assigned card, from the worktree:
    devtools/cpp-tier.sh --preset <preset> -j <J>
  Bash needs dangerouslyDisableSandbox to see the GPU. With --preset auto the
  run prints "gpu: auto -> <preset> (<backend>)" with each card's load and
  slots; quote that preset in the PR. The ctest phase takes one of that card's
  slots (with_gpu_slot, devtools/lib.sh) and says it is waiting when the
  slots are full or the card is busy — that is expected, not a hang. Read the
  ***Skipped list: if the gpu cases skip with "no GPU device visible", your
  card is down (nvidia-smi -L / rocminfo) — rerun on the other preset and say
  so in the PR. If the issue touches code the FBF_SKIP_IF_FAST_TESTS cases
  validate, also run devtools/cpp-slow-tier.sh --preset <the same preset>.
  Say in the PR which backend you did NOT run. If you touched a .cu, also run
  devtools/cross-backend-check.sh (exit 2 means it could not run — not a
  pass).
- Python changes: pytest -m "not slow" -n auto -rs, and read the skip reasons.
- Before committing: devtools/format.sh --check (devtools/format.sh fixes) and
  devtools/cmake-lint.sh if you touched CMake.
- If a suite fails, or passing would mean loosening a tolerance or moving a
  ratchet, stop and reply "STUCK: <reason>" — no PR needed.
- Commit and push with `git -C <abs worktree path> …`. Never --no-verify,
  never CLAUDE_ALLOW_MAIN_EDITS. End the commit message with the
  Co-Authored-By line from your system reminder.
- Open the PR with `gh pr create --body-file <file in your scratchpad>`, with
  `Closes #<n>`, and under "What was run" the preset(s), card, and ctest
  pass/skip counts (plus pytest counts if you ran it).
- You own these files and ONLY these: <the agent's partition>. Siblings are
  working in parallel, and a generated kernel has no clean 3-way merge — if
  you need to change a file outside your partition, reply "STUCK: <what>"
  instead. If asked to rebase, REGENERATE emitted kernels (the FelizJueves
  tools/emit_*_kernels.py scripts) rather than hand-merging their text.
- If the issue is ambiguous in a way the code can't answer, reply
  "STUCK: <question>" rather than guessing.
```
