---
name: pr
description: >-
  Ship the current work: commit the relevant changes, push the branch, open a
  GitHub PR with gh, and merge it. Use when the user asks to "PR this", "ship
  it", "commit/push/merge", or otherwise wants the working-tree changes landed
  on main end to end. Requires gh + GH_TOKEN.
---

# Commit → push → PR → merge

Drive the current changes all the way to `main` in one pass. Each step has a
gotcha; follow them in order.

## 1. Commit the relevant changes

- **Never commit on `main`.** If the session is on `main`, create a branch
  first (`git switch -c <topic>`). If the session is in a `.claude/worktrees/`
  worktree, the branch already exists — use it.
- Stage **relevant** changes, not everything: read `git status` and `git diff`,
  stage what belongs to this task, and leave unrelated untracked files behind.
  No blanket `git add -A` without looking.
- The pre-commit hook lints every touched file — a finding blocks the commit;
  fix it, don't `--no-verify`.
- If `pyproject.toml` changed, run `uv lock` and commit the lock with it. The
  lock is the only place exact versions live, and the image installs it with
  `--frozen`, so a stale lock breaks the container rather than the host.

## 2. Check the repo's own invariants

Before pushing, look at what the diff touches and whether this repo guards it:

- a **ratchet** (a test pinning a measured number or a floor) may only move in
  the improving direction, in its own commit, saying what moved. A loosened
  threshold produces a green run — that is exactly why it needs a human.
- the **reference chain** (whatever independently checks correctness here) is
  not something to adjust to make a test pass.

A repo with neither has nothing to do in this step.

## 3. Push

```bash
git push -u origin <branch>
```

- A push touching any `.py` auto-runs the fast test gate
  (`devtools/prepush-tests.sh`). A successful push therefore *implies* that
  gate passed. Do not bypass with `--no-verify`.
- If the worktree's git dir has gone missing (`fatal: not a git repository:
  …/worktrees/<name>`), push from the main checkout instead:
  `git -C <repo-root>/main push origin <branch>`.

## 4. Create the PR

```bash
gh pr create --title "<concise title>" --body "<what and why>"
```

Summarise what changed and why, and note any test implications. The harness's
PR-body footer rules apply.

## 5. Merge — GitHub's native auto-merge

```bash
gh pr merge <number> --auto --squash
```

`--auto` hands the merge to GitHub: it squash-merges server-side the moment the
checks the `main` ruleset requires (`lint`, `test (3.11)`, `test (3.13)` from
`ci.yml`) pass, and the repo's "delete branch on merge" setting removes the
remote branch. If they have already passed, it merges immediately. Do it for
every PR you intend to land; **skip it** for the "stop after step 4" cases in
"When NOT to merge" below (CI expected red, a ratchet/reference touch, or the
user asked for a PR but not a merge). `gh pr merge <number> --disable-auto`
takes it back.

- **Only required checks gate it.** Auto-merge waits for what the ruleset
  requires and nothing else. A job missing from that list, or renamed in
  `ci.yml` so it no longer matches the ruleset's name, is not waited for — and
  a required name that no job produces any more blocks every PR forever.
- **If `ci.yml` is manually disabled** (`gh workflow list --all` — see
  CLAUDE.md's "What is gated, and what is not"), the required checks never
  report and an auto-merge PR sits open indefinitely. Merging then needs an
  admin bypass in the GitHub UI; say so rather than working around it.
- **Never pass `--delete-branch` to a local `gh pr merge`**: it attempts a local
  `git checkout main`, which fails when `main` is held by another worktree.
  The server deletes the branch on merge; if it ever does not, use
  `git push origin --delete <branch>`.
- If `gh` reports the PR as not mergeable (conflicts), fix the branch and push
  again — auto-merge stays armed across pushes. Do not force.

## 6. After the merge

**First confirm the PR actually merged** — watch with `gh pr checks <number>
--watch`, then `gh pr view <number> --json state` must read `MERGED`. Auto-merge
lands it a short while **after** the checks go green; green CI alone is not
"merged", and armed auto-merge is not merged either. Everything below assumes `origin/main` has already advanced, so
doing it before the merge lands is a no-op at best.

`origin/main` has moved but local `main` has not — and `worktree.sh add` forks
from local HEAD, so a stale `main` silently seeds stale branches:

```bash
devtools/worktree.sh sync
```

It finds whichever checkout has `main` on it and fast-forwards it there, so it
runs from any worktree; it refuses (`--ff-only`) rather than merging if that
`main` has diverged. Use it (or `git -C <main> merge --ff-only origin/main`) —
**not** `git update-ref refs/heads/main origin/main`: `update-ref` moves the
branch pointer without touching the checked-out index or working tree, so the
just-merged files read as *reverted* on disk until a `git reset --hard HEAD`,
and a careless commit then un-does the merge.

If the work happened in a `.claude/worktrees/` worktree, whether to tear it
down depends on how this skill was invoked:

- **Bare `/pr`** — leave the worktree standing. The work is merged, but the
  checkout stays for follow-up commits; do not remove it unless asked.
- **`/pr full`** — reclaim it now, per the `worktree` skill's "`/pr full` —
  ship, then tear the worktree down" section: exit with
  `ExitWorktree {"action": "keep"}` if the session is inside it, then
  `git worktree remove` + `git branch -d <name>` (`-D` after a squash merge).

Otherwise — a plain topic branch on an ordinary checkout — the local branch can
go with `git branch -d <branch>` (or `-D` after a squash merge, since the SHAs
differ; the merge already confirmed the patches landed).

## When NOT to merge

Stop after step 4 and hand the PR to the user instead of merging when:

- CI is red, or the push gate was bypassed with `--no-verify`;
- the diff moves a ratchet floor or touches the correctness references in ways
  only a human should sign off on;
- the user asked for a PR but not a merge.
