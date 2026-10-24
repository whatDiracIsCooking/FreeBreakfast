---
name: milestone
description: >-
  Turn a plan — a roadmap/design .md, or the current conversation — into a
  GitHub milestone plus a DAG of issues, each scoped to a single PR, wired
  with native "blocked by" dependencies and labelled (area + wave-N) so
  independent issues can be worked in parallel.
  Use when the user asks to "make a milestone", "break this into issues",
  "plan out the work as issues", or wants a design decomposed into trackable,
  parallelizable GitHub work. Requires gh + GH_TOKEN (present in this repo's
  containers). Creating issues is outward-facing — draft and confirm first.
---

# Plan → milestone → DAG of PR-sized issues

Take a plan and turn it into one GitHub milestone whose issues each map to a
single PR, wired into a dependency DAG so the independent ones can run at once.
The hard part is not the `gh` calls — it is the decomposition. Do that well,
show it, then create.

## 1. Get the plan

The input is one of:

- **A file** — `/milestone path/to/plan.md` (a design doc, proposal, or
  roadmap). Read it in full.
- **The conversation** — `/milestone` with no path. Reconstruct the plan from
  what has been discussed and decided in this session. If the discussion is
  thin or ambiguous, say so and ask for the file rather than inventing scope.

Extract two things: the **overall goal** (becomes the milestone) and the
**discrete pieces of work** (become the issues). If the plan is one
undifferentiated wall, your job in step 2 is to cut it up; if it already has
sections/steps, use them as the starting seams but re-check their size.

## 2. Decompose into PR-sized issues

Each issue must be addressable by **one coherent, reviewable PR**. Calibrate:

- **Too big** if you cannot state the PR's diff in a sentence or two, if it
  touches unrelated subsystems, or if a reviewer would have to context-switch
  mid-review. Split it.
- **Too small** if it cannot stand as its own commit with its own test/check —
  a one-line tweak that only makes sense bundled with its neighbour. Merge it
  up.
- **Just right**: one bounded change, an independently statable acceptance
  check, a named set of files/modules it will touch. That usually means one
  module and its test, landing via the `pr` skill as a single squash-merge.

Write each issue with a consistent body so a future implementer (or subagent)
needs nothing else:

```
## Scope
<one-PR change, stated as the diff it will produce>

## Depends on
- #<n>   (or: "None — ready to start")

## Acceptance
- <the check that proves it done: a test name, a benchmark, a build>

## Files / area
<paths or package the PR will touch>
```

## 3. Wire the DAG — maximise what can run in parallel

Add an edge **A → B** only when B genuinely cannot start until A lands — B
imports A's new symbol, builds on a file A creates, or asserts against an
interface A defines. Do **not** add edges for mere preference of ordering; a
spurious edge serialises work that could have run at once.

Then compute **waves** (topological layers): wave 0 = every issue with no
unmet dependency, wave 1 = issues whose deps are all in wave 0, and so on.
Wide-and-shallow beats deep-and-narrow — if everything chains off one root,
look for a smaller shared prerequisite to split out so the rest fan out. The
wave map is what you show the user and what tells them what to start today.

## 4. Draft, then get approval — do not create yet

Creating issues is public and awkward to undo. Present the whole plan first:

- the **milestone** title + one-line description;
- a **wave table** — for each issue: proposed title, its wave, its `depends on`
  set, its labels, and a one-line scope;
- any **labels to be created** (new area labels, a `wave-N` beyond the ones
  that exist), with description and colour;
- call out the **wave-0 set** explicitly: "these N can be started in parallel
  immediately."

Get a clear go-ahead (or edits) before any `gh` write. If the user tweaks the
cut, revise the table and re-confirm.

## 5. Create the milestone

No `gh milestone` subcommand exists — use the API. Capture the number.

```bash
gh api repos/{owner}/{repo}/milestones \
  -f title="<milestone title>" \
  -f description="<goal + the wave map, in markdown>" \
  --jq '.number'
```

Put the wave map in the description (markdown renders on the milestone page) so
the DAG is visible without opening every issue.

## 6. Make sure the labels exist

Every issue gets labels, and `gh issue create --label` fails on one that does
not exist — so create the missing ones first. Two kinds:

- **`wave-N`** — exactly one per issue, its wave from step 3. `wave-0` …
  `wave-4` already exist; add a higher one only when the DAG is that deep,
  following the same pattern:

  ```bash
  gh label create wave-5 --description "Milestone wave 5" --color "<hex>"
  ```

- **Area labels** — one or more per issue, naming the subsystem it touches
  (`scf`, `pipeline`, `ci`, `coulomb`, `exchange`, `density-fitting`, …, plus
  the stock `bug` / `enhancement` / `documentation` where they fit). Reuse
  what exists — `gh label list --limit 200` — before inventing a near
  duplicate. A new subsystem gets a new label with a one-line description
  naming its module, e.g. `--description "The CI module (src/ci)"`.

Create only the labels the approved draft listed.

## 7. Create the issues in topological order

Create dependencies **before** dependents, so every blocker already has a real
number when its dependent is created. Each issue carries three things that
encode the graph:

- the body's `## Depends on` lines — the **source of truth**, which
  `milestone-run` reads and which survives any tooling;
- native **"blocked by"** links (`--blocked-by`), which GitHub shows in the
  issue sidebar as *Blocked by* / *Blocking* and on project boards — set them
  on every issue with dependencies, matching the body exactly;
- its **labels** (`wave-N` + area).

```bash
gh issue create \
  --title "<title>" \
  --milestone "<milestone title>" \
  --label "wave-1,scf" \
  --blocked-by 12,14 \
  --body "$(cat <<'EOF'
## Scope
...
## Depends on
- #12
- #14
## Acceptance
...
## Files / area
...
EOF
)"
```

Keep a title→number map as you go so later issues can reference earlier ones.
Notes:

- `--milestone` takes the milestone **title**, which must match what step 5
  created. It is `-m, --milestone name` — a bare number is not accepted.
- A wave-0 issue has no `--blocked-by` and `None — ready to start` in its
  body.
- `--blocked-by` takes issue numbers or URLs, comma-separated. It needs a
  recent gh: check that `gh issue create --help` lists it (the container's
  gh does; a distro gh such as Ubuntu 24.04's 2.45 does not). Without it, add
  each link after creation through the REST API. That endpoint wants the
  blocker's **database id**, not its number:

  ```bash
  id=$(gh api repos/{owner}/{repo}/issues/<blocker> --jq .id)
  gh api -X POST repos/{owner}/{repo}/issues/<n>/dependencies/blocked_by \
    -F issue_id="$id"
  ```

- A forgotten or wrong link is fixed in place:
  `gh issue edit <n> --add-blocked-by 12` / `--remove-blocked-by 12`.
- `--parent <n>` makes a **sub-issue** — a parent/child tree, not a
  dependency. Use it only when the relationship really is containment, never
  in place of `--blocked-by`.

Verify the graph before reporting. For every issue, the native list must equal
the body's `Depends on` set:

```bash
gh api repos/{owner}/{repo}/issues/<n>/dependencies/blocked_by --jq '[.[].number]'
```

Fix any mismatch with `gh issue edit`.

## 8. Report

Finish with the milestone URL and a short wave summary: which issues (by number
now) are in each wave, and the explicit **"ready now"** list — the wave-0
issues with no blocker. That is the answer to "what can we work on
simultaneously."

## Undo, if it came out wrong

Issue creation is reversible only by hand, so this is the escape hatch:

```bash
gh issue delete <number> --yes                                   # per issue
gh api -X DELETE repos/{owner}/{repo}/milestones/<number>        # the milestone
```

Deleting the milestone does **not** delete its issues — they just detach.
Remove the issues first if the whole plan is being scrapped. Deleting an issue
drops its blocked-by links with it. Labels from step 6 stay behind; delete a
new one (`gh label delete <name> --yes`) only if nothing else uses it.
