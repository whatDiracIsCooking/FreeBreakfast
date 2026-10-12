---
name: devbox
description: >-
  Drive this repo's containers and its container-backed sibling worktrees —
  devtools/devcontainer.sh (up/rebuild/shell/test/down) and
  devtools/worktree.sh (add/rm/sync/gc/list). Use when the user wants to bring
  a container up, get a shell or run the suite inside it, rebuild after editing
  a Dockerfile or devcontainer.json, create or tear down a full container-backed
  worktree, or clean up leaked docker state. NOT for the lightweight
  .claude/worktrees/ checkouts — those are the `worktree` skill.
---

# Containers and container-backed worktrees

Two scripts, one model: **a container is keyed on its workspace folder path**,
so every checkout gets its own container, its own named volumes and its own
build image. That is what makes several worktrees usable at once, and it is
also why removal needs `worktree.sh rm` rather than `git worktree remove`.

Both scripts resolve their workspace from their own location, so they act on
*the checkout they live in*, from any cwd. `devtools/devcontainer.sh up` run
from worktree `foo` brings up `foo`'s container, never main's.

## The container

```bash
devtools/devcontainer.sh up        # reuse if it exists, else create
devtools/devcontainer.sh rebuild   # recreate from scratch
devtools/devcontainer.sh shell     # bash inside
devtools/devcontainer.sh test      # the fast tier inside (see the `test` skill)
devtools/devcontainer.sh down      # stop and remove the container
```

- **Use `rebuild`, never `up`, after editing `devcontainer.json` or a
  `Dockerfile`.** `up` reuses the running container, applies none of the change,
  and reports success with the same container id — the single most expensive
  mistake here, because everything downstream looks like a code bug. The
  workspace is a bind mount and the caches are named volumes, so both survive a
  rebuild; anything written to the container's own filesystem does not.
- `down` removes the container but **keeps the volumes** (the pytest tmp tree,
  the agent's plugin state). That is deliberate: `up` again is cheap and
  nothing is lost. Reclaiming the volumes is `worktree.sh rm`'s job.
- `shell -c "<cmd>"` runs one command inside — how you inspect a container
  without an interactive session, and how the C++ tier is run:
  `devtools/devcontainer.sh shell -c devtools/cpp-tier.sh`.

### Three images, one per GPU backend

`DEVCONTAINER_CONFIG` in `devtools/config.sh` picks which `devcontainer.json`
every subcommand drives. A relative path resolves against the repo root, so it
works from any cwd and any worktree.

There are **three** configs, one per backend, each on its own Dockerfile:

| Variant | Config | Image | What it has |
| --- | --- | --- | --- |
| `cuda` | `.devcontainer/cuda/` | `docker/Dockerfile.cuda` | clang-20 + libc++, CMake 4.2, Ninja, CUDA 13, sccache, the `/opt` deps. **The default.** |
| `hip` | `.devcontainer/hip/` | `docker/Dockerfile.hip` | The same toolchain with ROCm and no CUDA. Pins `CMAKE_PRESET`/`CTEST_PRESET` to `hip` in `containerEnv`, so `cpp-tier.sh` needs no flag inside. |
| `combined` | `.devcontainer/combined/` | `docker/Dockerfile.combined` | Both SDKs, ~40GB. Stays on the `default` (CUDA) preset; `--preset hip` for the other. |

The variant flag goes **before** the command:

```bash
devtools/devcontainer.sh --hip rebuild
devtools/devcontainer.sh --combined shell
devtools/devcontainer.sh --cuda up        # the default; the flag is optional
```

For a whole session, export `DEVCONTAINER_CONFIG` instead; precedence is flag,
then variable, then `config.sh`. Containers are keyed on the workspace folder
**and** this path, so the variants of one worktree are separate containers and
can be up at once — but they share that worktree's volumes, so run the suite in
one variant at a time.

The CUDA image wants an NVIDIA GPU and the container toolkit for `--gpus all`,
but it starts and builds without either — only tests needing a live device are
lost. The `hip` and `combined` variants additionally pass `--device=/dev/kfd`
and `--device=/dev/dri` with `--group-add` for the host's `render` and `video`
gids, which `devcontainer.sh` resolves numerically (`host_gid` in `lib.sh`).
**That resolution matters:** `--group-add render` would resolve the name
*inside* the container, where the gid differs or the group is absent, and the
run then fails at the first device call with `hipErrorNoDevice (no ROCm-capable
device is detected)` — which reads exactly like a box with no GPU.

**The parent image must exist first.** The three Dockerfiles chain by tag
(`FROM ${PARENT_IMAGE}`), so each config carries an `initializeCommand` that
runs `docker/build.sh` for its parent before the CLI builds the leaf. Building
a leaf by hand with no parent tagged does not fall back to building one — it
tries to *pull*, and fails with `pull access denied for free-breakfast`.

`doctor.sh` prints which config is active; a path that does not exist is a FAIL,
and every container command refuses until it is fixed.

### Bounding a container's CPU

`CPUSET` (host cores) and `CPUS` (a core-count quota) come from
`devtools/config.sh`, or from the environment for one call, and are applied by
`up` and `rebuild` with `docker update` — live, no restart:

```bash
CPUSET=0-11 devtools/devcontainer.sh up      # pin to host cores 0-11
CPUS=8      devtools/devcontainer.sh up      # cap at 8 cores' worth
CPUSET=0-11 devtools/devcontainer.sh up      # re-pin one already running
```

This is **not** `JOBS`, and not `BUILD_JOBS`. Those are two runners' worker
counts — pytest's and `cmake --build`'s — and neither reaches anything else, so
a compile started from a `shell` by hand still takes the whole box.
`CPUSET`/`CPUS` bound everything in the container.

`BUILD_JOBS` deserves its own note: a C++23 module build is memory-hungry per
job (the scanner plus a BMI cache), so the right value is usually *lower* than
the core count and lower than what you would give pytest. Empty lets Ninja pick
cores + 2, which is what OOM-kills a 16-core box on a module-heavy tree — and an
OOM-killed compiler reads as a mysterious `ninja: build stopped`, not as a
memory problem.

Reach for it when two worktree containers are up at once: pinned to disjoint
cores (`CPUSET=0-11` in one, `CPUSET=12-23` in the other) they physically
cannot contend, which is the only way a timing measurement in one of them means
anything. Limits belong to the **container**, and a container is per workspace
folder — so pin per worktree, not per shell; two `shell`s from the same
checkout share one container and one set of limits.

Two things to know:

- `docker update` changes only what it is passed, so a second call with just
  `CPUSET` leaves an earlier `CPUS` quota in place, and running `up` with both
  unset **clears nothing** — it only stops applying. `rebuild` is how you get
  an unbounded container back; a fresh one starts with no limits.
- If nothing is running, the limits are skipped with a `cpu-limit: no running
  container …` line on stderr rather than an error — `up` had already done its
  work.

### Killing a container test run on the host does not kill it inside

`npx` dies; the pytest master and its xdist workers keep running at 100% CPU,
and nothing says so — the next run just comes out slower and you measure *that*.
After interrupting:

```bash
devtools/devcontainer.sh shell -c "ps -eo pid,pcpu,cmd --sort=-pcpu | head"
```

`pkill -f pytest` reaches only the master (a worker's command line is a bare
`python3 -u -c import sys;...`), so kill the workers **by PID**.

## Container-backed worktrees

Sibling directories under the repo root, named after their branch —
`<root>/main`, `<root>/<name>` — each with a container, three named volumes and
a build image.

```bash
devtools/worktree.sh list
devtools/worktree.sh add <name> [start-point] [--up]
devtools/worktree.sh rm  <name> [--force]
devtools/worktree.sh sync
devtools/worktree.sh gc [--dry-run] [--yes]
```

**`add`** creates or attaches branch `<name>` at `<root>/<name>`; `--up` also
brings the container up and drops you into a shell. It forks from the calling
checkout's **local HEAD**, and warns (offline, from the last fetch) when that
HEAD is behind its upstream — heed it, or the new branch starts on stale code.
It refuses the name `main`.

If it fails with *"$ROOT is not writable"*: the repo root, not the checkout,
needs to be writable. Fix once with the **numeric** uid —

```bash
sudo chown $(id -u) <root>
```

— because the host user and the container's `ubuntu` are typically both uid
1000 under different names, so `chown ubuntu` fails on the host.

**`rm`** is the one that matters. `git worktree remove` knows nothing about the
container, the three `<PROJECT_NAME>-{pytest-tmp,claude-plugins,claude-backups}-<name>`
volumes or the `vsc-<name>-<64hex>` image; `rm` tears down all of it, then
removes the worktree and deletes the branch — with `-d`, so an unmerged branch
is left in place and reported rather than lost. It refuses to remove the
checkout you are calling it from: run it from another one (typically main).

**`sync`** fast-forwards `main` to `origin/main`. Nothing else does this: after
a PR merges `origin/main` moves and local `main` does not, and `add` forks from
local HEAD. It finds whichever checkout has `main` on it (so it works from any
worktree, including a `.claude/worktrees/` one) and is `--ff-only` — a diverged
`main` is refused rather than turned into a merge commit.

**`gc`** reconciles docker against the worktrees that actually exist, for
everything an earlier plain `git worktree remove` leaked. Always show
`--dry-run` first; it prints a `orphans: N (…)` line and deletes nothing.
It is scoped to this repo — volumes by `PROJECT_NAME` prefix, images by the
`devcontainer.project` label, containers by folders under this repo's root or
its `.claude/worktrees/` — so it is safe beside other docker workloads. An
image with no label is never touched.

`doctor.sh` runs `gc --dry-run` for its "docker hygiene" line, which is how
leaked state surfaces without anyone going looking.

## The other front end: docker compose

`docker/compose.yaml` runs **the same `Dockerfile.cuda` image** as one-shot
batch jobs — configure, build, test, exit — teeing everything to `.log/`. It is
not layered on the devcontainer and neither is deprecated:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)     # required; compose.yaml uses :? not a default
alias dc='docker compose -f docker/compose.yaml run --rm'
dc build ; dc test ; dc asan ; dc compute-sanitizer
```

Use compose when you want to read a result and throw the container away — CI,
nightlies, a sanitizer sweep, a clean reproducible run. Use the devcontainer
when you want to *stay inside*: iterating, agents, debugging. Compose services
are global (not per worktree) and honour no `CPUSET`/`CPUS`, so two concurrent
compose runs will contend. `docker/README.md` has the full variable list.

The images are built from the **repo root**, not from `docker/` — their `COPY`
and bind-mount paths are spelled relative to it — and `docker/build.sh` is what
walks the tag chain in order:

```bash
docker/build.sh cuda        # base, then cuda -> <project>:cuda and :latest
docker/build.sh hip
docker/build.sh combined    # base, cuda, then combined
```

## When NOT to use this skill

For a quick isolated checkout, a spike, or agent scratch space, use the
lightweight `.claude/worktrees/<name>` layout and the **`worktree`** skill — no
container, no volumes, no image, and cheap to throw away. Reach for the
container-backed ones only when the work needs the image: the full suite, a
toolchain that exists only inside, or a GPU.
