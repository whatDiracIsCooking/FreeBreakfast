# Docker — the batch path

This directory is **one of two front ends onto the same image**
(`./Dockerfile.cuda`). Which one you want depends on what you are doing:

| | `docker/compose.yaml` (here) | `.devcontainer/cuda/` + `devtools/` |
|---|---|---|
| Shape | one-shot: configure, build, test, exit | long-lived container you stay in |
| Logs | tees everything to `.log/` | on your terminal; `cpp-tier.sh` also logs |
| Good for | CI, nightlies, sanitizer sweeps, a clean reproducible run | iterating, agents, debugging |
| Per worktree? | no — services are global | yes, container + volumes keyed on the workspace path |
| CPU bounds | none | `CPUSET`/`CPUS` in `devtools/config.sh` |
| Entry point | `dc test` | `devtools/devcontainer.sh shell -c devtools/cpp-tier.sh` |

They are not layered on each other and neither is deprecated. The rule of
thumb: if you want to read the result and throw the container away, use
compose; if you want to stay inside, use the devcontainer.

Every workflow here runs through `docker/compose.yaml`. All services mount the
repo root at `/workspace` and run as your host UID/GID, so nothing lands
root-owned.

Because the `user:` field is required, always export these two first:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
```

A shell alias keeps the rest short:

```bash
alias dc='docker compose -f docker/compose.yaml run --rm'
```

## Configuration

Compose reads variables from your shell and, if the file exists, from
`docker/.env` — auto-loaded because it sits beside `compose.yaml`. That file is
gitignored and there is deliberately **no committed template** for it: every
variable except `HOST_UID`/`HOST_GID` already carries a default in
`compose.yaml` (`${VAR:-default}`), and those two are written `${VAR:?...}`
*without* one so a forgotten export fails loudly. A committed `.env` pinning
`HOST_UID=1000` would satisfy that check and then quietly write build output
owned by uid 1000 on any host where you are not uid 1000 — precisely the failure
the `:?` exists to prevent.

If you want one anyway, write only the part that is machine-specific:

```bash
printf 'HOST_UID=%s\nHOST_GID=%s\n' "$(id -u)" "$(id -g)" > docker/.env
```

The tables below list every variable the services read.

## Build the Image

Four files beside this one, in a diamond:

```
                Dockerfile.base          the vendor-neutral C++23 toolchain
                 /           \
  Dockerfile.cuda             Dockerfile.hip
            |                       :
  Dockerfile.combined ..............:  (reuses the HIP install scripts, not the image)
```

`Dockerfile.cuda` is what these services and
`.devcontainer/cuda/devcontainer.json` both build, so the devcontainer and
compose cannot drift onto two different toolchains.

**They chain by tag, not by stage.** A child opens with
`FROM ${PARENT_IMAGE}`, so its parent has to be built and tagged first —
`./build.sh` is what walks the chain, and is the way to build any of them:

```bash
docker/build.sh cuda        # base, then cuda -> free-breakfast:cuda and free-breakfast:latest
docker/build.sh hip         # base, then hip  -> free-breakfast:hip
docker/build.sh combined    # base, cuda, then combined -> free-breakfast:combined
docker/build.sh base        # just the toolchain -> free-breakfast:base
```

It takes the tag prefix from `PROJECT_NAME` in `devtools/config.sh`, so
`free-breakfast:latest` — what `FREE_BREAKFAST_IMAGE` below defaults to — is always one of the
two tags the CUDA image gets. Run it from anywhere; the context is always the
repo root, because the files read `pyproject.toml`, `uv.lock` and the
`docker/install-*.sh` scripts relative to it.

Anything after the target is passed through to every `docker build` in the
chain, and any build arg set in the environment is forwarded to the file that
declares it — so widening the CUDA architecture for a mixed fleet is:

```bash
CUDA_ARCH="80;86;90" docker/build.sh cuda
docker/build.sh cuda --no-cache --progress=plain   # flags reach every step
```

`BUILD_DRY_RUN=1` prints the `docker build` commands without running them.

By hand is still fine, as long as you build the parent yourself first:

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.base -t free-breakfast:base .
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.cuda -t free-breakfast:latest .
```

Skip that first line and docker does not fall back to building the parent — it
tries to *pull* it, and fails with `pull access denied for free_breakfast, repository
does not exist`, which reads like a registry problem rather than a missing local
build. `docker/Dockerfile.base`'s header has the rest of the reasoning.

### Claude Code is pinned, and bumped on purpose

Each GPU file ends with `docker/install-claude-code.sh`: Node 22 from
nodesource, then `npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}`,
then a `test` that what landed is what was asked for — a typo'd dist-tag fails
the build instead of quietly installing something else.

```bash
devtools/claude-version.sh              # pinned / registry / installed here
devtools/claude-version.sh --apply      # rewrite the pin to the registry's latest
devtools/claude-version.sh --apply 2.1.300
devtools/claude-version.sh --rebuild    # ...and rebuild the images already built here
```

`ARG CLAUDE_CODE_VERSION` is declared in **all three** GPU files — the third
version to bump in more than one place, after `ROCM_VERSION` and `GPU_TARGETS`
— which is why `devtools/claude-version.sh` exists, and why `doctor.sh` warns
when the three disagree or have fallen behind the registry.

**It is deliberately not in `Dockerfile.base`.** A bump there invalidates the
CUDA toolkit and the ~30GB ROCm install below it, plus the three `/opt`
dependencies `base` compiles; at the leaves it is one ~10s layer.

This replaced the `claude-code` devcontainer feature, whose `install.sh`
npm-installs an **unpinned** `@anthropic-ai/claude-code` into a layer that is a
permanent cache hit. `devcontainer-lock.json` pins that feature's digest, not
the version it installs: measured on this box before the change, four images
off one lockfile carried claude 2.1.197 (×3, built over three days) and 2.1.289
— the version was a function of build date, and nothing reported it.

### The Antigravity CLI is pinned too — by *release*, not version

Each GPU file also ends with `docker/install-agy.sh`, which installs Google's
Antigravity CLI as `/usr/local/bin/agy`: download the pinned tarball, verify its
sha512, extract the one binary, then check `agy --version` and that `agy` on
`PATH` is the copy just installed. A 404 or a renamed archive member fails the
build rather than producing a container with no `agy` in it.

```bash
devtools/agy-version.sh              # pinned / upstream / installed here
devtools/agy-version.sh --apply      # rewrite the pin to upstream's current release
devtools/agy-version.sh --apply 1.3.3-5524738307653632
devtools/agy-version.sh --rebuild    # ...and rebuild the images already built here
```

**The pin is a release id — `1.3.3-5524738307653632` — and two lines, not one.**
Upstream's updater endpoint serves only the *current* release, so a version on
its own is not addressable the way `claude-code@2.1.296` is; what is addressable
is the storage URL, whose path carries the version plus an opaque build id. So
`ARG AGY_RELEASE` pins that segment and `ARG AGY_SHA512` pins the tarball it
names, `install-agy.sh` verifies the hash, and `agy-version.sh --apply` rewrites
both in all three files. The build then needs no manifest at all: an upstream
release neither changes what the image installs nor breaks the build.

Two consequences worth knowing:

- **Rolling back needs a release id you already have** — a previous value of the
  pin, which is in git. Upstream publishes no index of older releases.
  `--apply <release>` downloads that tarball to hash it (~60MB), because the
  manifest carries a hash for the current release only.
- **`ENV AGY_CLI_DISABLE_AUTO_UPDATE=1` is load-bearing.** `agy` self-updates in
  the background during normal runs, which would walk the container away from
  its pin; the binary is root-owned so the attempt fails anyway, but failing
  quietly on every run is not the same as not being attempted. This is the same
  decision as `autoUpdates: false` in `.devcontainer/seed-claude-config.sh`.
  Bump with `devtools/agy-version.sh`, not `agy update`.

`Dockerfile.combined` runs the script on top of the `cuda` image, which already
ran it, and the binary is 214MB — so `install-agy.sh` **inherits** a copy that is
already at the pinned release instead of laying down a second 214MB layer. The
early exit is keyed on the version matching, so a `combined` pinned to something
other than its parent still gets its own.

`agy`'s own config lives in `$HOME/.gemini`, which no image provides. The
devcontainer binds the host's copy — `HOST_MOUNTS` in `devtools/config.sh`, see
`.devcontainer/README.md`.

### The `-ci` variants, and the registry

Three more environment knobs turn a local build into a published one. Only
`.github/workflows/images.yml` sets them, and only to publish the two images
CI pulls:

```bash
IMAGE_TAG_SUFFIX=-ci IMAGE_REGISTRY=ghcr.io/<owner> BUILD_PUSH=1 \
  ROCM_PRUNE=1 docker/build.sh hip      # -> free-breakfast:hip-ci, pushed to GHCR
```

| knob | effect |
|---|---|
| `IMAGE_TAG_SUFFIX` | appended to every tag in the chain (`free-breakfast:hip-ci`) |
| `IMAGE_REGISTRY` | adds `<registry>/free_breakfast:<tag>` as a second tag |
| `BUILD_PUSH=1` | pushes the **final target's** registry tags, not its parents' |

`:latest` is dropped when a suffix is set — `free-breakfast:latest-ci` would be a lie,
since `latest` is what `FREE_BREAKFAST_IMAGE` resolves to and must keep meaning the
full CUDA dev image. Parents are not pushed because a child image is
self-contained; publishing `:base` too would upload 1.45GB nothing pulls.

**`ROCM_PRUNE=1` drops the bulk of ROCm that only a running kernel needs** —
Tensile/rocFFT kernel objects, composable-kernel archives and rocalution, none
of which a *compile* links. `docker/install-rocm.sh` carries the measured list
and the reason each entry is safe.

Note this project sets `WWR_WITH_CCL`/`WWR_WITH_COMP`/`WWR_WITH_TENSOR` **OFF**
in `deps/CMakeLists.txt`, so RCCL, hipCOMP and hipTENSOR are not required at
configure time and are not installed here. Turn one of those options on and the
matching library has to be added to this image in the same change — and
re-checked against the prune list, since pruning a library `find_package` now
needs would fail at configure.

It is an **optimisation**, worth ~13GB less to pull on every CI run and a
3-minute push instead of many. It is not what makes a HIP job possible: a
GitHub-hosted runner has a 145GB root with 86GB free before any cleanup, so the
unpruned image fits too. (This paragraph previously claimed otherwise, on an
estimate the first real run disproved.)

It has to happen inside that script's own `RUN`. Layers are additive, so an
`rm` in a later layer hides the files and frees nothing; `:hip` and `:hip-ci`
are therefore two full installs rather than a shared layer plus a delta.

## Build the Project

```bash
dc build
```

### Build Options

Pass environment variables before the command:

```bash
# Debug build
BUILD_PRESET=debug dc build

# Clean rebuild (recompile everything)
CLEAN=1 dc build

# Reconfigure only (e.g. after changing CMake options)
RECONFIGURE=1 dc build

# Full rebuild from scratch
REBUILD=1 dc build

# Skip configure, just recompile after code edits
BUILD_ONLY=1 dc build
```

| Env Var | Effect |
|---------|--------|
| `FREE_BREAKFAST_IMAGE` | Docker image to use (default: `free-breakfast:latest`) |
| `BUILD_PRESET` | Select cmake preset: `default` (Release), `debug`, `asan`, `compute-sanitizer` (default: `default`) |
| `CLEAN=1` | Remove compiled objects before building |
| `RECONFIGURE=1` | Wipe cmake cache and reconfigure from scratch |
| `REBUILD=1` | Both clean and reconfigure (nuclear) |
| `BUILD_ONLY=1` | Skip configure, just build |
| `FREE_BREAKFAST_SCCACHE_HOST_DIR` | Host path of the shared sccache store, bound to `/sccache` in every service (default: this checkout's own `../.sccache`). From a worktree, point it at main's to share **hits** as well as the store — every service mounts the tree at `/workspace`, so the keys match whichever worktree you ran from. Must already exist; `create_host_path: false` makes compose say so rather than let docker create it as root. See `docs/architecture.md` §7. |

## Run Tests

The `test` service configures, builds, then runs gtest followed by pytest.

The pytest half runs the whole suite — today one file,
`.claude/hooks/test_protect_main.py` — the same thing the push gate and
`devcontainer.sh test` run. There is no fast/slow split: the Python side is
checker scripts, not numerics. The `compute-sanitizer` service still deselects
`no_sanitizer`, because instrumented runs turn a merely-slow case into an
hours-long one.

The gtest half builds the whole tree, **WarpWraps included** — it is a submodule
built from source, so a first run in a cold container pays for the dependency's
module surface as well as this project's. `dc build` once before `dc test` if
you want those two costs reported separately.

```bash
# All tests (gtest + pytest)
dc test

# Gtest only
SKIP_PYTEST=1 dc test

# Gtest with filter (a ctest -R regex over gtest SUITE names)
SKIP_PYTEST=1 TEST_FILTER='Getrf' dc test

# Pytest only
SKIP_GTEST=1 dc test

# Pytest with filter
SKIP_GTEST=1 PYTEST_ARGS='-k smoke' dc test
```

| Env Var | Effect |
|---------|--------|
| `TEST_FILTER` | Regex passed to `ctest -R` (NOT `--gtest_filter`: compose drives ctest, which registers one entry per suite) |
| `PYTEST_ARGS` | Extra arguments appended to the pytest invocation |
| `SKIP_GTEST=1` | Skip the C++ suite |
| `SKIP_PYTEST=1` | Skip the Python suite |
| `TIMEOUT_MULTIPLIER` | Integer scale on test timeouts, forwarded as `-DFREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER` (unset: the preset's value — `1`, or `10` for `compute-sanitizer`) |
| `CUDA_VISIBLE_DEVICES` | Which GPU to run on. Passed through from your shell with no default — unset means every visible device. |

### pytest cannot run from a `.claude/worktrees/` checkout

Every service mounts `..:/workspace` and nothing else. In one of the
lightweight worktrees the repo's `.git` is a *file* pointing at
`<repo-root>/main/.git/worktrees/<name>`, which is outside that mount, so
`git rev-parse --git-common-dir` fails inside the container and
`.claude/hooks/test_protect_main.py` raises during collection — which takes
the whole pytest run with it, every other Python test included. The gtest half
is unaffected, and `dc test` exits 1.

Run the C++ half here and the Python half on the host, where git resolves:

```bash
SKIP_PYTEST=1 dc test     # in the worktree
uv run pytest -n auto -rs # on the host
```

No volume in `compose.yaml` fixes this: the path that would have to be mounted
is absolute and host-specific, so it cannot be committed. The
`.devcontainer/*.json` files do carry exactly such a path, hand-written per
machine, which is why the devcontainer has no such problem. Running compose
from the primary checkout also has none — there `.git` is a real directory
inside the mount.

## Run Tests under AddressSanitizer

```bash
dc asan
```

Uses the `asan` preset, built into `build-compose-asan/`. Accepts the same `TEST_FILTER`,
`SKIP_GTEST`, and `SKIP_PYTEST` variables.

## Run Tests under Compute Sanitizer

```bash
dc compute-sanitizer

# Select a tool other than memcheck
COMPUTE_SANITIZER_TOOL=racecheck dc compute-sanitizer
```

Uses the `compute-sanitizer` preset, built into `build-compose-compute-sanitizer/`.
The preset itself wraps each ctest entry in the tool (`CMAKE_TEST_LAUNCHER`), so
the service passes `COMPUTE_SANITIZER_TOOL` through at configure time and runs
plain ctest; only the pytest half is wrapped by the service. CUDA only.
`racecheck` and `synccheck` check only free_breakfast's own kernels (`--kernel-name
kns=free_breakfast`); `memcheck` and `initcheck` check every kernel, vendor ones included.

Every suppression, narrowed scope and known coverage gap of both sanitizer
services — that kernel filter, `ASAN_OPTIONS=protect_shadow_gap=0`, the timeout
multiplier, the `no_sanitizer` exclusion — is registered in
[`docs/sanitizers.md`](../docs/sanitizers.md).

## Interactive Shell

```bash
dc build bash
```

## Logs

All output is logged to `.log/` with timestamps:

```
.log/configure.{ts}.out.txt / .log/configure.{ts}.err.txt
.log/build.{ts}.out.txt     / .log/build.{ts}.err.txt
.log/gtest.{ts}.out.txt     / .log/gtest.{ts}.err.txt
.log/pytest.{ts}.out.txt    / .log/pytest.{ts}.err.txt
```

## Requirements

- Docker with BuildKit support
- NVIDIA Container Toolkit (for GPU access)
