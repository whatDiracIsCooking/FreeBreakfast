#!/usr/bin/env bash
# Per-project settings for everything in devtools/. EDIT THIS FILE, not the
# scripts -- the scripts are meant to survive being copied into the next repo
# unchanged, and this is the one place that knows what the project is called
# and how its tests run.
#
# Sourced (never executed) by devcontainer.sh, worktree.sh, doctor.sh,
# prepush-tests.sh and slow-tier.sh. Every value uses ${VAR:-default} so an
# environment variable still wins for a one-off:
#
#   JOBS=16 devtools/prepush-tests.sh
#
# The one thing this file CANNOT reach is the .devcontainer/<variant>/
# devcontainer.json files --
# JSON cannot source shell. The volume names, the container name and the .git
# bind path are spelled out there too; PROJECT_NAME below must match the
# `<name>-pytest-tmp-` style prefixes used there, or `worktree.sh rm` and
# `worktree.sh gc` will not recognise this project's volumes as its own.
# doctor.sh checks that the two agree and warns when they have drifted.
#
# The C++ side has a second config file with the same job: CMakePresets.json
# holds the build settings (compiler, build type, CUDA architecture, where the
# prebuilt dependencies are). What lives HERE is only which preset the scripts
# should drive -- see CMAKE_PRESET / CTEST_PRESET below.

# --- identity -------------------------------------------------------------

# Prefix for this project's docker volumes and per-worktree build images, and
# the name of the devcontainer. Must match devcontainer.json (see above).
# It ends up in docker resource names, so keep it lowercase.
PROJECT_NAME=${PROJECT_NAME:-free-breakfast}

# Which devcontainer.json devcontainer.sh drives. Relative paths resolve
# against the repo root, so this works from any cwd and from any worktree.
# There is one per GPU file of docker/ (Dockerfile.cuda / .hip / .combined):
#
#   .devcontainer/cuda/devcontainer.json     THE development container:
#                                            clang-20 + libc++, CMake 4.2,
#                                            CUDA 13. Needs an NVIDIA GPU and
#                                            nvidia-container-toolkit.
#   .devcontainer/hip/devcontainer.json      the same toolchain with ROCm and
#                                            no CUDA at all. Needs an AMD card
#                                            and the amdgpu kernel driver. It
#                                            pins CMAKE_PRESET/CTEST_PRESET to
#                                            `hip` in containerEnv, so
#                                            cpp-tier.sh needs no flag in there.
#   .devcontainer/combined/devcontainer.json both SDKs, ~40GB. For working on
#                                            both backends in one shell; `auto`
#                                            (below) picks the less-loaded
#                                            card, `--preset default|hip` pins.
#
# This is only the FALLBACK. For one call, use the variant flag instead --
# `devtools/devcontainer.sh --hip up` -- and for a whole shell session, export
# the variable. Precedence is flag, then variable, then this.
#
# Points at the CUDA variant because that is where the project builds by
# default. Each worktree sources its OWN copy of this file (lib.sh resolves the
# repo root from its own location), so changing the line below switches the
# default for that checkout alone -- useful for a worktree dedicated to one
# backend, as long as you do not commit it.
#
# Containers are keyed on the workspace folder AND this path, so the variants
# of one worktree are separate containers and can be up at once. They share
# that worktree's volumes, so run the suite in one at a time.
DEVCONTAINER_CONFIG=${DEVCONTAINER_CONFIG:-.devcontainer/cuda/devcontainer.json}

# --- C++ build (CMake) ------------------------------------------------------

# Which CMakePresets.json presets devtools/cpp-tier.sh drives. The presets
# themselves -- compiler, build type, CUDA arch, dependency prefixes -- live in
# CMakePresets.json; only the CHOICE lives here.
#
#   default     Release, CUDA backend, in-container /opt paths,
#               CUDA_ARCHITECTURES=native
#   workstation the same on a bare host (build-workstation/, nothing prebuilt)
#   debug / asan / compute-sanitizer / coverage / experimental
#   hip         the ROCm backend (build-hip/); hip-asan is its asan twin
#   ci-cuda     what .github/workflows/ci.yml builds: the full CUDA tree with
#               the architecture PINNED, and a test preset that excludes the
#               `gpu` label. Runnable here too, and worth it to reproduce a CI
#               failure exactly.
#   ci-hip      the HIP half, deliberately the SAME SHAPE: the whole tree
#               through clang's -x hip front end, then the same -LE gpu.
#               Compiling the sources is the point -- clang is the stricter
#               front end. Neither CI preset needs a device.
#
# Read either CI preset as compile-and-link plus whatever CPU-only tests exist,
# NOT as a test of GPU behaviour: `-LE gpu` is what holds the device cases back.
#
# THERE ARE NO PER-ARCHITECTURE PRESETS. `base` uses native and ci-cuda pins 86.
# Target something else with a -D and a build dir of its own:
#
#   cmake --preset default -B build-h100 -DCMAKE_CUDA_ARCHITECTURES="90"
#
# It has to be -D. That overrides a preset's cacheVariables, where an
# environment variable does not: CUDAARCHS is consulted only when the cache
# variable is unset, and `base` always sets it.
#
# NOTHING MAY GO BELOW sm_80. The WarpWraps dependency wraps bf16 tensor-core
# WMMA, an Ampere feature, and rejects a sub-80 CMAKE_CUDA_ARCHITECTURES at
# CONFIGURE time; the root CMakeLists checks it too, so the message names this
# project. CUDA 13's own nvcc floor is lower (compute_75) but is not the binding
# one.
#
# Override for one run rather than editing:
#   CMAKE_PRESET=asan CTEST_PRESET=asan devtools/cpp-tier.sh
#
# CTEST_PRESET is separate because not every configure preset has a matching
# test preset -- only default, workstation, debug and asan do. Set it empty to
# configure and build without running ctest.
#
# THE DEFAULT IS `auto`: one backend passing is usually enough, and several
# sessions run at once, so the scripts pick whichever live card is least
# loaded (gpu_auto_preset in lib.sh) and print the choice. CTEST_PRESET=auto
# follows whatever CMAKE_PRESET resolved to. Name a preset -- `--preset
# default` (CUDA), `--preset hip` -- when the backend matters, as it does for
# a change touching a .cu: then run BOTH.
CMAKE_PRESET=${CMAKE_PRESET:-auto}
CTEST_PRESET=${CTEST_PRESET:-auto}

# What `auto` chooses between, `backend:preset`, in tie-break order. A backend
# counts only where its device nodes and compiler both exist, so in the
# single-backend `cuda`/`hip` containers `auto` is simply that backend.
GPU_AUTO_PRESETS=${GPU_AUTO_PRESETS:-"cuda:default hip:hip"}
# What `auto` resolves to when no card is live (a bare host, a dropped card).
GPU_AUTO_FALLBACK=${GPU_AUTO_FALLBACK:-default}
# Presets that build the HIP backend; every other preset is CUDA. Decides
# which card a test run takes.
GPU_HIP_PRESETS=${GPU_HIP_PRESETS:-"hip hip-asan ci-hip"}

# Parallel compile jobs for `cmake --build`. Distinct from JOBS below, which is
# the pytest worker count: a C++23 module build is memory-hungry per job (the
# scanner plus a BMI cache), so the right number is usually LOWER than the core
# count, and lower still than what you would give pytest. Empty lets Ninja pick
# (cores + 2), which is what OOM-kills a 16-core box on a module-heavy tree.
BUILD_JOBS=${BUILD_JOBS:-12}

# Which preset devtools/coverage.sh configures, builds and runs ctest under to
# collect clang source-based coverage. Its own build directory (build-coverage/)
# and FREE_BREAKFAST_COVERAGE=ON live in CMakePresets.json; only the CHOICE
# lives here. The report and merged .profdata are written under that build dir,
# so they are covered by the build*/ line in .gitignore and need no cleanup of
# their own.
COVERAGE_PRESET=${COVERAGE_PRESET:-coverage}

# Source files devtools/coverage.sh drops from its llvm-cov summary, as an
# -ignore-filename-regex over the full path.
#
# deps/ is everything this project did not write: the WarpWraps submodule's
# src/ (covered by ITS repository's own suites) and GoogleTest. Without this the
# report is dominated by a dependency larger than src/, and the number stops
# meaning anything about this project.
#
# Set empty to report over everything the build compiled. Add a src/ pattern
# here only for code that is verified at COMPILE time and cannot execute -- and
# say which check covers it instead.
COVERAGE_IGNORE_REGEX=${COVERAGE_IGNORE_REGEX:-'/deps/'}

# --- cross-backend check --------------------------------------------------

# devtools/cross-backend-check.sh compiles the tree for the backend this build
# is NOT targeting. It exists because the two backends go through DIFFERENT
# FRONT ENDS -- a .cu is nvcc's problem under CUDA and clang's under HIP -- and
# nvcc is the more permissive of the two, so a whole class of error is invisible
# until someone builds the other side: a functor with a `const` member of class
# type, say, which clang rejects for a device-functor concept while nvcc waves
# it through.
#
# A build targets exactly one backend and CMAKE_PRESET above picks it, so these
# describe the OTHER one. Flip them together if this project's default backend
# ever becomes HIP.
CROSS_CHECK_BACKEND=${CROSS_CHECK_BACKEND:-HIP}
CROSS_CHECK_PRESET=${CROSS_CHECK_PRESET:-hip}
CROSS_CHECK_TOOL=${CROSS_CHECK_TOOL:-hipconfig}

# The image carrying that toolchain, used when this machine has no hipconfig.
# `docker/build.sh hip` builds and tags it -- that script derives the tag from
# PROJECT_NAME exactly as the line below does, so the two agree by construction
# rather than by convention.
CROSS_CHECK_IMAGE=${CROSS_CHECK_IMAGE:-${PROJECT_NAME}:hip}

# Its own build directory, separate from every preset's, because two presets
# sharing one directory silently reconfigure it back and forth -- a full rebuild
# each way. Matches the build*/ line in .gitignore.
CROSS_CHECK_BUILD_DIR=${CROSS_CHECK_BUILD_DIR:-build-cross-check}

# The device-kernel libraries, one per line -- what --device-only builds when
# you want the narrow check instead of the whole tree. The umbrella target
# collects every free_breakfast_add_gpu_device_library() target (the macro adds
# each one), so a new .cu needs no edit here. Overridden to empty,
# --device-only refuses rather than building nothing and reporting PASS.
CROSS_CHECK_DEVICE_TARGETS=${CROSS_CHECK_DEVICE_TARGETS-free_breakfast_device_libraries}

# --- the ROCm runtime tier -------------------------------------------------
#
# cross-backend-check.sh above proves the other backend COMPILES. On a machine
# with a real AMD card, `cpp-tier.sh --rocm` is the stronger statement: it runs
# that backend's ctest against the hardware, by re-execing cpp-tier.sh inside
# ROCM_IMAGE with the device nodes and group ids below.
#
# It exists because getting that passthrough right is fiddly and its failure
# mode is indistinguishable from having no card at all -- see ROCM_GROUPS.

# The image with the ROCm toolchain, and the preset to run in it. Same image
# cross-backend-check.sh uses; named separately so a machine can point the two
# at different builds without one silently following the other.
ROCM_IMAGE=${ROCM_IMAGE:-${PROJECT_NAME}:hip}
ROCM_PRESET=${ROCM_PRESET:-hip}

# The device nodes ROCm needs, one per line. /dev/kfd is the compute driver
# interface and /dev/dri carries the render nodes; both are required, and
# neither is passed into a container by default the way --gpus all handles
# NVIDIA.
ROCM_DEVICES=${ROCM_DEVICES:-"/dev/kfd
/dev/dri"}

# The HOST groups owning those nodes, one per line. Both container paths resolve
# these to NUMERIC gids with getent before passing them to --group-add, and that
# matters more than it looks: `--group-add render` resolves the name INSIDE the
# container, where it has a different id or does not exist, so the run then
# fails at the first device call with "hipErrorNoDevice (no ROCm-capable device
# is detected)" -- which reads exactly like a box with no GPU, and has been
# misread as one.
#
# cpp-tier.sh --rocm builds its own --group-add args from this list and fails
# loudly when a name does not resolve. devcontainer.sh instead exports one
# FREE_BREAKFAST_<NAME>_GID per entry (render -> FREE_BREAKFAST_RENDER_GID) for
# the ${localEnv:...} references in the hip/combined runArgs, and only warns:
# the json carries a fallback gid so that opening it WITHOUT this script still
# passes something. Adding a name here reaches the first automatically; the
# second also needs the reference written into the json, since JSON cannot loop.
ROCM_GROUPS=${ROCM_GROUPS:-"render
video"}

# --- tests ----------------------------------------------------------------

# How the test suite is invoked, as an array so arguments with spaces survive.
# Anything goes here -- pytest, go test, cargo test, make check -- as long as
# it takes extra arguments on the end and exits non-zero on failure. Scripts
# that need a pytest specifically (prepush, slow-tier) resolve it through
# VENV_PATHS below instead, so change both if you leave Python behind.
TEST_CMD=${TEST_CMD:-pytest}

# Default xdist worker count. 8 rather than `auto` on purpose: several worktree
# containers can be up at once, and `auto` in each would claim every core on
# the box. Raise it for a single-checkout project on a big machine. Set to
# empty to drop the -n flag entirely (a suite with no pytest-xdist).
JOBS=${JOBS:-8}

# Host CPU bounds for a devcontainer, applied by devcontainer.sh on `up` and
# `rebuild`. Both empty = unbounded, and nothing is applied.
#
#   CPUSET  pin the container to specific host cores, e.g. 0-11
#   CPUS    cap it at n cores' worth of CPU quota, e.g. 8
#
# These bound EVERYTHING the container runs -- a compiler, a linker, a build
# system, anything a shell inside it starts -- where JOBS above only bounds the
# test runner. Two worktree containers pinned to disjoint cores (CPUSET=0-11
# here, CPUSET=12-23 there) physically cannot contend, which is what makes a
# timing measurement in one of them mean anything. A container is per workspace
# folder, so these are per worktree: two `shell`s from the same checkout share
# one container and one set of limits.
#
# Left empty here by default because the right values are machine-specific --
# set them per invocation (CPUSET=0-11 devtools/devcontainer.sh up), or write
# them in here once the box this project builds on is settled.
CPUSET=${CPUSET:-}
CPUS=${CPUS:-}

# Host directories the devcontainer bind-mounts that must exist before `up`.
# One per line. A relative path resolves against the MAIN checkout -- the one
# holding the common .git -- not the calling worktree, so every worktree names
# the same directory, which is the whole point for a shared compiler cache.
#
# Why this needs a knob at all: docker does not refuse a bind whose source is
# missing, it CREATES it, owned by root. The container then runs as uid 1000,
# and the first compile dies with `failed to create directory
# /home/ubuntu/.sccache/preprocessor: Permission denied` -- a message that
# never mentions a mount -- while clearing the leftover needs sudo.
# devcontainer.sh makes these first so that cannot happen.
#
# Only list directories that are correct when EMPTY. The .git bind mount is not
# one of them: an empty .git is worse than a missing one, which is why it gets
# a loud check in devcontainer.sh rather than a mkdir.
HOST_BIND_DIRS=${HOST_BIND_DIRS:-.sccache}

# Host directories to bind into the devcontainer that are NOT written into
# devcontainer.json, one `host_path:container_path` per line. Applied by
# devcontainer.sh on `up` and `rebuild` as `--mount` arguments to the
# devcontainer CLI.
#
# WHY NOT IN THE JSON, like the sccache and .git mounts: those two are mounts
# the container cannot work without, so they are declared there and merely have
# their host path injected. These are OPTIONAL and personal -- a tool's config
# directory under your own $HOME -- and JSON has no way to say "mount this only
# if it exists". A `${localEnv:...}` entry with a placeholder fallback is worse
# than nothing here: when the variable is unset docker does not skip the mount,
# it CREATES the placeholder path as root. Keeping the list in shell means an
# empty value mounts nothing, and no host path or username is ever committed.
#
# `~` is expanded against the HOST's $HOME. A relative path resolves against
# the MAIN checkout (the one holding the common .git), not the calling
# worktree, exactly as HOST_BIND_DIRS does. A missing source directory is
# CREATED before the mount, for the same reason HOST_BIND_DIRS exists at all:
# docker would otherwise create it as root and the container, running as uid
# 1000, could not write in it.
#
# The default is the Antigravity CLI's config tree. `agy` is installed in the
# images (docker/install-agy.sh) and keeps everything in $HOME/.gemini --
# settings.json, hooks, plugins, skills, the project cache and whatever
# credential material it can store outside a keyring. Binding the host's copy
# is what makes the `agy` in a container the same logged-in `agy` as the one on
# the host, and is the counterpart of the ~/.claude mount that each
# devcontainer.json declares for Claude Code.
#
# NOTE a mount takes effect only when the container is CREATED. Editing this
# list and running `up` reuses the running container and applies none of it --
# the same trap as editing a Dockerfile. Use `rebuild`.
HOST_MOUNTS=${HOST_MOUNTS:-"~/.gemini:/home/ubuntu/.gemini"}

# Host environment variables to forward into commands run inside the container,
# space-separated. devcontainer.sh passes each one that is NON-EMPTY as a
# `--remote-env NAME=value`, on `up`/`rebuild` and on every `shell`/`test`.
#
# Deliberately not `containerEnv` in the JSON: an entry there is baked into the
# container at CREATE time, which is how GH_TOKEN ends up in the `docker run`
# line the CLI prints when a mount fails (run_cli in devcontainer.sh redacts
# that, and these names too). `--remote-env` reaches the process instead of the
# container, so a key rotated on the host is picked up by the next `shell`
# rather than at the next rebuild, and an unset variable is simply not passed
# -- where `${localEnv:FOO}` would define FOO as the empty string, which some
# tools read as "configured, but blank".
#
# The two defaults are the Antigravity CLI's API-key path: with `modelProvider`
# set to `gemini` in ~/.gemini/antigravity-cli/settings.json (mounted above),
# GEMINI_API_KEY is what it authenticates with, and GOOGLE_GEMINI_BASE_URL
# points it at a non-default endpoint. Neither is needed when `agy` is logged
# in through a Google account.
#
# ANTHROPIC_API_KEY_FREE_BREAKFAST is .claude/batch/batch.py's key, under a
# project-specific name for the opposite reason from GH_TOKEN_FREE_BREAKFAST
# below: an ANTHROPIC_API_KEY in the environment Claude Code starts in can be
# taken up by Claude Code itself, billing the API instead of the subscription.
# So it is never mapped here; batch.py reads this name itself.
HOST_ENV_PASSTHROUGH=${HOST_ENV_PASSTHROUGH:-"GEMINI_API_KEY GOOGLE_GEMINI_BASE_URL ANTHROPIC_API_KEY_FREE_BREAKFAST"}

# Cross-session GPU admission (see with_gpu_card() and gpu_auto_preset() in
# lib.sh). One run's own concurrency is GPU_TEST_JOBS below, but nothing in
# that run coordinates with a second worktree's cpp-tier.sh or cpp-smoke.sh
# running its own GPU binary against the same physical card at the same time.
#
# ONE RUN PER CARD. Two runs sharing the 10 GB RTX 3080 ran it out of memory
# (2026-10-28), even at 3 jobs each and admitted at 29% VRAM: the first run
# grew after the second joined, and each shard keeps the stack of the largest
# kernel it has launched, up to about 1.4 GB ((dd|g),
# src/integrals/eri/density_fitting/three_center/README.md). A failed
# allocation ABORTS here, it does not queue. And sharing
# bought no speed: one run's shards already keep the card busy, so two runs
# sharing it each got about half of it. So a run takes the whole card, and the
# next one waits for it.
#
# GPU_LOCK_DIR holds each card's lock files. Resolved the same way
# HOST_BIND_DIRS is: a relative path sits under the MAIN checkout, not the
# calling worktree, which is the point -- every worktree needs to see the same
# files. Defaults inside .sccache because that directory is already a bind
# mount shared by every compose service and every worktree's devcontainer
# (see docker/compose.yaml and .devcontainer/<variant>/devcontainer.json);
# reusing it costs no new mount.
GPU_LOCK_DIR=${GPU_LOCK_DIR:-.sccache/gpu-locks}
# VRAM use, in percent, at or above which a card is taken by something that
# holds no lock -- another user's job, or a binary run by hand. A run holding
# the card's lock waits until it drops below this, with no timeout: starting
# into memory that is not there only trades a wait for an abort. Compute load
# alone never makes a run wait; a foreign job that uses little memory slows it
# down but cannot crash it. 20 leaves this tree's worst case room on the 3080
# (an idle card reads about 8%); raise it on a card with more memory.
GPU_FOREIGN_VRAM_PCT=${GPU_FOREIGN_VRAM_PCT:-20}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-5}
# Concurrent GPU test processes WITHIN the run that holds a card: cpp-smoke.sh
# splits the gpu binary into this many gtest shards, cpp-tier.sh does the same
# for the GPU entries it batches and passes it to ctest as -j for the rest. 1
# is fully serial. Measured on idle cards (2026-10-25), N=1/2/3, when
# cpp-tier.sh still ran every entry through ctest:
#   cpp-tier   cuda 343/249/219s  hip 378/270/243s
#   cpp-smoke  cuda 143/137/137s  hip 193/177/180s  (GPU phase)
# ctest gained because every entry was its own process paying GPU runtime init,
# which overlaps; the smoke binary pays that once, so its cases already keep
# the card busy and shards barely help. Past 3 neither moves.
GPU_TEST_JOBS=${GPU_TEST_JOBS:-3}
# The watchdog on those shards (gpu_shard_watchdog in lib.sh). A shard that
# prints nothing for GPU_SHARD_STALL_SECONDS, or is still running
# GPU_SHARD_BUDGET_SECONDS after it started, is killed and the case it was in
# is named, so a wedged card (issue #477: gfx1200, every process spinning in a
# HIP sync, nothing printed) fails the gate instead of holding it -- and the
# card lock -- for hours. Both are generous on purpose: every case also has
# ctest's 120s backstop TIMEOUT (test/gtest/CMakeLists.txt), and a whole
# unsharded smoke run is about 3 minutes, so a trip means a hang, not a slow
# card. 0 turns either off.
GPU_SHARD_STALL_SECONDS=${GPU_SHARD_STALL_SECONDS:-600}
GPU_SHARD_BUDGET_SECONDS=${GPU_SHARD_BUDGET_SECONDS:-2700}
# Before it kills a hung shard the watchdog writes down the shard's state
# (gpu_wedge_snapshot in lib.sh): the vendor tools, every thread's backtrace.
# A wedged card can hang those as well, so each is cut off after this long.
GPU_WEDGE_TOOL_SECONDS=${GPU_WEDGE_TOOL_SECONDS:-120}
# Readings averaged into one load figure (0.3s apart); one is too noisy. Load
# only ranks the cards for `auto`.
GPU_LOAD_SAMPLES=${GPU_LOAD_SAMPLES:-4}

# The marker expression separating the fast iteration loop from the expensive
# tier. Empty disables the split -- everything is "fast", and slow-tier.sh then
# has nothing of its own to run.
# NB the single quotes around the defaults: inside ${VAR:-default} bash strips
# one level of quoting, so an unquoted `-m "not slow"` would reach pytest as
# the three words `-m`, `not`, `slow` -- a marker expression of `not` and a
# path called `slow`, which collects nothing and says only "no tests ran".
FAST_TEST_ARGS=${FAST_TEST_ARGS:-'-m "not slow"'}
SLOW_TEST_ARGS=${SLOW_TEST_ARGS:-'-m slow'}

# What the pre-push gate runs. Deliberately a curated list rather than the
# whole suite: the gate should cost seconds, so anything driving a compiler, a
# GPU or the network belongs in CI instead. The value is passed to pytest
# verbatim after FAST_TEST_ARGS, so paths, node ids and -k expressions all
# work. Empty means "whatever pytest collects by default", which is the right
# starting point for a small project.
PREPUSH_PATHS=${PREPUSH_PATHS:-}

# Node ids for `slow-tier.sh --skip-huge` to deselect: the cases with a hard
# memory floor or an outsized runtime that a smaller box cannot survive. One
# per line, exact node ids -- a -k filter cannot express a param containing
# pytest's own operators. Blank when the project has none.
SLOW_HUGE_CASES=${SLOW_HUGE_CASES:-}

# Where slow-tier.sh and cpp-tier.sh write their run logs. Relative paths are
# resolved against the repo root. Gitignored by default.
#
# NB docker/compose.yaml writes its own logs to .log/ instead -- it tees
# configure/build/test output there, and is not driven by this file at all.
SLOW_TIER_REPORTS=${SLOW_TIER_REPORTS:-.slow-tier-reports}

# --- environment ----------------------------------------------------------

# gh reads GH_TOKEN (then GITHUB_TOKEN) and nothing else, but this project's
# PAT is scoped to this one repository -- exporting it as a global GH_TOKEN on
# the host would break gh for every other checkout. So the host keeps it under
# a project-specific name and the mapping happens here.
#
# `export` is load-bearing twice over: devtools/devcontainer.sh runs
# `npx @devcontainers/cli` as a child of this shell, and devcontainer.json
# passes "GH_TOKEN": "${localEnv:GH_TOKEN}" -- localEnv being the
# environment of that CLI process, not the login shell.
#
# This does NOT reach a bare `gh` you type yourself, which sources nothing from
# here. For that, launch the session with the token already mapped:
#   GH_TOKEN=$GH_TOKEN_FREE_BREAKFAST claude
if [ -n "${GH_TOKEN_FREE_BREAKFAST:-}" ]; then
  export GH_TOKEN=$GH_TOKEN_FREE_BREAKFAST
fi


# Where to look for the project's virtualenv, in order; the first entry with a
# bin/pytest wins. `.venv` is what `uv sync` makes on the host, `/opt/venv` is
# what the Dockerfile bakes into the image.
VENV_PATHS=${VENV_PATHS:-.venv /opt/venv}

# Optional tools doctor.sh reports on, one `name:what is lost without it` per
# line. A missing one is a WARN (a slice of the suite will skip), never a FAIL.
#
# Most of this list is absent on the HOST and present in the CUDA container,
# which is the point: run doctor.sh on both sides and the warnings tell you
# which half of the workflow you are on. A host with none of the C++ toolchain
# is a perfectly good place to edit, lint and run the Python suite.
DOCTOR_OPTIONAL_TOOLS=${DOCTOR_OPTIONAL_TOOLS:-"docker:devcontainer, compose services and worktree containers
gh:the PR flow (also needs GH_TOKEN)
cmake:configuring and building the C++ tree at all
ninja:the generator every preset uses
clang++:the C++23 module build (must be clang -- CMakeLists.txt refuses gcc)
clang-scan-deps:module dependency scanning; a FATAL_ERROR at configure without it
nvcc:compiling .cu translation units under the CUDA backend
hipconfig:the HIP backend, and devtools/cross-backend-check.sh; present only in the \`hip\`/\`combined\` image
compute-sanitizer:the GPU memcheck/racecheck run (docker compose run --rm compute-sanitizer)
nsys:Nsight Systems profiling from inside the container
sccache:warm rebuilds (the compiler cache the default preset wires in as CMAKE_*_COMPILER_LAUNCHER); without it every configure recompiles from scratch
flock:cross-session GPU admission (one run per card, with_gpu_card in lib.sh); without it cpp-tier.sh/cpp-smoke.sh run unlocked
clang-tidy:the advisory lint pass (cpp-tier.sh --tidy, and CI's cuda leg)
clang-format:the C++ formatting pass (see CLAUDE.md -- it is not a git hook)
cmake-format:the CMake formatting pass; run by hand, not by any hook
cmake-lint:the CMake lint pass CI runs on every PR
agy:the Antigravity CLI the GPU images ship (docker/install-agy.sh); absent on a bare host unless you installed it there yourself"}

# Which GPU vendors doctor.sh probes for a LIVE DEVICE, space-separated:
# `nvidia`, `amd`, or empty to drop the section entirely (a CPU-only repo).
#
# This is about the DEVICE, not the toolchain -- the two are independent and
# both matter. nvcc present with no card still cannot configure the `default`
# preset, because CMAKE_CUDA_ARCHITECTURES=native asks the driver what is
# installed at CONFIGURE time; and a card present with no toolchain is the
# normal host state. Both vendors are listed because this project builds for
# either and a box can have both, while a build targets exactly one.
#
# The same list drives doctor's `gpu kernel modules` section, which maps the
# vendor to its DKMS module (nvidia -> nvidia, amd -> amdgpu) and asks whether
# that module is built for every bootable kernel -- a card that works today is
# still lost at the next reboot if the rebuild failed. Adding a vendor here
# means teaching that mapping too.
DOCTOR_GPU_VENDORS=${DOCTOR_GPU_VENDORS:-"nvidia amd"}

# Paths that must exist for some slice of the suite to run, one
# `path:what skips without it` per line -- typically submodules or fixture
# trees.
#
# The three /opt trees are what CMakePresets.json's `default` preset points
# CMAKE_PREFIX_PATH at. They exist only inside the CUDA image, so on the host
# they warn -- and the answer to that warning is the `workstation` preset,
# which fetches everything instead. A `default` configure on a host without
# them fails at `find_package(spdlog)` with a message that does not mention
# containers at all, which is why they are listed.
DOCTOR_REQUIRED_PATHS=${DOCTOR_REQUIRED_PATHS:-"/opt/spdlog:the \`default\` CMake preset's prebuilt spdlog (use --preset workstation on a bare host)
/opt/taskflow:the \`default\` preset's prebuilt taskflow
/opt/tomlplusplus:the \`default\` preset's prebuilt toml++
deps/WarpWraps/CMakeLists.txt:the WarpWraps submodule -- the GPU layer everything under src/ imports; nothing configures without it"}

# --- the agent in the images -----------------------------------------------

# Where the Claude Code version pin lives, one path per line, relative to the
# repo root: every file carrying an `ARG CLAUDE_CODE_VERSION=` line. These are
# the three GPU Dockerfiles, each of which runs docker/install-claude-code.sh as
# its last layer -- NOT Dockerfile.base, because a bump there would rebuild the
# CUDA toolkit and the ~30GB ROCm install downstream of it.
#
# devtools/claude-version.sh reads, compares and rewrites exactly these, and
# doctor.sh warns when they disagree or have fallen behind the npm registry.
# Set it empty to drop both checks (a project that does not ship the agent in
# its image); a path listed here that does not exist is an error rather than a
# skip, because a Dockerfile renamed out of this list IS the drift.
CLAUDE_PIN_FILES=${CLAUDE_PIN_FILES:-"docker/Dockerfile.cuda
docker/Dockerfile.hip
docker/Dockerfile.combined"}

# The npm package those files install, and how long to wait on the registry
# before giving up and reporting the pin alone.
CLAUDE_NPM_PACKAGE=${CLAUDE_NPM_PACKAGE:-@anthropic-ai/claude-code}
CLAUDE_REGISTRY_TIMEOUT=${CLAUDE_REGISTRY_TIMEOUT:-8}

# The same three files carry the second agent's pin: `ARG AGY_RELEASE=` and
# `ARG AGY_SHA512=`, the Antigravity CLI (`agy`) installed by
# docker/install-agy.sh. devtools/agy-version.sh reads, compares and rewrites
# exactly these -- BOTH lines, since the hash belongs to the release -- and
# doctor.sh warns when they disagree or have fallen behind upstream. Set empty
# to drop both checks.
#
# It is a separate list from CLAUDE_PIN_FILES on purpose, even though the two
# name the same files today: dropping one agent from the images should not have
# to mean dropping the other.
AGY_PIN_FILES=${AGY_PIN_FILES:-"docker/Dockerfile.cuda
docker/Dockerfile.hip
docker/Dockerfile.combined"}

# Where `agy` releases are published, and for which platform. There is no
# version index -- the manifest endpoint serves only the CURRENT release, and
# the tarball URL is the only thing that carries a release's build id -- which
# is why the pin is a release id rather than a version. See
# docker/install-agy.sh.
#
# MANIFEST_PLATFORM is the updater's own spelling (`linux_amd64`) and PLATFORM
# is the storage path's (`linux-x64`); they differ upstream, not here. PLATFORM
# must match the AGY_PLATFORM default in docker/install-agy.sh, or this script
# would hash one tarball while the build downloads another.
AGY_MANIFEST_BASE=${AGY_MANIFEST_BASE:-https://antigravity-cli-auto-updater-974169037036.us-central1.run.app}
AGY_MANIFEST_PLATFORM=${AGY_MANIFEST_PLATFORM:-linux_amd64}
AGY_URL_BASE=${AGY_URL_BASE:-https://storage.googleapis.com/antigravity-public/antigravity-cli}
AGY_PLATFORM=${AGY_PLATFORM:-linux-x64}
AGY_REGISTRY_TIMEOUT=${AGY_REGISTRY_TIMEOUT:-8}
