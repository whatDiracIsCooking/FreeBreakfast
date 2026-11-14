#!/usr/bin/env bash
# Compile the tree for the backend you are NOT building, to catch the class of
# bug that a green build on one backend cannot see.
#
#   devtools/cross-backend-check.sh [--fresh] [-j N] [--native] [--device-only]
#                                   [--plan]
#
# WHY THIS EXISTS. A build targets exactly one backend, and the two are
# compiled by different front ends: a .cu goes through nvcc under CUDA and
# through clang's `-x hip` under HIP. nvcc is the more permissive of the two,
# so code that only clang rejects compiles clean, passes ctest, and ships --
# and the first sign of trouble is a ROCm build weeks later.
#
# A concrete example of the class: a functor that carries a `const` member of
# CLASS type is non-trivially-copyable under clang, so it fails the
# device_functor concept WarpWraps's parallel_for requires -- yet nvcc accepts it.
# Such code compiles clean, stays green across the whole CUDA tier, and only
# fails when someone builds ROCm. See WarpWraps's docs/architecture.md for the rule.
#
# WHAT IT COSTS. This is a COMPILE-ONLY check -- no GoogleTest fetch, no runtime
# tests, no device -- which is why it is worth running every time where
# devtools/test.py pr costs minutes. NO TIMINGS ARE QUOTED HERE ON PURPOSE:
# this tree compiles WarpWraps's module surface as well as its own, so the cost
# tracks the dependency and a number measured today would be stale by the next
# submodule bump. Measure it on your box if you need one.
#
# WHAT IT DOES NOT DO. It proves the other backend COMPILES, not that it runs:
# no kernel is launched and no result is checked. A real ROCm box running
# `devtools/test.py pr --preset hip` is still the stronger statement, and is
# what you want before trusting the backend in anger. This is the cheap check
# you can afford every time.
#
# Flags:
#   --fresh         wipe the CMake cache first. Needed after changing a
#                   toolchain variable, which a plain reconfigure keeps.
#   -j N            parallel compile jobs (default: BUILD_JOBS from
#                   devtools/config.sh, or Ninja's own choice when empty).
#   --device-only   build only the device .cu libraries instead of the whole
#                   compile-time tier. Narrower: it catches a kernel-side
#                   divergence but not one in a module unit, and it only knows
#                   the targets CROSS_CHECK_DEVICE_TARGETS names (default:
#                   the free_breakfast_device_libraries umbrella). Empty, and this
#                   flag refuses rather than building nothing and calling it a
#                   pass.
#   --native        skip the container and run cmake here. Implied when this
#                   host already has the target backend's toolchain.
#   --plan          print the backend, the root and rust/ presets, the build
#                   directory, the probe and the image this run would use, and
#                   exit without building.
#
# WHICH BACKEND. One variable is enough (#1528). CROSS_CHECK_PRESET alone
# decides it: the backend is that preset's FREE_BREAKFAST_GPU_BACKEND in
# CMakePresets.json, following `inherits` (ci-cuda: CUDA, ci-hip: HIP).
# CROSS_CHECK_BACKEND alone picks its preset (HIP: hip, CUDA: ci-cuda). Both
# set and disagreeing is refused, exit 2, rather than one quietly winning.
# Neither set is devtools/config.sh's pair, HIP and hip. Unless set, the probe
# (hipconfig or nvcc) and the image (<PROJECT_NAME>:hip or :cuda) follow the
# backend, and a CUDA run builds into CROSS_CHECK_BUILD_DIR with -cuda
# appended, so the two backends never reconfigure one cache back and forth.
# After configuring, the backend in each tree's CMakeCache.txt is checked
# against the one resolved here, and a mismatch stops the run with exit 2.
#
# rust/'s .cu are checked too, with or without --device-only. rust/ is a CMake
# project of its own (rust/CMakeLists.txt) that the root build never adds, so
# after the root tree this configures it with its preset for the same backend
# (the resolved backend lowercased: hip or cuda) into <build dir>/rust, and
# builds all of it: the forked kernel libraries and the C ABI library they
# link. rust/build-hip and rust/build-cuda, which cargo links, are not touched.
#
# Exit status: 0 the other backend compiles, 1 it does not, 2 the check could
# not be run at all (no toolchain and no usable image) -- never a silent skip,
# because the machines least likely to have ROCm are exactly the ones where
# this is worth knowing.
#: -- help stops here --
#
# Deliberately NOT a git hook. It needs docker and a ~20GB image, so on a host
# without them it can only skip -- and a gate that silently does nothing on the
# machine that most needs it is worse than no gate, which is the same reasoning
# .pre-commit-config.yaml gives for keeping the C++ tier off the push gate.
# Run it before opening a PR that touches a .cu, or anything under src/ that a
# .cu includes.
#
# Deliberately NOT devcontainer.sh --hip either: that brings up a full
# devcontainer with its own volumes and per-worktree state, which is the right
# tool for working IN the HIP backend and far too much machinery for a
# four-second compile.
set -euo pipefail

# What the caller set, before config.sh fills in its defaults: the backend is
# resolved from these ("WHICH BACKEND" above).
user_backend=${CROSS_CHECK_BACKEND-}
user_preset=${CROSS_CHECK_PRESET-}
user_tool=${CROSS_CHECK_TOOL-}
user_image=${CROSS_CHECK_IMAGE-}
user_build_dir=${CROSS_CHECK_BUILD_DIR-}

# shellcheck source=./lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

fresh=0
native=0
device_only=0
plan=0
jobs=$BUILD_JOBS

while [ $# -gt 0 ]; do
  case "$1" in
    --fresh) fresh=1 ;;
    --native) native=1 ;;
    --device-only) device_only=1 ;;
    --plan) plan=1 ;;
    -j) shift; jobs=${1:?-j needs a value} ;;
    -j*) jobs=${1#-j} ;;
    -h|--help) usage; exit 0 ;;
    *) echo "cross-backend-check: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# The FREE_BREAKFAST_GPU_BACKEND a configure preset sets in CMakePresets.json,
# following `inherits` in order, as preset_binary_dir (lib.sh) does for its
# binaryDir. On failure, prints why to stderr and returns 1.
preset_backend() {
  python3 -I - "$REPO_ROOT" "$1" <<'EOF'
import json, sys
root, name = sys.argv[1], sys.argv[2]
presets = {p["name"]: p for p in
           json.load(open(f"{root}/CMakePresets.json"))["configurePresets"]}
def find(n, seen=()):
    if n in seen or n not in presets:
        return None
    p = presets[n]
    v = p.get("cacheVariables", {}).get("FREE_BREAKFAST_GPU_BACKEND")
    if isinstance(v, dict):
        v = v.get("value")
    if v:
        return v
    inh = p.get("inherits", [])
    for parent in [inh] if isinstance(inh, str) else inh:
        b = find(parent, seen + (n,))
        if b:
            return b
    return None
if name not in presets:
    sys.exit(f"cross-backend-check: no configure preset '{name}' in CMakePresets.json")
b = find(name)
if not b:
    sys.exit(f"cross-backend-check: preset '{name}' sets no FREE_BREAKFAST_GPU_BACKEND")
print(b)
EOF
}

refuse() {
  echo "cross-backend-check: $*" >&2
  exit 2
}

# The backend and the preset, from whichever the caller set ("WHICH BACKEND").
if [ -n "$user_preset" ]; then
  preset=$user_preset
elif [ -n "$user_backend" ]; then
  case "${user_backend^^}" in
    HIP) preset=hip ;;
    CUDA) preset=ci-cuda ;;
    *) refuse "CROSS_CHECK_BACKEND='$user_backend' is neither CUDA nor HIP" ;;
  esac
else
  preset=$CROSS_CHECK_PRESET
fi
backend=$(preset_backend "$preset") ||
  refuse "cannot tell which backend CROSS_CHECK_PRESET='$preset' builds"
case "$backend" in
  CUDA|HIP) ;;
  *) refuse "preset '$preset' sets FREE_BREAKFAST_GPU_BACKEND='$backend', neither CUDA nor HIP" ;;
esac
# Against the backend the caller set; with neither set, against config.sh's.
# A preset set alone is not checked against config.sh's default backend: that
# default is exactly what it overrides.
wanted=$user_backend
[ -z "$user_preset" ] && [ -z "$wanted" ] && wanted=$CROSS_CHECK_BACKEND
if [ -n "$wanted" ] && [ "${wanted^^}" != "$backend" ]; then
  cat >&2 <<EOF
cross-backend-check: CROSS_CHECK_BACKEND=$wanted but CROSS_CHECK_PRESET=$preset
builds $backend.

  Set one of the two: the preset alone decides the backend, and the backend
  alone picks its preset (HIP: hip, CUDA: ci-cuda).
EOF
  exit 2
fi

if [ "$backend" = CUDA ]; then default_probe=nvcc; else default_probe=hipconfig; fi
probe=${user_tool:-$default_probe}
image=${user_image:-${PROJECT_NAME}:${backend,,}}
build_dir=$CROSS_CHECK_BUILD_DIR
if [ -z "$user_build_dir" ] && [ "$backend" = CUDA ]; then
  build_dir=$build_dir-cuda
fi
rust_preset=${backend,,}

if [ "$plan" = 1 ]; then
  echo "backend=$backend"
  echo "preset=$preset"
  echo "rust_preset=$rust_preset"
  echo "build_dir=$build_dir"
  echo "probe=$probe"
  echo "image=$image"
  exit 0
fi

# The targets that exist only to be compiled. Named rather than built wholesale
# so --device-only has something narrower to ask for; with neither, `cmake
# --build` builds everything the compile-time configuration generated.
declare -a device_targets=()
config_lines "$CROSS_CHECK_DEVICE_TARGETS"
device_targets=("${CONFIG_LINES[@]}")

# An empty list with --device-only would reach `cmake --build --target` with no
# target, whose failure says nothing about the backend -- and a version of this
# that swallowed it would report PASS for having compiled nothing. Refuse, with
# the reason, and exit 2 (the check could not be run) rather than 1 (it failed).
if [ "$device_only" = 1 ] && [ "${#device_targets[@]}" -eq 0 ]; then
  cat >&2 <<EOF
cross-backend-check: --device-only has nothing to build.

  CROSS_CHECK_DEVICE_TARGETS is empty -- its default in devtools/config.sh is
  the free_breakfast_device_libraries umbrella, so something overrode it. Unset the
  override, or run without the flag to compile the whole compile-time tier for
  $backend.
EOF
  exit 2
fi

# The backend a configured tree's cache holds must be the one resolved above:
# a PASS line naming a backend that was not compiled is what this replaced
# (#1528).
check_cached_backend() {
  local cached
  cached=$(sed -n 's/^FREE_BREAKFAST_GPU_BACKEND:[A-Z]*=//p' "$1/CMakeCache.txt")
  [ "$cached" = "$backend" ] ||
    refuse "$2 in $1 is configured for '${cached:-nothing}', not $backend"
}

# ---------------------------------------------------------------------------
# Phase 1: run the check, here, with cmake
# ---------------------------------------------------------------------------
run_here() {
  local -a cfg=(cmake --preset "$preset" -B "$build_dir" -DFREE_BREAKFAST_COMPILE_TIME_ONLY=ON)
  [ "$fresh" = 1 ] && cfg+=(--fresh)

  echo "cross-backend-check: configuring $backend (preset=$preset) -> $build_dir"
  "${cfg[@]}"
  check_cached_backend "$build_dir" "the root tree"

  local -a bld=(cmake --build "$build_dir")
  [ -n "$jobs" ] && bld+=(-j "$jobs")
  if [ "$device_only" = 1 ]; then
    bld+=(--target "${device_targets[@]}")
    echo "cross-backend-check: building ${#device_targets[@]} device librar(ies)"
  else
    echo "cross-backend-check: building the compile-time tier"
  fi
  "${bld[@]}"

  # rust/'s own project, from its own directory, where its presets are.
  local rust_build=$build_dir/rust
  case "$rust_build" in /*) ;; *) rust_build=$REPO_ROOT/$rust_build ;; esac
  local -a rcfg=(cmake --preset "$rust_preset" -B "$rust_build")
  [ "$fresh" = 1 ] && rcfg+=(--fresh)
  echo "cross-backend-check: configuring rust/ for $backend (preset=$rust_preset) -> $rust_build"
  (cd rust && "${rcfg[@]}")
  check_cached_backend "$rust_build" "rust/"

  local -a rbld=(cmake --build "$rust_build")
  [ -n "$jobs" ] && rbld+=(-j "$jobs")
  echo "cross-backend-check: building rust/'s kernel libraries"
  "${rbld[@]}"
}

# ---------------------------------------------------------------------------
# Phase 0: decide where to run it
# ---------------------------------------------------------------------------
# The toolchain for the OTHER backend is by definition not in the container you
# develop in -- the cuda image has no ROCm, and a HIP build needs no CUDA. So
# the normal path is a one-shot `docker run` against the other backend's image,
# from wherever you are, with no devcontainer involved.
if [ "$native" = 1 ] || command -v "$probe" >/dev/null 2>&1; then
  run_here
  if [ "$device_only" = 1 ]; then
    what="device libraries (${device_targets[*]})"
  else
    what="compile-time tier"
  fi
  echo "cross-backend-check: PASS ($backend compiles: the root tree's $what" \
    "with preset $preset, and rust/'s kernel libraries with preset $rust_preset)"
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  cat >&2 <<EOF
cross-backend-check: cannot run -- no '$probe' on this machine and no docker.

  This needs either the $backend toolchain here, or the image that has it.
  devtools/doctor.sh reports which tools this machine actually has.
EOF
  exit 2
fi

if ! docker image inspect "$image" >/dev/null 2>&1; then
  cat >&2 <<EOF
cross-backend-check: cannot run -- docker image '$image' is not built.

  Build it once (it is large; build.sh walks the chain, since
  docker/Dockerfile.${backend,,} needs docker/Dockerfile.base tagged first):

    docker/build.sh ${backend,,}

  That tags <PROJECT_NAME>:${backend,,}, which is what CROSS_CHECK_IMAGE
  defaults to. Or set CROSS_CHECK_IMAGE to an image with the $backend toolchain.
EOF
  exit 2
fi

# --native inside, so the container does not re-derive any of this. The repo is
# bind-mounted at /workspace; .git is deliberately NOT mounted, since configure
# and build read none of it and a worktree's .git points outside the workspace.
# Running as the caller's uid keeps the build directory host-owned.
echo "cross-backend-check: no '$probe' here -- running in $image"
declare -a docker_args=(
  run --rm
  -u "$(id -u):$(id -g)"
  -v "$REPO_ROOT:/workspace"
  -w /workspace
  -e "CROSS_CHECK_BACKEND=$backend"
  -e "CROSS_CHECK_PRESET=$preset"
  -e "CROSS_CHECK_BUILD_DIR=$build_dir"
  "$image"
  devtools/cross-backend-check.sh --native
)
[ "$fresh" = 1 ] && docker_args+=(--fresh)
[ "$device_only" = 1 ] && docker_args+=(--device-only)
[ -n "$jobs" ] && docker_args+=(-j "$jobs")

docker "${docker_args[@]}"
