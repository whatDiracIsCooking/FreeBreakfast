#!/usr/bin/env bash
# The C++ smoke gate, run at `git push` by pre-commit's pre-push stage -- but
# ONLY when the push touches C++. That condition is the `files:` filter on the
# hook in .pre-commit-config.yaml, not anything in here.
#
# What it does, and why it is shaped this way:
#   * INCREMENTAL configure + build of the whole tree (both test binaries),
#     reusing devtools/build.sh's cpp_build -- the one copy of the cmake phases.
#     Warm (you have been building), this is seconds; cold (a fresh worktree),
#     ~1 min. It never passes --fresh.
#   * runs free_breakfast_tests_host in full (~0.3s), then
#     free_breakfast_tests_gpu with FREE_BREAKFAST_FAST_TESTS=1,
#     which makes every reference-grade case GTEST_SKIP (see
#     test/gtest/slow_test.h) -- not just end-to-end capstones, but any case
#     that checks a result against an external oracle to tight tolerance, or
#     repeats a production-scale solve for a timing/convergence comparison.
#     Those are correctness/perf regressions for cpp-tier.sh and CI to catch,
#     not smoke, and a live GPU the push gate does not have exclusive use of
#     makes their cost unpredictable.
#
# The split made the BUILD cheap; FREE_BREAKFAST_FAST_TESTS is what
# keeps reference-grade validation off the push path. This is a BUILD + SMOKE
# gate on purpose: it catches the one class with no other server-side check
# here -- does the tree still compile and link, do the host tests and a GPU
# smoke pass -- without the minutes that get a gate bypassed. The full
# validation (every reference case) stays in devtools/cpp-tier.sh and the
# nightly / self-hosted path.
#
# The GPU suite runs under with_gpu_slot (lib.sh), holding one of its card's
# slots so concurrent cpp-tier.sh runs in other worktrees cannot overload that
# card. The host suite does not touch the GPU and runs unlocked.
#
# It SKIPS (exit 0, not a failure) when the toolchain or a GPU is absent: the
# clang/CMake/nvcc stack lives in docker/Dockerfile.{cuda,hip,combined}, and
# the `default` preset's
# `native` arch queries a live device at CONFIGURE time, so a push from a bare
# host cannot run this. Skipping keeps that push off --no-verify; the gate is
# real where it can run (inside the container, on the GPU box), and CI /
# cpp-tier remain the backstop elsewhere.
#
# Bypass a one-off with `git push --no-verify`. Runnable by hand
# (`devtools/cpp-smoke.sh`) to see exactly what a push will run.
#: -- help stops here --
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
# After lib.sh: build.sh skips re-sourcing it (REPO_ROOT is set) and only
# defines cpp_build, cpp_build_dir, cpp_require_build_tools and run_phase.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build.sh"
cd "$REPO_ROOT"

[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { usage_from_header "${BASH_SOURCE[0]}"; exit 0; }

preset=$CMAKE_PRESET
jobs=$BUILD_JOBS
# `auto` (the default) picks the least-loaded live card (gpu_auto_preset in
# lib.sh); the device-node check below follows the backend it chose.
[ "$preset" = auto ] && preset=$(gpu_auto_preset)
backend=$(gpu_backend_of_preset "$preset")

skip() {
  echo "cpp-smoke: $1"
  echo "cpp-smoke: skipping (CI and devtools/cpp-tier.sh remain the backstop)"
  exit 0
}

# The toolchain is in the image, not on a bare host.
for tool in cmake ninja; do
  command -v "$tool" >/dev/null 2>&1 ||
    skip "$tool not found -- the C++ toolchain lives in the GPU container"
done

# PRESENT IS NOT THE SAME AS USABLE, and this gate runs on the HOST -- it is a
# git pre-push hook, so it gets whatever cmake the distro installed. A cmake
# older than the tree's floor fails in a way that looks nothing like a version
# problem: it cannot even parse CMakePresets.json, because that file declares
# `cmakeMinimumRequired` and an older cmake rejects the whole document with
#
#   Error: @4,14: "cmakeMinimumRequired" major version 4 must be less than 3
#
# so the gate reported BUILD FAILED and blocked the push on a machine that was
# never expected to build at all. Checking only `command -v cmake` is what let
# that through: Ubuntu 24.04 ships 3.28, which is present and unusable.
#
# The floor comes from CMakeLists.txt rather than being hardcoded here, so the
# two cannot drift. `sort -V` does the comparison -- no bash arithmetic on
# dotted versions.
cmake_floor=$(sed -n 's/^cmake_minimum_required(VERSION \([0-9.]*\)).*/\1/p' \
  "$REPO_ROOT/CMakeLists.txt" | head -1)
cmake_have=$(cmake --version 2>/dev/null | sed -n '1s/.* \([0-9][0-9.]*\).*/\1/p')
if [ -n "$cmake_floor" ] && [ -n "$cmake_have" ] &&
   [ "$(printf '%s\n%s\n' "$cmake_floor" "$cmake_have" | sort -V | head -1)" \
     != "$cmake_floor" ]; then
  skip "cmake $cmake_have is below this tree's $cmake_floor floor -- the C++ toolchain lives in the GPU container"
fi
# `default`'s native arch reads a live device at configure time, and the GPU
# smoke needs one to run; without a card this gate cannot proceed. Probe the
# driver device NODES, not nvidia-smi: NVML (nvidia-smi) routinely fails with
# "Failed to initialize NVML: Unknown Error" inside a container whose CUDA
# runtime -- and thus /dev/nvidia* -- works perfectly, which is exactly what
# native configure and the tests use.
if [ "$backend" = hip ]; then
  [ -e /dev/kfd ] ||
    skip "no HIP device node (/dev/kfd) -- the GPU smoke needs one"
elif [ ! -e /dev/nvidiactl ] || ! ls /dev/nvidia[0-9]* >/dev/null 2>&1; then
  skip "no CUDA device node (/dev/nvidia*) -- native configure and the GPU smoke need one"
fi

report_dir=$SLOW_TIER_REPORTS
case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac
mkdir -p "$report_dir"
stamp=$(date +%Y%m%d-%H%M%S)
rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_LOG="$report_dir/smoke-$stamp-$rev-$preset.log"

echo "cpp-smoke: $REPO_ROOT @ $rev  preset=$preset  (incremental build + host + GPU smoke)"
echo "cpp-smoke: full log -> $BUILD_LOG"

build_dir=$(cpp_build_dir "$preset")

rc=0
set +e

# Incremental build of the whole tree (both binaries). Quiet to the terminal --
# run_phase still tees everything into BUILD_LOG; only its stdout is dropped.
echo "cpp-smoke: building (incremental, preset=$preset, -j ${jobs:-auto}) ..."
cpp_build "$preset" 0 0 "$jobs" >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
  echo "cpp-smoke: BUILD FAILED (rc=$rc) -- see $BUILD_LOG" >&2
  tail -25 "$BUILD_LOG" >&2
  set -e
  exit "$rc"
fi

host_bin="$build_dir/test/gtest/free_breakfast_tests_host"
gpu_bin="$build_dir/test/gtest/free_breakfast_tests_gpu"

# Run one binary: append its full output to the log, echo its summary lines,
# return its status. `lock` gates with_gpu_slot (lib.sh) -- the host suite
# doesn't touch the GPU, so it shouldn't queue behind another session's GPU
# tests, but the gpu suite must serialize against them.
run_suite() {
  local label=$1 bin=$2 lock=$3
  shift 3
  [ -x "$bin" ] || {
    echo "cpp-smoke: missing binary $bin -- see $BUILD_LOG" >&2
    return 1
  }
  echo "--- $label: $bin ---" >>"$BUILD_LOG"
  local out rc2
  if [ "$lock" -eq 1 ]; then
    out=$(with_gpu_slot "$backend" "$bin" "$@" 2>&1)
  else
    out=$("$bin" "$@" 2>&1)
  fi
  rc2=$?
  printf '%s\n' "$out" >>"$BUILD_LOG"
  printf '%s\n' "$out" | grep -E 'ran\.|PASSED|SKIPPED|FAILED' | sed "s/^/cpp-smoke:   /"
  return $rc2
}

echo "cpp-smoke: host tests ..."
run_suite host "$host_bin" 0 || rc=$?

echo "cpp-smoke: GPU smoke (heavy reference cases skipped via FREE_BREAKFAST_FAST_TESTS) ..."
export FREE_BREAKFAST_FAST_TESTS=1
run_suite gpu "$gpu_bin" 1 || rc=$?
unset FREE_BREAKFAST_FAST_TESTS

set -e

if [ "$rc" -ne 0 ]; then
  echo "cpp-smoke: FAILED (rc=$rc) -- see $BUILD_LOG (bypass one push with --no-verify)" >&2
  exit "$rc"
fi
echo "cpp-smoke: PASS"
