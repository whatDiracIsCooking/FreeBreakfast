#!/usr/bin/env bash
# Configure and build the C++ tree -- the build half of devtools/cpp-tier.sh,
# on its own so a build-without-ctest run has a name of its own and cpp-tier
# does not carry its own copy of the two cmake phases.
#
#   devtools/build.sh [--preset NAME] [--fresh] [--clean] [-j N] [--report DIR]
#
# THE TOOLCHAIN IS NOT ON YOUR HOST. clang-20 with libc++'s module manifest,
# CMake 4.2 and nvcc live in docker/Dockerfile.{cuda,hip,combined}, so the
# normal way to call this is
# from inside that container (which DEVCONTAINER_CONFIG in config.sh already
# selects):
#
#   devtools/devcontainer.sh shell -c devtools/build.sh
#
# On a bare host with its own clang and CUDA, use --preset workstation: the
# `default` preset expects the prebuilt /opt dependencies the image bakes in,
# and fails at find_package(spdlog) without them. `devtools/doctor.sh` lists
# which of the two situations you are in.
#
# Flags (a subset of cpp-tier.sh's, minus everything about running tests):
#   --preset NAME   configure/build preset (default: CMAKE_PRESET from
#                   devtools/config.sh).
#   --fresh         wipe the CMake cache first (`--fresh`). Needed after
#                   changing a toolchain variable; a plain reconfigure keeps
#                   the old value and the change appears to do nothing.
#   --clean         rebuild every object (`--clean-first`), keeping the cache.
#   -j N            parallel compile jobs (default: BUILD_JOBS from config.sh,
#                   or Ninja's own choice when that is empty). Module builds
#                   are memory-hungry per job -- a too-high N OOMs rather than
#                   failing cleanly.
#   --report DIR    timestamped log + one-line summary (default:
#                   SLOW_TIER_REPORTS from config.sh).
#
# Exit status is the first failing phase's, so a scheduler can alert on it.
#: -- help stops here --
#
# Two lives. Run directly, this is the build-only tier: configure + build,
# logged, with its own PASS/FAIL summary line (tagged `build`, so it is
# distinct from cpp-tier's `cpp-tier` line in the same summary.log). SOURCED --
# which is how devtools/cpp-tier.sh uses it -- it defines run_phase,
# cpp_build_dir, cpp_require_build_tools and cpp_build and does nothing else, so
# the tier's configure, build and ctest all tee into ONE log (BUILD_LOG) and the
# tier writes the single summary line. Nothing here duplicates a phase cpp-tier
# also owns; there is exactly one copy of the cmake invocations, and it is here.
set -euo pipefail

# Sourced from cpp-tier.sh, which has already loaded lib.sh (and with it
# config.sh); only pull it in when run standalone, so we do not re-run its git
# probes on every source.
if [ -z "${REPO_ROOT:-}" ]; then
  . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
fi

# Where run_phase tees. A caller that sources this sets it to its own log so the
# whole tier lands in one file; left as /dev/null a stray source still runs.
: "${BUILD_LOG:=/dev/null}"

# Echo one phase's header and streamed output into BUILD_LOG, and return the
# command's status (not tee's -- PIPESTATUS[0]). Shared with cpp-tier.sh's
# ctest and clang-tidy phases, which is why it lives here rather than in either
# script.
run_phase() {
  local name=$1; shift
  echo "--- $name: $* ---" | tee -a "$BUILD_LOG"
  "$@" 2>&1 | tee -a "$BUILD_LOG"
  return "${PIPESTATUS[0]}"
}

# The binary directory a preset configures into. Asked of CMake rather than
# guessed: `default` uses build/ but the others use build-<preset>/, and a
# hardcoded build/ would point --tidy's compile_commands.json at the previous
# preset's database. Falls back to build/ when CMake cannot say.
cpp_build_dir() {
  local preset=$1 d
  d=$(cmake --preset "$preset" -N 2>/dev/null |
    sed -n 's/^.*Binary directory: *//p' | head -1)
  [ -n "$d" ] && echo "$d" || echo "$REPO_ROOT/build"
}

# Fail early, and toward the fix, when the toolchain is not on this machine --
# the whole clang/CMake/nvcc stack lives in the image, not on a bare host.
cpp_require_build_tools() {
  local tool
  for tool in cmake ninja; do
    command -v "$tool" >/dev/null 2>&1 || {
      echo "build: $tool not found -- this needs the CUDA container:" >&2
      echo "  devtools/devcontainer.sh shell -c devtools/build.sh" >&2
      echo "(or --preset workstation, on a host with its own clang and CUDA)" >&2
      return 1
    }
  done
}

# Configure then build one preset, teeing both phases into BUILD_LOG. Returns
# the first failing phase's status; runs with the caller's errexit setting, so
# wrap the call in `set +e; cpp_build ... || rc=$?` as both entry points do.
#   cpp_build <preset> <fresh 0|1> <clean 0|1> <jobs-or-empty>
cpp_build() {
  local preset=$1 fresh=$2 clean=$3 jobs=$4

  # pre-commit's pre-push stage runs this (via cpp-smoke.sh) with GIT_DIR /
  # GIT_WORK_TREE / GIT_INDEX_FILE set for its own hook dispatch. Those leak
  # into every child process, including the `git clone`/`git checkout` that
  # CMake's FetchContent shells out to for deps/CMakeLists.txt's
  # nlohmann_json -- and GIT_DIR overrides that nested git's normal
  # cwd-based repo discovery, so it resolves refs against THIS repo instead
  # of the freshly cloned one. Symptom: "fatal: invalid reference: v3.11.3"
  # on any push from a worktree whose deps cache isn't already warm, even
  # though the same configure succeeds outside the hook. Nothing in this
  # tree reads these vars, so unsetting is safe.
  unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE

  local configure=(cmake --preset "$preset")
  [ "$fresh" -eq 1 ] && configure+=(--fresh)
  run_phase configure "${configure[@]}" || return $?

  local build=(cmake --build --preset "$preset")
  [ -n "$jobs" ] && build+=(-j "$jobs")
  [ "$clean" -eq 1 ] && build+=(--clean-first)
  run_phase build "${build[@]}" || return $?
}

# --- standalone entry point ------------------------------------------------
# Only when executed, never when sourced.

build_main() {
  local preset=$CMAKE_PRESET
  local fresh=0 clean=0 jobs=$BUILD_JOBS
  local report_dir=$SLOW_TIER_REPORTS

  while [ $# -gt 0 ]; do
    case "$1" in
      --preset)  preset=$2; shift 2;;
      --fresh)   fresh=1; shift;;
      --clean)   clean=1; shift;;
      -j|--jobs) jobs=$2; shift 2;;
      --report)  report_dir=$2; shift 2;;
      -h|--help) usage_from_header "${BASH_SOURCE[0]}"; exit 0;;
      *)         echo "build: unknown flag: $1" >&2; exit 2;;
    esac
  done

  cpp_require_build_tools || exit 1

  case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac
  mkdir -p "$report_dir"
  local stamp rev
  stamp=$(date +%Y%m%d-%H%M%S)
  rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
  BUILD_LOG="$report_dir/build-$stamp-$rev-$preset.log"

  echo "build: $REPO_ROOT @ $rev  preset=$preset -> $BUILD_LOG"

  local rc=0
  set +e
  cpp_build "$preset" "$fresh" "$clean" "$jobs" || rc=$?
  set -e

  local verdict summary
  verdict=$([ "$rc" -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")
  summary="build $verdict  rev=$rev  preset=$preset  stamp=$stamp"
  echo "$summary" | tee -a "$BUILD_LOG"
  echo "$summary" >> "$report_dir/summary.log"
  exit "$rc"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  build_main "$@"
fi
