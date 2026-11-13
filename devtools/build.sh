#!/usr/bin/env bash
# Configure and build the C++ tree -- the build half of `devtools/test.py pr`,
# on its own so a build-without-ctest run has a name of its own and the gates
# do not carry their own copy of the two cmake phases.
#
#   devtools/build.sh [--preset NAME] [--fresh] [--clean] [-j N] [--report DIR]
#
# THE TOOLCHAIN IS NOT ON YOUR HOST. clang-23 with libc++'s module manifest,
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
# Flags (a subset of `test.py pr`'s, minus everything about running tests):
#   --preset NAME   configure/build preset (default: CMAKE_PRESET from
#                   devtools/config.sh -- `auto`, the least-loaded live
#                   card's preset; the choice is printed).
#   --fresh         wipe the CMake cache first (`--fresh`). Needed after
#                   changing a toolchain variable; a plain reconfigure keeps
#                   the old value and the change appears to do nothing.
#                   Implied, with a line saying so, when the tree's cached
#                   C++ compiler is gone or now reports another version.
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
# logged, with its own PASS/FAIL summary line (tagged `build`, so it is distinct
# from `test.py pr`'s line in the same summary.log). SOURCED -- which is how
# devtools/test.py uses it, through bash -- it defines
# run_phase, cpp_build_dir, cpp_require_build_tools and cpp_build and does
# nothing else, so a gate's configure, build and ctest all tee into ONE log
# (BUILD_LOG) and the gate writes the single summary line. Nothing here
# duplicates a phase a gate also owns; there is exactly one copy of the cmake
# invocations, and it is here.
set -euo pipefail

# Sourced after lib.sh, which has already loaded (and with it
# config.sh); only pull it in when run standalone, so we do not re-run its git
# probes on every source.
if [ -z "${REPO_ROOT:-}" ]; then
  . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
fi

# Where run_phase tees. A caller that sources this sets it to its own log so the
# whole tier lands in one file; left as /dev/null a stray source still runs.
: "${BUILD_LOG:=/dev/null}"

# Echo one phase's header and streamed output into BUILD_LOG, and return the
# command's status (not tee's -- PIPESTATUS[0]). devtools/test.py has its own
# port, for the phases it runs itself (ctest, pytest).
run_phase() {
  local name=$1; shift
  echo "--- $name: $* ---" | tee -a "$BUILD_LOG"
  "$@" 2>&1 | tee -a "$BUILD_LOG"
  return "${PIPESTATUS[0]}"
}

# The binary directory a preset configures into: `default` uses build/ but the
# others use build-<preset>/, and a hardcoded build/ would point --tidy's
# compile_commands.json at the previous preset's database, and the push gate
# at the wrong test binaries. preset_binary_dir (lib.sh) reads it out of
# CMakePresets.json; falls back to build/ when that cannot say.
cpp_build_dir() {
  local d
  d=$(preset_binary_dir "$1")
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
# Anything after the jobs goes to the configure: the weekly nightly's
# -DFREE_BREAKFAST_SCCACHE=OFF, and the warm nightly's -U of it.
#   cpp_build <preset> <fresh 0|1> <clean 0|1> <jobs-or-empty> [cmake args...]
cpp_build() {
  local preset=$1 fresh=$2 clean=$3 jobs=$4
  shift 4

  # pre-commit's pre-push stage runs this (via test.py push) with GIT_DIR /
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

  # After the unset, for the same reason: a leaked GIT_DIR would point
  # `git submodule` at the hook's repo. A failed init is not fatal here --
  # deps/CMakeLists.txt then stops the configure with an error naming the fix.
  init_submodules "$REPO_ROOT" 2>&1 | tee -a "$BUILD_LOG" ||
    echo "build: WARNING -- submodule init failed; configuring anyway" >&2

  local configure=(cmake --preset "$preset" "$@")
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

  [ "$preset" = auto ] && preset=$(gpu_auto_preset)
  cpp_require_build_tools || exit 1

  # A tree configured by a compiler that has since changed cannot configure
  # (#1333); reconfigure it fresh, as devtools/test.py does.
  local drift
  if [ "$fresh" -eq 0 ] &&
       drift=$(build_tree_compiler_drift "$(cpp_build_dir "$preset")"); then
    echo "build: compiler changed ($drift); reconfiguring with --fresh"
    fresh=1
  fi

  case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac
  mkdir -p "$report_dir"
  local stamp rev
  stamp=$(date +%Y%m%d-%H%M%S)
  tree_rev "$REPO_ROOT"  # lib.sh: HEAD, +dirty when tracked changes (#694)
  rev=$TREE_REV
  BUILD_LOG="$report_dir/build-$stamp-$rev-$preset.log"

  echo "build: $REPO_ROOT @ $rev  preset=$preset -> $BUILD_LOG"

  tree_rev_header build "$BUILD_LOG"

  local rc=0
  set +e
  cpp_build "$preset" "$fresh" "$clean" "$jobs" || rc=$?
  set -e

  local verdict summary
  verdict=$([ "$rc" -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")
  summary="build $verdict  rev=$rev  preset=$preset  stamp=$stamp${TREE_DIFF_HASH:+  diff=$TREE_DIFF_HASH}"
  echo "$summary" | tee -a "$BUILD_LOG"
  echo "$summary" >> "$report_dir/summary.log"
  exit "$rc"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  build_main "$@"
fi
