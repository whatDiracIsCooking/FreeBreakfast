#!/usr/bin/env bash
# Configure, build and ctest the C++ tree WITHOUT skipping anything -- the
# deep-validation counterpart to devtools/cpp-tier.sh, the same way
# devtools/slow-tier.sh is the expensive counterpart to the fast pytest tier.
#
#   devtools/cpp-slow-tier.sh [--preset NAME] [--fresh] [--clean] [--no-test]
#                             [--tidy] [-j N] [--report DIR] [-- <ctest args>]
#
# devtools/cpp-tier.sh ALWAYS runs with FREE_BREAKFAST_FAST_TESTS=1,
# which GTEST_SKIPs every case guarded by
# FREE_BREAKFAST_SKIP_IF_FAST_TESTS() (see test/gtest/slow_test.h and
# cpp-tier.sh's own header) -- the reference-grade and end-to-end validation
# cases, which in a mature tree are a handful of entries and the majority of a
# full run's wall time. This script is where they run: every gtest case,
# unskipped.
#
# NOT on any per-commit or per-PR path, same reasoning as devtools/slow-tier.sh:
# a gate that costs the better part of an hour is one people learn to bypass.
# Reach for it the way you reach for the `asan` / `compute-sanitizer` compose
# services -- an occasional deep-validation sweep for a change with real risk
# of reaching code you did not directly touch (a shared core module, a
# generated-kernel regeneration, a cross-cutting rename), or as a periodic
# health check. See CLAUDE.md's "What is gated, and what is not".
#
# THE TOOLCHAIN IS NOT ON YOUR HOST. clang-20 with libc++'s module manifest,
# CMake 4.2 and nvcc live in docker/Dockerfile.{cuda,hip,combined}, so the
# normal way to call this is
# from inside that container (which DEVCONTAINER_CONFIG in config.sh already
# selects):
#
#   devtools/devcontainer.sh shell -c devtools/cpp-slow-tier.sh
#
# On a bare host with its own clang and CUDA, use --preset workstation: the
# `default` preset expects the prebuilt /opt dependencies the image bakes in,
# and fails at find_package(spdlog) without them. `devtools/doctor.sh` lists
# which of the two situations you are in.
#
# Flags: identical to cpp-tier.sh, minus anything about skipping tests -- there
# is no flag here that turns skipping back on; use cpp-tier.sh for that.
#   --preset NAME   configure/build preset (default: CMAKE_PRESET from
#                   devtools/config.sh). The test preset follows it when one of
#                   the same name exists, and is skipped with a note when not
#                   -- only default, workstation, debug and asan have one.
#   --fresh         wipe the CMake cache first (`--fresh`). Needed after
#                   changing a toolchain variable; a plain reconfigure keeps
#                   the old value and the change appears to do nothing.
#   --clean         rebuild every object (`--clean-first`), keeping the cache.
#   --no-test       configure and build only.
#   --tidy          after a successful build, run clang-tidy over the .cpp
#                   implementation units using the build's
#                   compile_commands.json. Reports findings; does NOT fail
#                   the run -- see cpp-tier.sh's header for why.
#   -j N            parallel compile jobs (default: BUILD_JOBS from config.sh,
#                   or Ninja's own choice when that is empty). Module builds
#                   are memory-hungry per job -- a too-high N OOMs rather than
#                   failing cleanly.
#   --report DIR    timestamped log + one-line summary (default:
#                   SLOW_TIER_REPORTS from config.sh).
#   -- <args>       everything after `--` is passed to ctest verbatim
#                   (e.g. `-- -R Constants` to scope the run). ctest has
#                   one entry per gtest case, so -R matches suite names.
#
# Exit status is the first failing phase's, so a scheduler can alert on it.
#: -- help stops here --
#
# The configure and build phases themselves are NOT here: they live in
# devtools/build.sh, which this sources for cpp_build, cpp_build_dir,
# cpp_require_build_tools and run_phase -- the same functions cpp-tier.sh
# sources, so there is exactly one copy of the cmake invocations.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
# After lib.sh: build.sh skips re-sourcing it (REPO_ROOT is already set) and
# only defines its functions. run_phase comes from here too, so the ctest and
# clang-tidy phases below tee into the same BUILD_LOG the build did.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build.sh"
cd "$REPO_ROOT"

preset=$CMAKE_PRESET
test_preset=$CTEST_PRESET
fresh=0
clean=0
run_tests=1
tidy=0
jobs=$BUILD_JOBS
report_dir=$SLOW_TIER_REPORTS
declare -a passthrough=()
preset_explicit=0

while [ $# -gt 0 ]; do
  case "$1" in
    --preset)  preset=$2; preset_explicit=1; shift 2;;
    --fresh)   fresh=1; shift;;
    --clean)   clean=1; shift;;
    --no-test) run_tests=0; shift;;
    --tidy)    tidy=1; shift;;
    -j|--jobs) jobs=$2; shift 2;;
    --report)  report_dir=$2; shift 2;;
    --)        shift; passthrough=("$@"); break;;
    -h|--help) usage_from_header "${BASH_SOURCE[0]}"; exit 0;;
    *)         echo "cpp-slow-tier: unknown flag: $1" >&2; exit 2;;
  esac
done

# --preset switches the test preset with it, or the run silently tests the
# wrong build directory: ctest --preset default reads build/, while
# --preset asan built into build-asan/.
[ "$preset_explicit" -eq 1 ] && test_preset=$preset

cpp_require_build_tools || exit 1

case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac
mkdir -p "$report_dir"
stamp=$(date +%Y%m%d-%H%M%S)
rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
log="$report_dir/cpp-slow-$stamp-$rev-$preset.log"
# run_phase (from build.sh) tees here, so configure, build, ctest and tidy all
# land in this one file.
BUILD_LOG=$log

echo "cpp-slow-tier: $REPO_ROOT @ $rev  preset=$preset -> $log"

# Where this preset puts compile_commands.json, for --tidy below. build.sh's
# cpp_build_dir asks CMake rather than guessing, because `default` uses build/
# but the others use build-<preset>/, and a hardcoded build/ would have --tidy
# silently lint the previous preset's database.
build_dir=$(cpp_build_dir "$preset")

# `jq -e` is not assumed to be present, so the "does this test preset exist"
# question is answered by asking ctest, which is. --show-only lists the tests a
# preset would run and exits non-zero when the preset is unknown.
has_test_preset() {
  [ -n "$test_preset" ] || return 1
  ctest --preset "$test_preset" --show-only >/dev/null 2>&1
}

rc=0
set +e

# Configure + build, both phases from build.sh (the one place they are spelled
# out). ctest and the advisory tidy below are this tier's own.
cpp_build "$preset" "$fresh" "$clean" "$jobs" || rc=$?

if [ "$rc" -eq 0 ] && [ "$run_tests" -eq 1 ]; then
  if has_test_preset; then
    # Deliberately no FREE_BREAKFAST_FAST_TESTS here -- this is the
    # tier where the reference-grade/end-to-end cases run.
    run_phase ctest with_gpu_lock ctest --preset "$test_preset" "${passthrough[@]}" || rc=$?
  else
    echo "cpp-slow-tier: no test preset '${test_preset:-<empty>}' -- built only" \
      | tee -a "$log"
    echo "               (CMakePresets.json defines test presets for default," \
      "workstation, debug and asan)" | tee -a "$log"
  fi
fi

# clang-tidy is deliberately advisory. `.clang-tidy` is configured and the two
# .cpp units are close to clean, but the .cppm module interfaces are not -- and
# the remaining findings there are the design rather than defects
# (cppcoreguidelines-special-member-functions on the CRTP buffer wrappers,
# which correctly inherit BufferBase's destructor; misc-unused-parameters on
# the deliberately symmetric deallocate(T*, size_t)). Making the run fail on
# those would mean bending the module hierarchy to a linter. Suppress them
# properly, then turn this into a gate.
if [ "$rc" -eq 0 ] && [ "$tidy" -eq 1 ]; then
  if ! command -v clang-tidy >/dev/null 2>&1; then
    echo "cpp-slow-tier: clang-tidy not found -- skipping --tidy" | tee -a "$log"
  else
    mapfile -t tidy_files < <(git -C "$REPO_ROOT" ls-files 'src/*.cpp')
    if [ "${#tidy_files[@]}" -eq 0 ]; then
      echo "cpp-slow-tier: no src/*.cpp to tidy" | tee -a "$log"
    else
      # Advisory: the phase's own status is reported but never folded into rc.
      run_phase clang-tidy clang-tidy -p "$build_dir" --quiet "${tidy_files[@]}" \
        || echo "cpp-slow-tier: clang-tidy reported findings (advisory)" | tee -a "$log"
    fi
  fi
fi

set -e

verdict=$([ "$rc" -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")
summary="cpp-slow-tier $verdict  rev=$rev  preset=$preset  stamp=$stamp"
echo "$summary" | tee -a "$log"
echo "$summary" >> "$report_dir/summary.log"
exit "$rc"
