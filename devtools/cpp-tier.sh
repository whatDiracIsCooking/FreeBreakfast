#!/usr/bin/env bash
# Configure, build and ctest the C++ tree -- the FAST C++ tier, beside the
# fast pytest gate and devtools/cpp-slow-tier.sh.
#
#   devtools/cpp-tier.sh [--preset NAME] [--fresh] [--clean] [--no-test]
#                        [--tidy] [-j N] [--report DIR]
#                        [-- <ctest args>]
#
# It is NOT on the push gate -- cpp-smoke.sh is -- but it is the gate before
# opening a PR (milestone-run requires it), and like the smoke gate it is
# budgeted at under five minutes on a busy card: test/support/slow_test.h keeps
# the fast set inside it, and the GPU entries run batched (run_tier_tests).
#
# ALWAYS runs with FREE_BREAKFAST_FAST_TESTS=1 -- the same env var
# cpp-smoke.sh sets -- so every case guarded by
# FBF_SKIP_IF_FAST_TESTS() (test/support/slow_test.h) GTEST_SKIPs.
# That guard is for reference-grade and end-to-end validation: cases that check
# a result against an external oracle to tight tolerance, or re-run a
# production-scale solve to compare timing or convergence. In a mature tree
# those are a handful of cases and the majority of a full run's wall time.
#
# This tier CANNOT run them -- there is no flag that turns it back on.
# devtools/cpp-slow-tier.sh is the un-skipped run, and a change touching the
# code those cases exercise still needs one before a PR.
#
# THE TOOLCHAIN IS NOT ON YOUR HOST. clang-20 with libc++'s module manifest,
# CMake 4.2 and nvcc live in docker/Dockerfile.{cuda,hip,combined}, so the
# normal way to call this is
# from inside that container (which DEVCONTAINER_CONFIG in config.sh already
# selects):
#
#   devtools/devcontainer.sh shell -c devtools/cpp-tier.sh
#
# On a bare host with its own clang and CUDA, use --preset workstation: the
# `default` preset expects the prebuilt /opt dependencies the image bakes in,
# and fails at find_package(spdlog) without them. `devtools/doctor.sh` lists
# which of the two situations you are in.
#
# Flags:
#   --preset NAME   configure/build preset (default: CMAKE_PRESET from
#                   devtools/config.sh -- `auto`, the least-loaded live
#                   card's preset; the choice is printed). The test preset follows it when one of
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
#                   the run, because the tree is not clean yet (see
#                   .clang-tidy). Off by default so the normal tier stays
#                   a build-and-test.
#   -j N            parallel COMPILE jobs (default: BUILD_JOBS from config.sh,
#                   or Ninja's own choice when that is empty). Module builds
#                   are memory-hungry per job -- a too-high N OOMs rather than
#                   failing cleanly.
#   --report DIR    timestamped log + one-line summary (default:
#                   SLOW_TIER_REPORTS from config.sh).
#   -- <args>       everything after `--` is passed to ctest verbatim
#                   (e.g. `-- -R Constants` to scope the run). ctest has
#                   one entry per gtest case, so -R matches suite names, and
#                   the selection also picks which GPU entries are batched
#                   into one gtest process (run_tier_tests below). ctest
#                   itself runs CTEST_JOBS (config.sh) entries at once;
#                   `-- -j 1` overrides that for one run.
#
# Exit status is the first failing phase's, so a scheduler can alert on it.
#: -- help stops here --
#
# Deliberately NOT what docker/compose.yaml's `build`/`test` services do, even
# though the commands overlap: compose owns the batch path (it brings its own
# container, tees to .log/, and is the thing CI calls), this owns the
# already-inside-a-container path. Changing the preset list means touching
# CMakePresets.json, which both read -- neither hardcodes one.
#
# The configure and build phases themselves are NOT here: they live in
# devtools/build.sh, which this sources for cpp_build, cpp_build_dir,
# cpp_require_build_tools and run_phase. `devtools/build.sh` run directly is the
# same two phases without a ctest -- one copy of the cmake invocations, used by
# both. This tier adds ctest and the advisory clang-tidy on top.
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
    *)         echo "cpp-tier: unknown flag: $1" >&2; exit 2;;
  esac
done

# --preset switches the test preset with it, or the run silently tests the
# wrong build directory: ctest --preset default reads build/, while
# --preset asan built into build-asan/.
[ "$preset_explicit" -eq 1 ] && test_preset=$preset

# `auto` (the default) picks the least-loaded live card's preset, before the
# build; the test preset follows it (gpu_auto_preset in lib.sh).
[ "$preset" = auto ] && preset=$(gpu_auto_preset)
[ "$test_preset" = auto ] && test_preset=$preset

cpp_require_build_tools || exit 1

case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac
mkdir -p "$report_dir"
stamp=$(date +%Y%m%d-%H%M%S)
rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
log="$report_dir/cpp-$stamp-$rev-$preset.log"
# run_phase (from build.sh) tees here, so configure, build, ctest and tidy all
# land in this one file.
BUILD_LOG=$log

echo "cpp-tier: $REPO_ROOT @ $rev  preset=$preset -> $log"

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

# The GPU binaries, one batch each: test/core's, then test/gtest's.
gpu_bins=(
  "$build_dir/test/core/free_breakfast_core_tests_gpu"
  "$build_dir/test/gtest/free_breakfast_tests_gpu"
)

# The test phase, inside the card the caller holds. Each ctest entry is its
# own process, and a GPU one spends about 0.45s creating the runtime's context
# before its case starts -- most of the phase once the cases are cheap. So the
# GPU entries that can share a process (devtools/split-gpu-entries.py says
# which: no launcher, not RUN_SERIAL) run as one gtest process per GPU binary,
# as in cpp-smoke.sh, and ctest runs everything else: the host entries, the
# RUN_SERIAL ones it keeps alone on the card, and every entry when a launcher
# such as compute-sanitizer wraps them. The split reads ctest's own listing, so
# the preset's filters and a `-- -R ...` pick the batches too. When the listing
# cannot be read, ctest runs every entry, as before the batch existed.
# The batches run one after another, never at once: processes on one card
# time-slice it (config.sh, by CTEST_JOBS), and each is its own watched run
# with its own `--- gpu batch` and `[gpu]` lines.
# ctest bounds each of its entries with the TIMEOUT property; each batch is
# bounded by run_gpu_batch's watchdog (GPU_BATCH_* in config.sh, issue #477).
run_tier_tests() {
  local dir trc=0 n i bin
  local -a split_args=()
  dir=$(mktemp -d)
  for i in "${!gpu_bins[@]}"; do
    split_args+=("${gpu_bins[$i]}" "$dir/filter.$i")
  done
  if ! ctest --preset "$test_preset" --show-only=json-v1 "${passthrough[@]}" 2>/dev/null |
      python3 "$REPO_ROOT/devtools/split-gpu-entries.py" "$dir/exclude" "${split_args[@]}"; then
    echo "cpp-tier: could not split the GPU entries -- ctest runs them all" | tee -a "$log"
    : >"$dir/exclude"
    for i in "${!gpu_bins[@]}"; do : >"$dir/filter.$i"; done
  fi
  # Before the passthrough, so a `-- -j 1` still wins.
  run_phase ctest ctest --preset "$test_preset" --parallel "$CTEST_JOBS" \
    --exclude-from-file "$dir/exclude" "${passthrough[@]}" || trc=$?
  for i in "${!gpu_bins[@]}"; do
    bin=${gpu_bins[$i]}
    [ -s "$dir/filter.$i" ] || continue
    n=$(awk -F: '{ print NF }' "$dir/filter.$i")
    echo "--- gpu batch: $n entries in one process of $bin ---" | tee -a "$log"
    # ctest's WORKING_DIRECTORY (the binary's own build directory), and its
    # --gtest_also_run_disabled_tests.
    (
      cd "$(dirname "$bin")" &&
        GTEST_FILTER=$(<"$dir/filter.$i") GTEST_ALSO_RUN_DISABLED_TESTS=1 \
          run_gpu_batch cpp-tier "$bin"
    ) | tee -a "$log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || trc=1
  done
  rm -rf "$dir"
  return "$trc"
}

rc=0
set +e

# Configure + build, both phases from build.sh (the one place they are spelled
# out). ctest and the advisory tidy below are this tier's own.
cpp_build "$preset" "$fresh" "$clean" "$jobs" || rc=$?

if [ "$rc" -eq 0 ] && [ "$run_tests" -eq 1 ]; then
  if has_test_preset; then
    # Always fast: skip the reference-grade/end-to-end validation cases (see
    # the header comment). devtools/cpp-slow-tier.sh is the un-skipped run.
    echo "cpp-tier: skipping reference-grade validation cases (FREE_BREAKFAST_FAST_TESTS=1)" \
      | tee -a "$log"
    export FREE_BREAKFAST_FAST_TESTS=1
    with_gpu_card "$(gpu_backend_of_preset "$test_preset")" run_tier_tests || rc=$?
    unset FREE_BREAKFAST_FAST_TESTS
  else
    echo "cpp-tier: no test preset '${test_preset:-<empty>}' -- built only" \
      | tee -a "$log"
    echo "          (CMakePresets.json defines test presets for default," \
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
    echo "cpp-tier: clang-tidy not found -- skipping --tidy" | tee -a "$log"
  else
    mapfile -t tidy_files < <(git -C "$REPO_ROOT" ls-files 'src/*.cpp')
    if [ "${#tidy_files[@]}" -eq 0 ]; then
      echo "cpp-tier: no src/*.cpp to tidy" | tee -a "$log"
    else
      # Advisory: the phase's own status is reported but never folded into rc.
      run_phase clang-tidy clang-tidy -p "$build_dir" --quiet "${tidy_files[@]}" \
        || echo "cpp-tier: clang-tidy reported findings (advisory)" | tee -a "$log"
    fi
  fi
fi

set -e

verdict=$([ "$rc" -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")
summary="cpp-tier $verdict  rev=$rev  preset=$preset  stamp=$stamp"
echo "$summary" | tee -a "$log"
echo "$summary" >> "$report_dir/summary.log"
exit "$rc"
