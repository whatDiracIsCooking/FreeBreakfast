#!/usr/bin/env bash
# Compile each generated kernel ON ITS OWN, under a per-file timeout, and
# report the wall time of every one.
#
#   devtools/kernel-compile-watchdog.sh [--preset P] [--limit SECONDS]
#                                       [-j N] [--keep] [REGEX ...]
#
# WHY THIS EXISTS. The emitted kernels under src/integrals/*/kernels/ are one
# translation unit per angular-momentum class, and their cost is not linear in
# the class: an unrolled high class is hundreds of KB of straight-line
# arithmetic, and the back end's time in it grows faster than its text. One
# such file can hold the whole build while ninja's other jobs finish and idle,
# and `cmake --build` reports nothing until it is done -- so the symptom is a
# build that looks hung, with no indication of which file or whether it will
# ever land. That has to be caught per file, which is what this does.
#
# It is also the check behind the aux ladder's limit: raising AUX_MAX_L in
# tools/emit/df_kernels.py adds classes whose compile time nobody has
# measured, and the answer to "can this ladder be built" is this script's
# table, not a guess from the source size -- which does not even order the
# times, since a rolled low-stack kernel can be a third the text of an
# unrolled one and take longer, its loops being where the optimiser works.
#
# WHAT IT DOES. For each matching object: delete it, then run ninja on that
# object alone under `timeout --kill-after`, and time it. Deleting first is
# what makes the number real -- an up-to-date object compiles in no time and
# would report a pass having compiled nothing. Nothing else is built: the
# configure before the loop regenerates build.ninja, so each measured
# invocation is one compiler process.
#
# WHAT IT DOES NOT DO. It does not link, run, or check a result -- a kernel
# that compiles here can still be wrong on the card, which is test/gtest/'s
# job. And it measures ONE backend, the preset's: nvcc and clang `-x hip`
# differ by more than a constant on these files, so a ladder that fits under
# CUDA is not thereby known to fit under HIP. Run it again with --preset hip,
# or see devtools/cross-backend-check.sh for the compile-only cross check.
#
# Flags:
#   --preset P      configure/build preset (default: CMAKE_PRESET from
#                   devtools/config.sh; `auto` resolves as elsewhere).
#   --limit N       seconds one kernel may take (default
#                   KERNEL_COMPILE_LIMIT, 300). A file that goes over is
#                   killed and reported TIMEOUT, and the run FAILs.
#   -j N            compile N at a time instead of one (default 1). Faster
#                   over a long ladder, but the times are then inflated by
#                   contention and only the timeout still means anything --
#                   so -j is for sweeping, -j 1 for a number worth quoting.
#   --keep          leave each object in place instead of deleting it first,
#                   which makes the run a no-op where it is up to date.
#   REGEX ...       extended regexes against the object path; an object
#                   matching any of them is measured. With none, every object
#                   under a `kernels/` directory is.
#
# Exit status: 0 every kernel compiled within the limit, 1 one or more went
# over or failed to compile, 2 the check could not run at all (no toolchain,
# failed configure, or no object matched) -- never a silent skip.
#: -- help stops here --
#
# Deliberately not a hook and not part of cpp-tier.sh: a serial sweep of the
# DF ladder is minutes of compiler time, which is the slow tier's budget and
# not the commit gate's. Run it when you change what is emitted -- after
# tools/emit/*.py, or after bumping deps/FelizJueves -- narrowed to the
# family you touched:
#
#   devtools/kernel-compile-watchdog.sh 'density_fitting/.*kernels/'
set -euo pipefail

# shellcheck source=./lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

# cpp_build_dir and cpp_require_build_tools, so the build directory and the
# missing-toolchain message are the ones every other entry point uses.
# shellcheck source=./build.sh
. "$REPO_ROOT/devtools/build.sh"

preset=$CMAKE_PRESET
limit=${KERNEL_COMPILE_LIMIT:-300}
jobs=1
keep=0
patterns=()

while [ $# -gt 0 ]; do
  case "$1" in
    --preset)  preset=$2; shift 2;;
    --limit)   limit=$2; shift 2;;
    -j|--jobs) jobs=$2; shift 2;;
    --keep)    keep=1; shift;;
    -h|--help) usage; exit 0;;
    -*)        echo "kernel-compile-watchdog: unknown flag: $1" >&2; exit 2;;
    *)         patterns+=("$1"); shift;;
  esac
done

[ "${#patterns[@]}" -eq 0 ] && patterns=('/kernels/')
[ "$preset" = auto ] && preset=$(gpu_auto_preset)
cpp_require_build_tools || exit 2

build_dir=$(cpp_build_dir "$preset")
echo "kernel-compile-watchdog: preset=$preset limit=${limit}s -j$jobs -> $build_dir"
cmake --preset "$preset" >/dev/null || {
  echo "kernel-compile-watchdog: configure failed -- rerun it to see why" >&2
  exit 2
}

# Every .cu object ninja knows, narrowed to the patterns. `-t targets all`
# prints "<path>: <rule>", the path relative to the build directory.
pattern_re=$(IFS='|'; echo "${patterns[*]}")
mapfile -t objects < <(
  ninja -C "$build_dir" -t targets all |
    sed -n 's/\.cu\.o:.*/.cu.o/p' |
    grep -E -- "$pattern_re" |
    sort -u
)

if [ "${#objects[@]}" -eq 0 ]; then
  echo "kernel-compile-watchdog: no .cu object matched $pattern_re" >&2
  exit 2
fi

echo "kernel-compile-watchdog: ${#objects[@]} kernels"

results=$(mktemp)
trap 'rm -f "$results" "$results".*' EXIT

# One object: delete it (so the compile is real), compile it under the
# timeout, and append "<seconds> <status> <object>" to the results file.
# SIGKILL follows 10s after SIGTERM, because a back end wedged in an
# optimiser pass does not always take the first one.
compile_one() {
  local obj=$1 start end secs status=ok rc=0
  [ "$keep" -eq 1 ] || rm -f "$build_dir/$obj"
  start=${EPOCHREALTIME/,/.}
  timeout --kill-after=10 "$limit" ninja -C "$build_dir" "$obj" \
    >"$results.$$.log" 2>&1 || rc=$?
  end=${EPOCHREALTIME/,/.}
  secs=$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.1f", b - a }')
  case "$rc" in
    0)       ;;
    124|137) status=TIMEOUT;;
    *)       status=FAILED;;
  esac
  [ "$status" = ok ] || { echo "--- $obj: $status ---"; cat "$results.$$.log"; }
  rm -f "$results.$$.log"
  printf '%s %s %s\n' "$secs" "$status" "$obj" >>"$results"
  printf '  %7ss  %-8s %s\n' "$secs" "$status" "${obj##*/}"
}

if [ "$jobs" -gt 1 ]; then
  export build_dir limit keep results
  export -f compile_one
  printf '%s\n' "${objects[@]}" |
    xargs -P "$jobs" -I{} bash -c 'compile_one "$1"' _ {}
else
  for obj in "${objects[@]}"; do compile_one "$obj"; done
fi

over=$(awk '$2 != "ok"' "$results" | wc -l)
total=$(awk '{ s += $1 } END { printf "%.0f", s }' "$results")
worst=$(sort -gr "$results" | head -1)

echo "--- slowest 10 ---"
sort -gr "$results" | head -10 |
  while read -r secs status obj; do
    printf '  %7ss  %-8s %s\n' "$secs" "$status" "$obj"
  done

if [ "$over" -gt 0 ]; then
  echo "kernel-compile-watchdog: FAIL -- $over of ${#objects[@]} over ${limit}s" \
    "or broken (preset=$preset, ${total}s total)"
  awk '$2 != "ok" { printf "  %s %s %ss\n", $2, $3, $1 }' "$results"
  exit 1
fi

echo "kernel-compile-watchdog: PASS -- ${#objects[@]} kernels, all under" \
  "${limit}s (preset=$preset, ${total}s total, worst ${worst%% *}s," \
  "${worst##* })"
