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
# WHAT IT DOES. For each matching object: delete it, then run that object's
# compile command from compile_commands.json under `timeout --kill-after`, and
# time it. Deleting first is what makes the number real -- an up-to-date
# object compiles in no time and would report a pass having compiled nothing.
#
# The command is run DIRECTLY, not through ninja, because ninja would go
# through the compiler launchers -- sccache, and devtools/gpu-object-cache.sh
# in front of it for a `.cu` -- and with either one warm a deleted object comes
# back as a ~0.3s cache hit, which is the opposite of a measurement (#838).
# compile_commands.json carries the bare compiler line, no launcher, so no
# cache is read, none is written, and the shared ones are left exactly as they
# were. (SCCACHE_RECACHE plus an empty BUILD_OBJECT_CACHE_DIR would also miss,
# but it depends on the sccache daemon honouring a client variable, still
# times the object cache's include scan, and rewrites the shared sccache
# entry.) Each object's prerequisites -- ninja's implicit and order-only
# inputs -- are built once through ninja before the loop, untimed. The object
# is then newer than ninja's record of it, so the next build redoes it once,
# through the caches.
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
#   --keep          leave each object in place and build it through ninja,
#                   launchers and caches included: a no-op where it is up to
#                   date, a cache hit where a cache holds it. Not a timing.
#   --summarize F   print the summary of an existing results file F
#                   ("<seconds> <status> <object>" per line) and exit with
#                   its status, compiling nothing. What the tests drive.
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
summarize=
patterns=()

while [ $# -gt 0 ]; do
  case "$1" in
    --preset)    preset=$2; shift 2;;
    --limit)     limit=$2; shift 2;;
    -j|--jobs)   jobs=$2; shift 2;;
    --keep)      keep=1; shift;;
    --summarize) summarize=$2; shift 2;;
    -h|--help)   usage; exit 0;;
    -*)          echo "kernel-compile-watchdog: unknown flag: $1" >&2; exit 2;;
    *)           patterns+=("$1"); shift;;
  esac
done

# The table and the verdict, from a results file of "<seconds> <status>
# <object>" lines. Sorted ONCE into a file and read back with awk, never
# `sort | head`: under pipefail, head closing the pipe early kills sort with
# SIGPIPE once the list outgrows a pipe buffer, and the script then exited
# 141 with no summary at all (#838).
summarize() {
  local results=$1 sorted n over total worst
  sorted=$(mktemp)
  sort -gr "$results" >"$sorted"
  n=$(wc -l <"$sorted")
  over=$(awk '$2 != "ok"' "$sorted" | wc -l)
  total=$(awk '{ s += $1 } END { printf "%.0f", s }' "$sorted")
  worst=$(awk 'NR == 1' "$sorted")

  echo "--- slowest 10 ---"
  awk 'NR <= 10 { printf "  %7ss  %-8s %s\n", $1, $2, $3 }' "$sorted"
  rm -f "$sorted"

  if [ "$over" -gt 0 ]; then
    echo "kernel-compile-watchdog: FAIL -- $over of $n over ${limit}s" \
      "or broken (preset=$preset, ${total}s total)"
    awk '$2 != "ok" { printf "  %s %s %ss\n", $2, $3, $1 }' "$results"
    return 1
  fi
  echo "kernel-compile-watchdog: PASS -- $n kernels, all under" \
    "${limit}s (preset=$preset, ${total}s total, worst ${worst%% *}s," \
    "${worst##* })"
}

if [ -n "$summarize" ]; then
  [ -s "$summarize" ] || {
    echo "kernel-compile-watchdog: no results in $summarize" >&2
    exit 2
  }
  rc=0
  summarize "$summarize" || rc=$?
  exit "$rc"
fi

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
commands=$results.commands

if [ "$keep" -eq 1 ]; then
  echo "kernel-compile-watchdog: --keep: built through ninja and its" \
    "launchers; a cache hit is timed as one"
else
  # "<object>\t<directory>\t<command>" for every matched object, the bare
  # compiler line CMake records -- the launchers are not in it. A compile
  # line holds no tab or newline, so the fields need no escaping.
  printf '%s\n' "${objects[@]}" |
    python3 -I -c '
import json, os, sys
build, wanted = sys.argv[1], set(sys.stdin.read().split())
for e in json.load(open(os.path.join(build, "compile_commands.json"))):
    out = os.path.relpath(os.path.join(e["directory"], e.get("output", "")), build)
    if out in wanted:
        print(out, e["directory"], e["command"], sep="\t")
' "$build_dir" >"$commands" || {
    echo "kernel-compile-watchdog: cannot read" \
      "$build_dir/compile_commands.json" >&2
    exit 2
  }
  # A launcher in the recorded line would put the caches back in the loop.
  if awk -F'\t' '$3 ~ /(^|\/)(sccache|ccache|gpu-object-cache\.sh)[ ]/' \
    "$commands" | grep -q .; then
    echo "kernel-compile-watchdog: compile_commands.json names a compiler" \
      "launcher; refusing to time through a cache" >&2
    exit 2
  fi
  echo "kernel-compile-watchdog: cold: each command run from" \
    "compile_commands.json, bypassing sccache and the object cache" \
    "(${BUILD_OBJECT_CACHE_DIR:-${PRIMARY_ROOT:-$REPO_ROOT}/.build-object-cache})"

  # What ninja would build first for each object: its implicit (`|`) and
  # order-only (`||`) inputs. Built once, untimed, through ninja as usual.
  mapfile -t prereqs < <(
    for obj in "${objects[@]}"; do
      ninja -C "$build_dir" -t query "$obj"
    done | awk '/^  input:/ { f = 1; next } /^  [a-z]/ { f = 0 }
                f && ($1 == "|" || $1 == "||") { print $2 }' | sort -u
  )
  if [ "${#prereqs[@]}" -gt 0 ]; then
    echo "kernel-compile-watchdog: building ${#prereqs[@]} prerequisites" \
      "(untimed)"
    ninja -C "$build_dir" "${prereqs[@]}" >"$results.prereqs.log" 2>&1 || {
      cat "$results.prereqs.log"
      echo "kernel-compile-watchdog: prerequisites failed to build" >&2
      exit 2
    }
  fi
fi

# One object: delete it (so the compile is real), compile it under the
# timeout, and append "<seconds> <status> <object>" to the results file.
# SIGKILL follows 10s after SIGTERM, because a back end wedged in an
# optimiser pass does not always take the first one.
compile_one() {
  local obj=$1 start end secs status=ok rc=0 dir cmd
  if [ "$keep" -eq 1 ]; then
    start=${EPOCHREALTIME/,/.}
    timeout --kill-after=10 "$limit" ninja -C "$build_dir" "$obj" \
      >"$results.$$.log" 2>&1 || rc=$?
    end=${EPOCHREALTIME/,/.}
  else
    rm -f "$build_dir/$obj"
    IFS=$'\t' read -r _ dir cmd < <(
      awk -F'\t' -v o="$obj" '$1 == o' "$commands"
    ) || true
    start=${EPOCHREALTIME/,/.}
    if [ -z "${cmd:-}" ]; then
      echo "no entry in compile_commands.json" >"$results.$$.log"
      rc=1
    else
      (cd "$dir" && exec timeout --kill-after=10 "$limit" bash -c "$cmd") \
        >"$results.$$.log" 2>&1 || rc=$?
    fi
    end=${EPOCHREALTIME/,/.}
    # Its -MD depfile is ninja's to read after ninja's own compile, not ours.
    rm -f "$build_dir/$obj.d"
  fi
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
  export build_dir limit keep results commands
  export -f compile_one
  # shellcheck disable=SC2016 # "$1" is the inner shell's argument
  printf '%s\n' "${objects[@]}" |
    xargs -P "$jobs" -I{} bash -c 'compile_one "$1"' _ {}
else
  for obj in "${objects[@]}"; do compile_one "$obj"; done
fi

rc=0
summarize "$results" || rc=$?
exit "$rc"
