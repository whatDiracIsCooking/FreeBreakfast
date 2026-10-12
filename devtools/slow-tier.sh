#!/usr/bin/env bash
# The SLOW-TIER verification job -- deliberately OFF the per-commit path.
#
#   devtools/slow-tier.sh [--skip-huge] [--report DIR] [-j N] [-- <pytest args>]
#
# The push gate and the iteration loop run the fast tier on purpose: the slow
# tier is where the multi-minute compiles, the device-heavy cases and anything
# with a hard memory floor live, and a gate that costs hours is one nobody
# runs. But nothing expensive should ship *un*verified either. This is that
# missing job: run the slow tier on the CURRENT checkout (intended: a
# freshly-synced `main`, after a batch of merges) and report pass/fail plus the
# skip reasons.
#
# Meant to be driven by a scheduled job or a nightly, not by hand and not by a
# per-PR gate -- it verifies what already merged, it does not block a merge.
#
# Flags:
#   --skip-huge     deselect the cases listed in SLOW_HUGE_CASES (devtools/
#                   config.sh) -- the ones with a hard memory floor that OOM a
#                   smaller box rather than failing cleanly. Off by default:
#                   the whole point is to verify the expensive cases.
#   --report DIR    write a timestamped log + one-line summary here
#                   (default: SLOW_TIER_REPORTS from config.sh).
#   -j N            xdist workers (default: JOBS from config.sh). Note that a
#                   high worker count beside a memory-hungry case can itself
#                   OOM -- pass --skip-huge or a small -j on a small box.
#   -- <args>       everything after `--` is passed to pytest verbatim
#                   (e.g. `-- -k slow_thing` to scope to part of the tier).
#
# Exit status is pytest's, so a scheduler can alert on it. A run where optional
# tools are missing SKIPS rather than fails -- read the `-rs` summary before
# trusting green.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

skip_huge=0
report_dir=$SLOW_TIER_REPORTS
jobs=$JOBS
declare -a passthrough=()

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-huge) skip_huge=1; shift;;
    --report)    report_dir=$2; shift 2;;
    -j|--jobs)   jobs=$2; shift 2;;
    --)          shift; passthrough=("$@"); break;;
    -h|--help)   usage_from_header "${BASH_SOURCE[0]}"; exit 0;;
    *)           echo "slow-tier: unknown flag: $1" >&2; exit 2;;
  esac
done

case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac

PYTEST=$(find_pytest) || {
  echo "slow-tier: no pytest found in ${VENV_PATHS} or on PATH -- run 'uv sync'" >&2
  exit 1
}

config_args "$SLOW_TEST_ARGS"
declare -a sel=("${CONFIG_ARGS[@]}" -rs -p no:cacheprovider)
[ -n "$jobs" ] && sel+=(-n "$jobs")

if [ "$skip_huge" -eq 1 ]; then
  # Deselecting by node id is exact; a -k filter cannot express a param that
  # contains pytest -k's own operators.
  config_lines "$SLOW_HUGE_CASES"
  if [ "${#CONFIG_LINES[@]}" -eq 0 ]; then
    echo "slow-tier: --skip-huge given but SLOW_HUGE_CASES is empty in config.sh"
  fi
  for case_id in "${CONFIG_LINES[@]}"; do
    sel+=(--deselect "$case_id")
    echo "slow-tier: deselecting $case_id"
  done
fi

mkdir -p "$report_dir"
stamp=$(date +%Y%m%d-%H%M%S)
rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
log="$report_dir/slow-$stamp-$rev.log"

echo "slow-tier: $REPO_ROOT @ $rev -> $log"
echo "slow-tier: $PYTEST ${sel[*]} ${passthrough[*]:-}"

set +e
"$PYTEST" "${sel[@]}" "${passthrough[@]}" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e

# A one-line summary at the tail of the log, greppable by a scheduler.
verdict=$([ "$rc" -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")
summary="slow-tier $verdict  rev=$rev  stamp=$stamp  skip_huge=$skip_huge"
echo "$summary" | tee -a "$log"
echo "$summary" >> "$report_dir/summary.log"
exit "$rc"
