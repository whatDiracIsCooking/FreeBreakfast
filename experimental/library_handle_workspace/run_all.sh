#!/usr/bin/env bash
# Run every case of free_breakfast_library_handle_workspace on one backend's
# card, one process per case, holding the card the way the gates do
# (with_gpu_card, devtools/lib.sh) so no other run moves the free-memory
# readings. A case that dies prints its exit code and the rest go on.
#
#   experimental/library_handle_workspace/run_all.sh cuda   # build-experimental
#   experimental/library_handle_workspace/run_all.sh hip    # build-hip-experimental
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source devtools/lib.sh

backend=${1:?usage: run_all.sh cuda|hip}
case $backend in
  cuda) build=build-experimental ;;
  hip) build=build-hip-experimental ;;
  *) echo "run_all.sh: backend must be cuda or hip" >&2; exit 2 ;;
esac
bin=$build/experimental/library_handle_workspace/free_breakfast_library_handle_workspace
[ -x "$bin" ] || { echo "run_all.sh: $bin not built" >&2; exit 2; }

run_cases() {
  local routine mode
  for routine in syevd potrf gesvd; do
    for mode in exact exact_guarded short short_guarded null; do
      "$bin" "$routine" "$mode" 2>&1 || echo "routine=$routine mode=$mode exit=$?"
    done
  done
  for routine in dgemm_square dgemm_splitk; do
    for mode in default user tiny reset; do
      "$bin" "$routine" "$mode" 2>&1 || echo "routine=$routine mode=$mode exit=$?"
    done
  done
}

with_gpu_card "$backend" run_cases
