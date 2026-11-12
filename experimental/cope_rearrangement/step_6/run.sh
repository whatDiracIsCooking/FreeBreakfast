#!/usr/bin/env bash
# `experiment.py run NAME ...` holding the CUDA card, as step 4's run.sh and
# the gates hold it (with_gpu_card, devtools/lib.sh): one run per card, so
# another session's job cannot share it, nor its memory samples.
#
#   experimental/cope_rearrangement/step_6/run.sh NAME ...
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
cd "$root"
# shellcheck source=/dev/null
source devtools/config.sh
# shellcheck source=/dev/null
source devtools/lib.sh
# At most 8 CPU threads for the binary's host side (the box's share).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
echo "gpu lock dir: $(gpu_lock_dir)"
with_gpu_card cuda python3 "$here/experiment.py" run "$@"
