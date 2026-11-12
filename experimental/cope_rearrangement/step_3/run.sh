#!/usr/bin/env bash
# `surface.py run METHOD [NAME ...]` holding the CUDA card, as step 4's and
# step 6's run.sh hold it (with_gpu_card, devtools/lib.sh): one run per card.
#
#   experimental/cope_rearrangement/step_3/run.sh pc
#   experimental/cope_rearrangement/step_3/run.sh casscf
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
cd "$root"
# shellcheck source=/dev/null
source devtools/config.sh
# shellcheck source=/dev/null
source devtools/lib.sh
echo "gpu lock dir: $(gpu_lock_dir)"
with_gpu_card cuda python3 "$here/surface.py" run "$@"
