#!/usr/bin/env bash
# `energies.py run` (or, named first, another step 4 script's `run`, such as
# stationary.py's) holding the CUDA card, as the gates hold it (with_gpu_card,
# devtools/lib.sh): one run per card, so another session's job cannot share it.
#
#   experimental/cope_rearrangement/step_4/run.sh [R ...]
#   experimental/cope_rearrangement/step_4/run.sh stationary.py NAME ...
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
cd "$root"
# shellcheck source=/dev/null
source devtools/config.sh
# shellcheck source=/dev/null
source devtools/lib.sh
echo "gpu lock dir: $(gpu_lock_dir)"
script=energies.py
if [[ ${1:-} == *.py ]]; then
  script=$1
  shift
fi
with_gpu_card cuda python3 "$here/$script" run "$@"
