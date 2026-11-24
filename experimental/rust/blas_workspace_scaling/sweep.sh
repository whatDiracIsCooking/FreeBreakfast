#!/usr/bin/env bash
# Build routine-sweep for one backend and run "routine n" cases on that
# backend's card, one process per case, holding the card (with_gpu_card).
#
#   experimental/rust/blas_workspace_scaling/sweep.sh hip "dtrmm 4096" "dtrmm 4100"
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
cd "$root"
source devtools/lib.sh

backend=${1:?usage: sweep.sh cuda|hip \"routine n\" ...}
shift
case $backend in
  cuda | hip) ;;
  *) echo "sweep.sh: backend must be cuda or hip" >&2; exit 2 ;;
esac
cases=("$@")
export FB_RUST_BUILD_DIR=$root/rust/build-$backend
export CARGO_TARGET_DIR=$here/target-$backend
cargo build --release --manifest-path "$here/Cargo.toml" || exit 2
bin=$CARGO_TARGET_DIR/release/routine-sweep

run_cases() {
  local c
  for c in "${cases[@]}"; do
    # shellcheck disable=SC2086  # a case is two words
    timeout 600 "$bin" $c 2>&1 || echo "case=\"$c\" exit=$?"
  done
}

with_gpu_card "$backend" run_cases
