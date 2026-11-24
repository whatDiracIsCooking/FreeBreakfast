#!/usr/bin/env bash
# Build handle-ring for one backend and run every case on that backend's card,
# one process per case, holding the card the way the gates do (with_gpu_card,
# devtools/lib.sh) so no other run moves the free-memory readings. A case that
# dies prints its exit code and the rest go on.
#
#   experimental/rust/handle_ring/run_all.sh cuda   # links rust/build-cuda
#   experimental/rust/handle_ring/run_all.sh hip    # links rust/build-hip
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
cd "$root"
source devtools/lib.sh

backend=${1:?usage: run_all.sh cuda|hip}
case $backend in
  cuda | hip) ;;
  *) echo "run_all.sh: backend must be cuda or hip" >&2; exit 2 ;;
esac
export FB_RUST_BUILD_DIR=$root/rust/build-$backend
export CARGO_TARGET_DIR=$here/target-$backend
cargo build --release --manifest-path "$here/Cargo.toml" || exit 2
bin=$CARGO_TARGET_DIR/release/handle-ring

run_cases() {
  local routine mode
  for routine in handles-blas handles-solver; do
    "$bin" "$routine" 4 2>&1 || echo "routine=$routine count=4 exit=$?"
  done
  for routine in dgemm_square dgemm_splitk; do
    for mode in stream_then_ws ws_then_stream stream_only \
      stream_then_ws_tiny ws_then_stream_tiny stream_only_tiny; do
      "$bin" ring "$routine" "$mode" 2>&1 || echo "routine=$routine mode=$mode exit=$?"
    done
  done
}

with_gpu_card "$backend" run_cases
