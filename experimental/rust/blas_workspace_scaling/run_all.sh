#!/usr/bin/env bash
# Build blas-workspace-scaling for one backend and run its shapes on that
# backend's card, one process per shape, holding the card the way the gates do
# (with_gpu_card, devtools/lib.sh) so no other run moves the free-memory
# readings. A shape that dies prints its exit code and the rest go on.
#
#   experimental/rust/blas_workspace_scaling/run_all.sh cuda   # rust/build-cuda
#   experimental/rust/blas_workspace_scaling/run_all.sh hip    # rust/build-hip
#   ... run_all.sh hip "right 1024 4096"                       # other shapes
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
cd "$root"
source devtools/lib.sh

backend=${1:?usage: run_all.sh cuda|hip [\"side m n\" ...]}
shift
case $backend in
  cuda) shapes=("left 4096 4096" "left 4100 4096" "left 16400 1024") ;;
  hip)
    shapes=("left 4096 4096" "left 4100 4096" "left 4096 1024"
      "left 16384 1024" "left 16400 1024")
    ;;
  *) echo "run_all.sh: backend must be cuda or hip" >&2; exit 2 ;;
esac
[ $# -gt 0 ] && shapes=("$@")
export FB_RUST_BUILD_DIR=$root/rust/build-$backend
export CARGO_TARGET_DIR=$here/target-$backend
cargo build --release --manifest-path "$here/Cargo.toml" || exit 2
bin=$CARGO_TARGET_DIR/release/blas-workspace-scaling

run_shapes() {
  local shape
  for shape in "${shapes[@]}"; do
    # shellcheck disable=SC2086  # a shape is three words
    "$bin" $shape 2>&1 || echo "shape=\"$shape\" exit=$?"
  done
}

with_gpu_card "$backend" run_shapes
