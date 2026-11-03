#!/usr/bin/env bash
# Seed a cloud GPU VM's compiler caches from this machine's, so its first build
# is warm rather than cold. Run it on the workstation, against a VM that
# gpu-vm-setup.sh has already cloned onto:
#
#   devtools/vm-cache-push.sh <ssh-dest> [<remote-root>]
#   devtools/vm-cache-push.sh --no-sccache <ssh-dest> [<remote-root>]
#
# <ssh-dest> is anything ssh and rsync accept (`ubuntu@1.2.3.4`, a Host alias).
# <remote-root> is the VM's primary checkout, ~/fb/main by default. Run it
# before `devcontainer.sh up` there, or `sccache --stop-server` in the VM's
# container afterwards. docs/gpu-vm.md, "Seeding the compiler caches", has
# which hits to expect.
#: --- end of `usage` output; what follows is for a reader of the file --------
#
# TWO STORES, AND THEY TRAVEL DIFFERENTLY:
#
#   .build-object-cache   devtools/gpu-object-cache.sh's, the device `.cu`
#                         objects -- the slow half of a cold build, and a few
#                         hundred MB. Keyed on content, the nvcc version and
#                         the salt (host processor, and the card's arch under
#                         `native`; root CMakeLists.txt), never on a path. So it
#                         always goes, and it hits wherever the VM's processor,
#                         CUDA toolkit and card arch match the ones that filled
#                         it: a 3080 store serves an A10 (both 86) but not an
#                         H100 (90).
#   .sccache              everything else: host C++ and the module units. Tens
#                         of GB, and sccache keys on the absolute -I paths CMake
#                         passes, so it hits only when the VM's checkout sits at
#                         THIS checkout's absolute path. Elsewhere it would be
#                         gigabytes of guaranteed misses, so it is skipped and
#                         the reason printed. --no-sccache skips it anyway.
#
# Neither store is pruned on the VM (no --delete): this only adds entries.
# gpu-locks/ under .sccache is the VM's own state and is never copied.
#
# A different processor (a GH200 against an x86-64 workstation) is refused:
# nothing in either store can hit there.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }
say() { echo "vm-cache-push: $*"; }
die() { echo "vm-cache-push: error: $*" >&2; exit 1; }

sccache=1
args=()
while [ $# -gt 0 ]; do
  case $1 in
    --no-sccache) sccache= ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "vm-cache-push: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    *) args+=("$1") ;;
  esac
  shift
done
if [ "${#args[@]}" -lt 1 ] || [ "${#args[@]}" -gt 2 ]; then
  usage >&2
  exit 2
fi
dest=${args[0]}
remote_root=${args[1]:-'~/fb/main'}

command -v rsync >/dev/null 2>&1 ||
  die "no rsync here -- run this on the workstation host, not in the container"

root=${PRIMARY_ROOT:-$REPO_ROOT}
object_cache=${FREE_BREAKFAST_OBJECT_CACHE_DIR:-$root/.build-object-cache}
sccache_dir=${FREE_BREAKFAST_SCCACHE_DIR:-$root/.sccache}

# One round trip for everything the decisions need. The remote path is
# resolved there (~ is the VM user's home), and need not exist yet.
# shellcheck disable=SC2029  # $remote_root is meant to expand on the VM
probe=$(ssh "$dest" "uname -m; command -v rsync >/dev/null && echo rsync; \
  realpath -m $remote_root") || die "cannot reach $dest over ssh"
remote_arch=$(sed -n 1p <<<"$probe")
remote_rsync=$(sed -n 2p <<<"$probe")
remote_abs=$(sed -n 3p <<<"$probe")

[ "$remote_rsync" = rsync ] || die "$dest has no rsync (sudo apt-get install rsync)"
[ "$remote_arch" = "$(uname -m)" ] ||
  die "$dest is $remote_arch, this is $(uname -m): nothing in either store can hit there"

push() {
  local from=$1 to=$2
  shift 2
  # shellcheck disable=SC2029  # $to is resolved here, on purpose
  ssh "$dest" "mkdir -p '$to'"
  rsync -a --info=stats1 "$@" "$from/" "$dest:$to/"
}

if [ -d "$object_cache" ]; then
  say "GPU object cache: $(du -sh "$object_cache" | cut -f1) -> $dest:$remote_abs/.build-object-cache"
  push "$object_cache" "$remote_abs/.build-object-cache"
else
  say "no GPU object cache at $object_cache; nothing to send"
fi

if [ -z "$sccache" ]; then
  say "sccache: skipped (--no-sccache)"
elif [ "$remote_abs" != "$root" ]; then
  say "sccache: skipped -- its keys carry absolute paths, and the VM's checkout"
  say "  is $remote_abs, not $root. Clone it at $root on the VM for host hits."
elif [ ! -d "$sccache_dir" ]; then
  say "sccache: no store at $sccache_dir; nothing to send"
else
  say "sccache: $(du -sh "$sccache_dir" | cut -f1) -> $dest:$remote_abs/.sccache"
  push "$sccache_dir" "$remote_abs/.sccache" --exclude=/gpu-locks/
fi
