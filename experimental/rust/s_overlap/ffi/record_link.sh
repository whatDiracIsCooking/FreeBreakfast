#!/usr/bin/env bash
# LINKER_LAUNCHER of free_breakfast_rust_s_overlap_linkprobe: record the link
# command, then run it.
#
# Usage: record_link.sh <out-file> <linker> <args...>
#
# <out-file> gets the working directory on its first line (the command's
# relative paths are relative to it), then the linker and each argument on a
# line of its own. build.rs reads it back.
set -euo pipefail
out=$1
shift
{
  pwd
  printf '%s\n' "$@"
} >"$out.tmp"
mv "$out.tmp" "$out"
exec "$@"
