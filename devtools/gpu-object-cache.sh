#!/usr/bin/env bash
# Compiler launcher for the device-pass `.cu` sources of either backend: a
# stateless, content-addressed object cache that sits IN FRONT of sccache.
# Under CUDA it is CMAKE_CUDA_COMPILER_LAUNCHER (nvcc); under HIP, where a
# `.cu` is compiled as CXX by clang's `-x hip`, it is the CXX_COMPILER_LAUNCHER
# of each free_breakfast_add_gpu_device_library target only. It must never
# front a module unit: those write a BMI beside the .o, and a hit here restores
# only the .o.
#
# sccache already caches these object compiles, but CMake bakes each worktree's
# absolute path into the -I flags it hands the compiler, and that path ends up
# in sccache's cache key -- so identical source in a different worktree/checkout
# is a guaranteed miss (measured: eri_K_2222.cu.o, 597s in a fresh worktree
# despite an identical compile already cached elsewhere). SCCACHE_BASEDIRS
# looks like the fix but is not: it is read once when the sccache daemon
# spawns and is sticky for that daemon's whole life, so whichever worktree
# does not spawn it silently gets no benefit -- see CMakePresets.json history.
#
# This script computes its OWN key from file CONTENT instead of paths: the
# source file, every LOCAL header it #includes (resolved recursively, hashed
# by content, never by path), the compiler version, an optional caller salt,
# and the compile flags with this checkout's own absolute root stripped out.
# Identical content hits regardless of which worktree or container built it
# first, because nothing worktree-specific ever enters the key.
#
# The -I/-isystem search dirs are deliberately NOT part of the key. They exist
# only so the include scan below can FIND each header; once found, that header's
# bytes are already in the key. Keeping the raw dir paths too would double-count
# them -- and since they carry this checkout's own repo-relative layout, a .cu
# that merely moves to a different target (new -I set, byte-identical compile)
# would needlessly miss. Excluding them makes a file move a cache HIT, the way
# linking a prebuilt cuBLAS never recompiles it. Resolution stays content-
# checked: if a different -I set resolves a different header, that header's
# content hash moves and busts the key on its own.
#
# Headers that resolve OUTSIDE this checkout are not hashed; the compiler
# version stands in for them. That holds for nvcc, whose --version names the
# whole toolkit, but not under HIP: the compiler is apt's clang while the
# headers and device libraries are ROCm's, so a ROCm upgrade under the same
# clang would hit stale objects. --salt=<text> is how CMake puts that version
# (hip_VERSION) into the key.
#
# Usage (as a compiler launcher):
#   this script [--salt=<text>] <compiler> <the real compiler args...>
# Every failure mode here falls through to a real compile -- a cache-layer bug
# must never be a build failure.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

# The shared cache lives beside the one main checkout, regardless of which
# worktree invoked this -- PRIMARY_ROOT (lib.sh) is empty exactly when
# REPO_ROOT already IS that checkout, so the fallback covers both.
CACHE_ROOT=${BUILD_OBJECT_CACHE_DIR:-${PRIMARY_ROOT:-$REPO_ROOT}/.build-object-cache}

# ---------------------------------------------------------------------------
# Parse the real compiler invocation: <compiler> <args...>, e.g.
#   nvcc -forward-unknown-to-host-compiler $DEFINES $INCLUDES $FLAGS \
#        -MD -MT $out -MF $DEP_FILE -x cu -rdc=true -c $in -o $out
#   clang++ $DEFINES $INCLUDES $FLAGS -MD -MT $out -MF $DEP_FILE -o $out -c $in
# (the second is HIP, $FLAGS carrying `-x hip --offload-arch=...`).
# -I is glued (-I/path), -isystem is a separate following argument -- CMake's
# CUDA rule uses both shapes in the same command line (see CMakeLists.txt for
# a device-heavy library: -I<private dirs>, -isystem <CUDA toolkit includes>).
# ---------------------------------------------------------------------------
salt=
case "${1:-}" in
  --salt=*) salt=${1#--salt=}; shift ;;
esac
compiler=$1
shift
args=("$@")

input= output= depfile=
include_dirs=()
hashable_args=()
prev=
for a in "${args[@]}"; do
  case "$prev" in
    -c) input=$a ;;
    -o) output=$a ;;
    -MF) depfile=$a ;;
    -I | -isystem) include_dirs+=("$a") ;;
  esac
  case "$a" in
    -I?*) include_dirs+=("${a#-I}") ;;
  esac
  # -o/-MT/-MF/-c and their values only say WHERE ninja wants things written
  # (build-dir-relative, so it can differ between presets like default/debug
  # for an otherwise byte-identical compile) or repeat $input (already hashed
  # by content above) -- none of it affects the compiled bytes, so none of it
  # belongs in the key. -I/-isystem and their dirs are dropped for the reason in
  # the file header: the headers they locate are already content-hashed, so the
  # raw dir paths only add this checkout's layout and make a file move miss.
  case "$prev" in
    -c | -o | -MT | -MF | -I | -isystem) prev=$a; continue ;;
  esac
  case "$a" in
    -c | -o | -MT | -MF | -I | -isystem | -I?*) prev=$a; continue ;;
  esac
  hashable_args+=("$a")
  prev=$a
done

# sccache fronts the real compile when it is installed. It is not always: this
# launcher is wired into the bare-host `workstation` preset too (CMakePresets),
# where sccache may be absent -- so fall back to the compiler directly rather
# than let a missing OPTIONAL tool fail the build. A cache HIT below is a plain
# cp and needs neither; only a genuine miss pays the compile, sccache or not.
if command -v sccache >/dev/null 2>&1; then
  launcher=(sccache)
else
  launcher=()
fi

real_compile() {
  exec "${launcher[@]}" "$compiler" "${args[@]}"
}

# Nothing to key on -- an argv shape this script does not understand. Compile
# for real rather than guess.
[ -n "$input" ] && [ -n "$output" ] || real_compile

# ---------------------------------------------------------------------------
# Recursively resolve #include targets, quote and angle form alike. A target
# that resolves OUTSIDE this checkout (the CUDA toolkit, CCCL, libc++) is the
# compiler's problem, not ours -- the compiler-version string below already
# keys on it. A target that resolves INSIDE this checkout is hashed by content
# and re-scanned, so editing it correctly busts the cache. Note a project
# header can legitimately be included with <> syntax, which is why angle form
# is resolved against the project too rather than assumed external. Quote-form should search the including file's own directory first;
# checking it for angle-form too is harmless -- worst case one extra stat.
# ---------------------------------------------------------------------------
declare -A visited
header_hashes=()

resolve_include() {
  local from_dir=$1 name=$2 dir candidate
  candidate="$from_dir/$name"
  [ -f "$candidate" ] && { printf '%s\n' "$candidate"; return 0; }
  for dir in "${include_dirs[@]}"; do
    candidate="$dir/$name"
    [ -f "$candidate" ] && { printf '%s\n' "$candidate"; return 0; }
  done
  return 1
}

scan() {
  local file=$1 resolved dir target inc
  resolved=$(cd "$(dirname "$file")" && pwd)/$(basename "$file")
  [ -n "${visited[$resolved]:-}" ] && return 0
  visited[$resolved]=1
  case "$resolved" in
    "$REPO_ROOT"/*) ;;
    *) return 0 ;;
  esac
  header_hashes+=("$(sha256sum "$resolved" | cut -d' ' -f1)")
  dir=$(dirname "$resolved")
  while IFS= read -r inc; do
    target=$(resolve_include "$dir" "$inc") || continue
    scan "$target"
  done < <(grep -oE '#include[[:space:]]*["<][^">]+[">]' "$resolved" 2>/dev/null |
    sed -E 's/#include[[:space:]]*["<]([^">]+)[">]/\1/')
}

scan "$input" || real_compile

# ---------------------------------------------------------------------------
# The key. hashable_args already excludes -c/-o/-MT/-MF and their values (see
# the parse loop above): those only say WHERE ninja wants things written, in a
# path that can differ between build dirs (default/ vs debug/) for an
# otherwise byte-identical compile, and $input is already hashed by content.
# What is left still carries this checkout's own absolute root in every -I
# flag; stripping it (not the shared PRIMARY_ROOT -- see the file header)
# reduces every checkout's compile line to the same repo-relative text,
# exactly the normalisation SCCACHE_BASEDIRS promises and cannot reliably
# deliver here.
# ---------------------------------------------------------------------------
compiler_id=$("$compiler" --version 2>&1 | tr '\n' ' ') || compiler_id="unknown"
normalized_args=${hashable_args[*]//$REPO_ROOT/}
key_input=$(
  sha256sum "$input" | cut -d' ' -f1
  printf '%s\n' "${header_hashes[@]}" | sort -u
  printf 'compiler:%s\n' "$compiler_id"
  printf 'salt:%s\n' "$salt"
  printf 'args:%s\n' "$normalized_args"
)
key=$(printf '%s' "$key_input" | sha256sum | cut -d' ' -f1)
shard=${key:0:2}
cached="$CACHE_ROOT/$shard/$key.o"

mkdir -p "$(dirname "$output")"

if [ -f "$cached" ] && cp "$cached" "$output"; then
  :
else
  "${launcher[@]}" "$compiler" "${args[@]}"
  # Storing is best-effort. CACHE_ROOT can be unwritable -- a worktree's
  # container that does not mount the main checkout sees it as a root-owned
  # mount point -- and the compile above has already succeeded by now.
  tmp="$cached.tmp.$$"
  { mkdir -p "$(dirname "$cached")" && cp "$output" "$tmp" &&
    mv -f "$tmp" "$cached"; } 2>/dev/null || rm -f "$tmp" 2>/dev/null || true
fi

# Always (re)written, hit or miss: ninja needs this object's real dependencies
# for its NEXT incremental decision, and we already have the exact list from
# the scan above -- cheaper and more direct than caching nvcc's own -MD output.
if [ -n "$depfile" ]; then
  {
    printf '%s:' "$output"
    printf ' %s' "$input" "${!visited[@]}"
    printf '\n'
  } >"$depfile"
fi
