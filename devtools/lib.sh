#!/usr/bin/env bash
# Shared plumbing for the devtools scripts. Sourced, never run:
#
#   ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
#   . "$ROOT/devtools/lib.sh"
#
# Sourcing this also sources devtools/config.sh, so a script gets the project's
# settings and these helpers in one line. Nothing project-specific belongs in
# here -- that is config.sh's job.

# REPO_ROOT is the checkout this file lives in, resolved from its own location
# rather than from $PWD, so every script works from any cwd and in any worktree.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# The checkout that owns the shared .git directory -- empty when REPO_ROOT is
# already it, or when git cannot say.
#
# A .claude/worktrees/ checkout has no virtualenv of its own: `uv sync` is run
# once, in the main checkout, and a lightweight worktree deliberately gets no
# container and no setup step. Without this fallback every `language: script`
# pre-commit hook fails inside such a worktree with "run 'uv sync'" -- at
# commit time, on the one kind of checkout agents work in most.
PRIMARY_ROOT=$(cd "$REPO_ROOT" && git rev-parse --path-format=absolute \
  --git-common-dir 2>/dev/null) || PRIMARY_ROOT=""
[ -n "$PRIMARY_ROOT" ] && PRIMARY_ROOT=$(dirname "$PRIMARY_ROOT")
[ "$PRIMARY_ROOT" = "$REPO_ROOT" ] && PRIMARY_ROOT=""

# shellcheck source=./config.sh
. "$REPO_ROOT/devtools/config.sh"

# Split a config value that may contain quoted arguments -- FAST_TEST_ARGS is
# `-m "not slow"`, three characters of which are quoting -- into an array.
# Plain word splitting would hand pytest `"not` and `slow"`.
#
#   config_args "$FAST_TEST_ARGS"; cmd "${CONFIG_ARGS[@]}"
config_args() {
  CONFIG_ARGS=()
  [ -n "${1:-}" ] || return 0
  eval "CONFIG_ARGS=($1)"
}

# Split a newline-separated config value (the `name:description` lists) into an
# array, dropping blank lines. Values are taken literally -- no quote handling,
# because these are descriptions rather than command lines.
#
#   config_lines "$DOCTOR_OPTIONAL_TOOLS"; for e in "${CONFIG_LINES[@]}"; ...
config_lines() {
  CONFIG_LINES=()
  local line
  while IFS= read -r line; do
    [ -n "${line// /}" ] || continue
    CONFIG_LINES+=("$line")
  done <<<"${1:-}"
}

# Print a script's leading comment block as its help text: from line 2 down to
# the first `#:` sentinel, or to the first non-comment line if there is none,
# with the leading `# ` stripped.
#
#   usage() { usage_from_header "${BASH_SOURCE[0]}"; }
#
# Hardcoded line ranges (`sed -n '2,16p'`) were what this replaced, and all
# three scripts had already lost their last line to one: a sentence added to a
# header silently truncates the help nobody re-reads. Put a `#:` line where the
# help should stop when the header continues into notes for a reader of the
# file, as devtools/worktree.sh does.
usage_from_header() {
  sed -n '2,${
    /^#:/q
    /^[^#]/q
    s/^# \{0,1\}//p
  }' "$1"
}

# The binary directory a configure preset names in CMakePresets.json, following
# `inherits`, with ${sourceDir} and ${presetName} expanded; empty when it
# cannot say. Read from the file because CMake will not tell: `cmake --preset X
# -N` lists the cache variables but, in 4.x, no "Binary directory" line --
# scraping for one came back empty for every preset, and the build/ fallback
# then sent the push gate under `hip` looking for build/'s test binaries.
preset_binary_dir() {
  python3 -I - "$REPO_ROOT" "$1" 2>/dev/null <<'EOF' || true
import json, sys
root, name = sys.argv[1], sys.argv[2]
presets = {p["name"]: p for p in
           json.load(open(f"{root}/CMakePresets.json"))["configurePresets"]}
def find(n):
    p = presets[n]
    if "binaryDir" in p:
        return p["binaryDir"]
    inh = p.get("inherits", [])
    for parent in [inh] if isinstance(inh, str) else inh:
        d = find(parent)
        if d:
            return d
    return None
d = find(name)
if d:
    print(d.replace("${sourceDir}", root).replace("${presetName}", name))
EOF
}

# Whether the build tree $1 was configured by a C++ compiler that is no longer
# the one it would get (#1333). The devcontainer's clang went from 20 to 23
# behind every existing build*/: /usr/bin/clang++ still exists, but the cache
# still names llvm-20's libc++ module sources, and the configure fails on a
# missing std.cppm without saying the compiler changed. CMake catches a
# changed CMAKE_CXX_COMPILER itself; what it misses is the same path running
# a different compiler. So: the compiler the tree recorded
# (CMakeFiles/<cmake version>/CMakeCXXCompiler.cmake) is gone, or now reports
# another version, or the libc++ module manifest the cache names is gone.
# On drift, prints one line, `<old> -> <new>`, and returns 0. Returns 1 when
# the tree matches, and when it cannot say (no cache, no record of the
# compiler, a compiler that will not report its version): a false alarm would
# throw away a working cache.
build_tree_compiler_drift() {
  local dir=$1 cache="$1/CMakeCache.txt" ver rec cxx old_v new_v json
  [ -f "$cache" ] || return 1
  ver=$(sed -n 's/^CMAKE_CACHE_\(MAJOR\|MINOR\|PATCH\)_VERSION:INTERNAL=//p' \
          "$cache" | paste -sd.)
  rec="$dir/CMakeFiles/$ver/CMakeCXXCompiler.cmake"
  [ -n "$ver" ] && [ -f "$rec" ] || return 1
  cxx=$(sed -n 's/^set(CMAKE_CXX_COMPILER "\(.*\)")$/\1/p' "$rec")
  old_v=$(sed -n 's/^set(CMAKE_CXX_COMPILER_VERSION "\(.*\)")$/\1/p' "$rec")
  [ -n "$cxx" ] || return 1
  if [ ! -x "$cxx" ]; then
    echo "$cxx $old_v -> gone ($cxx no longer exists)"
    return 0
  fi
  # gcc's -dumpversion is the major alone; clang has no -dumpfullversion.
  new_v=$("$cxx" -dumpfullversion 2>/dev/null) ||
    new_v=$("$cxx" -dumpversion 2>/dev/null) || new_v=""
  if [ -n "$new_v" ] && [ -n "$old_v" ] && [ "$new_v" != "$old_v" ]; then
    echo "$cxx $old_v -> $cxx $new_v"
    return 0
  fi
  json=$(sed -n 's/^CMAKE_CXX_STDLIB_MODULES_JSON:[A-Z]*=//p' "$cache")
  if [ -n "$json" ] && [ ! -e "$json" ]; then
    echo "$cxx $old_v -> $cxx ${new_v:-?}, and the cached $json is gone"
    return 0
  fi
  return 1
}

# --- GPU cards: one run per card, load, and the `auto` preset --------------
#
# Admission of GPU-bound commands across every worktree/session on this host.
# One run is one GPU process at a time (devtools/test.py's run_gpu_batch) plus
# `test.py pr`'s ctest -j CTEST_JOBS for the rest, but nothing in that run
# coordinates with a second session's gate running its own ctest/gtest
# binary against the same physical GPU at the same time, which is what
# actually timed out a full-tier run under cross-worktree contention.
#
# A run holds its card for the whole GPU phase, and no other run shares it
# (config.sh says why); the two cards never block each other. A card's lock is
# two files under GPU_LOCK_DIR, `<backend>.0.lock` and `<backend>.1.lock`,
# held together: branches from before 2026-10-28 let two runs share a card,
# each holding one of those, and holding both keeps such a run from joining
# this one. Drop the second once no worktree carries the old lib.sh.
# A relative GPU_LOCK_DIR resolves under the MAIN checkout (see config.sh), an
# absolute one is used as given (#1332). The default sits inside the
# .sccache directory that is already bind-mounted into every compose service
# and every worktree's devcontainer -- so this needs no new mount, and the
# kernel enforces the locks across containers. flock's lock is held against
# the open file descriptor, so it releases on process exit, including SIGKILL:
# no stale-lock file to detect or clean up.
#
# Load is the larger of the card's compute utilization and its VRAM use, in
# percent; it only ranks the cards for `auto`. What makes a run wait is VRAM
# alone (GPU_FOREIGN_VRAM_PCT), because a failed device allocation ABORTS in
# this tree (CreatePolicy, src/core/wwr/policy.cppm): a run beside a busy
# foreign job slows down, a run beside a full card crashes. A figure the
# driver tool cannot report is `?`, and the lock alone then decides.

GPU_CARD_LOCKS="0 1"

gpu_lock_dir() {
  if [[ $GPU_LOCK_DIR == /* ]]; then
    echo "$GPU_LOCK_DIR"
  else
    echo "${PRIMARY_ROOT:-$REPO_ROOT}/$GPU_LOCK_DIR"
  fi
}

# `hip` for a preset named in GPU_HIP_PRESETS, else `cuda`.
gpu_backend_of_preset() {
  local p
  for p in $GPU_HIP_PRESETS; do
    [ "$p" = "$1" ] && { echo hip; return 0; }
  done
  echo cuda
}

# Whether this machine can build AND run a backend: its device nodes exist and
# its compiler is on PATH. Device nodes rather than nvidia-smi, for the reason
# devtools/test.py's push gate gives: NVML routinely fails inside a container
# whose CUDA runtime works.
gpu_card_alive() {
  case "$1" in
    cuda) [ -e /dev/nvidiactl ] && compgen -G '/dev/nvidia[0-9]*' >/dev/null &&
            command -v nvcc >/dev/null 2>&1;;
    hip)  [ -e /dev/kfd ] && command -v hipconfig >/dev/null 2>&1;;
    *)    return 1;;
  esac
}

# The card's load in percent (see above), or `?`: the mean of GPU_LOAD_SAMPLES
# readings 0.3s apart, because one reading is a sub-second window and swings
# from 10% to 98% between test cases. Device 0 of each vendor -- this host has
# one card per backend.
gpu_card_load() {
  local i one sum=0 n=0
  for ((i = 0; i < GPU_LOAD_SAMPLES; i++)); do
    [ "$i" -gt 0 ] && sleep 0.3
    one=$(gpu_card_load_once "$1")
    [ "$one" = "?" ] && continue
    sum=$((sum + one)) n=$((n + 1))
  done
  if [ "$n" -eq 0 ]; then echo "?"; else echo $((sum / n)); fi
}

gpu_card_load_once() {
  local util mem
  IFS=, read -r util mem <<<"$(gpu_card_util_mem "$1")"
  [[ "${util:-}" =~ ^[0-9]+$ ]] || { echo "?"; return 0; }
  [[ "${mem:-}" =~ ^[0-9]+$ ]] || mem=0
  echo $((util > mem ? util : mem))
}

# The card's VRAM use in percent, or `?`. SEPARATE from gpu_card_load on
# purpose: that one is max(compute, VRAM), and compute pegs near 100% under
# any GPU job, so the combined figure says nothing about how much memory is
# left -- which is the one thing that decides whether a run can start.
gpu_card_mem() {
  local util mem
  IFS=, read -r util mem <<<"$(gpu_card_util_mem "$1")"
  if [[ "${mem:-}" =~ ^[0-9]+$ ]]; then echo "$mem"; else echo "?"; fi
}

# `<utilization %>,<VRAM %>` for a card, either field empty when the vendor
# tool could not say. Device 0 of each vendor -- this host has one card per
# backend. One place reads each tool, so load and memory cannot come from
# two different samples of the same card.
gpu_card_util_mem() {
  local out util="" mem="" used total
  case "$1" in
    cuda)
      out=$(nvidia-smi -i 0 --format=csv,noheader,nounits \
        --query-gpu=utilization.gpu,memory.used,memory.total 2>/dev/null) || out=""
      IFS=', ' read -r util used total <<<"$out"
      if [[ "$used" =~ ^[0-9]+$ && "$total" =~ ^[1-9][0-9]*$ ]]; then
        mem=$((used * 100 / total))
      fi;;
    hip)
      # `device,GPU use (%),GPU Memory Allocated (VRAM%),...` then one row.
      out=$(rocm-smi --showuse --showmemuse --csv 2>/dev/null | sed -n 2p) || out=""
      IFS=, read -r _ util mem _ <<<"$out";;
  esac
  echo "${util},${mem}"
}

# The CUDA the `cuda` image installs, as `13.0`: CUDA_VERSION when set (what
# docker/build.sh forwards), else docker/Dockerfile.cuda's ARG default -- read
# from the file, so this holds no version number of its own.
cuda_image_version() {
  local v=${CUDA_VERSION:-}
  [ -n "$v" ] ||
    v=$(sed -n 's/^ARG CUDA_VERSION=//p' "$REPO_ROOT/docker/Dockerfile.cuda" | head -1)
  echo "${v//-/.}"
}

# The newest CUDA the NVIDIA driver can run, as `13.0`, from nvidia-smi's
# banner; empty without a working nvidia-smi. The same on the host and in a
# container, since the driver is the host's either way. An image newer than
# this will not start under --gpus (see docker/Dockerfile.cuda's header).
cuda_driver_ceiling() {
  nvidia-smi 2>/dev/null |
    sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1
}

# Whether dotted version $1 is at least $2.
version_ge() {
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]
}

# Whether another run holds the card: any of its lock files is locked. Probes
# each with a non-blocking flock that is released at once.
gpu_card_held() {
  local dir i fd
  dir=$(gpu_lock_dir)
  mkdir -p "$dir"
  for i in $GPU_CARD_LOCKS; do
    exec {fd}>"$dir/$1.$i.lock"
    if flock -n "$fd"; then
      flock -u "$fd"
      exec {fd}>&-
    else
      exec {fd}>&-
      return 0
    fi
  done
  return 1
}

# Resolve CMAKE_PRESET=auto to a preset: the live card no run holds and no
# foreign job fills (VRAM under GPU_FOREIGN_VRAM_PCT), least loaded first;
# when none qualifies, the least loaded live card, which the run then waits
# for at test time. With no live card at all, GPU_AUTO_FALLBACK. Ties go to
# the earlier entry of GPU_AUTO_PRESETS. The decision and every card's state
# go to stderr, so a log says which card a run used and why.
#
# The choice is made BEFORE the build, because the build is per backend; it
# can be minutes stale by the time ctest starts, which is why with_gpu_card
# re-checks and waits.
#
#   [ "$preset" = auto ] && preset=$(gpu_auto_preset)
gpu_auto_preset() {
  local entry backend p load mem held key
  local best="" best_key="" report=""
  for entry in $GPU_AUTO_PRESETS; do
    backend=${entry%%:*} p=${entry#*:}
    gpu_card_alive "$backend" || { report+=" $backend=absent"; continue; }
    load=$(gpu_card_load "$backend")
    mem=$(gpu_card_mem "$backend")
    held=free
    command -v flock >/dev/null 2>&1 && gpu_card_held "$backend" && held=held
    report+=" $backend=load:${load}%,vram:${mem}%,$held"
    # Sort key, smallest wins: free cards first (0), then held or full ones
    # (1), by load within each.
    local l=${load/\?/0} m=${mem/\?/0}
    if [ "$held" = free ] && [ "$m" -lt "$GPU_FOREIGN_VRAM_PCT" ]; then
      key=$(printf '0-%03d' "$l")
    else
      key=$(printf '1-%03d' "$l")
    fi
    if [ -z "$best" ] || [[ "$key" < "$best_key" ]]; then
      best=$p best_key=$key
    fi
  done
  if [ -z "$best" ]; then
    echo "gpu: auto -> $GPU_AUTO_FALLBACK (no live card:$report)" >&2
    echo "$GPU_AUTO_FALLBACK"
    return 0
  fi
  echo "gpu: auto -> $best ($(gpu_backend_of_preset "$best")) [$report ]" >&2
  echo "$best"
}

# Take every lock file of a card, non-blocking. Sets GPU_HELD_FDS to what it
# got and returns 0 only when it got them all; on failure the caller drops
# what it has. A lock belongs to the open file DESCRIPTION, so a second open
# of the same file does not share it -- which is what lets gpu_card_held see
# our own lock alongside everyone else's.
gpu_take_card() {
  local dir fd i
  dir=$(gpu_lock_dir)
  mkdir -p "$dir"
  GPU_HELD_FDS=()
  for i in $GPU_CARD_LOCKS; do
    exec {fd}>"$dir/$1.$i.lock"
    flock -n "$fd" || { exec {fd}>&-; return 1; }
    GPU_HELD_FDS+=("$fd")
  done
}

gpu_drop_card() {
  local fd
  for fd in "${GPU_HELD_FDS[@]:-}"; do
    [ -n "$fd" ] || continue
    flock -u "$fd"
    exec {fd}>&-
  done
  GPU_HELD_FDS=()
}

# Run a GPU-bound command holding a whole card:
#
#   with_gpu_card "$(gpu_backend_of_preset "$test_preset")" ctest --preset ...
#
# Waits, polling every GPU_POLL_SECONDS, while another run holds the card.
# Then, holding it, waits while the card's VRAM is at or above
# GPU_FOREIGN_VRAM_PCT: with the lock held, that memory belongs to something
# that takes no lock, and it is waited out with no timeout, because starting
# into a full card only ends in an abort. Compute load never makes it wait.
with_gpu_card() {
  local backend=$1
  shift
  if ! command -v flock >/dev/null 2>&1; then
    echo "devtools: flock not found -- running without cross-session GPU admission" >&2
    "$@"
    return $?
  fi

  local dir mem said=""
  dir=$(gpu_lock_dir)
  until gpu_take_card "$backend"; do
    gpu_drop_card
    [ -n "$said" ] || echo "devtools: waiting for the $backend card -- another run holds it ($dir)" >&2
    said=held
    sleep "$GPU_POLL_SECONDS"
  done
  said=""
  while mem=$(gpu_card_mem "$backend") && [ "$mem" != "?" ] &&
        [ "$mem" -ge "$GPU_FOREIGN_VRAM_PCT" ]; do
    [ -n "$said" ] || echo "devtools: waiting -- $backend VRAM ${mem}% >=" \
      "${GPU_FOREIGN_VRAM_PCT}% held by a job outside these locks" >&2
    said=mem
    sleep "$GPU_POLL_SECONDS"
  done

  echo "devtools: $backend card held (VRAM ${mem}%)" >&2
  local rc
  "$@" && rc=0 || rc=$?
  gpu_drop_card
  return "$rc"
}

# The xdist worker flag, or nothing when JOBS is empty (a suite without
# pytest-xdist installed). Used as `${XDIST[@]}`.
xdist_args() {
  XDIST=()
  [ -n "${JOBS:-}" ] && XDIST=(-n "$JOBS")
  return 0
}

# Locate a tool without assuming it is on PATH: each VENV_PATHS entry's bin/ in
# turn (relative ones resolved against the repo root), then PATH. Echoes the
# path; returns 1 with nothing echoed when there is none, so callers can report
# the project's own setup command rather than a generic error.
#
# A bare `command -v` is wrong for anything `uv sync` installs. On the host
# those live in .venv/bin, which is NOT on PATH unless you activated the venv,
# so doctor.sh reported cmake-format missing on a machine where it worked
# perfectly; inside the image /opt/venv/bin IS on PATH, so the same check
# passed there. Opposite verdicts for the same working tool, neither of them
# about whether the tool works.
find_tool() {
  local v p r
  for r in "$REPO_ROOT" ${PRIMARY_ROOT:+"$PRIMARY_ROOT"}; do
    for v in $VENV_PATHS; do
      case "$v" in /*) p="$v" ;; *) p="$r/$v" ;; esac
      [ -x "$p/bin/$1" ] && { echo "$p/bin/$1"; return 0; }
    done
  done
  command -v "$1" 2>/dev/null && return 0
  return 1
}

# Resolve a group NAME to its numeric gid on THIS host, echoing the gid or
# nothing. The `|| true` is load-bearing: getent exits 2 when the key is not
# found, and under `set -euo pipefail` that 2 would kill the CALLER here --
# before its own diagnostic, and with an exit code indistinguishable from a
# deliberate "cannot run". The ROCm-on-a-real-card path needs the numeric HOST
# gid of the device groups (render, video), so the trap lives here once rather
# than copied into each; see devcontainer.sh.
host_gid() {
  getent group "$1" 2>/dev/null | cut -d: -f3 || true
}

# The pytest the project's scripts should drive. Thin wrapper kept because the
# callers read better for it, and because "which pytest" is a question worth
# naming.
find_pytest() { find_tool pytest; }

# Same search, for the venv itself -- doctor.sh reports which one is in use.
# Echoes the directory, or nothing.
find_venv() {
  local v p r
  for r in "$REPO_ROOT" ${PRIMARY_ROOT:+"$PRIMARY_ROOT"}; do
    for v in $VENV_PATHS; do
      case "$v" in /*) p="$v" ;; *) p="$r/$v" ;; esac
      [ -x "$p/bin/python" ] && { echo "$p"; return 0; }
    done
  done
  return 1
}

# Initialise every submodule that has never been initialised, and touch nothing
# else. Returns git's status, or 0 when there was nothing to do.
#
#   init_submodules <checkout>
#
# `git worktree add` gives a new checkout the gitlink and an EMPTY directory,
# and a .claude/worktrees/ checkout gets no setup step -- so the C++ entry
# points call this before configuring instead of relying on whoever made the
# worktree to remember. Lazy on purpose: initialising in every worktree would
# clone into docs- and Python-only checkouts too, and a worktree with an
# initialised submodule can only be removed with --force.
#
# Only the `-` (uninitialised) entries of `git submodule status` are passed on.
# A bare `git submodule update --init` would also check an INITIALISED
# submodule back out at the pinned commit, detaching whatever branch or commits
# someone has in it -- the state a build must not disturb. No-op without
# .gitmodules, so this survives being copied into a repo that has none.
init_submodules() {
  local root=$1
  local -a paths
  [ -f "$root/.gitmodules" ] || return 0
  mapfile -t paths < <(git -C "$root" submodule status 2>/dev/null |
    sed -n 's/^-[0-9a-f]* \([^ ]*\).*/\1/p')
  [ "${#paths[@]}" -gt 0 ] || return 0
  echo "initialising submodules: ${paths[*]}"
  git -C "$root" submodule update --init --recursive -- "${paths[@]}"
}

# The revision a run tested, for its log name, header and PASS line.
#
#   tree_rev "$REPO_ROOT"   # sets TREE_REV, TREE_DIFF_HASH (and TREE_ROOT)
#   tree_rev_header build "$log"
#
# TREE_REV is HEAD's short sha -- on a clean tree exactly the `rev=` every
# tier always printed -- with `+dirty` appended when the tested tree differs
# from HEAD (#694): a tier run before the commit would otherwise name the base
# commit for code that is not in it. TREE_DIFF_HASH is then the first 12 hex
# of the sha256 of `git diff HEAD`, so two dirty runs on one base can be told
# apart, and is empty on a clean tree.
#
# Dirty means a TRACKED change, staged or not, in the tree or in a submodule:
# a moved submodule pointer, or edited content inside one (deps/FelizJueves
# is often both in the main checkout), counts, and --submodule=diff puts the
# submodule's own content diff into the hash rather than only the pointer.
# UNTRACKED files do not count, in the tree or in a submodule: build trees,
# logs and scratch are untracked, and a stamp that read +dirty on every
# checkout would say nothing. A new source file only reaches the build through
# a tracked CMakeLists.txt edit, which does mark the tree; tree_rev_header
# still lists the untracked files, so a reader can see them. "unknown" (no
# git) is never marked.
tree_rev() {
  local root=$1
  TREE_ROOT=$root
  TREE_DIFF_HASH=""
  TREE_REV=$(git -C "$root" rev-parse --short HEAD 2>/dev/null) || {
    TREE_REV=unknown
    return 0
  }
  [ -n "$(git -C "$root" status --porcelain --untracked-files=no \
    --ignore-submodules=untracked 2>/dev/null)" ] || return 0
  TREE_REV="$TREE_REV+dirty"
  TREE_DIFF_HASH=$(git -C "$root" diff HEAD --binary --no-color --no-ext-diff \
    --submodule=diff --ignore-submodules=untracked | sha256sum | cut -c1-12)
}

# What made a tree dirty, into the head of its log (and to stdout) -- nothing
# at all on a clean tree, so a clean run's log is unchanged. Call it after
# tree_rev and before the first phase writes to the log.
tree_rev_header() {
  local tag=$1 log=$2 untracked
  [ -n "$TREE_DIFF_HASH" ] || return 0
  untracked=$(git -C "$TREE_ROOT" ls-files --others --exclude-standard | wc -l)
  {
    echo "$tag: tree is dirty -- rev=$TREE_REV  diff=$TREE_DIFF_HASH" \
      "(sha256 of git diff HEAD --submodule=diff)"
    git -C "$TREE_ROOT" status --short --untracked-files=no \
      --ignore-submodules=untracked | sed "s/^/$tag:   /"
    git -C "$TREE_ROOT" diff HEAD --stat --ignore-submodules=untracked |
      sed "s/^/$tag:   /"
    # sed -n, not head: head exiting early would fail the pipe under pipefail.
    [ "$untracked" -eq 0 ] ||
      echo "$tag:   $untracked untracked file(s), not in the hash:"
    git -C "$TREE_ROOT" ls-files --others --exclude-standard |
      sed -n "1,10s/^/$tag:     /p"
  } | tee -a "$log"
}
