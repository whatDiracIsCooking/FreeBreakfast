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
# then sent cpp-smoke under `hip` looking for build/'s test binaries.
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

# --- GPU cards: slots, load, and the `auto` preset -------------------------
#
# Admission of GPU-bound commands across every worktree/session on this host.
# RUN_SERIAL (test/gtest/CMakeLists.txt) only serializes test cases within one
# ctest process -- it does nothing when a second session's cpp-tier.sh or
# cpp-smoke.sh runs its own ctest/gtest binary against the same physical GPU
# at the same time, which is what actually timed out a full-tier run under
# cross-worktree contention.
#
# Each card (`cuda`, `hip`) has GPU_SLOTS lock files, `<backend>.<i>.lock`
# under GPU_LOCK_DIR. A run holds one slot of its card for the whole GPU
# phase, so up to GPU_SLOTS sessions share a card and the two cards never
# block each other. GPU_LOCK_DIR resolves under the MAIN checkout (see
# config.sh), inside the .sccache directory that is already bind-mounted into
# every compose service and every worktree's devcontainer -- so this needs no
# new mount, and the kernel enforces the locks across containers. flock's lock
# is held against the open file descriptor, so it releases on process exit,
# including SIGKILL: no stale-lock file to detect or clean up.
#
# Load is the larger of the card's compute utilization and its VRAM use, in
# percent. VRAM counts because a failed device allocation ABORTS in this tree
# (CreatePolicy, src/core/wwr/policy.cppm): two runs that together exhaust a
# card crash rather than slow down. A load the driver tool cannot report is
# `?`, and the slots alone then decide.

gpu_lock_dir() {
  echo "${PRIMARY_ROOT:-$REPO_ROOT}/$GPU_LOCK_DIR"
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
# cpp-smoke.sh gives: NVML routinely fails inside a container whose CUDA
# runtime works.
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
  local out util mem used total
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
  [[ "${util:-}" =~ ^[0-9]+$ ]] || { echo "?"; return 0; }
  [[ "${mem:-}" =~ ^[0-9]+$ ]] || mem=0
  echo $((util > mem ? util : mem))
}

# How many of a card's slots are held right now. Probes each with a
# non-blocking flock that is released at once.
gpu_slots_held() {
  local dir i fd held=0
  dir=$(gpu_lock_dir)
  mkdir -p "$dir"
  for ((i = 0; i < GPU_SLOTS; i++)); do
    exec {fd}>"$dir/$1.$i.lock"
    if flock -n "$fd"; then flock -u "$fd"; else held=$((held + 1)); fi
    exec {fd}>&-
  done
  echo "$held"
}

# Resolve CMAKE_PRESET=auto to a preset: pick the live card that has a free
# slot and is under GPU_BUSY_PCT, least loaded first; when none qualifies,
# the one with the fewest held slots (then least loaded), whose slot the run
# then waits for at test time. With no live card at all, GPU_AUTO_FALLBACK.
# Ties go to the earlier entry of GPU_AUTO_PRESETS. The decision and every
# card's state go to stderr, so a log says which card a run used and why.
#
# The choice is made BEFORE the build, because the build is per backend; it
# can be minutes stale by the time ctest starts, which is why with_gpu_slot
# re-checks and waits.
#
#   [ "$preset" = auto ] && preset=$(gpu_auto_preset)
gpu_auto_preset() {
  local entry backend p load held key
  local best="" best_key="" report=""
  for entry in $GPU_AUTO_PRESETS; do
    backend=${entry%%:*} p=${entry#*:}
    gpu_card_alive "$backend" || { report+=" $backend=absent"; continue; }
    load=$(gpu_card_load "$backend")
    held=$(command -v flock >/dev/null 2>&1 && gpu_slots_held "$backend" || echo 0)
    report+=" $backend=load:${load}%,slots:$held/$GPU_SLOTS"
    # Sort key, smallest wins: free-and-cool cards first (0), then busy ones
    # (1); within a class by held slots only when busy, then by load.
    local l=${load/\?/0}
    if [ "$held" -lt "$GPU_SLOTS" ] && [ "$l" -lt "$GPU_BUSY_PCT" ]; then
      key=$(printf '0-%03d' "$l")
    else
      key=$(printf '1-%03d-%03d' "$held" "$l")
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

# Run a GPU-bound command holding one slot of a card:
#
#   with_gpu_slot "$(gpu_backend_of_preset "$test_preset")" ctest --preset ...
#
# Takes the first free slot, but only while the card's load is under
# GPU_BUSY_PCT; otherwise polls every GPU_POLL_SECONDS. A card that stays
# busy with a slot free -- loaded by something that takes no slot, such as
# another user's job -- is taken anyway after GPU_BUSY_WAIT seconds (0: never),
# so a pegged card delays a run rather than hanging it.
with_gpu_slot() {
  local backend=$1
  shift
  if ! command -v flock >/dev/null 2>&1; then
    echo "devtools: flock not found -- running without cross-session GPU admission" >&2
    "$@"
    return $?
  fi

  local dir fd i got load busy_since="" said=""
  dir=$(gpu_lock_dir)
  mkdir -p "$dir"
  while :; do
    got=""
    for ((i = 0; i < GPU_SLOTS; i++)); do
      exec {fd}>"$dir/$backend.$i.lock"
      if flock -n "$fd"; then got=$i; break; fi
      exec {fd}>&-
    done
    if [ -n "$got" ]; then
      load=$(gpu_card_load "$backend")
      if [ "$load" = "?" ] || [ "$load" -lt "$GPU_BUSY_PCT" ]; then
        break
      fi
      busy_since=${busy_since:-$SECONDS}
      if [ "$GPU_BUSY_WAIT" -gt 0 ] &&
         [ $((SECONDS - busy_since)) -ge "$GPU_BUSY_WAIT" ]; then
        echo "devtools: $backend still at ${load}% after ${GPU_BUSY_WAIT}s -- taking slot $got anyway" >&2
        break
      fi
      flock -u "$fd"
      exec {fd}>&-
      [ "$said" = busy ] || echo "devtools: waiting -- $backend load ${load}% >= ${GPU_BUSY_PCT}%" >&2
      said=busy
    else
      [ "$said" = full ] || echo "devtools: waiting for a $backend slot -- all $GPU_SLOTS held ($dir)" >&2
      said=full
    fi
    sleep "$GPU_POLL_SECONDS"
  done

  echo "devtools: $backend slot $got (load ${load}%)" >&2
  local rc
  "$@" && rc=0 || rc=$?
  flock -u "$fd"
  exec {fd}>&-
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
# deliberate "cannot run". Both ROCm-on-a-real-card paths need the numeric HOST
# gid of the device groups (render, video), so the trap lives here once rather
# than copied into each; see cpp-tier.sh --rocm and devcontainer.sh.
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
