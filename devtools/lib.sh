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

# --- GPU cards: one run per card, load, and the `auto` preset --------------
#
# Admission of GPU-bound commands across every worktree/session on this host.
# One run's own concurrency is GPU_TEST_JOBS (config.sh: gtest shards in
# both, and ctest -j in cpp-tier.sh), but nothing in that run coordinates with
# a second session's cpp-tier.sh or cpp-smoke.sh running its own ctest/gtest
# binary against the same physical GPU at the same time, which is what
# actually timed out a full-tier run under cross-worktree contention.
#
# A run holds its card for the whole GPU phase, and no other run shares it
# (config.sh says why); the two cards never block each other. A card's lock is
# two files under GPU_LOCK_DIR, `<backend>.0.lock` and `<backend>.1.lock`,
# held together: branches from before 2026-10-28 let two runs share a card,
# each holding one of those, and holding both keeps such a run from joining
# this one. Drop the second once no worktree carries the old lib.sh.
# GPU_LOCK_DIR resolves under the MAIN checkout (see config.sh), inside the
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

# Run a gtest binary as `n` concurrent shards, inside the card the caller
# holds -- call it under with_gpu_card:
#
#   with_gpu_card "$backend" run_gpu_sharded cpp-smoke "$bin" "$GPU_TEST_JOBS"
#
# One process per shard, not per case, so the GPU runtime's context creation
# (about 0.45s on the 3080) is paid n times rather than once per case. gtest
# deals cases to shards round-robin, so each gets a slice of every suite; a
# GTEST_FILTER in the environment narrows what is dealt. Every shard's output
# is appended to BUILD_LOG under its own header and its summary lines printed,
# tagged `<tag>:   [shard i/n]`. Each pid's status is waited for, so a crashed
# shard fails the run even when its siblings pass, and for a crash, where gtest
# prints no FAILED line, the case that was running (a [ RUN ] with no verdict)
# is named.
#
# A shard that HANGS is killed, not waited on forever: gpu_shard_watchdog
# below. On gfx1200 a card can wedge with every process on it spinning in a
# HIP sync at full CPU, printing nothing, until killed (issue #477) -- and
# before this a wedge held the gate, and the card lock, for hours with no test
# named. Each shard is started through devtools/ptraceable-exec.py so that the
# watchdog's debugger may attach to it before the kill: gpu_wedge_snapshot.
run_gpu_sharded() {
  local tag=$1 bin=$2 n=$3 dir i src rc=0
  local -a pids=() launch=()
  [ -x "$bin" ] || {
    echo "$tag: missing binary $bin -- see $BUILD_LOG" >&2
    return 1
  }
  command -v python3 >/dev/null &&
    launch=(python3 -I "$REPO_ROOT/devtools/ptraceable-exec.py" "$$")
  dir=$(mktemp -d)
  for ((i = 0; i < n; i++)); do
    GTEST_TOTAL_SHARDS=$n GTEST_SHARD_INDEX=$i "${launch[@]}" "$bin" \
      >"$dir/$i.log" 2>&1 &
    pids+=("$!")
  done
  gpu_shard_watchdog "$tag" "$dir" "${pids[@]}"
  for ((i = 0; i < n; i++)); do
    wait "${pids[$i]}" && src=0 || src=$?
    echo "--- gpu shard $i/$n: $bin ---" >>"$BUILD_LOG"
    cat "$dir/$i.log" >>"$BUILD_LOG"
    grep -E 'ran\.|PASSED|SKIPPED|FAILED' "$dir/$i.log" |
      sed "s|^|$tag:   [shard $i/$n] |"
    if [ -e "$dir/$i.killed" ]; then
      rc=1
      echo "$tag:   [shard $i/$n] KILLED by the watchdog: $(<"$dir/$i.killed")" >&2
      echo "$tag:   [shard $i/$n] hung in $(gtest_running_case "$dir/$i.log")" >&2
      [ -s "$dir/$i.wedge" ] &&
        echo "$tag:   [shard $i/$n] state before the kill: $(<"$dir/$i.wedge")" >&2
      continue
    fi
    [ "$src" = 0 ] && continue
    rc=1
    echo "$tag:   [shard $i/$n] exited $src" >&2
    gtest_running_case "$dir/$i.log" | sed "s|^|$tag:   [shard $i/$n] died in |" >&2
  done
  if compgen -G "$dir/*.killed" >/dev/null; then
    echo "$tag: a GPU shard hung -- possibly a card wedge (issue #477: host" \
      "spinning, GPU 100% busy, no memory activity). Check the card" \
      "(devtools/doctor.sh; dmesg on the HOST), and rerun the gate on the" \
      "other backend if it is the card." >&2
  fi
  rm -rf "$dir"
  return $rc
}

# The gtest case a log was in the middle of: the last `[ RUN ]` with no verdict
# after it, or `<unknown>` when every case it started finished.
gtest_running_case() {
  awk '/^\[ RUN      \]/ { t = $NF } /^\[ *(OK|FAILED|SKIPPED) *\]/ { t = "" }
       END { print (t != "" ? t : "<unknown>") }' "$1"
}

# Wait for the shards in `pids` (their logs are <dir>/<i>.log), killing any
# that prints nothing for GPU_SHARD_STALL_SECONDS or is still running
# GPU_SHARD_BUDGET_SECONDS after the start (config.sh; 0 turns either off).
# A killed shard gets <dir>/<i>.killed holding the reason, which
# run_gpu_sharded reports. It is TERMed, then KILLed if it has not gone in 10s.
# Returns once every shard has exited or been killed.
#
# The kill is reported the moment it happens, not only in the summary: a
# process stuck in the driver can outlive even SIGKILL, and then the wait that
# follows is the hang, but by then the case has been named. Before the kill the
# shard's state is written down (gpu_wedge_snapshot); <dir>/<i>.wedge holds
# the path.
gpu_shard_watchdog() {
  local tag=$1 dir=$2 i t live size reason start=$SECONDS
  shift 2
  local -a pids=("$@") sizes=() quiet=()
  for i in "${!pids[@]}"; do
    sizes[i]=-1
    quiet[i]=$SECONDS
  done
  while :; do
    live=0
    for i in "${!pids[@]}"; do
      [ -e "$dir/$i.killed" ] && continue
      kill -0 "${pids[i]}" 2>/dev/null || continue
      live=1
      size=$(stat -c %s "$dir/$i.log" 2>/dev/null || echo 0)
      if [ "$size" != "${sizes[i]}" ]; then
        sizes[i]=$size
        quiet[i]=$SECONDS
      fi
      reason=""
      if [ "$GPU_SHARD_STALL_SECONDS" -gt 0 ] &&
         [ $((SECONDS - quiet[i])) -ge "$GPU_SHARD_STALL_SECONDS" ]; then
        reason="no output for ${GPU_SHARD_STALL_SECONDS}s (GPU_SHARD_STALL_SECONDS)"
      elif [ "$GPU_SHARD_BUDGET_SECONDS" -gt 0 ] &&
           [ $((SECONDS - start)) -ge "$GPU_SHARD_BUDGET_SECONDS" ]; then
        reason="still running after ${GPU_SHARD_BUDGET_SECONDS}s (GPU_SHARD_BUDGET_SECONDS)"
      fi
      [ -n "$reason" ] || continue
      echo "$reason" >"$dir/$i.killed"
      echo "$tag:   [shard $i/${#pids[@]}] watchdog: $reason -- killing pid" \
        "${pids[i]}, in $(gtest_running_case "$dir/$i.log") (see issue #477)" >&2
      gpu_wedge_snapshot "$i" "${pids[i]}" "$dir/$i.log" "$reason" \
        >"$dir/$i.wedge"
      kill -TERM "${pids[i]}" 2>/dev/null
      for ((t = 0; t < 10; t++)); do
        kill -0 "${pids[i]}" 2>/dev/null || break
        sleep 1
      done
      kill -KILL "${pids[i]}" 2>/dev/null
    done
    [ "$live" -eq 1 ] || return 0
    sleep 1
  done
}

# Write down what a hung shard (index, pid, log, reason) was doing, before the
# watchdog kills it, and echo the file's path. The kill frees the card at once,
# so this is the only moment a wedge (issue #477) can be read: what each GPU
# tool reports about its card, every thread's scheduler state, CPU time and
# kernel wait channel, a debugger's backtrace of every thread (rocgdb, else
# gdb; the shard was started through ptraceable-exec.py, which is what allows
# it), and the end of the shard's log. A wedge can hang the vendor tools and
# the debugger too, so each is cut off after GPU_WEDGE_TOOL_SECONDS.
#
# The file sits beside BUILD_LOG, as <log>.shard<i>-wedge.txt, so it outlives
# the shard directory; with no BUILD_LOG file it goes in TMPDIR. What it cannot
# hold is the kernel's side -- an amdgpu ring timeout or MES fault is in dmesg,
# which the container cannot read -- so it ends with the host command for that.
gpu_wedge_snapshot() {
  local i=$1 pid=$2 log=$3 reason=$4 out t tool secs=$GPU_WEDGE_TOOL_SECONDS
  case "${BUILD_LOG:-}" in
    "" | /dev/null) out=$(mktemp "${TMPDIR:-/tmp}/gpu-wedge-shard$i.XXXXXX");;
    *) out="${BUILD_LOG%.log}.shard$i-wedge.txt";;
  esac
  {
    echo "=== GPU shard $i, pid $pid, $(date -u '+%F %T') UTC"
    echo "watchdog: $reason; in $(gtest_running_case "$log")"
    for tool in "rocm-smi --showuse --showmemuse --showmeminfo vram" \
                "nvidia-smi"; do
      command -v "${tool%% *}" >/dev/null || continue
      echo; echo "=== $tool"
      # shellcheck disable=SC2086  # the tool and its flags, split on purpose
      timeout "$secs" $tool || echo "($tool: exit $?)"
    done
    echo; echo "=== threads: tid state utime stime (clock ticks) wchan"
    for t in /proc/"$pid"/task/*; do
      # stat's comm is parenthesised and may hold spaces: read after the `)`.
      echo "${t##*/} $(sed 's/.*) //' "$t/stat" | awk '{print $1, $12, $13}')" \
        "$(cat "$t/wchan")"
    done
    tool=$(command -v rocgdb || command -v /opt/rocm/bin/rocgdb ||
      command -v gdb) || tool=""
    echo; echo "=== backtrace (${tool:-no debugger found})"
    [ -z "$tool" ] || timeout "$secs" "$tool" -p "$pid" -batch \
      -ex "info threads" -ex "thread apply all bt 30" </dev/null
    echo; echo "=== the last 40 lines of the shard's log"
    tail -n 40 "$log"
    echo
    echo "=== the kernel's side is in the HOST's dmesg; there, before a reboot:"
    echo "  sudo dmesg -T | grep -iE 'amdgpu|kfd|mes|ring|timeout|reset|NVRM|Xid'"
  } >"$out" 2>&1
  echo "$out"
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
