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

# Serialize a GPU-bound command against every other worktree/session on this
# host. RUN_SERIAL (test/gtest/CMakeLists.txt) only serializes test cases
# within one ctest process -- it does nothing when a second session's
# cpp-tier.sh or cpp-smoke.sh runs its own ctest/gtest binary against the same
# physical GPU at the same time, which is what actually timed out a full-tier
# run under cross-worktree contention.
#
# GPU_LOCK_FILE resolves under the MAIN checkout (see config.sh), inside the
# .sccache directory that is already bind-mounted into every compose service
# and every worktree's devcontainer -- so this needs no new mount. flock's
# lock is held by the kernel against the open file descriptor, so it releases
# automatically on process exit, including SIGKILL: no stale-lock file to
# detect or clean up.
#
#   with_gpu_lock ctest --preset "$test_preset" "${passthrough[@]}"
with_gpu_lock() {
  if ! command -v flock >/dev/null 2>&1; then
    echo "devtools: flock not found -- running without cross-session GPU serialization" >&2
    "$@"
    return $?
  fi

  local lock_file="${PRIMARY_ROOT:-$REPO_ROOT}/$GPU_LOCK_FILE"
  mkdir -p "$(dirname "$lock_file")"

  local gpu_lock_fd
  exec {gpu_lock_fd}>"$lock_file"
  if ! flock -n "$gpu_lock_fd"; then
    echo "devtools: waiting for the GPU lock ($lock_file) -- another worktree/session is running GPU tests" >&2
    flock "$gpu_lock_fd"
  fi

  local rc
  "$@" && rc=0 || rc=$?
  flock -u "$gpu_lock_fd"
  exec {gpu_lock_fd}>&-
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
