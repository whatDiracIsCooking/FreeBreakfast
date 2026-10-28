#!/usr/bin/env bash
# Drive this worktree's devcontainer. There is usually no `devcontainer` binary
# on PATH, so every call goes through npx.
#
#   devtools/devcontainer.sh up          bring it up, reusing an existing one
#   devtools/devcontainer.sh rebuild     recreate from scratch
#   devtools/devcontainer.sh shell       open a bash shell inside
#   devtools/devcontainer.sh test [...]  run the test suite inside
#   devtools/devcontainer.sh down        stop and remove the container
#   devtools/devcontainer.sh down --all  ... every GPU variant of this worktree
#
# WHICH VARIANT: a flag BEFORE the command, naming a directory under
# .devcontainer/ -- so this repo's three configs are reachable as:
#
#   devtools/devcontainer.sh --hip shell        .devcontainer/hip/
#   devtools/devcontainer.sh --combined up      .devcontainer/combined/
#   devtools/devcontainer.sh --cuda rebuild     .devcontainer/cuda/ (the default)
#
# The flag must come before the command, because everything after `shell` and
# `test` is passed through to what runs inside. Use it on EVERY call in a
# session, teardown included -- `down` matches on the config file, so
# `--hip up` followed by a bare `down` takes down the CUDA container and leaves
# the HIP one running. `down --all` is the escape hatch.
#
# CPUSET / CPUS (from config.sh, or set for one call) bound the container's
# share of the host on `up` and `rebuild`:
#
#   CPUSET=0-11 devtools/devcontainer.sh up      pin to host cores 0-11
#   CPUS=8      devtools/devcontainer.sh up      cap at 8 cores' worth
#
# Unlike JOBS, which only bounds the test runner, these bound everything the
# container runs -- compiles started from a `shell` included.
#
# HOST_MOUNTS / HOST_ENV_PASSTHROUGH (also from config.sh) add host state that
# is deliberately NOT written into devcontainer.json -- a personal config
# directory, an API key -- as `--mount` and `--remote-env` arguments. The
# mounts need a `rebuild` to take effect, like any other mount; the env reaches
# every `shell` and `test` too. host_mount_args and remote_env_args have the
# reasoning, config.sh the defaults (both exist for the `agy` the images ship).
# The host's git user.name/user.email always go in this way too, as
# GIT_AUTHOR_*/GIT_COMMITTER_*, so commits made inside are attributed.
#
# The three configs, and what each is:
#
#   .devcontainer/cuda/devcontainer.json      CUDA 13 (the default)
#   .devcontainer/hip/devcontainer.json       ROCm, no CUDA at all
#   .devcontainer/combined/devcontainer.json  both SDKs, ~40GB
#
# Three ways to choose, highest precedence first: the flag above, the
# DEVCONTAINER_CONFIG environment variable, and its default in config.sh (the
# CUDA variant -- the one that can actually build the C++ tree). The variable
# is still the way to point at a config that is not one of these, and the way
# to make a whole shell session use one:
#
#   export DEVCONTAINER_CONFIG=.devcontainer/hip/devcontainer.json
#
# Containers key on the workspace folder path AND the config file, so each
# worktree gets its own container per variant, and the variants of one worktree
# can be up at the same time. They SHARE that worktree's named volumes (see the
# comment in the json), so run the suite in one variant at a time. The script
# resolves the workspace from its own location, so it does the right thing
# whichever checkout you call it from, and from any cwd.
#
# IMAGES do not key on the config file, and that asymmetry surprises people:
# the CLI names a built image `vsc-<workspace basename>-<sha256 of the
# workspace PATH>`, so all three variants of one checkout build to that ONE
# name and each `up` retags it away from the last. The containers are
# unharmed -- each holds its image by id -- but the variant that lost the tag
# goes untagged and unexplainable in `docker images`, so `up` and `rebuild`
# add a `:<variant>` tag beside the CLI's `:latest`. See tag_variant_image.
#
# Project-specific values (the test command, the worker count) come from
# devtools/config.sh -- edit that, not this.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

WORKSPACE=$REPO_ROOT
CLI=(npx -y @devcontainers/cli)

# A leading variant flag selects a config by DIRECTORY NAME under
# .devcontainer/. Resolved by convention rather than from a list, so adding a
# .devcontainer/<name>/ needs no edit here and this script stays copyable into
# the next repo -- the same convention doctor.sh globs for.
#
# A leading `--foo` that does not name an existing config is NOT silently
# treated as a variant: it falls through to the command dispatch and is
# reported as unknown, so a typo says so instead of failing later with a
# confusing path. Two variant flags in one call is an error rather than
# last-one-wins, which would make a copy-pasted command line quietly do
# something other than what it reads as.
variant_flag=
while [ $# -gt 0 ]; do
  case "$1" in
    # The name is restricted to a single plain directory name, which is what
    # makes interpolating it into a path safe: a name with slashes or dots
    # (`--../../elsewhere`) would otherwise reach outside .devcontainer/, and a
    # bare `--` would yield `.devcontainer//devcontainer.json`.
    --[A-Za-z0-9]*)
      name=${1#--}
      case "$name" in
        *[!A-Za-z0-9._-]*) break ;;
        *..*) break ;;
      esac
      candidate=$REPO_ROOT/.devcontainer/$name/devcontainer.json ;;
    *) break ;;
  esac
  [ -f "$candidate" ] || break
  if [ -n "$variant_flag" ]; then
    echo "error: two variant flags: $variant_flag and $1" >&2
    exit 2
  fi
  variant_flag=$1
  DEVCONTAINER_CONFIG=$candidate
  shift
done

# Which devcontainer.json to drive. Defaults in config.sh to the CUDA variant
# -- the one with the C++ toolchain -- and the flag above or an environment
# variable still wins for one call. A RELATIVE path is resolved against the
# repo root here, not the cwd: the CLI resolves --config relative to where you
# invoked it, so a bare `.devcontainer/cuda/devcontainer.json` would otherwise
# only work from the root. Empty would leave it to the CLI, which finds no
# top-level .devcontainer/devcontainer.json to fall back to -- but config.sh
# always sets it.
DC_CONFIG=()
if [ -n "${DEVCONTAINER_CONFIG:-}" ]; then
  case "$DEVCONTAINER_CONFIG" in
    /*) dc_config_path=$DEVCONTAINER_CONFIG ;;
    *) dc_config_path=$REPO_ROOT/$DEVCONTAINER_CONFIG ;;
  esac
  if [ ! -f "$dc_config_path" ]; then
    echo "error: DEVCONTAINER_CONFIG does not exist: $dc_config_path" >&2
    exit 1
  fi
  DC_CONFIG=(--config "$dc_config_path")
fi

# The variant this call drives: the directory name under .devcontainer/, which
# is what names a config here -- cuda, hip, combined. Empty only if
# DEVCONTAINER_CONFIG was cleared by hand, since config.sh always sets it.
# Derived from the RESOLVED path rather than from the flag, so it is still
# right when the config came from the environment or from config.sh's default.
VARIANT=
if [ -n "${dc_config_path:-}" ]; then
  VARIANT=$(basename "$(dirname "$dc_config_path")")
fi

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

# This workspace's container ids, one per line. Empty when nothing matches.
#
# TWO labels, not one. Every GPU variant of a worktree -- cuda, hip, combined
# -- is built for the SAME workspace folder and so carries the same
# `devcontainer.local_folder`; only `devcontainer.config_file` tells them
# apart, and the CLI sets both on every container it creates. Matching on the
# folder alone would make `down` remove a sibling variant's container and
# `apply_cpu_limits` pin whichever one `head -1` happened to return.
#
#   --stopped      include exited containers -- what `down` wants, since a
#                  crashed run keeps the labels. The bare form is the running
#                  container, the only thing `docker update` can act on.
#   --any-config   every variant for this workspace, not just the active one.
container_ids() {
  local ps=(docker ps -q) any_config=
  while [ $# -gt 0 ]; do
    case $1 in
      --stopped) ps=(docker ps -aq) ;;
      --any-config) any_config=1 ;;
    esac
    shift
  done
  local filters=(--filter "label=devcontainer.local_folder=$WORKSPACE")
  # dc_config_path is set whenever DEVCONTAINER_CONFIG is, and config.sh always
  # defaults it. With neither, the CLI picks the config itself and there is no
  # path to match on, so fall back to the folder alone -- which is the old
  # behaviour, and only reachable by clearing DEVCONTAINER_CONFIG by hand.
  if [ -z "$any_config" ] && [ -n "${dc_config_path:-}" ]; then
    filters+=(--filter "label=devcontainer.config_file=$dc_config_path")
  fi
  "${ps[@]}" "${filters[@]}"
}

# Apply CPUSET/CPUS to the running container with `docker update`, then restart
# it if it holds an NVIDIA GPU. A no-op when neither is set, and a warning
# rather than an error when nothing is running: the limits are a refinement of
# `up`, not a precondition for it.
#
# `docker update` changes only the limits it is given, so a later call with
# just CPUSET leaves an earlier CPUS quota in place, and unsetting both here
# clears nothing -- it just stops applying. To get back to an unbounded
# container, `rebuild` it (a fresh container starts with no limits).
#
# THE RESTART IS LOAD-BEARING, and only for `--gpus`. `docker update`
# regenerates the container's device cgroup from HostConfig.Devices -- and for
# `--gpus all` that list is EMPTY. The ask lives in HostConfig.DeviceRequests
# and is honoured by the NVIDIA container runtime's OCI hook at container
# START, so regenerating the cgroup silently revokes GPU access while leaving
# every /dev/nvidia* node visible inside. The symptom is
# `nvidia-smi: Failed to initialize NVML: Unknown Error`, and because the
# device nodes are still there it reads like a driver problem rather than a
# permissions one. Worse here: CMAKE_CUDA_ARCHITECTURES=native then fails at
# CONFIGURE time, so a perfectly good tree looks like a broken change.
#
# Restarting re-runs the hook and re-injects the devices; the limits persist
# across it (they are in the container config, not the cgroup). The HIP
# variant needs none of this -- `--device=/dev/kfd` populates
# HostConfig.Devices, which `docker update` preserves -- so the restart is
# gated on DeviceRequests rather than applied blindly.
apply_cpu_limits() {
  [ -z "${CPUSET:-}" ] && [ -z "${CPUS:-}" ] && return 0
  local id flags=()
  # `|| true`: docker may be absent or its daemon down, and a missing CPU
  # pin is not a reason to fail a command that has already done its work.
  id=$(container_ids | head -1 || true)
  if [ -z "$id" ]; then
    echo "cpu-limit: no running container for $WORKSPACE (skipped)" >&2
    return 0
  fi
  [ -n "${CPUSET:-}" ] && flags+=(--cpuset-cpus "$CPUSET")
  [ -n "${CPUS:-}" ] && flags+=(--cpus "$CPUS")
  docker update "${flags[@]}" "$id" >/dev/null
  echo "cpu-limit: ${CPUSET:+cpuset-cpus=$CPUSET }${CPUS:+cpus=$CPUS }-> $id" >&2

  # See the comment above: only a hook-injected GPU needs this. `tr -cd` rather
  # than a bare test, so a container that died between the two calls yields an
  # empty string and not a stray newline that would read as non-zero. The
  # restart is non-fatal for the same reason the pin is: `up` has already done
  # its work by this point.
  local requests
  requests=$(docker inspect "$id" \
    --format '{{len .HostConfig.DeviceRequests}}' 2>/dev/null | tr -cd '0-9')
  if [ -n "$requests" ] && [ "$requests" != "0" ]; then
    if docker restart "$id" >/dev/null 2>&1; then
      echo "cpu-limit: restarted $id to restore its GPU device cgroup" >&2
    else
      echo "cpu-limit: could not restart $id -- its GPU may be unusable until you do" >&2
    fi
  fi
}

# The .git bind mount in devcontainer.json reads its host path from
# ${localEnv:FREE_BREAKFAST_GIT_DIR}, which this resolves and exports before the CLI
# runs -- so no machine-specific path is ever written into the tracked json.
# (The literal after the colon there is only a fallback for a direct
# `devcontainer up` / VS Code "Reopen in Container", which does not run this.)
#
# WHY A GUARD, NOT JUST AN EXPORT: an unresolved mount source makes docker's
# --mount refuse a path that does not exist, and the devcontainer CLI reports
# that by printing the ENTIRE failing `docker run` line -- which by then has
# -e GH_TOKEN=<your token> in it. So a bad mount puts a credential on the
# terminal. Resolving it here, before the CLI is invoked, is what stops that.
#
# The source is the repo's COMMON git dir: a worktree's own `.git` is a file
# pointing into <main>/.git/worktrees/<name>, so the whole common dir has to be
# visible inside at its host path (source == target) for git to work there.
require_git_dir() {
  local d
  d=$(git -C "$REPO_ROOT" rev-parse --path-format=absolute \
        --git-common-dir 2>/dev/null || true)
  if [ -z "$d" ] || [ ! -d "$d" ]; then
    echo "error: cannot resolve this repo's git dir for the container mount." >&2
    echo "       run devtools/devcontainer.sh from inside the FreeBreakfast checkout." >&2
    exit 1
  fi
  export FREE_BREAKFAST_GIT_DIR="$d"
}

# The sccache bind mount reads its host path from
# ${localEnv:FREE_BREAKFAST_SCCACHE_DIR} the same way, and for the same reason: no
# machine-specific path in the tracked json.
#
# ONE STORE FOR EVERY CHECKOUT, and it is the PRIMARY one's. The common git dir
# resolved above is <primary>/.git whether this runs from the main checkout or
# from .claude/worktrees/<name>, so its dirname is the primary root in both
# cases -- which is what makes every worktree name the same directory.
#
# WHY THE mkdir: docker does not refuse a bind whose source is missing, it
# CREATES it, owned by root. The container runs as uid 1000, and the first
# compile then dies with `failed to create directory /home/ubuntu/.sccache/
# preprocessor: Permission denied` -- a message that never mentions a mount --
# while clearing the leftover needs sudo. Making it here is what stops that.
# (compose says the same thing declaratively, with create_host_path: false.)
#
# Note the store is shared but HITS are not, between trees mounted at different
# absolute paths: a restored BMI keeps the path it was built under. That is
# correct behaviour, not a misconfiguration -- see CMakeLists.txt's "Compiler
# cache" section. Sharing the store still dedups what each tree rebuilds.
#
# FREE_BREAKFAST_OBJECT_CACHE_DIR is the second store, devtools/gpu-object-
# cache.sh's, made and exported the same way. Unlike sccache's it is bound at
# its HOST path (source == target, like the .git mount): the script finds it
# as <primary root>/.build-object-cache, and the .git mount is what makes that
# root resolve inside the container. Without this bind, a sibling worktree's
# container sees the primary root as a root-owned mount point holding only
# .git, and every device object misses. And unlike sccache's, its hits DO
# cross worktrees -- the key is content, not paths.
require_cache_dirs() {
  local root d o
  root=$(dirname "$FREE_BREAKFAST_GIT_DIR")
  d="${FREE_BREAKFAST_SCCACHE_DIR:-$root/.sccache}"
  o="${FREE_BREAKFAST_OBJECT_CACHE_DIR:-$root/.build-object-cache}"
  if ! mkdir -p "$d" 2>/dev/null; then
    echo "error: cannot create the sccache store at $d" >&2
    echo "       set FREE_BREAKFAST_SCCACHE_DIR to a writable directory." >&2
    exit 1
  fi
  if ! mkdir -p "$o" 2>/dev/null; then
    echo "error: cannot create the GPU object cache at $o" >&2
    echo "       set FREE_BREAKFAST_OBJECT_CACHE_DIR to a writable directory." >&2
    exit 1
  fi
  export FREE_BREAKFAST_SCCACHE_DIR="$d"
  export FREE_BREAKFAST_OBJECT_CACHE_DIR="$o"

  ensure_host_bind_dirs
}

# Every other host directory the devcontainer binds, from HOST_BIND_DIRS in
# config.sh. Driven by config rather than hardcoded here, so adding a bind to
# the json needs no edit to this script.
#
# Why this needs to exist at all: docker does not refuse a bind whose source is
# missing, it CREATES it, owned by root. The container then runs as uid 1000 and
# the first compile dies with `failed to create directory
# /home/ubuntu/.sccache/preprocessor: Permission denied` -- a message that never
# mentions a mount -- while clearing the leftover needs sudo.
#
# A relative path resolves against the MAIN checkout (the one holding the common
# .git), not the calling worktree, so every worktree names the same directory --
# which is the whole point for a shared compiler cache.
#
# Only list directories that are correct when EMPTY. The .git bind mount is not
# one of them: an empty .git is worse than a missing one, which is why it gets a
# loud check (require_git_dir) rather than a mkdir.
ensure_host_bind_dirs() {
  local dir root=${PRIMARY_ROOT:-$REPO_ROOT}
  while read -r dir; do
    [ -n "$dir" ] || continue
    case "$dir" in /*) ;; *) dir=$root/$dir ;; esac
    [ -d "$dir" ] && continue
    if ! mkdir -p "$dir"; then
      echo "error: cannot create bind source $dir" >&2
      exit 1
    fi
    echo "devcontainer: created bind source $dir"
  done <<< "${HOST_BIND_DIRS:-}"
}

# The optional, personal host mounts from HOST_MOUNTS in config.sh, as
# `--mount` arguments for the devcontainer CLI. Empty list, no arguments, no
# behaviour change -- which is what lets a knob like this default to something
# ($HOME/.gemini, for the `agy` the images install) without imposing it.
#
# WHY NOT IN devcontainer.json: config.sh's own comment has it in full. Short
# version -- JSON cannot express "mount this only if it exists", an unresolved
# `${localEnv:...}` source makes docker CREATE the placeholder path as root,
# and a personal path in a tracked file is a path in git.
#
# A missing source is created here, before the CLI runs, for the same reason
# ensure_host_bind_dirs exists: docker creates a missing bind source as root,
# the container runs as uid 1000, and the tool inside then fails on a
# permission error that never mentions a mount.
#
# These reach the CLI's `up` only. `exec` cannot add a mount to a container
# that already exists, so editing HOST_MOUNTS needs `rebuild`, not `up` -- the
# same trap as editing a Dockerfile, and the reason `up` reports what it
# mounted.
HOST_MOUNT_ARGS=()
host_mount_args() {
  HOST_MOUNT_ARGS=()
  local entry src dst root=${PRIMARY_ROOT:-$REPO_ROOT}
  config_lines "${HOST_MOUNTS:-}"
  for entry in ${CONFIG_LINES[@]+"${CONFIG_LINES[@]}"}; do
    src=${entry%%:*}
    dst=${entry#*:}
    if [ -z "$src" ] || [ -z "$dst" ] || [ "$dst" = "$entry" ]; then
      echo "error: HOST_MOUNTS entry is not host_path:container_path -- $entry" >&2
      exit 2
    fi
    case "$src" in
      '~')   src=$HOME ;;
      '~/'*) src=$HOME/${src#'~/'} ;;
      /*)    ;;
      *)     src=$root/$src ;;
    esac
    if [ ! -d "$src" ]; then
      if ! mkdir -p "$src"; then
        echo "error: cannot create mount source $src (HOST_MOUNTS)" >&2
        exit 1
      fi
      echo "devcontainer: created mount source $src"
    fi
    HOST_MOUNT_ARGS+=(--mount "type=bind,source=$src,target=$dst")
    echo "devcontainer: mounting $src -> $dst"
  done
}

# HOST_ENV_PASSTHROUGH from config.sh, as `--remote-env NAME=value` arguments.
# Only variables that are set AND non-empty are passed: an exported-but-empty
# GEMINI_API_KEY is worse than an absent one, since a tool reads it as
# configured-but-blank.
#
# Unlike the mounts above these work on `exec` too, so they apply to every
# `shell` and `test` as well as to `up` -- a key rotated on the host reaches
# the next command, with no rebuild. Values are redacted out of run_cli's
# output; see the note there.
#
# The host's git identity rides along the same way, always, whatever
# HOST_ENV_PASSTHROUGH says. The container's ~/.gitconfig is its own (only
# post-create.sh's safe.directory and gh's credential helper are in it), so
# without this every commit made inside dies with `Author identity unknown`.
# It goes in as git's own GIT_AUTHOR_*/GIT_COMMITTER_* variables, read from
# the host's `git config` here, rather than written into the container's
# config: that keeps a personal name and email out of the tracked files, and a
# changed host identity reaches the next `shell` with no rebuild. A variable
# already set on the host wins over the config it would be read from.
GIT_IDENTITY_ENV="GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL"
export_git_identity() {
  local user_name user_email
  user_name=$(git -C "$REPO_ROOT" config --get user.name 2>/dev/null || true)
  user_email=$(git -C "$REPO_ROOT" config --get user.email 2>/dev/null || true)
  export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-$user_name}"
  export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$user_name}"
  export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-$user_email}"
  export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$user_email}"
}

REMOTE_ENV_ARGS=()
remote_env_args() {
  REMOTE_ENV_ARGS=()
  export_git_identity
  local name
  for name in ${HOST_ENV_PASSTHROUGH:-} $GIT_IDENTITY_ENV; do
    [ -n "${!name:-}" ] || continue
    REMOTE_ENV_ARGS+=(--remote-env "$name=${!name}")
  done
}

# Resolve ROCM_GROUPS to numeric HOST gids and export one FREE_BREAKFAST_<NAME>_GID
# per group, which is what the `${localEnv:...}` references in the hip and
# combined runArgs read.
#
# WHY THE JSON CANNOT SIMPLY NAME THE GROUP: `--group-add render` resolves the
# name INSIDE the container, where docker/install-rocm.sh made `render` gid
# 110, while the bind-mounted /dev/kfd and /dev/dri/renderD* keep their HOST
# ownership. The container user then joins a group the devices do not grant,
# the container starts clean, and the first device call fails with
# hipErrorNoDevice -- indistinguishable from a machine with no AMD card.
# cpp-tier.sh --rocm has always resolved these with getent; this is the same
# resolution for the interactive path, off the same config.sh list.
#
# NON-FATAL, unlike cpp-tier.sh's, because each reference carries a baked-in
# default after the colon (`${localEnv:FREE_BREAKFAST_RENDER_GID:109}`). An
# unresolvable group leaves that value standing instead of passing an empty
# --group-add, and that default is also what a VS Code "Reopen in Container"
# gets -- nothing exports these for it. So the default is a last-known-good
# host GID, never a group NAME: a name there would be exactly the silent wrong
# answer this function exists to remove.
#
# The warning is gated on the ACTIVE config actually reading the variable, so
# the CUDA variant -- which passes no --group-add at all -- stays quiet on a
# host with no render group.
export_host_gids() {
  local grp gid var cfg=${dc_config_path:-}
  # Name the VARIANT in the warning: every config file is called
  # devcontainer.json, so the basename alone would not say which one.
  [ -z "$cfg" ] || cfg=$VARIANT/$(basename "$cfg")
  while IFS= read -r grp; do
    [ -n "$grp" ] || continue
    var=FREE_BREAKFAST_${grp//-/_}_GID
    var=${var^^}
    # A group name that does not survive into an identifier is skipped rather
    # than exported: `export 'FREE_BREAKFAST_A B_GID=1'` fails, and under `set -e` that
    # would take down an `up` over a config typo.
    case "$var" in
      *[!A-Za-z0-9_]*) continue ;;
    esac
    # Numeric HOST gid, or empty; host_gid (lib.sh) owns the getent exit-2 trap.
    gid=$(host_gid "$grp")
    if [ -n "$gid" ]; then
      export "$var=$gid"
    elif grep -q "$var" "${dc_config_path:-/dev/null}" 2>/dev/null; then
      echo "warning: no '$grp' group on this host (getent found none), so" >&2
      echo "         $var is unset and $cfg falls back" >&2
      echo "         to the gid baked into it. If the GPU is then invisible" >&2
      echo "         in there, fix ROCM_GROUPS in config.sh or that fallback." >&2
    fi
  done <<<"${ROCM_GROUPS:-}"
}

# Give every gid the container user holds a NAME inside the container.
#
# The --group-add values the hip and combined runArgs pass are numeric HOST
# gids and have to be (export_host_gids above owns that story), so the
# container user lands in a group its own /etc/group has no entry for: `render`
# is gid 109 on this host, while docker/install-rocm.sh made the container's
# own `render` 110. Nothing functional is wrong -- the gid is what the device
# nodes grant on, and a name is only a label -- but every getgrgid() miss
# surfaces as an error, and the one that greets you is the `groups` call in
# Ubuntu's /etc/bash.bashrc sudo hint, on every interactive shell:
#
#   groups: cannot find name for group ID 109
#
# which reads like a broken container rather than a working one. So add the
# missing entry as `host-<group>`, and NEVER by renumbering the container's own
# `render` to match: that is the mismatch-chasing the json comments warn
# against, and it is the bind-mounted nodes' HOST ownership that decides which
# gid actually works.
#
# Cosmetic, so never fatal. `docker exec` rather than a devcontainer lifecycle
# command because groupadd needs root and the remoteUser's sudo asks for a
# password; a VS Code "Reopen in Container", which never calls this script,
# keeps the warning.
name_host_gids() {
  command -v docker >/dev/null 2>&1 || return 0
  local id gid name gids=()
  # `|| true` for the same reason as apply_cpu_limits: a down daemon is not a
  # reason to fail an `up` that has already done its work.
  id=$(container_ids | head -1 || true)
  [ -n "$id" ] || return 0
  # `id -G` in there, rather than ROCM_GROUPS out here, so this covers whatever
  # the active config actually passed -- and stays a no-op for the cuda
  # variant, which passes no --group-add at all.
  read -ra gids <<<"$(docker exec "$id" id -G 2>/dev/null || true)"
  for gid in "${gids[@]}"; do
    docker exec "$id" getent group "$gid" >/dev/null 2>&1 && continue
    # The HOST's name for the gid, since that is what it means here; bare
    # `host-<gid>` when the host has no entry for it either.
    name=$(getent group "$gid" 2>/dev/null | cut -d: -f1 || true)
    docker exec -u root "$id" groupadd -g "$gid" "host-${name:-$gid}" \
      >/dev/null 2>&1 ||
      echo "warning: gid $gid has no name inside the container, and groupadd" \
        "could not add one -- harmless, but \`groups\` will complain" >&2
  done
}

# Fail before `exec` does, naming the variant and the command that fixes it.
#
# `devcontainer exec` neither creates a container nor STARTS a stopped one. Run
# against a variant that is merely stopped -- a reboot, a `docker stop`, a
# machine that slept -- it fails with the daemon's raw
# `container <id> is not running`, which names an id you have never seen, says
# nothing about WHICH variant it belongs to, and reads like a broken container
# rather than one that is simply down. A variant that was never built reads
# much the same way. Both are one `up` away.
#
# A diagnostic, not a gate: with no docker on PATH there is nothing to inspect
# here, so let the CLI produce its own error rather than inventing one.
# How many devices the CUDA driver can actually see from inside the container.
# Empty (and non-zero) when there is no container, no /dev/nvidia* nodes, or the
# driver cannot be initialised -- the caller treats all three as "nothing to
# say".
gpu_device_count() {
  local id
  id=$(container_ids | head -1 || true)
  [ -z "$id" ] && return 0
  docker exec "$id" sh -c 'ls /dev/nvidia[0-9]* >/dev/null 2>&1' || return 0
  docker exec "$id" python3 -c 'import ctypes,sys
lib=ctypes.CDLL("libcuda.so.1")
if lib.cuInit(0)!=0: sys.exit(1)
c=ctypes.c_int(0); lib.cuDeviceGetCount(ctypes.byref(c)); print(c.value)' 2>/dev/null
}

# The device nodes being present while the driver reports NO device is a
# specific, recoverable state: the container's --gpus device-cgroup grant was
# revoked out from under it by a host driver reload or a systemd
# daemon-reload. It reads exactly like a box with no GPU, and `up` will not fix
# it -- only recreating the container will -- so say which one is needed.
#
# CUDA-only by construction (it probes /dev/nvidia*): ROCm's device nodes are
# /dev/kfd and /dev/dri and are passed by --device rather than by a cgroup
# grant, so they do not fail this way.
warn_if_gpu_revoked() {
  local id
  id=$(container_ids | head -1 || true)
  [ -z "$id" ] && return 0
  docker exec "$id" sh -c 'ls /dev/nvidia[0-9]* >/dev/null 2>&1' || return 0
  gpu_device_count >/dev/null && return 0
  echo "devcontainer: WARNING -- GPU nodes are present but the CUDA driver reports no device." >&2
  echo "devcontainer:   the container's --gpus device-cgroup grant was revoked (host driver" >&2
  echo "devcontainer:   reload or a systemd daemon-reload). Recover by recreating it:" >&2
  echo "devcontainer:       devtools/devcontainer.sh rebuild" >&2
  echo "devcontainer:   (a plain 'up' reuses this container and will NOT restore the GPU.)" >&2
}

require_running() {
  command -v docker >/dev/null 2>&1 || return 0
  [ -n "$(container_ids || true)" ] && return 0
  if [ -n "$(container_ids --stopped || true)" ]; then
    echo "error: the ${VARIANT:-active} container for $WORKSPACE exists, but is stopped." >&2
  else
    echo "error: no ${VARIANT:-active} container for $WORKSPACE." >&2
  fi
  echo "       start it: devtools/devcontainer.sh ${variant_flag:+$variant_flag }up" >&2
  echo "       (rebuild instead if its devcontainer.json or Dockerfile has moved on)" >&2
  exit 1
}

# Tag the image this variant just built with the variant's name.
#
# WHY: the CLI derives a built image's name from the workspace folder ALONE --
# `vsc-<basename>-<sha256 of the path>`, with the config file nowhere in it --
# while it keys the CONTAINER on both. So the three variants of one checkout
# share a single image name, and every `up` retags it away from whichever
# variant built it last, leaving that one's image untagged: still on disk,
# still held by its container, and no longer attributable in `docker images`.
# There is no `--image-name` on `up` to head that off -- the CLI offers that
# flag on `build` only -- so the tag goes on afterwards.
#
# It is a SECOND tag on the SAME repository, deliberately: `vsc-<...>:cuda`
# beside the CLI's `vsc-<...>:latest`, not a name of our own. worktree.sh's
# `rm` and `gc` recognise this project's build images by that
# `vsc-<basename>-<64hex>` repository and delete by image id, so they keep
# sweeping these with no edit there; a parallel naming scheme would leak past
# both.
#
# Non-fatal throughout, like apply_cpu_limits: `up` has already done its work
# by this point, and a missing tag costs legibility rather than a container.
tag_variant_image() {
  [ -n "$VARIANT" ] || return 0
  local cid repo img
  cid=$(container_ids | head -1 || true)
  [ -n "$cid" ] || return 0
  repo=$(docker inspect "$cid" --format '{{.Config.Image}}' 2>/dev/null || true)
  img=$(docker inspect "$cid" --format '{{.Image}}' 2>/dev/null || true)
  # Only an image the CLI built for this workspace. A config that names a
  # prebuilt `image` runs someone else's tag, and stamping one of our variant
  # names onto that would claim something about an image we did not build.
  case "$repo" in vsc-*) ;; *) return 0 ;; esac
  [ -n "$img" ] || return 0
  if docker tag "$img" "${repo%%:*}:$VARIANT" 2>/dev/null; then
    echo "image: tagged ${repo%%:*}:$VARIANT" >&2
  fi
}

# Say so when the Claude Code the image installs has fallen behind, or when
# the three Dockerfiles have drifted apart.
#
# WHY HERE: `up` and `rebuild` are the only moments the version in the
# container can change, and the failure this guards against is a SILENT one --
# the agent in the image used to come from a devcontainer feature that
# npm-installed an unpinned version into a permanently cached layer, so every
# rebuild reported success and moved nothing (docker/install-claude-code.sh
# has the measurements). Now that the pin is a line in git, the only thing
# left to notice is that the line is old, and nothing else would.
#
# Quiet when there is nothing to say, non-fatal always -- the container is
# already up by this point, and a registry that cannot be reached is not a
# reason to colour the run red. The timeout is short for the same reason: this
# is on the path of every `up`.
report_claude_pin() {
  local cv out pinned registry files
  cv="$REPO_ROOT/devtools/claude-version.sh"
  [ -x "$cv" ] || return 0
  [ -n "${CLAUDE_PIN_FILES:-}" ] || return 0
  out=$(CLAUDE_REGISTRY_TIMEOUT=${CLAUDE_REGISTRY_TIMEOUT:-3} "$cv" --porcelain 2>/dev/null)
  pinned=$(printf '%s\n' "$out" | sed -n 's/^pinned //p')
  registry=$(printf '%s\n' "$out" | sed -n 's/^registry //p')
  files=$(printf '%s\n' "$out" | sed -n 's/^files //p')
  if [ -z "$pinned" ] || [ "$pinned" = "-" ]; then
    echo "claude: the version pin disagrees across the Dockerfiles -- $files" >&2
    echo "        fix: devtools/claude-version.sh --apply <version>" >&2
  elif [ -n "$registry" ] && [ "$pinned" != "$registry" ]; then
    echo "claude: image pins $pinned, npm has $registry" >&2
    echo "        bump: devtools/claude-version.sh --apply, then $0 rebuild" >&2
  fi
}

# The same, for the other agent the images ship. Same reasoning, same silence
# when there is nothing to say: `up`/`rebuild` is the only moment the `agy` in
# the container can change, and an old pin is invisible otherwise -- more so
# than claude's, since `agy` would normally keep itself current and the image
# deliberately turns that off (AGY_CLI_DISABLE_AUTO_UPDATE).
#
# It reports VERSIONS while the pin is a release id (version plus build id --
# docker/install-agy.sh has why); the ids are what get compared, the versions
# what get printed, because the build id means nothing to a reader.
report_agy_pin() {
  local av out pinned pinned_v registry registry_v files
  av="$REPO_ROOT/devtools/agy-version.sh"
  [ -x "$av" ] || return 0
  [ -n "${AGY_PIN_FILES:-}" ] || return 0
  out=$(AGY_REGISTRY_TIMEOUT=${AGY_REGISTRY_TIMEOUT:-3} "$av" --porcelain 2>/dev/null)
  pinned=$(printf '%s\n' "$out" | sed -n 's/^pinned //p')
  pinned_v=$(printf '%s\n' "$out" | sed -n 's/^pinned_version //p')
  registry=$(printf '%s\n' "$out" | sed -n 's/^registry //p')
  registry_v=$(printf '%s\n' "$out" | sed -n 's/^registry_version //p')
  files=$(printf '%s\n' "$out" | sed -n 's/^files //p')
  if [ -z "$pinned" ] || [ "$pinned" = "-" ]; then
    echo "agy: the release pin disagrees across the Dockerfiles -- $files" >&2
    echo "     fix: devtools/agy-version.sh --apply" >&2
  elif [ -n "$registry" ] && [ "$registry" != "-" ] && [ "$pinned" != "$registry" ]; then
    echo "agy: image pins $pinned_v, upstream has $registry_v" >&2
    echo "     bump: devtools/agy-version.sh --apply, then $0 rebuild" >&2
  fi
}

# The CLI prints the whole `docker run` invocation on failure, GH_TOKEN and
# all. Everything below routes through this so a secret cannot reach the
# terminal; the exit status is the CLI's, not sed's.
#
# GEMINI_API_KEY and ANTHROPIC_API_KEY_FREE_BREAKFAST are on the list for a
# second reason: remote_env_args puts them on the CLI's own command line as a
# `--remote-env`, so either can be echoed back by any error that quotes the
# invocation, not only by a failed `docker run`.
#
# `sed -u` is load-bearing, not a micro-optimisation. sed's own stdout is the
# terminal, so without it sed is LINE-buffered: anything not terminated by a
# newline is held indefinitely. That silently swallows every partial-line
# write the CLI passes through -- pytest's progress dots, a compiler's
# carriage-returned status, and (before `shell` stopped routing through here)
# the shell prompt itself. GNU sed only; the images and the host are both
# Ubuntu.
run_cli() {
  local rc=0
  "${CLI[@]}" "$@" > >(sed -u -E 's/(GH_TOKEN|GITHUB_TOKEN|GEMINI_API_KEY|ANTHROPIC_API_KEY_FREE_BREAKFAST)=[A-Za-z0-9_.-]+/\1=***REDACTED***/g') \
    2> >(sed -u -E 's/(GH_TOKEN|GITHUB_TOKEN|GEMINI_API_KEY|ANTHROPIC_API_KEY_FREE_BREAKFAST)=[A-Za-z0-9_.-]+/\1=***REDACTED***/g' >&2) || rc=$?
  # Let the redirections drain before the caller reads the next line.
  wait
  return "$rc"
}

case "${1:-}" in
  up)
    shift
    require_git_dir
    require_cache_dirs
    export_host_gids
    host_mount_args
    remote_env_args
    run_cli up --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" \
      "${HOST_MOUNT_ARGS[@]}" "${REMOTE_ENV_ARGS[@]}" "$@"
    name_host_gids
    tag_variant_image
    apply_cpu_limits
    # `up` may have reused a container whose GPU grant was revoked mid-life; a
    # reuse cannot restore it, so flag it here and point at `rebuild`.
    warn_if_gpu_revoked
    report_claude_pin
    report_agy_pin
    ;;
  rebuild)
    # Needed after editing devcontainer.json or the Dockerfile: a plain `up`
    # reuses the running container and silently applies none of it, while
    # reporting success and the same container id. The workspace is a bind
    # mount and the caches are named volumes, so both survive; anything in the
    # container's own filesystem does not.
    shift
    require_git_dir
    require_cache_dirs
    export_host_gids
    host_mount_args
    remote_env_args
    run_cli up --workspace-folder "$WORKSPACE" \
      --remove-existing-container "${DC_CONFIG[@]}" \
      "${HOST_MOUNT_ARGS[@]}" "${REMOTE_ENV_ARGS[@]}" "$@"
    # Again on `rebuild`, not only `up`: a fresh container means a fresh
    # /etc/group, so the entry has to be re-added rather than inherited.
    name_host_gids
    tag_variant_image
    apply_cpu_limits
    # A fresh container should have the GPU; if this still warns, the problem is
    # host-side (driver/cgroup), not a stale container -- see doctor.sh.
    warn_if_gpu_revoked
    report_claude_pin
    report_agy_pin
    ;;
  shell)
    shift
    require_running
    # An INTERACTIVE shell must not route through run_cli, and the reason is
    # not the redaction but the pipe it needs to do it. Two things break once
    # the CLI's stdout is not a terminal:
    #
    #   1. sed withholds the prompt. A prompt ends in "$ ", not a newline, so
    #      a line-buffered filter holds it forever -- you get a blank screen
    #      and no echo of what you type, on a container that is working
    #      perfectly. (`sed -u` above fixes this much.)
    #   2. The terminal SIZE is lost regardless. The CLI decides whether to
    #      ask docker for a pty from process.stdin.isTTY, so a pty is still
    #      allocated, but it is sized from process.stdout.columns -- undefined
    #      through a pipe. Every full-screen program in there then draws into
    #      a default 80x24 that does not match your window.
    #
    # There is nothing to redact on this path anyway: GH_TOKEN reaches the
    # terminal from the `docker run` line that `up`/`rebuild` print on
    # failure, and `exec` prints no such line -- the token was baked into the
    # container's environment at create time.
    #
    # WITH arguments (`shell -c '...'`) it is a one-shot command rather than a
    # terminal session, so keep the filter: that is the path scripts and
    # agents use, and its output is what gets pasted into an issue.
    remote_env_args
    if [ $# -eq 0 ]; then
      exec "${CLI[@]}" exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" \
        "${REMOTE_ENV_ARGS[@]}" bash
    fi
    run_cli exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" \
      "${REMOTE_ENV_ARGS[@]}" bash "$@"
    ;;
  test)
    shift
    require_running
    # TEST_CMD may carry its own arguments, so split it as a command line
    # rather than on whitespace alone.
    config_args "$TEST_CMD"
    cmd=("${CONFIG_ARGS[@]}")
    # -rs because tests that degrade-to-skip on a missing tool make a green run
    # meaningless until you have read the skip reasons.
    xdist_args
    cmd+=("${XDIST[@]}" -rs)
    # Default to the fast tier, unless the caller passed their own marker
    # expression -- then which tier runs is their call.
    case " $* " in
      *" -m "*) ;;
      *) config_args "$FAST_TEST_ARGS"; cmd+=("${CONFIG_ARGS[@]}") ;;
    esac
    remote_env_args
    run_cli exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" \
      "${REMOTE_ENV_ARGS[@]}" "${cmd[@]}" "$@"
    ;;
  down)
    shift
    # By default this takes down the ACTIVE variant only -- the one
    # DEVCONTAINER_CONFIG names -- because a cuda and a hip container for the
    # same worktree are two different containers that merely share a workspace
    # folder. `down --all` takes down every variant of this workspace, which
    # is what you want before `worktree.sh rm` or when you have lost track.
    scope=()
    [ "${1:-}" = --all ] && scope=(--any-config)
    ids=$(container_ids --stopped "${scope[@]}")
    if [ -z "$ids" ]; then
      echo "no container for $WORKSPACE${scope:+ (any config)}"
    else
      # One id per line, and there can be more than one, so loop: a single
      # quoted "$ids" would reach docker as one newline-joined name and fail
      # with "No such container" while leaving both behind.
      while read -r id; do
        [ -n "$id" ] || continue
        docker rm -f "$id"
      done <<<"$ids"
    fi
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "unknown command: $1" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
