#!/usr/bin/env bash
# Report what is degraded in this environment, so "why did half the suite skip"
# is one command rather than a green run you cannot trust. Runs on the host and
# inside the container. Exits non-zero only on a FAIL (the core suite cannot
# run); a WARN means a slice of the suite will skip, which is often fine.
#
#   devtools/doctor.sh
#
# What counts as optional here is project-specific and comes from
# devtools/config.sh: DOCTOR_OPTIONAL_TOOLS, DOCTOR_GPU_VENDORS and
# DOCTOR_REQUIRED_PATHS. Add the project's own device compiler, GPU vendor,
# submodule or fixture tree there rather than editing this script.
set -uo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

fails=0
warns=0
ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m[warn]\033[0m %s\n'  "$1"; warns=$((warns+1)); }
fail() { printf '  \033[31m[FAIL]\033[0m %s\n'  "$1"; fails=$((fails+1)); }
note() { printf '         %s\n' "$1"; }

# --- repo -----------------------------------------------------------------
echo "repo"
TOP=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$TOP" ]; then
  fail "not inside a git repository"
  echo; echo "cannot check anything else outside the repo"; exit 1
fi
ok "git repo at $TOP"

# The <root>/<branch> layout is anchored at the PRIMARY checkout, not at
# whichever worktree doctor happens to be run from: the parent of the shared
# git common dir is the primary worktree, and its parent is the root. Deriving
# these from $TOP was wrong in exactly the place it matters -- run from
# .claude/worktrees/<name> it called .claude/worktrees the "worktree root" and
# looked for main at .claude/worktrees/main.
COMMON=$(git -C "$TOP" rev-parse --git-common-dir 2>/dev/null)
case "$COMMON" in /*) : ;; *) COMMON="$TOP/$COMMON" ;; esac
PRIMARY=$(cd "$(dirname "$COMMON")" 2>/dev/null && pwd) || PRIMARY=$TOP

# The worktree root must be writable or `devtools/worktree.sh add` cannot make
# sibling checkouts.
ROOT=$(dirname "$PRIMARY")
if [ -w "$ROOT" ]; then
  ok "worktree root writable ($ROOT)"
else
  warn "worktree root not writable: $ROOT"
  note "worktree.sh add will fail; fix once: sudo chown $(id -u) $ROOT"
fi

# Every devcontainer.json in the tree, relative to $TOP: one per GPU variant
# (cuda, hip, combined). Discovered rather than listed, so adding a variant does
# not need an edit here -- the checks below are exactly the ones a new variant is
# most likely to get wrong.
dc_configs() {
  local p
  for p in "$TOP"/.devcontainer/*/devcontainer.json; do
    [ -f "$p" ] || continue          # no match: the glob comes back literal
    echo "${p#"$TOP"/}"
  done
}

# PROJECT_NAME is duplicated in every devcontainer.json (JSON cannot source
# shell), and if they drift, worktree.sh rm/gc stop recognising this project's
# volumes and silently leak them. Every variant is checked, not just the active
# one: a rename that missed any of them leaks from whichever side you forgot.
while read -r rel; do
  DCJSON="$TOP/$rel"
  [ -f "$DCJSON" ] || continue
  if grep -q "source=$PROJECT_NAME-" "$DCJSON"; then
    ok "PROJECT_NAME '$PROJECT_NAME' matches $rel volume names"
  elif grep -q 'type=volume' "$DCJSON"; then
    warn "$rel names volumes that do not start with '$PROJECT_NAME-'"
    note "worktree.sh rm/gc will not recognise them; align the two"
    note "  config: devtools/config.sh PROJECT_NAME   json: $DCJSON"
  fi
done < <(dc_configs)

# Which of them devcontainer.sh will actually drive. A path that does not exist
# is a hard stop for every container command, so it is a FAIL rather than a
# warning.
case "${DEVCONTAINER_CONFIG:-}" in
  "") note "DEVCONTAINER_CONFIG unset: there is no top-level .devcontainer/devcontainer.json; pass a variant flag or set it" ;;
  /*) DCACTIVE=$DEVCONTAINER_CONFIG ;;
  *) DCACTIVE="$TOP/$DEVCONTAINER_CONFIG" ;;
esac
if [ -n "${DCACTIVE:-}" ]; then
  if [ -f "$DCACTIVE" ]; then
    ok "devcontainer config: $DEVCONTAINER_CONFIG"
  else
    fail "DEVCONTAINER_CONFIG does not exist: $DCACTIVE"
    note "every devtools/devcontainer.sh command will refuse; fix devtools/config.sh"
  fi
fi

# The .git bind mount reads its host path from ${localEnv:FREE_BREAKFAST_GIT_DIR}, which
# devtools/devcontainer.sh injects on up/rebuild (the literal in the json is only
# a fallback for a direct VS Code "Reopen in Container"). So the thing worth
# checking is that the path it will inject actually resolves -- an unresolved
# mount makes the CLI print the failing `docker run` line, GH_TOKEN and all.
gitdir=$(git -C "$TOP" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
if [ -n "$gitdir" ] && [ -d "$gitdir" ]; then
  ok "container .git mount resolves via FREE_BREAKFAST_GIT_DIR ($gitdir)"
else
  warn "cannot resolve this repo's git dir for the container .git mount"
  note "run doctor from inside the FreeBreakfast checkout"
fi

# --- worktrees ------------------------------------------------------------
# Two failure modes that are local usability damage rather than lost work
# (committed work is safe on the remote), so these are warn/note, never fail:
# a primary checkout owned by another uid, and linked worktrees whose admin
# state has broken.
echo "worktrees"

owner=$(stat -c %u "$PRIMARY" 2>/dev/null)
if [ -n "$owner" ] && [ "$owner" != "$(id -u)" ]; then
  warn "primary checkout owned by uid $owner, not $(id -u): $PRIMARY"
  note "git and worktree ops there fail; fix: sudo chown -R $(id -u) $ROOT"
else
  ok "primary checkout owned by current uid ($PRIMARY)"
fi
if [ ! -d "$ROOT/main" ]; then
  note "no sibling main checkout at $ROOT/main (not the <root>/<branch> layout here)"
fi

# `prunable` means git found a linked worktree whose gitdir points nowhere --
# the working tree was removed out from under it. A worktrees/<name> admin dir
# with no readable `gitdir` file is the other half. `git worktree prune` clears
# both.
prunable=$(git worktree list --porcelain 2>/dev/null | grep -c '^prunable')
corrupt=0
if [ -d "$COMMON/worktrees" ]; then
  for wtd in "$COMMON"/worktrees/*/; do
    [ -d "$wtd" ] || continue
    [ -r "${wtd}gitdir" ] || corrupt=$((corrupt+1))
  done
fi
broken=$((prunable + corrupt))
if [ "$broken" -gt 0 ]; then
  warn "$broken broken/prunable linked worktree(s) ($prunable prunable, $corrupt corrupt admin dir)"
  note "clean up: git worktree prune; then recreate via devtools/worktree.sh"
else
  ok "no broken/prunable linked worktrees"
fi

# --- python / test tooling (the core suite) -------------------------------
echo "python / test tooling"
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
else
  warn "uv not found -- setup builds the venv from uv.lock with it"
fi
if venv=$(find_venv); then
  ok "venv: $venv"
else
  warn "no venv found in '$VENV_PATHS' -- run 'uv sync'"
fi

# find_venv's `-x bin/python` FOLLOWS the symlink, so a venv whose interpreter
# is unreachable reads as no venv at all -- and `uv sync`, the advice above,
# needs that same interpreter and fails with it. An interpreter that IS
# reachable can still be unusable: one whose stdlib sits behind a directory this
# uid cannot traverse prints --version happily, then dies on `import encodings`.
# Either way the message above points at the wrong thing, so name the real
# fault. Inside a container it means a stale image, nothing a command here fixes.
for vpath in $VENV_PATHS; do
  case "$vpath" in /*) vdir=$vpath ;; *) vdir=$TOP/$vpath ;; esac
  [ -L "$vdir/bin/python" ] || [ -f "$vdir/bin/python" ] || continue
  if "$vdir/bin/python" -c "import json" >/dev/null 2>&1; then
    ok "interpreter usable: $vdir/bin/python"
  else
    warn "'$vdir/bin/python' exists but cannot run as uid $(id -u)"
    note "target: $(readlink "$vdir/bin/python" 2>/dev/null || echo '<not a symlink>')"
    note "in a container: stale image -- rebuild with docker/build.sh <variant>"
    note "on the host: rm -rf '$vdir' && uv sync"
  fi
done
if pytest_bin=$(find_pytest); then
  ok "pytest: $pytest_bin"
else
  warn "pytest not found -- run 'uv sync' (dev group)"
fi

# uv.lock records the project name, and `uv sync --frozen` -- which is what both
# Dockerfiles and CI run -- refuses a lock that disagrees with pyproject.toml.
# The failure lands at IMAGE BUILD time with a message about the lock being out
# of date, a long way from the pyproject edit or the half-finished rename that
# caused it, so it is worth one grep here. Only the name is checked: anything
# deeper is uv's job, and `uv lock` is the fix for all of it.
if [ -f "$TOP/pyproject.toml" ] && [ -f "$TOP/uv.lock" ]; then
  py_name=$(sed -n 's/^name = "\(.*\)"/\1/p' "$TOP/pyproject.toml" | head -1)
  lock_name=$(sed -n 's/^name = "\(.*\)"/\1/p' "$TOP/uv.lock" |
    grep -Fx "$py_name" | head -1)
  if [ -n "$py_name" ] && [ "$lock_name" = "$py_name" ]; then
    ok "uv.lock names the project '$py_name'"
  else
    warn "uv.lock does not name the project '$py_name' -- it is stale"
    note "'uv sync --frozen' will fail in both Dockerfiles and in CI"
    note "fix: uv lock, then commit the result (never hand-edit uv.lock)"
  fi
fi

# --- optional tools (project-declared) ------------------------------------
echo "optional tools"
config_lines "$DOCTOR_OPTIONAL_TOOLS"
if [ "${#CONFIG_LINES[@]}" -eq 0 ]; then
  note "none declared (DOCTOR_OPTIONAL_TOOLS in devtools/config.sh)"
fi
for entry in "${CONFIG_LINES[@]}"; do
  tool=${entry%%:*}
  what=${entry#*:}
  # find_tool, not `command -v`: the cmakelang tools come from the project venv
  # and are on PATH inside the image but not on the host. See devtools/lib.sh.
  if toolpath=$(find_tool "$tool"); then
    ver=$("$toolpath" --version 2>/dev/null | head -1)
    ok "$tool${ver:+: $ver}"
    command -v "$tool" >/dev/null 2>&1 || note "not on PATH; using $toolpath"
  else
    warn "$tool not found -- lost:${what:+ $what}"
  fi
done

# gh is the one optional tool with a second half: installed but unauthenticated
# is a distinct state, and the PR flow fails at push time rather than at start.
if command -v gh >/dev/null 2>&1 && [ -z "${GH_TOKEN:-}" ]; then
  warn "gh present but GH_TOKEN empty -- gh is unauthenticated"
  note "export a fine-grained PAT (Contents + Pull requests: R/W) on the host"
# A token can authenticate as the right user and still be unable to see THIS
# repo: a fine-grained PAT names its repositories one by one, and one that has
# not been given this one 404s. `gh auth status` reports a clean login either
# way, so the first sign of trouble is `gh pr create` failing with "Could not
# resolve to a Repository" after the branch is already pushed. One API call
# here turns that into a warning you get before starting.
elif command -v gh >/dev/null 2>&1; then
  # Match on github.com FIRST, then extract. Stripping known URL prefixes and
  # accepting anything left containing a slash was wrong: a clone from a local
  # path has an origin like /home/you/projects/thing, which survives every
  # substitution and still looks like `owner/repo` to a `*/*` glob. The probe
  # then asked GitHub for `repos//home/you/...`, got a 404, and reported "the
  # token lacks access to this repository" about a repository that is not on
  # GitHub at all. Same for a GitLab remote, and for a fresh `git init` with no
  # origin yet.
  ORIGIN=$(git -C "$TOP" remote get-url origin 2>/dev/null || true)
  case "$ORIGIN" in
    *github.com[:/]*)
      SLUG=$(printf '%s' "$ORIGIN" | sed -E 's#^.*github\.com[:/]+##; s#\.git$##; s#/+$##')
      # owner/repo and nothing else: two non-empty segments, no extra slashes.
      if printf '%s' "$SLUG" | grep -qE '^[^/]+/[^/]+$'; then
        if ${GH_PROBE_TIMEOUT:-timeout 10} gh api "repos/$SLUG" >/dev/null 2>&1; then
          ok "gh can reach $SLUG"
        else
          warn "gh cannot see $SLUG -- the PR and milestone flows will fail"
          note "the token authenticates but lacks access to this repository;"
          note "add it under Repository access on the fine-grained PAT"
        fi
      else
        note "origin looks like github but is not owner/repo: $ORIGIN"
      fi
      ;;
    "") note "no origin remote yet -- nothing to check gh against" ;;
    *) note "origin is not a github remote; skipping the gh reachability check" ;;
  esac
fi

# --- gpu devices (is a card actually here?) --------------------------------
#
# Deliberately separate from the toolchain checks above, because the two fail
# independently and only this half is invisible without a probe. A missing
# nvcc announces itself; a missing CARD does not, and its symptoms are easy to
# read as a broken change:
#
#   - CMAKE_CUDA_ARCHITECTURES=native queries a live driver at CONFIGURE time,
#     so with no NVIDIA card the `default` preset cannot even configure -- for
#     a change that touches no CUDA.
#   - Nothing in test/ calls GTEST_SKIP, so the runtime suites FAIL on a
#     device-less box rather than skipping and naming themselves.
#
# Probed through the kernel (sysfs / procfs) before any vendor tool, so the
# answer is the same on the host and inside a container, with or without
# nvidia-smi or rocminfo installed.
if [ -n "${DOCTOR_GPU_VENDORS// /}" ]; then
  echo "gpu devices"
  gpus_found=0

  # gfx_target_version is major*10000 + minor*100 + step, and the step prints
  # in HEX in the ISA name -- 90010 is gfx90a, not gfx9010. Getting that wrong
  # produces an --offload-arch the compiler silently has no code for.
  amd_gfx_name() {
    printf 'gfx%d%d%x' "$(( $1 / 10000 ))" "$(( ($1 / 100) % 100 ))" "$(( $1 % 100 ))"
  }

  for vendor in $DOCTOR_GPU_VENDORS; do
    case $vendor in
      nvidia)
        # nvidia-smi first when present: it is the only one of these that also
        # reports the compute capability, which is the number the presets want.
        # The container runtime injects it under --gpus, so this is the usual
        # path on both sides.
        nv_listed=0
        if command -v nvidia-smi >/dev/null 2>&1; then
          while IFS=, read -r nv_name nv_cap; do
            nv_name=${nv_name# }; nv_cap=${nv_cap# }
            [ -n "$nv_name" ] || continue
            ok "nvidia: $nv_name (compute ${nv_cap} -> CMAKE_CUDA_ARCHITECTURES=${nv_cap//./})"
            nv_listed=$((nv_listed+1))
          done < <(nvidia-smi --query-gpu=name,compute_cap \
                     --format=csv,noheader 2>/dev/null)
        fi
        if [ "$nv_listed" -eq 0 ]; then
          # No nvidia-smi (or it failed): fall back to the driver's own procfs,
          # which names the card but not its compute capability.
          for info in /proc/driver/nvidia/gpus/*/information; do
            [ -r "$info" ] || continue
            nv_name=$(sed -n 's/^Model: *//p' "$info" | head -1)
            [ -n "$nv_name" ] || continue
            ok "nvidia: $nv_name"
            note "nvidia-smi absent -- compute capability not read; pin an arch preset"
            nv_listed=$((nv_listed+1))
          done
        fi
        if [ "$nv_listed" -gt 0 ]; then
          gpus_found=$((gpus_found+nv_listed))
          find_tool nvcc >/dev/null 2>&1 ||
            note "card present but nvcc is not -- the CUDA toolchain is in the \`cuda\` image"
        else
          note "no nvidia device -- the \`native\` CUDA presets cannot configure here"
        fi
        ;;
      amd)
        # /sys/class/kfd is amdkfd's own topology, the same source rocminfo
        # reads, and it is visible inside a container given --device=/dev/kfd.
        # Node 0 is the CPU node and reports gfx_target_version 0; skip it.
        amd_listed=0
        amd_wedged=0
        for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
          [ -r "$props" ] || continue
          gfxver=$(sed -n 's/^gfx_target_version //p' "$props" | head -1)
          [ -n "$gfxver" ] && [ "$gfxver" != 0 ] || continue
          nodedir=${props%/properties}
          isa=$(amd_gfx_name "$gfxver")
          # The PCI ids, not the node's `name`: amdkfd fills that with the
          # generic "ip discovery" on recent cards, and a marketing name is
          # exposed nowhere in sysfs (lspci's database may not know a new id
          # either). vendor/device are decimal in properties; print them hex,
          # which is how every lspci line and driver table spells them.
          amd_vid=$(sed -n 's/^vendor_id //p' "$props" | head -1)
          amd_did=$(sed -n 's/^device_id //p' "$props" | head -1)
          amd_pci=$(printf '%04x:%04x' "${amd_vid:-0}" "${amd_did:-0}")
          amd_name=$(tr -d '\0' < "$nodedir/name" 2>/dev/null)
          case $amd_name in ip\ discovery|'') amd_name= ;; *) amd_name=" $amd_name" ;; esac

          # ENUMERATED IS NOT THE SAME AS USABLE, and this is the one place
          # doctor could say `ok` about a card that can run nothing. The card
          # can wedge with amdgpu still bound and this topology node still
          # listing it, while every HIP program gets hipErrorNoDevice and
          # rocminfo gets HSA_STATUS_ERROR_OUT_OF_RESOURCES -- in which state a
          # HIP ctest run GTEST_SKIPs every GPU case and still prints 100%
          # passed. A green run nobody could distinguish from a real one is
          # exactly what this script exists to prevent.
          #
          # The tell is the PCI device's runtime-PM state, readable without
          # sudo and (unlike the dkms check below) from INSIDE the container
          # too, which is where the symptom is met. Reached through the KFD
          # node's own drm_render_minor rather than by guessing a cardN: the
          # numbering depends on probe order, and with two vendors' cards here
          # the amdgpu one is not card0. Only `error` is a fault -- `active`,
          # `suspended` and the transient `suspending`/`resuming` are all
          # normal, and a kernel without runtime PM says `unsupported`.
          amd_rminor=$(sed -n 's/^drm_render_minor //p' "$props" | head -1)
          amd_pm=
          if [ -n "$amd_rminor" ]; then
            amd_pm=$(cat "/sys/class/drm/renderD$amd_rminor/device/power/runtime_status" 2>/dev/null)
          fi
          if [ "$amd_pm" = error ]; then
            warn "amd:$amd_name $isa [$amd_pci] is enumerated but WEDGED (runtime_status error)"
            note "every HIP call returns hipErrorNoDevice; a hip ctest run skips every GPU case and still says 100% passed"
            note "host-side and not fixable from a container: sudo reboot (never modprobe -r amdgpu -- it restarts the display manager)"
            amd_wedged=$((amd_wedged+1))
          else
            ok "amd:$amd_name $isa [$amd_pci] (-> GPU_TARGETS=$isa)"
            amd_listed=$((amd_listed+1))
          fi
        done
        if [ "$amd_listed" -gt 0 ]; then
          # Only the usable ones count: a wedged card must not satisfy the
          # "some GPU is here" test at the bottom of this section.
          gpus_found=$((gpus_found+amd_listed))
          [ -e /dev/kfd ] ||
            note "/dev/kfd absent -- pass --device=/dev/kfd --device=/dev/dri to reach it"
          find_tool hipconfig >/dev/null 2>&1 ||
            note "card present but hipconfig is not -- ROCm is in the \`hip\`/\`combined\` image"
        elif [ "$amd_wedged" -gt 0 ]; then
          note "the \`hip\` preset builds but can run nothing until the host reboots"
        else
          note "no amd device -- the \`hip\` preset has nothing to run against"
        fi
        ;;
      *) note "unknown vendor '$vendor' in DOCTOR_GPU_VENDORS; skipping" ;;
    esac
  done

  if [ "$gpus_found" -eq 0 ]; then
    warn "no GPU of any declared vendor found -- runtime tests will FAIL, not skip"
    note "nothing in test/ calls GTEST_SKIP; read the BUILD as the result instead"
    note "use --preset compile-time (no device, no GoogleTest), or pin an arch"
  fi
fi

# --- gpu kernel modules (will the card survive the NEXT reboot?) ----------
#
# The section above answers "is a card here now". This one answers the
# question that costs a day when nobody asks it: will there still be one after
# the next reboot. Both vendors' drivers here are OUT-OF-TREE DKMS modules,
# rebuilt per kernel by the dkms apt trigger -- and a rebuild that FAILS is
# effectively silent. The kernel arrives through unattended-upgrades, apt's
# report of the failure goes to a log nobody reads, and the module already
# loaded for the RUNNING kernel keeps working, so the machine looks perfect
# until the reboot that picks the new kernel up. Then the card is simply gone:
# `rocminfo` lists only the CPU agent, `lspci -k` shows no driver for the card,
# and nothing in any of it names a kernel upgrade as the cause.
#
# Seen on amdgpu across 6.8.0-146 -> -147: Ubuntu gave pci_resize_resource() a
# fourth argument that amdgpu-dkms 6.16.13 does not pass, so the build broke
# and the card came back driverless (amdgpu-dkms 7.1.9 autoconf-tests for the
# signature). nvidia's DKMS is the same construction and can fail the same way.
#
# This is the HOST-side counterpart to the NVIDIA --device= lines in
# .devcontainer/{cuda,combined}/. Those stop a live card dropping out of a
# long-running container (a cgroup grant systemd erases on daemon-reload), and
# the AMD half never needed them because ../hip/ always named /dev/kfd and
# /dev/dri explicitly. What is left over is this, and it cannot be made
# declarative from in here -- no amount of container config rebuilds a kernel
# module. What a check CAN do is catch it in the one window where the fix is
# cheap: before the reboot, while the old kernel is still bootable and the card
# still works. So a warning here means exactly one thing -- do not reboot yet.
#
# Host-only by construction rather than by a probe: the container has no dkms
# and no /boot, so both guards below come back empty and the section skips.
if [ -n "${DOCTOR_GPU_VENDORS// /}" ] && command -v dkms >/dev/null 2>&1; then
  # The bootable set, oldest first: a kernel needs BOTH a /boot/vmlinuz- image
  # (what grub offers) and a /lib/modules/ tree (where dkms installs into). A
  # vmlinuz with no module tree is a half-removed kernel, not a reboot target,
  # and warning about one would be a false alarm -- images of that shape do
  # accumulate in /boot. `sort -V` because the ABI number is numeric:
  # lexically, -99 sorts after -147.
  kernels=()
  while read -r kver; do
    [ -n "$kver" ] && [ -d "/lib/modules/$kver" ] && kernels+=("$kver")
  done < <(for img in /boot/vmlinuz-*; do
             [ -f "$img" ] && echo "${img#/boot/vmlinuz-}"
           done | sort -V)

  if [ "${#kernels[@]}" -gt 0 ]; then
    echo "gpu kernel modules"
    dkms_status=$(dkms status 2>/dev/null)
    running=$(uname -r)
    karch=$(uname -m)
    newest=${kernels[${#kernels[@]}-1]}

    # dkms spells a usable module as
    #   <module>/<version>, <kernel>, <arch>: installed [(Original ...)]
    # with older dkms separating module and version by ", " instead of "/".
    # `installed` is the only state that is loadable -- a module can be `added`
    # or `built` and still not be in /lib/modules -- so the state is matched
    # too, not just the kernel appearing somewhere in the line.
    dkms_installed() {
      printf '%s\n' "$dkms_status" |
        grep -q "^$1[/,].*[,/ ]$2, *$karch: installed"
    }

    for vendor in $DOCTOR_GPU_VENDORS; do
      # The module name, which is not always the vendor name, and where a
      # replacement comes from -- the two differ per vendor and the warning is
      # worth little without the second. An unknown vendor has already been
      # reported by the section above.
      case $vendor in
        nvidia) kmod=nvidia; kfix="a newer nvidia-dkms package" ;;
        amd)    kmod=amdgpu
                kfix="a newer amdgpu-dkms from repo.radeon.com/amdgpu/latest" ;;
        *)      continue ;;
      esac

      if ! printf '%s\n' "$dkms_status" | grep -q "^$kmod[/,]"; then
        # Not a DKMS module here at all: either this vendor has no card on
        # this host, or its driver is the distro's in-tree one. Neither has a
        # per-kernel rebuild to fail, so neither is a warning.
        note "$kmod: not a DKMS module here -- no per-kernel rebuild to fail"
      elif ! dkms_installed "$kmod" "$running"; then
        warn "$kmod: no module built for the running kernel $running"
        note "this is the post-reboot state -- the card has no driver until it builds"
        note "why it failed: /var/lib/dkms/$kmod/*/build/make.log"
        note "fix: $kfix, then reboot"
      elif [ "$newest" != "$running" ] && ! dkms_installed "$kmod" "$newest"; then
        warn "$kmod: built for $running but NOT for $newest -- a reboot loses the card"
        note "why the rebuild failed: /var/lib/dkms/$kmod/*/build/make.log"
        note "fix before rebooting: $kfix, built against $newest"
        note "or boot $running from grub's advanced menu until it is fixed"
      elif [ "$newest" != "$running" ]; then
        ok "$kmod: built for $running (running) and $newest (next boot)"
      else
        ok "$kmod: built for $running (running, and the newest installed)"
      fi
    done
  fi
fi

# --- required paths (submodules, fixture trees) ---------------------------
config_lines "$DOCTOR_REQUIRED_PATHS"
if [ "${#CONFIG_LINES[@]}" -gt 0 ]; then
  echo "required paths"
  for entry in "${CONFIG_LINES[@]}"; do
    rpath=${entry%%:*}
    what=${entry#*:}
    # Absolute entries are taken as-is, for a path that lives outside the
    # checkout -- something baked into the image, say: this project's three
    # /opt trees (see config.sh for why they are listed). Relative ones stay
    # repo-relative (submodules, fixtures), so doctor works from any worktree.
    case "$rpath" in /*) abs=$rpath ;; *) abs="$TOP/$rpath" ;; esac
    if [ -e "$abs" ]; then
      ok "$rpath present"
    else
      warn "$rpath absent --${what:+ $what}"
      case "$rpath" in
        /*) ;;
        *) note "if it is a submodule: git submodule update --init --recursive" ;;
      esac
    fi
  done
fi

# --- pre-commit hook (the shared-hook footgun) ----------------------------
echo "pre-commit hook"
# The hook lives in the *common* git dir, shared by the main repo and every
# worktree, and bakes in an absolute INSTALL_PYTHON. Installed from the wrong
# side (container vs host) it points at a python that does not exist here and
# every commit fails with "No module named 'pre_commit'". Test that exact path
# -- the same check the hook itself does -- rather than guessing which side we
# are on, so a hybrid environment does not produce a false alarm.
HOOK="$COMMON/hooks/pre-commit"
if [ ! -e "$HOOK" ]; then
  warn "no pre-commit hook installed -- run 'pre-commit install' on your commit side"
else
  ip=$(sed -n 's/^INSTALL_PYTHON=//p' "$HOOK" | head -1)
  if [ -n "$ip" ] && [ -x "$ip" ]; then
    ok "pre-commit hook usable (baked python)"
  elif command -v pre-commit >/dev/null 2>&1; then
    ok "pre-commit hook usable (pre-commit on PATH; baked python absent here)"
  else
    warn "pre-commit hook unusable: baked python not executable and no pre-commit on PATH"
    note "commits fail with 'No module named pre_commit'; fix: pre-commit install -f (from this side)"
  fi
fi

# --- claude code pin (the agent the image ships) ---------------------------
# The GPU Dockerfiles install Claude Code at an explicit ARG
# CLAUDE_CODE_VERSION -- see docker/install-claude-code.sh for the drift that
# replaced: a devcontainer FEATURE npm-installs an unpinned version, its layer
# is a permanent cache hit, and `rebuild` recreates the container from the same
# image, so an image can sit ~90 releases behind with nothing reporting it.
# Three ways that can still go wrong, one check each. CLAUDE_PIN_FILES in
# config.sh is the list of files carrying the pin; empty drops the section.
CV="$TOP/devtools/claude-version.sh"
if [ -n "${CLAUDE_PIN_FILES:-}" ] && [ -x "$CV" ]; then
  echo "claude code pin"
  cv=$("$CV" --porcelain 2>/dev/null)
  cv_pinned=$(printf '%s\n' "$cv" | sed -n 's/^pinned //p')
  cv_files=$(printf '%s\n' "$cv" | sed -n 's/^files //p')
  cv_registry=$(printf '%s\n' "$cv" | sed -n 's/^registry //p')
  cv_installed=$(printf '%s\n' "$cv" | sed -n 's/^installed //p')

  # 1. the copies agree with each other.
  if [ -z "$cv_pinned" ] || [ "$cv_pinned" = "-" ]; then
    warn "the Claude Code pin disagrees across the Dockerfiles"
    note "$cv_files"
    note "fix: devtools/claude-version.sh --apply <version>"
  # 2. and with the registry. Behind is a WARN, never a FAIL: a pin is a
  #    deliberate choice, and the point is only that the gap is visible.
  elif [ -z "$cv_registry" ]; then
    ok "Claude Code pinned at $cv_pinned"
    note "npm registry unreachable from here -- cannot say whether it is current"
  elif [ "$cv_pinned" != "$cv_registry" ]; then
    warn "Claude Code pin $cv_pinned is behind the registry ($cv_registry)"
    note "bump: devtools/claude-version.sh --apply && docker/build.sh cuda"
  else
    ok "Claude Code pinned at $cv_pinned (current)"
  fi

  # 3. and, inside an image built from that pin, with what is on PATH.
  #    CLAUDE_CODE_VERSION in the environment is the image's own ENV, so its
  #    presence is what tells container from host.
  if [ -n "${CLAUDE_CODE_VERSION:-}" ]; then
    if [ "$cv_installed" != "$CLAUDE_CODE_VERSION" ]; then
      warn "claude on PATH is ${cv_installed:-none}, this image pinned $CLAUDE_CODE_VERSION"
      note "something npm-installed over the image's copy; rebuild the container"
    elif [ -n "$cv_pinned" ] && [ "$CLAUDE_CODE_VERSION" != "$cv_pinned" ]; then
      warn "this image was built at $CLAUDE_CODE_VERSION, the tree now pins $cv_pinned"
      note "rebuild: devtools/devcontainer.sh rebuild"
    else
      ok "claude in here is the pinned $CLAUDE_CODE_VERSION"
    fi
  else
    note "claude here is ${cv_installed:-not installed} (a host install, unrelated to the pin)"
  fi
fi

# --- antigravity cli pin (the other agent the image ships) -----------------
# The same three failure modes as the Claude pin above, one check each. Two
# differences, both from upstream rather than from choice: the pin is a RELEASE
# id (version plus an opaque build id) because the updater endpoint serves only
# the current release and the storage URL is the only addressable artifact; and
# `agy` normally self-updates in the background, which the image turns off
# (AGY_CLI_DISABLE_AUTO_UPDATE) precisely so that the pin means something --
# which in turn makes an unnoticed old pin more likely here than for claude.
# docker/install-agy.sh has the rest. AGY_PIN_FILES in config.sh is the list of
# files carrying the pin; empty drops the section.
AV="$TOP/devtools/agy-version.sh"
if [ -n "${AGY_PIN_FILES:-}" ] && [ -x "$AV" ]; then
  echo "antigravity cli pin"
  av=$("$AV" --porcelain 2>/dev/null)
  av_pinned=$(printf '%s\n' "$av" | sed -n 's/^pinned //p')
  av_pinned_v=$(printf '%s\n' "$av" | sed -n 's/^pinned_version //p')
  av_files=$(printf '%s\n' "$av" | sed -n 's/^files //p')
  av_upstream=$(printf '%s\n' "$av" | sed -n 's/^registry //p')
  av_upstream_v=$(printf '%s\n' "$av" | sed -n 's/^registry_version //p')
  av_installed=$(printf '%s\n' "$av" | sed -n 's/^installed //p')

  # 1. the copies agree with each other -- release AND sha512, since a release
  #    pinned against another release's hash is a build that fails its
  #    checksum, which reads as a corrupt download rather than as drift.
  if [ -z "$av_pinned" ] || [ "$av_pinned" = "-" ]; then
    warn "the Antigravity CLI pin disagrees across the Dockerfiles"
    note "$av_files"
    note "fix: devtools/agy-version.sh --apply"
  # 2. and with upstream. Behind is a WARN, never a FAIL: a pin is a
  #    deliberate choice, and the point is only that the gap is visible.
  elif [ -z "$av_upstream" ] || [ "$av_upstream" = "-" ]; then
    ok "Antigravity CLI pinned at $av_pinned_v"
    note "the release manifest is unreachable from here -- cannot say whether it is current"
  elif [ "$av_pinned" != "$av_upstream" ]; then
    warn "Antigravity CLI pin $av_pinned_v is behind upstream ($av_upstream_v)"
    note "bump: devtools/agy-version.sh --apply && docker/build.sh cuda"
  else
    ok "Antigravity CLI pinned at $av_pinned_v (current)"
  fi

  # 3. and, inside an image built from that pin, with what is on PATH.
  #    AGY_RELEASE in the environment is the image's own ENV, so its presence
  #    is what tells container from host -- and it is also the only place the
  #    build id survives, since `agy --version` reports the version alone.
  if [ -n "${AGY_RELEASE:-}" ]; then
    if [ "$av_installed" != "${AGY_RELEASE%%-*}" ]; then
      warn "agy on PATH is ${av_installed:-none}, this image pinned ${AGY_RELEASE%%-*}"
      note "something installed over the image's copy (\`agy update\`?); rebuild the container"
    elif [ -n "$av_pinned" ] && [ "$AGY_RELEASE" != "$av_pinned" ]; then
      warn "this image was built at $AGY_RELEASE, the tree now pins $av_pinned"
      note "rebuild: devtools/devcontainer.sh rebuild"
    else
      ok "agy in here is the pinned ${AGY_RELEASE%%-*}"
    fi
  else
    note "agy here is ${av_installed:-not installed} (a host install, unrelated to the pin)"
  fi
fi

# --- docker hygiene (report-only) -----------------------------------------
# Worktree containers/volumes/images leak when a worktree is removed with plain
# `git worktree remove` (bypassing worktree.sh rm), or via an agent's worktrees
# under .claude/worktrees/. Surface the count instead of letting it accrete.
# The enumeration lives in `worktree.sh gc` -- called in --dry-run, which
# neither prunes nor deletes, so there is one source of truth.
echo "docker hygiene"
if command -v docker >/dev/null 2>&1; then
  wt="$TOP/devtools/worktree.sh"
  if [ -x "$wt" ]; then
    n=$("$wt" gc --dry-run 2>/dev/null \
        | sed -n 's/^orphans: \([0-9]*\).*/\1/p')
    if [ "${n:-0}" -gt 0 ]; then
      warn "$n orphaned docker resource(s) from removed worktrees"
      note "review and remove: devtools/worktree.sh gc"
    else
      ok "no orphaned containers/volumes/images"
    fi
  else
    note "devtools/worktree.sh not executable -- skipping orphan check"
  fi
else
  note "docker not found here (expected inside the container; run on the host)"
fi

# --- summary --------------------------------------------------------------
echo
if [ "$fails" -gt 0 ]; then
  printf '\033[31m%d FAIL\033[0m, %d warn\n' "$fails" "$warns"
  exit 1
elif [ "$warns" -gt 0 ]; then
  printf 'no failures, \033[33m%d warn\033[0m (some tests will skip)\n' "$warns"
else
  printf '\033[32mall good\033[0m\n'
fi
