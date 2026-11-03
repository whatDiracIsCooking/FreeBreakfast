#!/usr/bin/env bash
# Turn a fresh single-GPU cloud VM (Ubuntu, NVIDIA, x86-64 -- e.g. a Lambda
# 1x A100, 1x H100 or 1x A10) into a host for this repo's CUDA devcontainer,
# then build and start it. Run it ON THE VM, from a clone at <root>/main:
#
#   git clone https://github.com/whatDiracIsCooking/FreeBreakfast.git ~/fb/main
#   cd ~/fb/main && devtools/gpu-vm-setup.sh
#
#   devtools/gpu-vm-setup.sh --no-container   host prerequisites only
#
# Every step is idempotent, so rerunning after a fix (or after a reboot) is the
# intended way to resume. It uses sudo for the system-level steps, which a
# cloud VM's default user has without a password. docs/gpu-vm.md has the
# whole walkthrough, and what this does not do.
#: --- end of `usage` output; what follows is for a reader of the file --------
#
# WHAT IT CHECKS, in the order a fresh VM fails them:
#
#   1. the card: present, compute >= 8.0 (WarpWraps' sm_80 floor, which a V100
#      or an RTX 6000 is under), and exactly one -- the devcontainer passes
#      /dev/nvidia0 alone and lib.sh's GPU lock is one card per backend.
#   2. the driver's CUDA ceiling against docker/Dockerfile.cuda's CUDA_VERSION.
#      Below it the container refuses to start, and the fix is a host driver,
#      which this does not install: that is a reboot, and the vendor image's
#      driver packaging is the vendor's call.
#   3. docker usable without sudo, and `--gpus all` working end to end (an
#      nvidia-smi inside a throwaway container), configuring the NVIDIA
#      container runtime when it is installed but not wired into docker.
#   4. the host tools the devcontainer path needs: node >= 18 for the npx-run
#      @devcontainers/cli, and uv for the host .venv.
#   5. ~/.claude and ~/.claude.json, which the devcontainer bind-mounts. A
#      missing bind source is CREATED by docker, as root, and ~/.claude.json
#      would come back a root-owned directory.
#
# Then submodules, `uv sync`, and `devtools/devcontainer.sh --cuda rebuild`,
# which builds docker/Dockerfile.base and Dockerfile.cuda on the VM.
set -euo pipefail
USER=${USER:-$(id -un)}

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

container=1
while [ $# -gt 0 ]; do
  case $1 in
    --no-container) container= ;;
    -h|--help) usage; exit 0 ;;
    *) echo "gpu-vm-setup: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

say()  { echo "gpu-vm-setup: $*"; }
die()  { echo "gpu-vm-setup: error: $*" >&2; exit 1; }
warn() { echo "gpu-vm-setup: WARNING -- $*" >&2; }

# --- where we are ----------------------------------------------------------
[ ! -e /.dockerenv ] ||
  die "this is a container; run it on the VM itself (it drives docker)"
[ "$(uname -m)" = x86_64 ] ||
  die "$(uname -m) host: the images and toolchain are x86-64 only (no GH200)"
# The images' `ubuntu` user is uid 1000 and the devcontainer does not remap it
# (updateRemoteUserUID: false), so a different uid owns bind mounts the
# container cannot write.
[ "$(id -u)" = 1000 ] ||
  warn "uid $(id -u), not 1000 -- the container's ubuntu user will not own the workspace"

# --- 1. the card -----------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 ||
  die "no nvidia-smi: this needs an NVIDIA VM image with the driver installed"
mapfile -t cards < <(nvidia-smi --query-gpu=name,compute_cap \
  --format=csv,noheader 2>/dev/null)
[ "${#cards[@]}" -gt 0 ] || die "nvidia-smi lists no GPU -- is the driver loaded?"
for card in "${cards[@]}"; do
  IFS=, read -r name cap <<<"$card"
  cap=${cap# }
  version_ge "$cap" 8.0 ||
    die "$name is compute $cap; WarpWraps' floor is sm_80 (A10/A100/H100 and up)"
  say "card: $name (compute $cap)"
done
[ "${#cards[@]}" -eq 1 ] ||
  warn "${#cards[@]} GPUs -- the devcontainer passes /dev/nvidia0 only, and the GPU lock is one card per backend"

# --- 2. the driver's CUDA ceiling ------------------------------------------
ceiling=$(cuda_driver_ceiling)
image=$(cuda_image_version)
[ -n "$ceiling" ] || die "could not read the driver's CUDA version from nvidia-smi"
version_ge "$ceiling" "$image" ||
  die "the driver runs CUDA <= $ceiling but the image is CUDA $image; install a newer NVIDIA driver (R580+ for 13.0) and reboot, or start from a newer VM image"
say "driver: CUDA <= $ceiling (image: $image)"

# --- 3. docker, and the GPU through it -------------------------------------
command -v docker >/dev/null 2>&1 ||
  die "docker is not installed; use a VM image that ships it (Lambda Stack, GPU Base)"
if ! docker info >/dev/null 2>&1; then
  if ! id -nG | tr ' ' '\n' | grep -qx docker; then
    getent group docker >/dev/null || die "docker info fails and there is no docker group"
    if ! getent group docker | cut -d: -f4 | tr ',' '\n' | grep -qx "$USER"; then
      say "adding $USER to the docker group"
      sudo usermod -aG docker "$USER"
    fi
    # The new group reaches only new logins. Rerun under it once rather than
    # making the user log out halfway through.
    [ -z "${GPU_VM_SETUP_REEXEC:-}" ] ||
      die "still cannot reach docker under the docker group"
    say "rerunning under the docker group (log in again later to keep it)"
    rerun=("${BASH_SOURCE[0]}")
    [ -n "$container" ] || rerun+=(--no-container)
    exec sg docker -c "GPU_VM_SETUP_REEXEC=1 $(printf '%q ' "${rerun[@]}")"
  fi
  die "docker info fails although $USER is in the docker group -- is dockerd running?"
fi

gpu_probe() {
  docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >/dev/null 2>&1
}
if ! gpu_probe; then
  command -v nvidia-ctk >/dev/null 2>&1 ||
    die "\`docker run --gpus all\` fails and nvidia-container-toolkit is not installed"
  say "configuring the NVIDIA container runtime for docker"
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  gpu_probe || die "\`docker run --gpus all ubuntu:24.04 nvidia-smi -L\` still fails"
fi
say "docker: --gpus all reaches the card"

# --- 4. host tools ---------------------------------------------------------
node_major=$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)
if [ "$node_major" -lt 18 ]; then
  # The same NodeSource repo docker/install-claude-code.sh uses in the images.
  say "installing node 22 (for npx @devcontainers/cli)"
  sudo install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key |
    sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/nodesource.gpg
  echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" |
    sudo tee /etc/apt/sources.list.d/nodesource.list >/dev/null
  sudo apt-get update -q
  sudo apt-get install -y -q nodejs
fi
say "node: $(node --version)"

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=$HOME/.local/bin:$PATH
say "uv: $(uv --version)"

# --- 5. what the devcontainer bind-mounts from $HOME -----------------------
mkdir -p "$HOME/.claude"
# An empty object, not an empty file: seed-claude-config.sh parses it as JSON.
[ -e "$HOME/.claude.json" ] || echo '{}' >"$HOME/.claude.json"
[ -f "$HOME/.claude.json" ] ||
  die "$HOME/.claude.json is not a file (a docker-created directory?) -- remove it and rerun"
[ -n "${GH_TOKEN:-}" ] ||
  warn "GH_TOKEN is unset -- the container gets no gh auth or git push; export it and rerun"

# --- the checkout ----------------------------------------------------------
say "submodules"
git -C "$REPO_ROOT" submodule update --init --recursive
say "uv sync"
(cd "$REPO_ROOT" && uv sync --frozen)

if [ -z "$container" ]; then
  say "host ready; next: devtools/devcontainer.sh --cuda rebuild"
  exit 0
fi

# --- the devcontainer ------------------------------------------------------
say "building and starting the cuda devcontainer (the first build takes a while)"
"$REPO_ROOT/devtools/devcontainer.sh" --cuda rebuild

say "done. Next:"
# Said again HERE, at the end, because the rerun's own note scrolls away under
# the image build: the shell this was started from still lacks the group.
[ -z "${GPU_VM_SETUP_REEXEC:-}" ] ||
  say "  log out and back in first -- this login is not in the docker group yet"
say "  devtools/devcontainer.sh shell"
say "  devtools/doctor.sh && devtools/cpp-tier.sh --preset default   # inside"
