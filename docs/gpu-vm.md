# Developing on a cloud GPU VM

A rented single-GPU VM (a Lambda on-demand 1× A100, 1× H100 or 1× A10, or
anything like it) runs the same CUDA devcontainer as the workstation. Nothing
about the build or the tiers changes; `devtools/gpu-vm-setup.sh` does the
host-side setup a fresh VM is missing, then builds and starts the container.

This is the **CUDA half only**. There is no AMD card on these VMs, so the HIP
runtime tier stays on the workstation. `devtools/cross-backend-check.sh` still
works on the VM, because it only compiles.

## Picking an instance

| GPU | sm | Use it for |
|---|---|---|
| H100 (SXM or PCIe) | 90 | fp64-heavy work: the slow tier, CASSCF/NEVPT2/Hessian validation |
| A100 (40 or 80 GB) | 80 | the same at a lower price; exactly WarpWraps' floor |
| A10 | 86 | another 3080-class card: more parallel capacity, no fp64 gain |

Do not use these:

- **V100 and RTX 6000 (Turing).** They are below the sm_80 floor, so the tree
  does not configure. `gpu-vm-setup.sh` and `doctor.sh` both refuse them.
- **GH200.** Its CPU is ARM, and the images and toolchain are x86-64 only.
- **Multi-GPU instances.** The devcontainer passes `/dev/nvidia0` alone, and
  the GPU lock in `devtools/lib.sh` assumes one card per backend. A 2×/4×/8×
  box would run as a 1× box.

Pick an Ubuntu image that ships Docker and the NVIDIA container toolkit
(Lambda Stack or GPU Base). The image is CUDA 13, so the host driver must be
R580 or newer. The script checks this before doing anything else, because an
older driver means the container refuses to start.

## Bring-up

On the VM:

```bash
export GH_TOKEN=...          # reaches the container: gh, git push
git clone https://github.com/whatDiracIsCooking/FreeBreakfast.git ~/fb/main
cd ~/fb/main
devtools/gpu-vm-setup.sh     # host checks, node + uv, submodules, then the container
devtools/devcontainer.sh shell
```

Clone to `<root>/main`, not straight into `<root>`: `devtools/worktree.sh`
creates sibling worktrees next to `main/`.

Inside the container, everything works as it does on the workstation:

```bash
devtools/doctor.sh
devtools/cpp-tier.sh --preset default
```

The `default` preset compiles for `native`, so the binary targets whichever
card the VM has. Neither `CUDA_ARCH` in the image nor anything else needs
editing.

The script is idempotent. Rerun it after fixing whatever it stopped on, and
after a reboot. It runs `rebuild`, which recreates the container. Once the
container exists, `devtools/devcontainer.sh up` is enough.

Claude Code is in the image. Run `claude` inside the container and log in
once; the login lives in the VM's `~/.claude`, which is bind-mounted.

## What the script handles

Each of these is a way a fresh VM fails, and each one is checked in this order:

1. **The card.** It must be present, at compute 8.0 or higher, and the only one.
2. **The driver's CUDA ceiling** against `docker/Dockerfile.cuda`'s
   `CUDA_VERSION`. The script stops here rather than installing a driver:
   upgrading the driver needs a reboot, and driver packaging is the VM image's
   business.
3. **Docker.** The script adds you to the `docker` group and reruns itself
   under it. It then proves `--gpus all` works by running `nvidia-smi` in a
   throwaway container, and wires the NVIDIA runtime into Docker when the
   toolkit is installed but not configured.
4. **Host tools.** It installs Node 22 (`devcontainer.sh` runs the
   devcontainer CLI through `npx`) and `uv`.
5. **`~/.claude` and `~/.claude.json`.** The devcontainer bind-mounts both,
   and Docker would otherwise create the missing `.json` as a root-owned
   directory.

One more check lives in `devtools/devcontainer.sh` itself, because it applies
on any headless host: before `up` and `rebuild` it creates `/dev/nvidia-uvm`
and `/dev/nvidia-uvm-tools` with `nvidia-modprobe` if they are missing. The
driver creates them on first CUDA use, so a fresh or rebooted VM often has
neither, and the devcontainer's `--device=` lines then make `docker run` fail.

## What is different from the workstation

- **Timings.** The five-minute gate budget and `slow_test.h`'s ten-second
  rule are measured on the workstation's 3080. An A100 or H100 runs the fp64
  cases far faster, so a case that looks cheap there may not be cheap on the
  3080. Read `[gpu]` times against the 3080 before moving a case in or out of
  `FBF_SKIP_IF_FAST_TESTS()`. Benchmark numbers do not transfer either way.
- **Build parallelism.** A 1× H100 or A100 VM has 26 to 30 vCPUs and over
  200 GiB of memory. `BUILD_JOBS` (default 12) is capped for memory, not
  cores, so `BUILD_JOBS=24 devtools/cpp-tier.sh` is safe there.
- **Nothing survives termination.** These VMs are terminated, not stopped,
  and the root volume goes with them: the images, the build directories,
  `.sccache` and the `~/.claude` login. Push branches before you terminate.
  To keep the compiler cache, attach a persistent filesystem and point
  `FREE_BREAKFAST_SCCACHE_DIR` at a directory on it before
  `devcontainer.sh up`. Keep build directories on the local disk, since a
  module build on NFS is slow.
- **Credentials live on rented hardware.** Give the VM a fine-grained
  `GH_TOKEN` scoped to this repository, not your general-purpose one.

## Not done yet

- **A prebuilt image.** Every new VM builds `Dockerfile.base` and
  `Dockerfile.cuda` from scratch. `docker/build.sh` can already tag and push
  to a registry (`IMAGE_REGISTRY`, `BUILD_PUSH`). What is missing is a
  published image and a pull path for the devcontainer. The same image would
  unblock the `cpp` job in `.github/workflows/ci.yml`.
- **Multi-GPU boxes.** This needs a lock per device and a `--device=` line per
  card; see above.

`devtools/cloud-env-setup.sh` is a different thing: it sets up the
**GPU-less** claude.ai cloud environment, without containers.
