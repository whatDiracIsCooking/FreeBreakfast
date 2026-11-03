# Developing on a cloud GPU VM

A rented single-GPU VM (a Lambda on-demand 1× A100, 1× H100, 1× A10 or
1× GH200, or anything like it) runs the same CUDA devcontainer as the workstation. Nothing
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
| GH200 | 90 | an H100 behind a 72-core ARM (Grace) CPU; see "On a GH200" below |

Do not use these:

- **V100 and RTX 6000 (Turing).** They are below the sm_80 floor, so the tree
  does not configure. `gpu-vm-setup.sh` and `doctor.sh` both refuse them.
- **Multi-GPU instances.** The devcontainer passes `/dev/nvidia0` alone, and
  the GPU lock in `devtools/lib.sh` assumes one card per backend. A 2×/4×/8×
  box would run as a 1× box.

Pick an Ubuntu image that ships Docker and the NVIDIA container toolkit
(Lambda Stack or GPU Base). The image is CUDA 13, so the host driver must be
R580 or newer. The script checks this before doing anything else, because an
older driver means the container refuses to start.

## Bring-up

Make the VM its own SSH key and its own GitHub token. Give the token
fine-grained access to this repository only: contents, pull requests and
issues read/write; actions and commit statuses read. A short expiry is fine,
and you can revoke it without touching your workstation's token.

On the VM:

```bash
git config --global user.name "<name>"
git config --global user.email "<id>+<user>@users.noreply.github.com"
read -rs GH_TOKEN && export GH_TOKEN   # paste; reaches the container at create
git clone https://github.com/whatDiracIsCooking/FreeBreakfast.git ~/fb/main
cd ~/fb/main
devtools/gpu-vm-setup.sh     # host checks, node + uv, submodules, then the container
devtools/devcontainer.sh shell
```

`devcontainer.sh` passes the host's git identity into the container, so set
it on the VM. Clone to `<root>/main`, not straight into `<root>`:
`devtools/worktree.sh` creates sibling worktrees next to `main/`.

Claude Code's memory is stored per project path, so copy it from the
workstation to the VM's path before the container is created:

```bash
ssh <vm> mkdir -p '~/.claude/projects/-home-ubuntu-fb-main'
scp -r ~/.claude/projects/<workstation-path-key>/memory \
    <vm>:'~/.claude/projects/-home-ubuntu-fb-main/'
```

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

## Seeding the compiler caches

A fresh VM builds cold. To warm it from the workstation's caches, run this on
the workstation host (it needs `rsync`, which the image does not carry):

```bash
devtools/vm-cache-push.sh <vm> [<vm checkout>]   # default ~/fb/main
```

Run it after `gpu-vm-setup.sh --no-container` and before the container
starts. Once sccache's server is running, stop it inside the container with
`sccache --stop-server` so the next server picks up the pushed entries.
It sends the two stores separately, and they hit under different conditions:

- **`.build-object-cache`** holds the device `.cu` objects: the slow half of a
  cold build, and a few hundred MB. It is keyed on content, the nvcc version,
  the host processor and, under the `native` preset, the card's resolved arch
  (the salt in the root `CMakeLists.txt`). So it is always sent, and it hits
  when the VM's card has the workstation's arch. A 3080 store serves an A10
  (both sm_86); an A100 or H100 misses every entry and compiles its own.
- **`.sccache`** holds the rest: host C++ and the module units. It is tens
  of GB, and sccache keys on the absolute paths CMake passes. **It hits only
  when the VM's checkout is at the workstation's absolute path**, so the
  script skips it otherwise and says why. Host code does not depend on the
  card, so with a matching path it hits on any x86-64 VM, an H100 included:
  clone there instead of `~/fb/main`, and use that path for the memory copy
  above as well:

  ```bash
  sudo install -d -o "$USER" -g "$USER" "$(dirname <workstation checkout>)"
  git clone https://github.com/whatDiracIsCooking/FreeBreakfast.git <workstation checkout>
  ```

  sccache also keys on the compiler binary, so a hit needs the VM's image to
  carry the same clang as the workstation's. An image built on a later day
  can pull a newer apt point release and miss every entry. `--no-sccache`
  sends only the object cache.

A GH200 is refused: the stores are x86-64 and nothing in them can hit there.

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
  A first try needs no persistent filesystem. If you keep coming back, one
  saves a rebuild per VM, but **decide at launch**: Lambda attaches a
  filesystem only when the instance is created, and only in its own region.
  On it, keep the compiler cache (point `FREE_BREAKFAST_SCCACHE_DIR` there
  before `devcontainer.sh up`) and `~/.claude` (a symlink keeps the login and
  memory). Cache hits need the same clone path on every VM (`~/fb/main`, or
  the workstation's path; see "Seeding the compiler caches"), because a
  cached module records the path it was built under. Keep build
  directories, Docker's data and the checkout on the local disk: a module
  build on NFS is slow.
- **On a GH200** the host and the images are aarch64. `Dockerfile.base` and
  `Dockerfile.cuda` build natively there (BuildKit's `TARGETARCH` picks the
  arm64 CMake, Ninja and sccache and NVIDIA's `sbsa` repo), with two
  differences. The image has no `agy`: its pin is the linux-x64 tarball, and
  `doctor.sh` notes the absence rather than warning. And host C++ is built
  with `-ffp-contract=off` (root `CMakeLists.txt`), because Grace always has
  an fma and clang would otherwise fuse where x86-64 does not, which
  `test/io/sad_host.cpp`'s bitwise check against the x86-built stamps would
  catch. That keeps the arithmetic the same; glibc's libm on aarch64 is not
  guaranteed to round `exp` and `sqrt` identically, so if that check still
  fails there, the cause is libm, not the build. Grace's 72 cores and large
  memory take a much higher `BUILD_JOBS` than the H100 VMs.
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
