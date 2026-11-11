#!/usr/bin/env bash
# Install the ROCm/HIP SDK from AMD's apt repository.
#
# Called by docker/Dockerfile.hip, and again by docker/Dockerfile.combined on
# top of the CUDA image. A script rather than an inline RUN so the block is
# written once -- Dockerfiles have no include, and `combined` cannot inherit
# from two parents, so it reuses THIS instead of the hip image (see that file's
# header).
#
# TWO CHANNELS, AND ROCM_VERSION PICKS ONE. AMD moved distribution to TheRock
# at 7.14.0; the legacy repo.radeon.com/rocm/apt ends at 7.2.4 and 7.11/7.14
# 404 there, while TheRock publishes to a different host, suite, package
# namespace and prefix. The pin is on TheRock, and the legacy branch is what
# still builds an image for anything <= 7.2.4 -- which is not idle: clang <= 22
# can only be paired with a ROCm from that side (docker/Dockerfile.base,
# "LLVM_VERSION AND ROCM_VERSION ARE ONE DECISION"), so the legacy branch is
# how you build a pre-23 toolchain image at all.
#
# THIS FILE IS A PORT OF WarpWraps' docker/install-rocm.sh, and the measurements
# in the comments below are wwr's -- its issues #305 (the channel move), #111
# and #149 (the floor image) and #100 (rccl). Unlike wwr, this tree declares no
# ROCm floor in CMakeLists.txt and its `cpp` CI legs are `if: false`, so nothing
# automated exercises the legacy branch here; keep the two files in step by
# hand when either moves.
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:?ROCM_VERSION must be set, e.g. 7.2.4 or 10.0.0}"

# The discontinuity started with the 7.9.0 preview, so 7.9 / 7.1x / 10.x are
# the TheRock track and anything below is legacy. `sort -V` so 10.0.0 compares
# above 7.9.0 rather than below it as a string would.
if [ "$ROCM_VERSION" = "$(printf '%s\n%s\n' "$ROCM_VERSION" 7.9.0 | sort -V | tail -1)" ]; then
  ROCM_CHANNEL=therock
else
  ROCM_CHANNEL=legacy
fi
echo "install-rocm.sh: ROCM_VERSION=$ROCM_VERSION -> $ROCM_CHANNEL channel"

# A signed-by keyring, not `apt-key add`: apt-key is deprecated in 24.04 and
# removed in 25.04. Both channels get one.
install -d -m 0755 /etc/apt/keyrings

if [ "$ROCM_CHANNEL" = legacy ]; then
  # --- the legacy channel: repo.radeon.com, ROCm <= 7.2.4 --------------------
  #
  # AMD spells an X.Y.0 release's repo directory X.Y. Listing
  # repo.radeon.com/rocm/apt/ shows 7.0, 7.0.1, 7.0.2, 7.0.3, 7.1, 7.2, 7.2.1
  # ... -- every .0 patch appears WITHOUT its trailing component, and there is
  # no 7.0.0 directory at all. So the floor image (ROCM_VERSION=7.0.0,
  # wwr#111/#149) 404s on the Release file unless the .0 comes off here; the old
  # pin 7.2.4 is unaffected, which is why nothing caught this until the floor
  # leg ran.
  #
  # ONLY the URL is rewritten. ROCM_VERSION stays the full MAJOR.MINOR.PATCH
  # everywhere else -- build.sh's version-carrying tag (:hip-7.0.0-ci) wants
  # the semver, and apt/7.0 does ship rocm-core 7.0.0.70000, so the two agree.
  # (wwr's copy also names its doctor's .info/version check and its CMake ROCm
  # floor here; this tree has neither, so the tag is the only consumer.)
  case "$ROCM_VERSION" in
    *.*.0) ROCM_APT_DIR="${ROCM_VERSION%.0}" ;;
    *)     ROCM_APT_DIR="$ROCM_VERSION" ;;
  esac

  # Only repo.radeon.com/rocm is added, NOT repo.radeon.com/amdgpu. That second
  # repo exists to ship the DKMS kernel driver and AMD's libdrm fork, and
  # neither belongs in a container -- the kernel driver comes from the host.
  #
  # The pin is load-bearing, not decoration. repo.radeon.com republishes its own
  # builds of packages that also exist in Ubuntu; at equal priority apt is free
  # to prefer either, and which one it picks can change between base-image
  # refreshes. Priority 600 makes it deterministic. `o=repo.radeon.com` is the
  # Origin its Release file declares.
  curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
  chmod 0644 /etc/apt/keyrings/rocm.gpg

  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_APT_DIR} noble main" \
    > /etc/apt/sources.list.d/rocm.list
  printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
    > /etc/apt/preferences.d/rocm-pin-600

  # rocm-hip-sdk is AMD's meta-package for "the HIP SDK" -- one name that tracks
  # whatever AMD decides belongs in it (hipBLAS, hipSOLVER, hipFFT, hipRAND,
  # hipSPARSE, rocm-smi, hipcc, rocm-cmake and the rest of the surface src/hip
  # wraps).
  #
  # The five after it are the ones it does NOT pull, checked against its
  # resolved dependency closure rather than assumed:
  #   rocm-hip-runtime-dev  hipcc, rocm-llvm and lib/cmake/hip -- the HIP
  #                   compiler itself. rocm-hip-sdk Depends on it at 7.0 and at
  #                   7.2+, so naming it is a no-op there; the 7.1 SERIES
  #                   DROPPED IT (verified in the apt index for both 7.1 and
  #                   7.1.1), and the omission surfaces only downstream, as
  #                   CMake's "Failed to find ROCm root directory" when
  #                   install-rocm-ds.sh calls enable_language(HIP). The floor
  #                   image is 7.1, so this is load-bearing, not
  #                   belt-and-braces -- and the tree needs hipcc regardless of
  #                   what a meta-package decides to carry.
  #   amd-smi-lib     the nvml analogue this tree wraps, as wwr.hip.amd_smi.
  #                   ROCm's older rocm-smi-lib is deliberately NOT wrapped (see
  #                   src/hip/README.md, "nvml's HIP counterpart") but stays
  #                   installed regardless, and nothing about the package set
  #                   turns on that choice: rocm-hip-libraries pulls it in with
  #                   the SDK, and librccl links it at this ROCm.
  #   roctracer-dev   the cupti analogue
  #   rocprofiler-sdk rocprofv3, its successor
  #   rocm-gdb        parity with cuda-gdb
  apt-get update
  apt-get install -y --no-install-recommends \
    rocm-hip-sdk \
    rocm-hip-runtime-dev \
    amd-smi-lib \
    roctracer-dev \
    rocprofiler-sdk \
    rocm-gdb

  # --- ROCM_PRUNE on the legacy channel: the compile-only variant ------------
  #
  # ROCm installs at ~20GB. .github/workflows/images.yml builds a second tag
  # (:hip-ci) with ROCM_PRUNE=1, which brings the image to 7.05GB; the dev image
  # is untouched, since the default is off.
  #
  # IT IS AN OPTIMISATION, NOT AN ENABLER -- this comment used to claim a hosted
  # runner has "20-25GB free" and that the prune was the only way the HIP job
  # could fit. Measured on the first real run: a GitHub-hosted runner has a
  # 145GB root with 86GB free BEFORE any cleanup, so the unpruned 20.5GB image
  # would have fit comfortably. What the prune actually buys is ~13GB less to
  # pull on every CI run, and a push that takes 3 minutes instead of many.
  #
  # What comes out, measured rather than guessed (`du -x -d1 /opt/rocm/lib` plus
  # `dpkg -S` on each of the largest files):
  #
  #   4.8G  composablekernel-dev   libdevice_{gemm,conv,reduction,contraction}
  #                                _operations.a -- static archives nothing in
  #                                this tree links. Pulled in by rocm-hip-sdk.
  #   4.5G  hipblaslt/library      Tensile kernel objects, loaded at RUN time by
  #   644M  rocblas/library        libhipblaslt.so / librocblas.so. The .so files
  #   1.7G  rocfft/                themselves stay; only the kernel data goes.
  #   459M  rocalution             sparse iterative solvers; nothing here wraps
  #
  # ~12.1GB of ~20.5GB, and none of it is a link-time dependency -- which is the
  # test that decides what may go on this list. Adding anything here that a
  # hipcc link actually needs surfaces as an undefined symbol in the ci-hip
  # tier, not as a silent wrong answer, so the failure mode is at least loud.
  #
  # rccl (572M) used to be on this list -- "multi-GPU collectives; nothing here
  # wraps them". wwr.hip.rccl now wraps it (wwr#100), so roc::rccl is a
  # link-time dependency of the ci-hip build and rccl/rccl-dev must STAY:
  # pruning it would fail find_package(rccl) at configure, exactly the loud
  # failure the rule above describes.
  #
  # hiptensor (226M) left this list the same way and for the same reason, and it
  # is the ONE half of the cuTENSOR/hipTensor pair that costs this image nothing
  # extra to keep -- rocm-hip-sdk already installs hiptensor-dev through
  # rocm-hip-libraries, where the CUDA side had to add a package (see
  # docker/install-cuda.sh).
  #
  # composablekernel-dev STAYS on the list even though hiptensor is built on CK:
  # what comes out is CK's static archives, which hiptensor consumed when IT was
  # compiled. Linking libhiptensor.so does not re-link them.
  #
  # Since ci-hip became a full build it also LINKS and LOADS the runtime test
  # binaries, so the SuiteListIsComplete guards now dlopen librocblas,
  # librocsolver, librocsparse and librocfft out of a pruned tree on every CI
  # run. They pass: those libraries read their kernel data lazily, on handle
  # creation, not at load. That is a stronger check of this list than the
  # compile-time tier could make.
  #
  # THE DELETION HAS TO HAPPEN IN THIS SCRIPT, not in a later Dockerfile layer.
  # Layers are additive: an `rm` in a child layer hides the files but keeps their
  # bytes in the parent, and the image does not shrink at all. That is also why
  # :hip and :hip-ci cannot share the ROCm layer -- each is a full install.
  #
  # The whole packages go through apt rather than rm, so that removing one
  # something else needs FAILS here instead of at link time. They pull out the
  # rocm-hip-sdk / rocm-hip-libraries meta-packages with them, which carry no
  # files of their own. --auto-remove is deliberately NOT passed: it would widen
  # the removal to whatever else those metas were the last reference to, which is
  # exactly the kind of quiet cascade this list is written to avoid.
  if [ "${ROCM_PRUNE:-0}" = "1" ]; then
    echo "install-rocm.sh: ROCM_PRUNE=1 -- building the compile-only variant"
    apt-get purge -y \
      composablekernel-dev \
      rocalution rocalution-dev
    rm -rf \
      /opt/rocm/lib/hipblaslt/library \
      /opt/rocm/lib/rocblas/library \
      /opt/rocm/lib/rocfft
  fi
else
  # --- TheRock channel: stable.repo.amd.com, ROCm >= 7.9 --------------------
  #
  # GPU_TARGETS is REQUIRED here and not on the legacy branch: TheRock splits
  # GPU kernels into per-arch packages whose names carry the arch, so the
  # package set cannot be written without knowing it. Both Dockerfiles declare
  # the ARG above their install-rocm.sh RUN for this reason.
  GPU_TARGETS="${GPU_TARGETS:?GPU_TARGETS must be set on the TheRock channel, e.g. gfx1200}"

  # Package names carry ROCm's MAJOR.MINOR (amdrocm-blas10.0-gfx1200), while the
  # arch-free metas do not (amdrocm-blas-dev). Only the versioned names need
  # this, and no unversioned alias exists for them.
  ROCM_MM="${ROCM_VERSION%.*}"

  # TheRock publishes per-distro: debian12/13, ubuntu2204/2404/2604, rhel8/9/10,
  # sles15/16, azl3. Derive the path from the image rather than hardcoding it --
  # Dockerfile.base's UBUNTU_TAG is an ARG, so the base distro can move.
  # shellcheck disable=SC1091
  . /etc/os-release
  ROCM_DISTRO="${ID}${VERSION_ID//./}"

  # THE KEY MUST GO THROUGH `gpg --dearmor`, AND NOT BECAUSE IT IS ARMORED.
  # stable.repo.amd.com/rocm/gpg/packages.gpg is a CONCATENATION: a valid
  # 2837-byte binary key block followed by 1086 bytes of ASCII-armored copy of
  # the same key. `gpg --import` reads the binary prefix and ignores the
  # trailer, so the file verifies fine by hand -- but `gpgv --keyring`, which is
  # what apt uses, walks packets strictly, hits the `-` of `-----BEGIN` at
  # offset 2837 and fails `invalid packet (ctb=2d)`. apt then reports the
  # repository as NOT SIGNED (NO_PUBKEY). Dearmor emits exactly the binary
  # prefix and gpgv accepts it. Writing the file through verbatim is the trap,
  # and is probably why AMD's published instructions use `[trusted=yes]` --
  # which this does not, since an unverified repo is not acceptable here.
  curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg \
    | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
  chmod 0644 /etc/apt/keyrings/rocm.gpg

  # Suite and codename are both `stable`; the Release file declares
  # `Origin: AMD ROCm`, which is what the pin must name -- the legacy
  # `o=repo.radeon.com` silently stops applying here, leaving apt free to prefer
  # an Ubuntu build of anything both publish. Verified with `apt-cache policy`:
  # 600 on the stable.repo.amd.com line against 500 for Ubuntu's.
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://stable.repo.amd.com/rocm/core/packages/${ROCM_DISTRO} stable main" \
    > /etc/apt/sources.list.d/rocm.list
  printf 'Package: *\nPin: release o=AMD ROCm\nPin-Priority: 600\n' \
    > /etc/apt/preferences.d/rocm-pin-600

  # THE PACKAGE SET IS DERIVED FROM THE find_package CALLS, not from a
  # meta-package: TheRock has no rocm-hip-sdk equivalent that is safe to name
  # (see the core-dev warning below). One entry per call in CMakeLists.txt and
  # src/hip/CMakeLists.txt.
  #
  # The split is THREE-way for the nine math/comm libraries, not the usual two:
  #   amdrocm-<lib>-dev          headers + lib/cmake/<pkg>, and NO .so
  #                              (amdrocm-blas-dev10.0 Depends on libc6 alone)
  #   amdrocm-<lib>-host<MM>     the host .so
  #   amdrocm-<lib><MM>-gfx<arch>  the device kernels
  # so naming only the -dev packages gives headers that cannot link. The -host
  # ones arrive transitively via the per-arch packages, but are named
  # EXPLICITLY below so that ROCM_PRUNE can drop kernels without taking a
  # link-time .so with them.
  #
  # Two names are not what they look like:
  #   amdrocm-ccl-dev     is rocPRIM + hipCUB + rocThrust ("ROCm primitive
  #                       header files"), NOT an RCCL companion. It is what
  #                       find_package(rocprim)/find_package(rocthrust) resolve
  #                       against. RCCL is amdrocm-rccl{,-dev} alone.
  #   amdrocm-profiler-base  carries roctracer64 + roctx64, which
  #                       src/hip/CMakeLists.txt find_library's directly.
  #                       amdrocm-profiler is rocprofiler-compute/-systems and
  #                       is NOT wanted.
  #
  # EVERY NAME CARRIES ${ROCM_MM}, INCLUDING THE -dev ONES. The unversioned
  # spellings (amdrocm-hiptensor-dev, ...) are metas that depend on the NEWEST
  # series the channel carries, which is only the pinned one until the day it is
  # not: with 10.0.0-4 and 10.1.0-3 both published, ROCM_VERSION=10.0.0 installed
  # the 10.0 versioned names beside 10.1 metas, leaving the image with
  # /opt/rocm/{core-10.0,core-10.1} and hipTENSOR's headers under the prefix
  # /opt/rocm/include does NOT point at -- a configure failure reading
  # `Could not find HIP_hiptensor_INCLUDE_DIR`. Every package here has a
  # <name><MM> / <name>-dev<MM> form, so the pin can simply be honoured.
  apt-get update
  apt-get install -y --no-install-recommends \
    "amdrocm-base${ROCM_MM}" \
    "amdrocm-runtime-dev${ROCM_MM}" \
    "amdrocm-llvm${ROCM_MM}" \
    "amdrocm-blas-dev${ROCM_MM}" \
    "amdrocm-hipblas-common-dev${ROCM_MM}" \
    "amdrocm-solver-dev${ROCM_MM}" \
    "amdrocm-sparse-dev${ROCM_MM}" \
    "amdrocm-fft-dev${ROCM_MM}" \
    "amdrocm-rand-dev${ROCM_MM}" \
    "amdrocm-ccl-dev${ROCM_MM}" \
    "amdrocm-rccl-dev${ROCM_MM}" \
    "amdrocm-hiptensor-dev${ROCM_MM}" \
    "amdrocm-amdsmi${ROCM_MM}" \
    "amdrocm-profiler-base${ROCM_MM}" \
    "amdrocm-debugger${ROCM_MM}" \
    "amdrocm-blas-host${ROCM_MM}" \
    "amdrocm-fft-host${ROCM_MM}" \
    "amdrocm-rand-host${ROCM_MM}" \
    "amdrocm-solver-host${ROCM_MM}" \
    "amdrocm-sparse-host${ROCM_MM}" \
    "amdrocm-rccl-host${ROCM_MM}" \
    "amdrocm-hiptensor-host${ROCM_MM}"

  # NEVER NAME THE UNVERSIONED `amdrocm-core-dev`. It hard-depends, with `=`
  # version equality, on all 25 amdrocm-core-dev<MM>-gfx<arch> packages, and
  # each amdrocm-core<MM>-gfx<arch> in turn pulls that arch's whole kernel set.
  # Measured: naming it produces a 27.9GB image with 212 gfx packages -- LARGER
  # than the 20.7GB legacy image it replaces. Naming the single arch instead
  # gives 11.3GB, which is what finally makes GPU_TARGETS control image size;
  # it never did on the legacy channel.
  #
  # GPU_TARGETS is CMake-spelled, so it may be a ;-separated list. Device code
  # is a different matter (install-rocm-ds.sh has the wavefront-size constraint
  # that keeps it single-valued in practice), but the loop costs nothing.
  for _arch in ${GPU_TARGETS//;/ }; do
    echo "install-rocm.sh: adding per-arch kernels for ${_arch}"
    apt-get install -y --no-install-recommends \
      "amdrocm-core-dev${ROCM_MM}-gfx${_arch#gfx}" \
      "amdrocm-hiptensor${ROCM_MM}-gfx${_arch#gfx}"
  done

  # --- ROCM_PRUNE on TheRock: almost nothing left to do ---------------------
  #
  # THE ARCH CHOICE REPLACED THIS LIST. The legacy prune took ~12.1GB of 20.5GB
  # off by deleting Tensile kernel data and CK's static archives; here the
  # default install is already 11.3GB because it carries ONE arch instead of 25,
  # and the old list has no package equivalent at all --
  # composablekernel/rocalution are different packages and the
  # hipblaslt/rocblas/rocfft `library` directories are not laid out the same.
  #
  # Per-library kernel pruning is IMPOSSIBLE by package, measured rather than
  # assumed: amdrocm-core<MM>-gfx<arch> hard-depends on that arch's entire
  # kernel set (blas, rand, fft, solver, sparse, dnn, rccl) plus ck, rocshmem,
  # hipify, hipfile, decode and jpeg, so `apt-get purge amdrocm-dnn<MM>-gfx...`
  # takes amdrocm-core<MM>-gfx... with it -- and with that, the
  # /etc/alternatives/rocm-* links that make /opt/rocm/{bin,lib,include} exist
  # at all. apt refuses loudly, which is the rule this list has always kept:
  # nothing on it may be a link-time dependency, and removals go through apt so
  # a mistake fails here rather than at link time.
  #
  # What IS separable is hipTensor's kernels, because hiptensor is absent from
  # core's Depends -- 45.9MB of 11.3GB. That is 0.4%, kept only so the flag's
  # contract (a compile-only image; hipTensor contractions will not RUN, which
  # `ctest -LE gpu` does not need) still means something. If a later ROCm makes
  # it smaller still, deleting this branch is defensible -- the 9.4GB that used
  # to justify the prune now comes from GPU_TARGETS.
  if [ "${ROCM_PRUNE:-0}" = "1" ]; then
    echo "install-rocm.sh: ROCM_PRUNE=1 -- building the compile-only variant"
    for _arch in ${GPU_TARGETS//;/ }; do
      apt-get purge -y "amdrocm-hiptensor${ROCM_MM}-gfx${_arch#gfx}"
    done
  fi
fi

# Nothing in either deb set registers the prefix with the dynamic loader, so
# without this every hipcc-linked binary would need an explicit RPATH.
#
# /opt/rocm/lib is correct on BOTH channels, for different reasons: on legacy,
# rocm-core provides /opt/rocm as a symlink to /opt/rocm-<version>; on TheRock,
# /opt/rocm is a real directory whose `lib` is a Debian-alternatives symlink to
# /opt/rocm/core-<MM>/lib. The same is true of bin/ and include/, which is why
# ROCM_PATH=/opt/rocm and PATH=/opt/rocm/bin in the Dockerfiles need no change.
echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
ldconfig

# Sanity-check the prefix. This used to be `test -L /opt/rocm`, which asserts
# the legacy layout and FAILS on TheRock, where /opt/rocm is a real directory.
# Asserting that the lib directory resolves is what both channels actually
# promise, and is what the line above depends on.
test -d /opt/rocm/lib

# Device access. /dev/kfd (the compute node) and /dev/dri/renderD* are
# root:render; /dev/dri/card* is root:video. The groups have to exist INSIDE the
# container for a non-root user to open those nodes once they are passed in, and
# `render` is not present in the stock Ubuntu image.
#
# gid 110 is Ubuntu 24.04's default for render, and the gid HERE is what a
# `--group-add render` by NAME would resolve to -- which is why nothing passes
# the name: the bind-mounted nodes keep the HOST's gid. Pass `--group-add
# <host-gid>` to docker run instead, which is what the devcontainer runArgs and
# cpp-tier.sh --rocm both do. Never rebuild the image to match a host.
(getent group render >/dev/null || groupadd -g 110 render || groupadd render)
(getent group video >/dev/null || groupadd video)
usermod -aG render,video ubuntu
