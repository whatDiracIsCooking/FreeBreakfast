# syntax=docker/dockerfile:1.7
# ============================================================================
# hip -- base + ROCm, with no CUDA toolkit at all
# ============================================================================
# A HIP build needs no CUDA, which is the point of this being a sibling of
# Dockerfile.cuda rather than a layer on it: ROCm alone is ~30GB, and neither
# backend should pay for the other.
#
#   docker/build.sh hip       # builds base first, then this; tags :hip
#
# By hand, and the parent must already be tagged (see Dockerfile.base, "WHY THE
# FOUR FILES CHAIN BY TAG"):
#
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.base -t free-breakfast:base .
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.hip  -t free-breakfast:hip .
#
# Both from the REPO ROOT: the trailing `.` is the context, and this file
# bind-mounts docker/install-rocm.sh from it.
#
# This is the image devtools/cross-backend-check.sh and `devtools/cpp-tier.sh
# --rocm` reach for (CROSS_CHECK_IMAGE / ROCM_IMAGE in devtools/config.sh), so
# it is worth having even on a machine with no AMD card -- the cross-backend
# check only compiles.
ARG PROJECT_NAME=free-breakfast
ARG PARENT_IMAGE=${PROJECT_NAME}:base
FROM ${PARENT_IMAGE}

ARG ROCM_VERSION=7.2.4

# Drop the ~12.1GB of ROCm that only a RUNNING kernel needs -- the Tensile and
# rocFFT kernel objects, composable-kernel's static archives, and rocalution.
# rccl and hiptensor are NOT dropped anymore: wwr.hip.rccl and wwr.hip.hiptensor
# link them, so pruning either would fail find_package at configure. See
# install-rocm.sh for the measured list and why nothing on it is a link-time
# dependency. OFF here: this is the dev image and it runs kernels.
# .github/workflows/images.yml sets it to 1 for the :hip-ci tag, whose only job
# is to answer "does the other backend still compile" on a GitHub-hosted runner
# that has neither an AMD card nor 20GB of free disk.
ARG ROCM_PRUNE=0
USER root

# Re-stamped rather than merely inherited from the parent, so that the
# PROJECT_NAME the devcontainer passes to THIS build is consumed where it is
# declared. See Dockerfile.base for what the label is for.
ARG PROJECT_NAME
LABEL devcontainer.project="${PROJECT_NAME}"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=bind,source=docker/install-rocm.sh,target=/tmp/install-rocm.sh \
    ROCM_VERSION="${ROCM_VERSION}" ROCM_PRUNE="${ROCM_PRUNE}" \
    bash /tmp/install-rocm.sh

# The LLVM target for the AMD card this image is built for. Exported as
# GPU_TARGETS / AMDGPU_TARGETS so a HIP CMake build has a default to read.
# gfx1200 is RDNA4 (Radeon RX 9060 XT); override for another card:
#   --build-arg GPU_TARGETS=gfx1100
#
# ONE target, not a list: hip-config reads this to decide what to compile
# device code for, and the warp size it implies has to match
# WWR_WARP_SIZE in the WarpWraps dependency -- 32 for RDNA (gfx10xx-gfx12xx),
# 64 for CDNA (gfx90a, gfx942).
ARG GPU_TARGETS=gfx1200
ENV GPU_TARGETS=${GPU_TARGETS} \
    AMDGPU_TARGETS=${GPU_TARGETS} \
    ROCM_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:${PATH}


# Claude Code, PINNED -- an agent layer at the END of this file on purpose: a
# version bump re-runs only this layer, never the SDK above it.
# docker/install-claude-code.sh carries the rest (why not Dockerfile.base, and
# which drift this ended). It is the fourth version to bump in more than one
# place: the same ARG is declared by Dockerfile.cuda, .hip and .combined, so
# bump it with `devtools/claude-version.sh --apply`, which rewrites all three
# at once -- doctor.sh warns when they have drifted apart.
#
# ENV as well as ARG so the INTENDED version is readable from inside a running
# container (`printenv CLAUDE_CODE_VERSION`), not merely inferable from the
# binary: that is what lets doctor.sh tell "pinned and installed" apart from
# "someone npm-installed over the top".
ARG CLAUDE_CODE_VERSION=2.1.296
ENV CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION}
RUN --mount=type=bind,source=docker/install-claude-code.sh,target=/tmp/install-claude-code.sh \
    bash /tmp/install-claude-code.sh

# The Antigravity CLI (`agy`), PINNED -- the same leaf-layer treatment as
# Claude Code above, declared in all three GPU files and bumped in all three at
# once by `devtools/agy-version.sh --apply`. docker/install-agy.sh carries the
# rest: why not upstream's install.sh, and why the pin is a RELEASE (version
# plus an opaque build id) rather than a version -- the manifest endpoint serves
# only the CURRENT release, so a version on its own is not addressable and an
# older one would be unfetchable.
#
# TWO lines to bump, not one: AGY_SHA512 belongs to the exact tarball
# AGY_RELEASE names, and install-agy.sh verifies it rather than recording it.
# devtools/agy-version.sh rewrites both, in every file, at once.
#
# ENV as well as ARG, exactly as with CLAUDE_CODE_VERSION above: it is what
# lets doctor.sh tell the image's own pinned copy from something installed over
# the top of it, which `agy --version` cannot -- that prints 1.3.3 and never
# the build id.
#
# AGY_CLI_DISABLE_AUTO_UPDATE is the binary's own opt-out of the background
# self-update it otherwise runs during normal use -- the counterpart of
# `autoUpdates: false` in .devcontainer/seed-claude-config.sh. The install is
# root-owned and the container runs as uid 1000, so an update attempt fails
# regardless; what this stops is it being attempted behind the image's back.
ARG AGY_RELEASE=1.3.3-5524738307653632
ARG AGY_SHA512=abe3ddf5cf30d3adaa586acf505221aa19702f5ca71af3c8b1f9b1a5976bd8e5c151c41ea33b39c41675d25843462a3e8c382c054f580e9d7e1688919d3b5110
ENV AGY_RELEASE=${AGY_RELEASE} \
    AGY_CLI_DISABLE_AUTO_UPDATE=1
RUN --mount=type=bind,source=docker/install-agy.sh,target=/tmp/install-agy.sh \
    AGY_SHA512="${AGY_SHA512}" bash /tmp/install-agy.sh

ENV PATH=/home/ubuntu/.local/bin:${PATH}

USER ubuntu
RUN hipconfig --version
