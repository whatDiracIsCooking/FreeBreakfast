#!/usr/bin/env bash
# Setup script for a claude.ai cloud environment (Ubuntu 24.04, no GPU) that
# can BUILD and RUN THE TESTS of both repos it serves:
#
#   FelizJueves    nvcc 13.0 + hipcc (ROCm 7.2.4)
#   FreeBreakfast  all of the above, plus what docker/Dockerfile.base bakes in
#                  (clang-20 + libc++, CMake 4.2, Ninja 1.13, /opt/{spdlog,
#                  taskflow,tomlplusplus}) and the SDK libraries WarpWraps
#                  links on each backend.
#
# Not the images' cuda-toolkit / rocm-hip-sdk metapackages: with
# composablekernel, Nsight and the per-arch kernel libraries they overflow the
# session's disk allowance before the build has started. This installs the
# -dev package of each library WarpWraps finds (deps/WarpWraps/src/{cuda,hip}/
# CMakeLists.txt) and nothing else. NCCL/RCCL, cuTENSOR/hipTensor and
# nvCOMP/hipcomp are left out because deps/CMakeLists.txt forces
# WWR_WITH_{CCL,TENSOR,COMP} OFF; turning one on means adding it here too.
#
# Paste it into the environment's setup script field. It runs as root, before
# the repos are cloned; the per-repo steps at the bottom run only for a
# checkout that is already there. Every step is idempotent.
#
# There is no GPU, so a test run builds everything and runs the host tests;
# every GPU case skips (FBF_REQUIRE_GPU). Use the presets that never query a
# device -- `native` architecture detection fails without one:
#
#   git submodule update --init --recursive
#   devtools/cpp-tier.sh --preset ci-cuda      # or ci-hip
#
# Versions are the image's (docker/Dockerfile.*); bump them together.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

LLVM_VERSION=20
CMAKE_VERSION=4.2.3
NINJA_VERSION=1.13.1
CUDA_VERSION=13-0
ROCM_VERSION=7.2.4
SPDLOG_TAG=v1.15.3
TASKFLOW_TAG=v4.0.0
TOMLPLUSPLUS_TAG=v3.4.0
# Pinned so configure never asks a driver (there is none). Both are the
# image's defaults; any sm_80+ / any gfx target compiles the same sources.
CUDA_ARCH=86
GPU_TARGETS=gfx1200

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg git jq pciutils shellcheck unzip \
  wget zlib1g-dev

# --- clang-20 + libc++ (C++23 `import std`) ---------------------------------
if ! command -v "clang-${LLVM_VERSION}" >/dev/null 2>&1 \
  || [ ! -f "/usr/lib/llvm-${LLVM_VERSION}/lib/libc++.modules.json" ]; then
  # Not add-apt-repository: it imports apt_pkg, which a non-distro python3
  # first on PATH does not have.
  curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/llvm.gpg
  echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" \
    > /etc/apt/sources.list.d/llvm.list
  apt-get update
  apt-get install -y --no-install-recommends \
    "llvm-${LLVM_VERSION}" "clang-${LLVM_VERSION}" \
    "clang-tidy-${LLVM_VERSION}" "clang-format-${LLVM_VERSION}" \
    "clang-tools-${LLVM_VERSION}" "libclang-rt-${LLVM_VERSION}-dev" \
    "libc++-${LLVM_VERSION}-dev" "libc++abi-${LLVM_VERSION}-dev"
fi
for tool in clang clang++ clang-tidy clang-format clang-scan-deps \
  llvm-objdump llvm-cxxfilt; do
  update-alternatives --install "/usr/bin/$tool" "$tool" \
    "/usr/bin/${tool}-${LLVM_VERSION}" 100
  update-alternatives --set "$tool" "/usr/bin/${tool}-${LLVM_VERSION}"
done
test -f "/usr/lib/llvm-${LLVM_VERSION}/lib/libc++.modules.json"

# --- CMake + Ninja (Ubuntu's 3.28 / 1.11 are too old for import std) -------
if ! /usr/local/bin/cmake --version 2>/dev/null | grep -q "${CMAKE_VERSION}"; then
  curl -fsSL -o /tmp/cmake.sh \
    "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh"
  sh /tmp/cmake.sh --skip-license --prefix=/usr/local
  rm -f /tmp/cmake.sh
fi
if ! /usr/local/bin/ninja --version 2>/dev/null | grep -qx "${NINJA_VERSION}"; then
  curl -fsSL -o /tmp/ninja.zip \
    "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip"
  unzip -o -q /tmp/ninja.zip -d /usr/local/bin
  chmod +x /usr/local/bin/ninja
  rm -f /tmp/ninja.zip
fi

# --- CUDA 13.0 and ROCm 7.2.4 apt repositories -----------------------------
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i /tmp/cuda-keyring.deb
  rm -f /tmp/cuda-keyring.deb
fi
install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
  | gpg --batch --yes --dearmor -o /usr/share/keyrings/rocm.gpg
chmod 0644 /usr/share/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
  > /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
  > /etc/apt/preferences.d/rocm-pin-600
apt-get update

# --- CUDA: nvcc + the libraries wwr's src/cuda links (~3.5 GB) --------------
cv=${CUDA_VERSION}
apt-get install -y --no-install-recommends \
  "cuda-nvcc-$cv" "cuda-cudart-dev-$cv" "cuda-driver-dev-$cv" "cuda-cccl-$cv" \
  "cuda-nvrtc-dev-$cv" "cuda-nvml-dev-$cv" "cuda-cupti-dev-$cv" "cuda-nvtx-$cv" \
  "cuda-profiler-api-$cv" \
  "libcublas-dev-$cv" "libcufft-dev-$cv" "libcurand-dev-$cv" \
  "libcusolver-dev-$cv" "libcusparse-dev-$cv" "libnvjitlink-dev-$cv" \
  "libnvfatbin-dev-$cv" "libcufile-dev-$cv" "libnvjpeg-dev-$cv"
apt-get clean

# --- ROCm: hipcc + the libraries wwr's src/hip links (~10 GB unpacked) ------
apt-get install -y --no-install-recommends \
  hipcc hip-dev hip-runtime-amd rocm-llvm rocm-device-libs rocminfo \
  amd-smi-lib roctracer-dev rocprim-dev rocthrust-dev \
  hipblas-dev hipblaslt-dev hipsolver-dev hipsparse-dev hipfft-dev hiprand-dev \
  rocrand-dev
apt-get clean
# No AMD card, so the per-arch kernel libraries (GBs, loaded only at run
# time on a device) are dead weight: Dockerfile.hip's ROCM_PRUNE=1. The .so
# files every link needs stay.
rm -rf /opt/rocm/lib/hipblaslt/library /opt/rocm/lib/rocblas/library \
  /opt/rocm/lib/rocfft

printf '%s\n' /usr/local/cuda/lib64 /opt/rocm/lib > /etc/ld.so.conf.d/gpu-sdks.conf
ldconfig

# --- the three prebuilt C++ deps, in /opt, built against libc++ -------------
export PATH=/usr/local/bin:$PATH
build_dep() { # name tag url header -- cmake args...
  local name=$1 tag=$2 url=$3 header=$4
  shift 5
  [ -f "/opt/$name/include/$header" ] && return 0
  rm -rf "/tmp/$name"
  git clone -q --depth 1 --branch "$tag" "$url" "/tmp/$name"
  cmake -G Ninja -S "/tmp/$name" -B "/tmp/$name/build" \
    -DCMAKE_INSTALL_PREFIX="/opt/$name" "$@"
  cmake --build "/tmp/$name/build"
  cmake --install "/tmp/$name/build"
  rm -rf "/tmp/$name"
  test -f "/opt/$name/include/$header"
}
build_dep spdlog "$SPDLOG_TAG" https://github.com/gabime/spdlog.git \
  spdlog/spdlog.h -- \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=23 \
  -DCMAKE_CXX_FLAGS=-stdlib=libc++ -DCMAKE_EXE_LINKER_FLAGS=-lc++abi \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPDLOG_BUILD_EXAMPLE=OFF -DSPDLOG_BUILD_TESTS=OFF
build_dep taskflow "$TASKFLOW_TAG" https://github.com/taskflow/taskflow.git \
  taskflow/taskflow.hpp -- -DTF_BUILD_TESTS=OFF -DTF_BUILD_EXAMPLES=OFF
build_dep tomlplusplus "$TOMLPLUSPLUS_TAG" https://github.com/marzer/tomlplusplus.git \
  toml++/toml.hpp -- \
  -DTOMLPLUSPLUS_BUILD_TESTS=OFF -DTOMLPLUSPLUS_BUILD_EXAMPLES=OFF

# --- the environment every later shell sees ---------------------------------
# CMAKE_PREFIX_PATH carries /opt for the ci-* presets, which (unlike
# `default`) do not name it themselves. BUILD_JOBS overrides config.sh's 12,
# sized for a workstation, with this machine's core count.
cat > /etc/profile.d/fj-toolchains.sh <<EOF
export CUDAToolkit_ROOT=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAARCHS=${CUDA_ARCH}
export CUDA_ARCH=${CUDA_ARCH}
export ROCM_PATH=/opt/rocm
export GPU_TARGETS=${GPU_TARGETS}
export AMDGPU_TARGETS=${GPU_TARGETS}
export BUILD_JOBS=\$(nproc)
export CMAKE_PREFIX_PATH=/opt/spdlog:/opt/taskflow:/opt/tomlplusplus
export PATH=/root/.local/bin:/usr/local/bin:/usr/local/cuda/bin:/opt/rocm/bin:\$PATH
EOF
grep -q fj-toolchains /etc/bash.bashrc \
  || echo '. /etc/profile.d/fj-toolchains.sh' >> /etc/bash.bashrc
# shellcheck source=/dev/null
. /etc/profile.d/fj-toolchains.sh

clang++ --version | head -1
cmake --version | head -1
echo "ninja $(ninja --version)"
nvcc --version | grep release
hipcc --version | grep -i 'HIP version'

# --- uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# --- repo steps: only for a checkout that is already there -----------------
for repo in /home/user/FelizJueves /home/user/FreeBreakfast; do
  if [ -d "$repo/.git" ]; then
    (
      cd "$repo"
      git submodule update --init --recursive
      uv sync --frozen
      uv run pre-commit install
      uv run pre-commit install --hook-type pre-push
    )
  else
    echo "setup: $repo not cloned yet; skipping submodules / uv sync / pre-commit"
  fi
done
