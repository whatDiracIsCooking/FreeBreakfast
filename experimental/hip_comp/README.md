# `experimental/hip_comp/`

GPU lossless compression on the HIP backend: hipCOMP's batched **LZ4** and
**Cascaded** codecs over device-resident arrays, compressed and decompressed on
the card and checked bit for bit against the input.

```sh
build-hip-experimental/experimental/hip_comp/free_breakfast_hip_comp_demo [mib = 64] [reps = 5]
```

Each input is cut into 64 KiB chunks and compressed as one batch; the
compressed chunks are decompressed as one batch into a fresh buffer. Times are
the median of `reps` event-timed calls after one warm-up; GB/s is uncompressed
bytes over that time, in both directions. It exits 1 on any round-trip mismatch
and 2 when there is no HIP device.

## Result on the gfx1200 (RX 9060 XT), 256 MiB per input, median of 7

| Input | Codec | Ratio | Compress | Decompress | Round trip |
|---|---|---:|---:|---:|---|
| sparse fp64, ~90% exact zeros | LZ4 | 5.55 | 7.5 GB/s | 42.9 GB/s | bit-exact |
| dense fp64, uniform random | LZ4 | 1.00 | 1.4 GB/s | 115.8 GB/s | bit-exact |
| sorted int64 indices (~10% pattern) | LZ4 | 1.98 | 1.8 GB/s | 14.4 GB/s | bit-exact |
| sorted int64 indices (~10% pattern) | Cascaded | 8.88 | 71.1 GB/s | 124.9 GB/s | bit-exact |

Reading it:

- **It works on RDNA4.** hipCOMP calls itself an early-access preview, and
  every codec here round-trips exactly at wave32.
- **Cascaded is the codec for integer index data.** Delta, then RLE, then
  bit-packing as 8-byte integers gives 4.5× LZ4's ratio at ~40× its
  compression speed.
- **LZ4 compression is slow here, at 1.4–7.5 GB/s.** That is below a PCIe 4
  host link, so compressing to save transfer time does not pay at these
  speeds. Decompression is far faster.
- **Random fp64 does not compress, by construction.** The ~1.00 ratio is the
  honest floor, and lossless byte codecs gain nothing on real-valued data with
  no exact repeats. Anything that does gain there (bit-plane splits or
  lossy/quantised schemes) is a different experiment.

## Getting hipCOMP

AMD publishes no hipCOMP package, and this tree's image does not build it.
WarpWraps' `deps/WarpWraps/docker/install-rocm-ds.sh` is the recipe: hipCOMP
`v2.2.0` from `github.com/ROCm/hipCOMP-core`, with the `CMAKE_CXX_STANDARD`/
`CMAKE_HIP_STANDARD` lines raised from 14 to 17 (ROCm's rocPRIM requires it),
then built with `-DCMAKE_HIP_ARCHITECTURES=gfx1200 -DUSE_WARPSIZE_32=ON`.
**The wave-size flag is not optional on RDNA.** Without it the library builds
fine and gives wrong answers. For a local prefix, run the script's
download/patch/cmake/install steps and skip its `ldconfig` step; the build
takes ~2.5 min on 12 jobs and installs 41 MB. Then configure with the prefix on
`CMAKE_PREFIX_PATH`:

```sh
CMAKE_PREFIX_PATH=$PWD/.temp/rocm-ds cmake --preset hip-experimental
cmake --build build-hip-experimental --target free_breakfast_hip_comp_demo
```

CMake gives the build-tree binary a RUNPATH to the prefix's `lib/`, so it needs
no `LD_LIBRARY_PATH`. Without hipCOMP, or on a CUDA build, the directory
configures to nothing, with a status line saying so.

## Why raw hipCOMP and not `wwr.comp`

WarpWraps already has a backend-neutral layer, `wwr.comp`: the LZ4/Snappy/
Cascaded intersection of nvCOMP 5.3 and hipCOMP 2.2. Building it needs
`WWR_WITH_COMP=ON`, a tree-wide switch that would also require nvCOMP 5.3 on
the CUDA side. This image's nvCOMP is a different version, and it fails to
compile there. To promote this, put hipCOMP (and a matching nvCOMP) in the
image, turn `WWR_WITH_COMP` on from the root `CMakeLists.txt`, and re-export
`wwr.comp` through `free_breakfast.core.wwr`. The calls in this demo map one to
one onto `wwrcompBatched*`.
