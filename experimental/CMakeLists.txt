# ---------------------------------------------------------------------------
# experimental/ -- opt-in scratch space
# ---------------------------------------------------------------------------
# Configured only under -DFREE_BREAKFAST_BUILD_EXPERIMENTAL=ON (the
# `experimental` preset), so nothing in here can slow or break a normal build.
#
# Add targets with the same three macros the rest of the tree uses, and
# add_subdirectory() them below. Registering a ctest entry works here too: the
# root CMakeLists calls enable_testing() before any add_subdirectory, so an
# add_test() in this directory is recorded rather than silently dropped.
#
# Keep this file even if every subdirectory below is promoted or deleted.
# `add_subdirectory(experimental)` in the root CMakeLists is unconditional
# within its option guard, so a missing directory fails the configure outright
# with CMake's "add_subdirectory given source ... which is not an existing
# directory" -- which is what the `experimental` preset did before this file
# existed.
#
# A GPU test of something here goes in test/experimental, not in this tree: the
# root CMakeLists adds that directory under this same option and also behind
# FREE_BREAKFAST_COMPILE_TIME_ONLY, which skips everything that needs
# GoogleTest.

# Stable LSD radix sort of double keys over core/kernel's block_reduce. Its
# GPU test is test/experimental, which the root CMakeLists adds under this
# same option.
add_subdirectory(sort/radix)

# Stable merge sort of double keys, radix_sort's contract and order. Its
# benchmark times it against radix_sort, so it comes after sort/radix.
add_subdirectory(sort/merge)

# Half-precision GEMM through cuBLAS / hipBLAS, both accumulators. Its GPU test
# is test/experimental.
add_subdirectory(fp16_gemm)

# Device arrays streamed into .npy files through a fixed ring of pinned slots,
# several files at once. Its GPU test is test/experimental.
add_subdirectory(output_ring)

# fp64 GEMM through the cuBLASLt / hipBLASLt heuristic, on the shapes
# nevpt2::rdms' digest runs. Benchmark only: there is nothing to assert about
# which algorithm a vendor heuristic picks.
add_subdirectory(blas_lt)

# The card's own fp64 ceiling, and fp64 GEMM composed from vendor Sgemm calls
# on a hi/lo fp32 split -- the speed-against-accuracy curve of spending fp32
# throughput to buy fp64 accuracy. Benchmark only, like blas_lt.
add_subdirectory(split_gemm)

# The DF exchange build cut into batches, B resident or streamed from pinned
# memory through a ring of device slots: what each batch costs on top of the
# one-call build. Benchmark only, like blas_lt.
add_subdirectory(gemm_host_offload)

# fp64 gemm driven from Rust: a C ABI over DeviceResources and device buffers
# as a shared library, and the cargo crate beside it that binds it. Cargo is
# not driven from here; the README has the two-step build.
add_subdirectory(rust/gemm)

# The spherical AO overlap S of a molecule in a basis, from Rust: a C ABI over
# the pipeline's setup and overlap steps, on rust/gemm's handles, so it comes
# after it. Checked against PySCF; cargo is run by hand, as for rust/gemm.
add_subdirectory(rust/s_overlap)
