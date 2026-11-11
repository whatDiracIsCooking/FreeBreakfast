/// @file fb_overlap.h
/// @brief A C ABI over free_breakfast's pipeline setup steps and its overlap
/// step: the spherical AO overlap matrix S of a molecule in a basis, for the
/// Rust crate beside it.
///
/// S is built on DeviceResources of this ABI's own, held by an fb_overlap
/// handle, and copied out into device memory the caller owns -- in practice a
/// rust/gemm DeviceBuffer, which comes from WarpWraps' C ABI rather than from
/// free_breakfast -- so it feeds that crate's gemm like any other matrix.
///
/// The crate's src/lib.rs declares these by hand; keep the two in step. Every
/// function is noexcept to C++: an exception unwinding into a Rust frame is
/// undefined behaviour, so one that escapes (only std::bad_alloc in practice)
/// terminates, as a failed GPU create or alloc aborts (CreatePolicy).
#ifndef FREE_BREAKFAST_EXPERIMENTAL_RUST_S_OVERLAP_FB_OVERLAP_H
#define FREE_BREAKFAST_EXPERIMENTAL_RUST_S_OVERLAP_FB_OVERLAP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define FB_NOEXCEPT noexcept
extern "C" {
#else
#define FB_NOEXCEPT
#endif

/// Plain int32_t on the wire: an enum's width is not fixed in C.
typedef int32_t fb_status;
enum {
  FB_OK = 0,
  /// A null handle or pointer, a negative device, or a length that is not
  /// nao * nao.
  FB_INVALID_ARGUMENT = 1,
  /// The GPU runtime returned an error (a copy or a synchronize).
  FB_GPU_ERROR = 2,
  /// A free_breakfast routine refused or failed; fb_last_error() says why.
  FB_FAILED = 3,
};

typedef struct fb_overlap fb_overlap;

/// The message of the last FB_FAILED on this thread, or "" if there has been
/// none. Valid until the next call on this thread that fails.
const char *fb_last_error(void) FB_NOEXCEPT;

/// S of the molecule in XYZ file `xyz_path` (Angstrom, as every XYZ here),
/// in `basis`: a name the build carries (resolve_basis's, e.g. "cc-pvdz") or a
/// path to a BSE JSON file. Runs the Molecule, BasisSet, SphericalTransform,
/// PrimitivePairClasses and Overlap steps on a new DeviceResources on device
/// `dev_idx`, and waits for them.
///
/// On FB_OK, `*out` holds S (and the resources it lives on) until
/// fb_overlap_free, and `*nao` is its dimension. On FB_FAILED (a missing
/// file, an unknown basis, an element the basis lacks, a step's refusal),
/// fb_last_error() says why, and nothing is allocated.
fb_status fb_overlap_new(int32_t dev_idx, const char *xyz_path, const char *basis, fb_overlap **out,
                         size_t *nao) FB_NOEXCEPT;

/// Copy S into `dst`, device memory on the same device holding `len` ==
/// nao * nao doubles: column-major, in PySCF's spherical AO order. The copy
/// runs on `s`'s own stream and is waited for, so `dst` must be ready to be
/// written (any work another stream queued against it already finished).
fb_status fb_overlap_copy(const fb_overlap *s, double *dst, size_t len) FB_NOEXCEPT;

void fb_overlap_free(fb_overlap *s) FB_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#endif // FREE_BREAKFAST_EXPERIMENTAL_RUST_S_OVERLAP_FB_OVERLAP_H
