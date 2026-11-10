/// @file fb_overlap.h
/// @brief A C ABI over free_breakfast's pipeline setup steps and its overlap
/// step: the spherical AO overlap matrix S of a molecule in a basis, for the
/// Rust crate beside it.
///
/// Built on rust/gemm's handles (fb_gemm.h): S comes back as an fb_buffer of
/// the fb_resources it was built on, so it frees, downloads and feeds fb_dgemm
/// like any other. The crate's src/lib.rs declares it by hand; keep the two in
/// step.
#ifndef FREE_BREAKFAST_EXPERIMENTAL_RUST_S_OVERLAP_FB_OVERLAP_H
#define FREE_BREAKFAST_EXPERIMENTAL_RUST_S_OVERLAP_FB_OVERLAP_H

#include "fb_gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

/// S of the molecule in XYZ file `xyz_path` (Angstrom, as every XYZ here),
/// in `basis`: a name the build carries (resolve_basis's, e.g. "cc-pvdz") or a
/// path to a BSE JSON file. Runs the Molecule, BasisSet, SphericalTransform,
/// PrimitivePairClasses and Overlap steps on `res`, and waits for them.
///
/// On FB_OK, `*out` is a new nao x nao buffer holding S, column-major in
/// PySCF's spherical AO order, and `*nao` is nao. On FB_FAILED (a missing
/// file, an unknown basis, an element the basis lacks, a step's refusal),
/// fb_last_error() says why, and nothing is allocated.
fb_status fb_overlap(fb_resources *res, const char *xyz_path, const char *basis, fb_buffer **out,
                     size_t *nao) FB_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#endif // FREE_BREAKFAST_EXPERIMENTAL_RUST_S_OVERLAP_FB_OVERLAP_H
