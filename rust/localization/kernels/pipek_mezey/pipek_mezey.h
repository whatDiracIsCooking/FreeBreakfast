/* The Pipek-Mezey launch wrappers: the C header forked from
 * src/localization/pipek_mezey/pipek_mezey_bridge.h. pipek_mezey.cu defines
 * them; fb_localization.h includes this, and devtools/rust_bindings.py reads
 * it into fb-localization-sys. Plain C: no templates, no vendor types.
 *
 * Every matrix is double, column-major. The population tensor holds Q^A_ij
 * for every atom A and every orbital pair (i, j), atom-fastest: element
 * ((i n_occ) + j) n_atoms + A, n_atoms n_occ^2 doubles in all. */
#pragma once

#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The largest n_occ the launches address: the population build and the
/// rotation put the orbital pair on two launch axes that both backends cap
/// at 65535. The C++ kMaxOccupied.
#define FB_PM_MAX_OCCUPIED ((int32_t)65535)

/// Builds the symmetrized Mulliken populations
/// Q^A_ij = 1/2 sum_{mu in A} (C_mu,i W_mu,j + C_mu,j W_mu,i) into pops, on
/// stream, for every atom and every occupied pair. c is the ld x n_occ
/// occupied block and w = S C, both column-major with leading dimension ld.
/// Atom A owns the AOs from entry A of atom_ao_offsets up to entry A + 1,
/// so atom_ao_offsets holds n_atoms + 1 entries, each in 0..ld and none
/// below the one before. The i <= j triangle is computed and mirrored, so pops is
/// symmetric in (i, j) to the last bit. All pointers are device pointers;
/// the kernel trusts the offsets.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for an n_atoms
/// below 1, an n_occ below 1 or past FB_PM_MAX_OCCUPIED, an ld below 1, or
/// a null pointer; otherwise the launch's status. Launch errors only: a
/// fault inside the kernel surfaces at the next synchronization.
fb_launch_status fb_pm_populations(wwr_stream stream, int32_t n_atoms, int32_t n_occ, int32_t ld,
                                   const double *c, const double *w, const int32_t *atom_ao_offsets,
                                   double *pops);

/// The rotation of each of one round's n_pairs disjoint orbital pairs that
/// raises L_exponent = sum_i sum_A (Q^A_ii)^exponent, exponent 2 (Pipek and
/// Mezey's exact angle) or 4 (Knizia's IBO angle, kept only where it raises
/// L_4 exactly), on stream. pairs holds the round's n_pairs (first, second)
/// orbital indices, 2 n_pairs entries; an index of n_occ is the padding of
/// an odd count. Each of the n_occ orbitals must be in exactly one pair.
///
/// The outputs are per orbital, n_occ each: a pair's two members share the
/// cosine in cos_out, carry opposite sines in sin_out and name each other
/// in partner_out; an orbital drawn against the padding index, or whose
/// pair gains no more than gain_threshold, gets the identity (cosine 1,
/// sine 0) and, against the padding, is its own partner. All pointers are
/// device pointers; the kernel trusts the pairs.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for an exponent
/// other than 2 or 4, an n_atoms or n_pairs below 1, an n_occ below 1 or
/// past FB_PM_MAX_OCCUPIED, or a null pointer; otherwise the launch's
/// status. Launch errors only.
fb_launch_status fb_pm_pair_angles(wwr_stream stream, int32_t exponent, int32_t n_atoms,
                                   int32_t n_occ, const double *pops, const int32_t *pairs,
                                   int32_t n_pairs, double gain_threshold, double *cos_out,
                                   double *sin_out, int32_t *partner_out);

/// pops_out = R^T pops_in R, atom by atom, on stream, for the block-diagonal
/// rotation R that fb_pm_pair_angles wrote as cos_v, sin_v and partner,
/// n_occ each. Out of place, since each output element reads four input
/// elements; pops_in and pops_out must not overlap. All pointers are device
/// pointers; the kernel trusts each partner to be in 0..n_occ.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for an n_atoms
/// below 1, an n_occ below 1 or past FB_PM_MAX_OCCUPIED, or a null
/// pointer; otherwise the launch's status. Launch errors only.
fb_launch_status fb_pm_rotate_populations(wwr_stream stream, int32_t n_atoms, int32_t n_occ,
                                          const double *pops_in, const double *cos_v,
                                          const double *sin_v, const int32_t *partner,
                                          double *pops_out);

/// u_out = u_in R for the same rotation, on the column-major n_occ x n_occ
/// matrices u_in and u_out, on stream. Out of place, as
/// fb_pm_rotate_populations is. All pointers are device pointers; the
/// kernel trusts each partner to be in 0..n_occ.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for an n_occ below
/// 1 or past FB_PM_MAX_OCCUPIED, or a null pointer; otherwise the launch's
/// status. Launch errors only.
fb_launch_status fb_pm_rotate_columns(wwr_stream stream, int32_t n_occ, const double *u_in,
                                      const double *cos_v, const double *sin_v,
                                      const int32_t *partner, double *u_out);

/// L_exponent = sum_i sum_A (Q^A_ii)^exponent of the population tensor pops
/// into the device scalar out, on stream, exponent 2 or 4. One fold in a
/// fixed order, so the value is the same bits on every run. Both pointers
/// are device pointers.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for an exponent
/// other than 2 or 4, an n_atoms below 1, an n_occ below 1 or past
/// FB_PM_MAX_OCCUPIED, or a null pointer; otherwise the launch's status.
/// Launch errors only.
fb_launch_status fb_pm_functional(wwr_stream stream, int32_t exponent, int32_t n_atoms,
                                  int32_t n_occ, const double *pops, double *out);

/// Writes the n x n identity into the column-major matrix, on stream. matrix
/// is a device pointer.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for an n below 1
/// or past FB_PM_MAX_OCCUPIED, or a null matrix; otherwise the launch's
/// status. Launch errors only.
fb_launch_status fb_pm_identity(wwr_stream stream, int32_t n, double *matrix);

#ifdef __cplusplus
}
#endif
