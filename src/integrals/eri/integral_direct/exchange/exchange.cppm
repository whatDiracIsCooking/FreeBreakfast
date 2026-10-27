/// @file
/// @brief The Cartesian exchange matrix K on the GPU, by McMurchie-Davidson,
/// over the kb kernels FelizJueves generates, screened by the dressed record
/// bound the ordered classes carry (OrderedPairClass::record_bound).
///
/// The records are
/// free_breakfast.integrals.eri.integral_direct.exchange.pairs's ordered
/// classes, re-exported here, so one import gives both the classes and the
/// build: every ordered class is both a bra class (a, c) and a ket class
/// (b, d).

export module free_breakfast.integrals.eri.integral_direct.exchange;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.recipes.matrix_ao;
export import free_breakfast.integrals.eri.integral_direct.exchange.pairs;

export namespace free_breakfast {

/// @brief The symmetry of @c exchange's density, which its K shares, and so
/// the sign of the mirror that completes K from its a <= b shell blocks.
///
/// Renaming c <-> d and using `(ac|bd) = (bd|ac)` gives `K[D]^T = K[D^T]`. So
/// a symmetric D has a symmetric K, `K_ba = K_ab^T`, and an antisymmetric D an
/// antisymmetric K, `K_ba = -K_ab^T`: either way only the a <= b blocks need
/// computing.
///
/// **A general D is two sweeps.** Split it as S = (D + D^T) / 2 plus
/// A = (D - D^T) / 2 (one @c geam), run @c exchange on S as Symmetric and on A
/// as Antisymmetric, and add the two K: the same kb work as one sweep over
/// every block, and each half fits its own decay, so it screens at least as
/// tightly. Codes usually have the two parts already: TDHF / RPA work with
/// X + Y and X - Y, which are exactly the symmetric and antisymmetric parts,
/// and so do CIS transition densities and the response to an imaginary
/// perturbation (a magnetic field's density is antisymmetric).
enum class DensitySymmetry { Symmetric, Antisymmetric };

/// @brief The Cartesian exchange matrix K_ab = sum_cd D_cd (ac|bd) of the AO
/// density @p density into @p k, screened at @p tau, by its a <= b shell
/// blocks, the rest mirrored as @p symmetry says.
///
/// @p classes are @c ordered_pair_classes(res, basis, primitive_pair_classes)
/// of the basis @p density and @p k were made for, built once per basis: the
/// whole set, in its order, since a bra's block tables are made against every
/// class of it.
///
/// Screened by kb's bound: a primitive quartet is skipped when
/// `Ip * Iq * F_ab < tau`, Ip and Iq the bra's and the ket's DRESSED record
/// bounds (OrderedPairClass::record_bound, FelizJueves's ao.k.record_bounds)
/// and F_ab the block's DENSITY FACTOR, a bound on every |D_cd| its K_ab block
/// can visit, from the density's decay with distance (arXiv:2407.21445
/// §4.3.1; FelizJueves #484): max|D| for a block whose shells are close, and
/// down to `H_off e^{-Λ d}` for one whose shells are d bohr further apart than
/// their pairs reach. FelizJueves's ao.k proves every skipped term under
/// @p tau with exactly that bound and factor.
/// @p tau = 0 skips nothing. The default is 1e-13, not @c coulomb's 1e-10:
/// each skipped term is under tau, but an element sums many, their number
/// growing with the record pairs that cover it, and at 1e-10 K misses libcint
/// by up to 6.9e-10 of max|K| (H2O/cc-pVDZ). 1e-13 is the largest round tau
/// at which H2O in STO-3G and cc-pVDZ and the H10 chain all meet 1e-12 max|K|.
///
/// Zeroes @p k and, on res.stream(), computes max|D|; then, per class on that
/// class's own stream, its largest record bound and, from a class (l, l), its
/// momentum's rank-to-AO table and AO centres; after a join, on res.stream(),
/// the density's decay (Λ, H_off) from the centres; then per class the (la, lb)
/// gather of the density into Dcd and its group geometry (each first shell's
/// centre and reach); then, after a join, each class as a bra on its own
/// stream: for every ket class, its block table's factors, one block per
/// (bra first shell a, ket first shell b) with a <= b, and one
/// kb<la, lc, lb, ld> launch over the table, one warp per block adding its
/// contracted K_ab straight into @p k; and after a last join, the a > b blocks
/// set to the a < b ones' transposes, negated for an antisymmetric density.
/// An a == b block is computed whole.
///
/// **@p symmetry is a contract: it is not checked** (refusing would need a
/// device pass and a host synchronization to read the verdict back). For a D
/// of neither symmetry, or the other one, the call returns its K's a <= b
/// blocks and the mirror of them @p symmetry names; @ref DensitySymmetry says
/// how to build such a D's K from two calls.
///
/// Queued on res.stream(), each phase's classes forked from it and joined back
/// (@c launch_each), with no host synchronization, and every class's stream
/// forked from it once more at the end, succeeded or not, so nothing queued
/// there later (a class's own frees, the next build's scratch) runs ahead of
/// another class's stream still reading it. Writes the classes' scratch
/// (@ref OrderedPairClass), so one call at a time per set of classes; it
/// allocates only one rank table per momentum, 3 n_ao doubles of AO centres
/// and n_ao int32s of shell starts, and four doubles of decay and bound
/// maximum, from res.device()'s pool in res.stream()'s order. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return Success; InvalidConfig if @p tau is negative or NaN, if @p res is
///         not on @p k's, @p density's or @p classes' device, if @p density and
///         @p k differ in order, if a class is above f, if the classes are of a
///         basis of another order than @p k's, if @p classes are not one whole
///         set in its order (a subset, or reordered), or if k.n() is too large
///         for the kernels' 32-bit index (k.n()^2 above INT32_MAX); or the
///         GpuRuntime error of the first memset, fork, launch or join that
///         failed, which leaves @p k unspecified.
[[nodiscard]] Status exchange(DeviceResources &res, std::span<OrderedPairClass> classes,
                              const CartesianMatrixAO &density, CartesianMatrixAO &k,
                              double tau = 1e-13,
                              DensitySymmetry symmetry = DensitySymmetry::Symmetric);

/// @brief The density factor F_ab of every block @c exchange would launch for
/// the same arguments, read back to the host: the decay, the reaches and the
/// factors made exactly as @c exchange makes them, and nothing of K itself
/// (no zeroing, no kb launch, no mirror). The factors do not depend on the
/// density's symmetry: both sweeps launch the same a <= b blocks.
///
/// For diagnostics and tests: how many blocks the decay screens below max|D|
/// is what decides whether it pays. The factors are bra class major (in
/// @p classes' order), then ket class, then each table's own order, bra group
/// major. It writes the classes' scratch as @c exchange does, and unlike it
/// synchronizes res.stream() before returning.
///
/// @return The factors (none for no @p classes or an empty basis); the
///         InvalidConfig errors of @c exchange that do not concern K; or the
///         GpuRuntime error of the first launch, copy or synchronization that
///         failed.
[[nodiscard]] Result<std::vector<double>>
exchange_block_factors(DeviceResources &res, std::span<OrderedPairClass> classes,
                       const CartesianMatrixAO &density, double tau = 1e-13);

} // namespace free_breakfast
