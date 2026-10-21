/// @file
/// @brief The Cartesian exchange matrix K on the GPU, by McMurchie-Davidson,
/// over the kb kernels FelizJueves generates, screened by the dressed record
/// bound the ordered classes carry (OrderedPairClass::record_bound).
///
/// The records are free_breakfast.integrals.exchange.pairs's ordered classes,
/// re-exported here, so one import gives both the classes and the build: every
/// ordered class is both a bra class (a, c) and a ket class (b, d).

export module free_breakfast.integrals.exchange;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.linalg.matrix_ao;
export import free_breakfast.integrals.exchange.pairs;

export namespace free_breakfast {

/// @brief The Cartesian exchange matrix K_ab = sum_cd D_cd (ac|bd) of the AO
/// density @p density into @p k, screened at @p tau, by the a <= b shell
/// blocks mirrored unless @p symmetric is false.
///
/// @p classes are @c ordered_pair_classes(res, basis, primitive_pair_classes)
/// of the basis @p density and @p k were made for, built once per basis.
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
/// @p tau = 0 skips nothing. The default, 1e-10, is @c coulomb's.
///
/// Zeroes @p k and, on res.stream(), computes max|D|; then, per class on that
/// class's own stream, its largest record bound and, from a class (l, l), its
/// momentum's rank-to-AO table and AO centres; after a join, on res.stream(),
/// the density's decay (Λ, H_off) from the centres; then per class the (la, lb)
/// gather of the density into Dcd and its group geometry (each first shell's
/// centre and reach); then, after a join, each class as a bra on its own
/// stream: for every ket class, its block table and the blocks' factors, one
/// block per (bra first shell a, ket first shell b), and one
/// kb<la, lc, lb, ld> launch over them, one warp per block adding its
/// contracted K_ab straight into @p k.
///
/// **@p symmetric (the default) is a contract: @p density must be symmetric,
/// and it is not checked** (refusing would need a device pass and a host
/// synchronization to read the verdict back). It halves the work by the
/// bra-ket swap `(ac|bd) = (bd|ac)`: for a symmetric D, K_ba = K_ab^T, so each
/// table holds only the blocks of a <= b, and after every launch the a > b
/// blocks of @p k are set to the a < b ones' transposes; an a == b block is
/// computed whole. With @p symmetric false every table holds every block and
/// nothing is mirrored: K_ab and K_ba are each computed directly, for any D,
/// at twice the work. For a non-symmetric D the symmetric sweep returns the
/// a <= b blocks of its K, mirrored.
///
/// Queued on res.stream(), each phase's classes forked from it and joined back
/// (@c launch_each), with no host synchronization; the scratch (per class 4
/// doubles per group, (l_max + 1) n_rec int32s, and 6 int32s and a double per
/// block of its tables; one Dcd per class; one rank table per momentum; 3 n_ao
/// doubles of AO centres and n_ao int32s of shell starts; four doubles of
/// decay and bound maximum) is drawn from
/// res.device()'s pool and returned to it in res.stream()'s order. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return Success; InvalidConfig if @p tau is negative or NaN, if @p res is
///         not on @p k's, @p density's or @p classes' device, if @p density and
///         @p k differ in order, if a class is above d, if a class's shell
///         reaches past @p k's order, or if k.n() is too large for the kernels'
///         32-bit index (k.n()^2 above INT32_MAX), or a launch's table is (6
///         int32s per block above INT32_MAX); or the GpuRuntime error of
///         the first memset, fork, launch or join that failed, which leaves
///         @p k unspecified.
[[nodiscard]] Status exchange(DeviceResources &res, std::span<const OrderedPairClass> classes,
                              const CartesianMatrixAO &density, CartesianMatrixAO &k,
                              double tau = 1e-10, bool symmetric = true);

/// @brief The density factor F_ab of every block @c exchange would launch for
/// the same arguments, read back to the host: the decay, the reaches and the
/// block tables made exactly as @c exchange makes them, and nothing of K
/// itself (no zeroing, no kb launch, no mirror).
///
/// For diagnostics and tests: how many blocks the decay screens below max|D|
/// is what decides whether it pays. The factors are bra class major (in
/// @p classes' order), then ket class, then each table's own order, bra group
/// major. Unlike @c exchange it synchronizes res.stream() before returning.
///
/// @return The factors (none for no @p classes or an empty basis); the
///         InvalidConfig errors of @c exchange that do not concern K; or the
///         GpuRuntime error of the first launch, copy or synchronization that
///         failed.
[[nodiscard]] Result<std::vector<double>>
exchange_block_factors(DeviceResources &res, std::span<const OrderedPairClass> classes,
                       const CartesianMatrixAO &density, double tau = 1e-10, bool symmetric = true);

} // namespace free_breakfast
