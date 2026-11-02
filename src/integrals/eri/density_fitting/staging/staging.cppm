/// @file
/// @brief The launch staging the DF derivative kernels share: the gradient
/// kernels and the Hessian kernels read one parameter block (the Hessian's
/// with H in place of G), so gradient.cpp and hessian.cpp stage through these.
/// The aux records and the metric's shell pairs with their atoms; mmd's bra
/// records regrouped by shell pair, with their exponents and atoms; the
/// three-center cells; and the spherical matrices made Cartesian.
///
/// Internal to density fitting: free_breakfast.integrals.eri.density_fitting
/// does not re-export it, and gradient and hessian link it PRIVATE. Two named
/// modules cannot share a partition, hence a module of its own. See README.md.

export module free_breakfast.integrals.eri.density_fitting.staging;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

export namespace free_breakfast {

// One extent per momentum the kernels cover, as in metric.cpp and
// three_center.cpp: a metric class (lP, lQ) is entry lP * kMetricMomenta + lQ,
// a three-center class (la, lb, lP) entry (la * kThreeCenterMomenta + lb) *
// kThreeCenterAuxMomenta + lP, of the per-class tables. The energy kernels'
// limits plus one; this interface cannot include their bridges, so
// staging.cpp checks these against them.
inline constexpr size_t kMetricMomenta = 7;
inline constexpr size_t kThreeCenterMomenta = 4;
inline constexpr size_t kThreeCenterAuxMomenta = 7;
inline constexpr size_t kMetricClasses = kMetricMomenta * kMetricMomenta;
inline constexpr size_t kThreeCenterClasses =
    kThreeCenterMomenta * kThreeCenterMomenta * kThreeCenterAuxMomenta;

/// @brief The atom each of a placed basis's shells sits on, by shell index.
[[nodiscard]] std::vector<int32_t> shell_atoms(const BasisSet &basis);

/// @brief Each shell's first record, one record per primitive, and the total
/// last.
[[nodiscard]] std::vector<int32_t> record_begins(const BasisSet &basis);

/// @brief Refuses an aux basis the metric kernels cannot take, each message
/// as @p who.
[[nodiscard]] Status check_metric_basis(const BasisSet &aux, std::string_view who);

/// @brief The metric's staging: the aux records, one per primitive of every
/// shell, field-major in kMetricRecordFields segments (shell i's are
/// [rec_begin[i], rec_begin[i + 1])), then U_aux.
[[nodiscard]] std::vector<double> stage_metric_records(const BasisSet &aux,
                                                       const std::vector<int32_t> &rec_begin);

/// @brief The metric's aux shell pairs, P <= Q by shell index with the
/// lower-momentum shell first, as metric.cpp stages them, each with its two
/// atoms in the staged order: one run per class c = lP kMetricMomenta + lQ,
/// count[c] pairs from pair begin[c], kMetricGradientPairFields segments of
/// count[c] each.
struct MetricPairs {
  std::vector<int32_t> fields;
  std::array<size_t, kMetricClasses> count{};
  std::array<size_t, kMetricClasses> begin{};
};

[[nodiscard]] MetricPairs stage_metric_pairs(const BasisSet &aux,
                                             const std::vector<int32_t> &rec_begin);

/// @brief W made Cartesian, w_cart = U_aux W U_aux^T, through t (n x n_sph),
/// on res.stream(); @p d_u_aux is U_aux, n x n_sph.
[[nodiscard]] Status metric_cartesian(DeviceResources &res, size_t n, size_t n_sph,
                                      const double *d_u_aux, const double *w, double *t,
                                      double *w_cart);

/// @brief One AO shell pair, as mmd stores it (the lower momentum first): its
/// two shells' first Cartesian AOs, their atoms and its run of the regrouped
/// bra records.
struct ThreeCenterPair {
  int32_t off_a;
  int32_t off_b;
  int32_t atom_a;
  int32_t atom_b;
  int32_t rec0;
  int32_t rec1;
};

/// @brief mmd's bra records regrouped by shell pair: n_rec records,
/// kThreeCenterGradientBraFields segments of n_rec in fields, and each pair
/// class's (la, lb) shell pairs at la kThreeCenterMomenta + lb of pairs.
struct ThreeCenterBra {
  size_t n_rec = 0;
  std::vector<double> fields;
  std::array<std::vector<ThreeCenterPair>, kThreeCenterMomenta * kThreeCenterMomenta> pairs;
};

/// @brief Refuses bases the three-center kernels cannot take, each message as
/// @p who.
[[nodiscard]] Status check_three_center_bases(const BasisSet &basis, const BasisSet &aux,
                                              std::string_view who);

/// @brief The bra records @p classes hold, from their sizes alone (so a
/// refusal on memory can come before anything is downloaded), refusing a
/// class that is not one of an orbital basis's, as @p who.
[[nodiscard]] Result<size_t> bra_record_count(std::span<const PrimitivePairClass> classes,
                                              std::string_view who);

/// @brief mmd's records, every class's in turn, copied back as they are (one
/// synchronization), then grouped by shell pair, (off_a, off_b) ascending,
/// mmd's order kept within a pair, as three_center.cpp groups them, each pair
/// with its two atoms; as @p who.
[[nodiscard]] Result<ThreeCenterBra>
stage_three_center_bra(DeviceResources &res, const BasisSet &basis,
                       std::span<const PrimitivePairClass> classes, size_t n_bra_rec,
                       std::string_view who);

/// @brief The three-center staging: the regrouped @p bra_fields, the aux
/// records, then U and U_aux.
[[nodiscard]] std::vector<double>
stage_three_center_records(std::vector<double> bra_fields, const BasisSet &basis,
                           const BasisSet &aux, const std::vector<int32_t> &aux_rec_begin);

/// @brief The cells of every shell pair of @p bra crossed with the aux shells
/// [first, first + count), one per (pair, aux shell), appended to @p cells
/// one run per class (la, lb, lP), the pair outer and the aux shell inner,
/// kThreeCenterGradientCellFields segments of each run's count; class c's run
/// holds n_cell[c] cells from cell cell_begin[c] of the whole of @p cells.
void stage_three_center_cells(const ThreeCenterBra &bra, const BasisSet &aux,
                              const std::vector<int32_t> &aux_rec_begin, uint32_t first,
                              uint32_t count, std::vector<int32_t> &cells,
                              std::array<size_t, kThreeCenterClasses> &n_cell,
                              std::array<size_t, kThreeCenterClasses> &cell_begin);

/// @brief Gam made Cartesian on all three indices, on res.stream(): the
/// spherical @p gamma, (m, n, P) at P + n_aux_sph (n + n_sph m), into @p cart,
/// (m, n, P) at (m n + n) n_aux + P, the kernels' layout, through @p work
/// (n_aux n n_sph). @p cart holds n^2 n_aux; @p d_u is U (n x n_sph),
/// @p d_u_aux U_aux (n_aux x n_aux_sph).
[[nodiscard]] Status three_center_cartesian(DeviceResources &res, size_t n, size_t n_sph,
                                            size_t n_aux, size_t n_aux_sph, const double *d_u,
                                            const double *d_u_aux, const double *gamma,
                                            double *cart, double *work);

} // namespace free_breakfast
