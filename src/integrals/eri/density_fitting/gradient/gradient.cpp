/// @file
/// @brief The DF gradients: the spherical matrices made Cartesian, the
/// energy builds' staging with each pair's or cell's atoms (and each bra
/// record's exponents), one launch per class into a zeroed G, and G
/// downloaded.

module;

// The kernel launchers, defined in gradient_launch.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "gradient_bridge.h"

module free_breakfast.integrals.eri.density_fitting.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne
import free_breakfast.core.primitives;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemmStridedBatched, wwrMemGetInfo, wwrMemsetAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.blas;

#include "diagnostics.h"

namespace free_breakfast {

// One extent per momentum the kernels cover, as in metric.cpp and
// three_center.cpp: a metric class (lP, lQ) is entry lP * kMetricMomenta + lQ,
// a three-center class (la, lb, lP) entry (la * kThreeCenterMomenta + lb) *
// kThreeCenterAuxMomenta + lP, of the per-class tables below.
constexpr auto kMetricMomenta = static_cast<size_t>(device::kMetricMaxMomentum + 1);
constexpr auto kThreeCenterMomenta = static_cast<size_t>(device::kThreeCenterMaxMomentum + 1);
constexpr auto kThreeCenterAuxMomenta = static_cast<size_t>(device::kThreeCenterAuxMaxMomentum + 1);
constexpr size_t kThreeCenterClasses =
    kThreeCenterMomenta * kThreeCenterMomenta * kThreeCenterAuxMomenta;

// The atom each of a placed basis's shells sits on, by shell index.
std::vector<int32_t> shell_atoms(const BasisSet &basis) {
  std::vector<int32_t> atom_of(basis.nshells());
  for (uint32_t a = 0; a < basis.natoms(); ++a) {
    const uint32_t first = basis.atom_shell_offset(a);
    std::fill_n(atom_of.begin() + first, basis.atom_shell_count(a), static_cast<int32_t>(a));
  }
  return atom_of;
}

// The aux records, one per primitive of every shell, field-major in
// kMetricRecordFields segments, into @p out; shell i's are
// [rec_begin[i], rec_begin[i + 1]).
void stage_aux_records(const BasisSet &aux, const std::vector<int32_t> &rec_begin, double *out) {
  const auto n_rec = static_cast<size_t>(rec_begin.back());
  for (size_t i = 0, r = 0; i < aux.shells().size(); ++i) {
    const Shell &shell = aux.shells()[i];
    const Vec<double, 3> &centre = shell.center();
    for (const Primitive &prim : shell.primitives()) {
      // The metric's record (its README): c carries every normalisation.
      const double weight = prim.coefficient * std::pow(std::numbers::pi / prim.exponent, 1.5);
      const std::array<double, device::kMetricRecordFields> fields{centre.x, centre.y, centre.z,
                                                                   prim.exponent, weight};
      for (size_t f = 0; f < fields.size(); ++f) {
        out[(f * n_rec) + r] = fields[f];
      }
      ++r;
    }
  }
}

// Each shell's first record, one record per primitive, and the total last.
std::vector<int32_t> record_begins(const BasisSet &basis) {
  std::vector<int32_t> rec_begin{0};
  rec_begin.reserve(basis.shells().size() + 1);
  for (const Shell &shell : basis.shells()) {
    rec_begin.push_back(rec_begin.back() + static_cast<int32_t>(shell.nprimitives()));
  }
  return rec_begin;
}

Result<std::vector<double>> metric_gradient(DeviceResources &res, const BasisSet &aux,
                                            const double *w) {
  if (aux.natoms() == 0) {
    return err_config("metric_gradient: the aux basis has no atoms; make it with place()");
  }
  const size_t n = aux.nao_cart();
  const size_t n_sph = aux.nao_sphe();
  // The kernels index W as a 32-bit a * n_aux + b.
  if (n * n > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("metric_gradient: {} Cartesian fitting functions is too many for the "
                      "kernels' 32-bit index",
                      n);
  }
  const std::vector<Shell> &shells = aux.shells();
  for (const Shell &shell : shells) {
    if (shell.angular_momentum() > device::kMetricMaxMomentum) {
      return err_config("metric_gradient: no kernel for an aux shell of momentum {}; the kernels "
                        "cover l <= {}",
                        shell.angular_momentum(), device::kMetricMaxMomentum);
    }
  }
  const size_t n_g = 3 * static_cast<size_t>(aux.natoms());
  if (shells.empty()) {
    return std::vector<double>(n_g, 0.0);
  }

  const std::vector<int32_t> atom_of = shell_atoms(aux);
  const std::vector<int32_t> rec_begin = record_begins(aux);
  const auto n_rec = static_cast<size_t>(rec_begin.back());

  // The pairs P <= Q by shell index, the lower-momentum shell first, as
  // metric.cpp stages them, with their atoms in the staged order.
  const auto class_of = [&](const size_t i, const size_t j) {
    const auto li = static_cast<size_t>(shells[i].angular_momentum());
    const auto lj = static_cast<size_t>(shells[j].angular_momentum());
    return (std::min(li, lj) * kMetricMomenta) + std::max(li, lj);
  };
  std::array<size_t, kMetricMomenta * kMetricMomenta> n_pair{};
  for (size_t i = 0; i < shells.size(); ++i) {
    for (size_t j = i; j < shells.size(); ++j) {
      ++n_pair[class_of(i, j)];
    }
  }
  std::array<size_t, kMetricMomenta * kMetricMomenta> pair_begin{};
  size_t n_pairs = 0;
  for (size_t c = 0; c < n_pair.size(); ++c) {
    pair_begin[c] = n_pairs;
    n_pairs += n_pair[c];
  }
  std::vector<int32_t> pairs(static_cast<size_t>(device::kMetricGradientPairFields) * n_pairs);
  std::array<size_t, kMetricMomenta * kMetricMomenta> filled{};
  for (size_t i = 0; i < shells.size(); ++i) {
    for (size_t j = i; j < shells.size(); ++j) {
      const bool swapped = shells[i].angular_momentum() > shells[j].angular_momentum();
      const size_t p = swapped ? j : i;
      const size_t q = swapped ? i : j;
      const size_t c = class_of(i, j);
      int32_t *const run =
          pairs.data() + (static_cast<size_t>(device::kMetricGradientPairFields) * pair_begin[c]);
      const std::array<int32_t, device::kMetricGradientPairFields> fields{
          rec_begin[p],
          rec_begin[p + 1],
          rec_begin[q],
          rec_begin[q + 1],
          static_cast<int32_t>(aux.ao_offset_cart(static_cast<uint32_t>(p))),
          static_cast<int32_t>(aux.ao_offset_cart(static_cast<uint32_t>(q))),
          atom_of[p],
          atom_of[q]};
      for (size_t f = 0; f < fields.size(); ++f) {
        run[(f * n_pair[c]) + filled[c]] = fields[f];
      }
      ++filled[c];
    }
  }

  // The staging: the aux records, then U_aux.
  const size_t n_records = static_cast<size_t>(device::kMetricRecordFields) * n_rec;
  std::vector<double> staging(n_records + (n * n_sph));
  stage_aux_records(aux, rec_begin, staging.data());
  std::ranges::copy(spherical_transform_matrix(aux), staging.data() + n_records);

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // t = U_aux W, then the Cartesian W = t U_aux^T.
  const auto [d_staging, d_pairs, t, w_cart, d_g] =
      res.device_regions<double, int32_t, double, double, double>(staging.size(), pairs.size(),
                                                                  n * n_sph, n * n, n_g);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_staging, staging.data(), sizeof(double) * staging.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_pairs, pairs.data(), sizeof(int32_t) * pairs.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  const double *const d_u = d_staging + n_records;
  const auto n32 = static_cast<int32_t>(n);
  const auto n_sph32 = static_cast<int32_t>(n_sph);
  FBF_TRY(gpu_status(
      gemm(res.blas(), n32, n_sph32, n_sph32, &kOne<double>, d_u, w, &kZero<double>, t)));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      res.blas(), n32, n32, n_sph32, &kOne<double>, t, d_u, &kZero<double>, w_cart)));

  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(d_g, 0, sizeof(double) * n_g, stream)));
  for (size_t lp = 0; lp < kMetricMomenta; ++lp) {
    for (size_t lq = lp; lq < kMetricMomenta; ++lq) {
      const size_t c = (lp * kMetricMomenta) + lq;
      if (n_pair[c] == 0) {
        continue;
      }
      FBF_TRY(gpu_status(device::metric_gradient_class(
          stream, static_cast<int32_t>(lp), static_cast<int32_t>(lq), static_cast<int32_t>(n_rec),
          d_staging, static_cast<int32_t>(n_pair[c]),
          d_pairs + (static_cast<size_t>(device::kMetricGradientPairFields) * pair_begin[c]), n32,
          static_cast<int32_t>(aux.natoms()), w_cart, d_g)));
    }
  }
  std::vector<double> g(n_g);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(g.data(), d_g, sizeof(double) * n_g,
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  // Also what lets the host staging go: a pageable upload has finished
  // reading its source only once the stream has.
  FBF_TRY(res.sync());
  return g;
}

// One AO shell pair, as mmd stores it (the lower momentum first): its two
// shells' first Cartesian AOs, their atoms and its run of the regrouped bra
// records.
struct ThreeCenterGradientPair {
  int32_t off_a;
  int32_t off_b;
  int32_t atom_a;
  int32_t atom_b;
  int32_t rec0;
  int32_t rec1;
};

Result<std::vector<double>> three_center_gradient(DeviceResources &res, const BasisSet &basis,
                                                  const std::span<const PrimitivePairClass> classes,
                                                  const BasisSet &aux, const double *gamma) {
  if (basis.natoms() == 0 || aux.natoms() != basis.natoms()) {
    return err_config("three_center_gradient: the bases have {} and {} atoms; make both with "
                      "place() over one molecule",
                      basis.natoms(), aux.natoms());
  }
  const std::vector<Shell> &shells = basis.shells();
  const std::vector<Shell> &aux_shells = aux.shells();
  for (size_t i = 0; i < shells.size(); ++i) {
    if (const int32_t l = shells[i].angular_momentum(); l > device::kThreeCenterMaxMomentum) {
      return err_unsupported("three_center_gradient: orbital shell {} has momentum {}; the "
                             "kernels cover orbital shells through l = {}",
                             i, l, device::kThreeCenterMaxMomentum);
    }
  }
  for (size_t i = 0; i < aux_shells.size(); ++i) {
    if (const int32_t l = aux_shells[i].angular_momentum();
        l > device::kThreeCenterAuxMaxMomentum) {
      return err_unsupported("three_center_gradient: aux shell {} has momentum {}; the kernels "
                             "cover aux shells through l = {}",
                             i, l, device::kThreeCenterAuxMaxMomentum);
    }
  }
  const size_t n_g = 3 * static_cast<size_t>(basis.natoms());
  if (shells.empty() || aux_shells.empty()) {
    return std::vector<double>(n_g, 0.0);
  }

  const size_t n = basis.nao_cart();
  const size_t n_sph = basis.nao_sphe();
  const size_t n_aux = aux.nao_cart();
  const size_t n_aux_sph = aux.nao_sphe();
  const size_t n_cart = n * n * n_aux;
  // The kernels index Gam as a 32-bit (m n_ao + n) n_aux + P.
  if (n_cart > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_unsupported("three_center_gradient: the Cartesian Gam, {} x {} x {} = {} elements "
                           "({} bytes), is past the kernels' 32-bit index",
                           n, n, n_aux, n_cart, sizeof(double) * n_cart);
  }

  // The counts, from the classes' sizes and the bases alone, so the refusal
  // below comes before anything is downloaded or allocated.
  size_t n_bra_rec = 0;
  for (const PrimitivePairClass &c : classes) {
    if (c.la < 0 || c.lb < c.la || c.lb > device::kThreeCenterMaxMomentum) {
      return err_config("three_center_gradient: a primitive-pair class ({}, {}) is not one of "
                        "the orbital basis's",
                        c.la, c.lb);
    }
    n_bra_rec += static_cast<size_t>(c.n_rec);
  }
  const std::vector<int32_t> aux_rec_begin = record_begins(aux);
  const auto n_aux_rec = static_cast<size_t>(aux_rec_begin.back());
  const size_t n_pairs = shells.size() * (shells.size() + 1) / 2;
  const size_t n_cells = n_pairs * aux_shells.size();
  const size_t n_bra = static_cast<size_t>(device::kThreeCenterGradientBraFields) * n_bra_rec;
  const size_t n_aux_fields = static_cast<size_t>(device::kMetricRecordFields) * n_aux_rec;
  const size_t n_staging = n_bra + n_aux_fields + (n * n_sph) + (n_aux * n_aux_sph);
  const size_t n_cell_fields =
      static_cast<size_t>(device::kThreeCenterGradientCellFields) * n_cells;
  // Gam's aux index made Cartesian lands in cart, its n index in work, its m
  // index back in cart.
  const size_t n_work = n_aux * n * n_sph;
  const auto region = [](const size_t bytes) {
    return math::align_up(bytes, DeviceResources::kWorkspaceAlignment);
  };
  const size_t need = region(sizeof(double) * n_staging) + region(sizeof(int32_t) * n_cell_fields) +
                      region(sizeof(double) * n_cart) + region(sizeof(double) * n_work) +
                      region(sizeof(double) * n_g);

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // Past the device's memory the workspace's growth would abort
  // (CreatePolicy), not return; three_center.cpp has the arithmetic.
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  const size_t available = std::max(res.device_workspace(0).num_elements(), free_bytes);
  if (need > available) {
    return err_unsupported("three_center_gradient: over {} Cartesian orbital and {} Cartesian "
                           "fitting functions it needs {} bytes of device scratch (the Cartesian "
                           "Gam alone {} bytes), and {} bytes are available",
                           n, n_aux, need, sizeof(double) * n_cart, available);
  }

  // mmd's records, every class's in turn, field-major over all of them, in
  // mmd's order: copied back as they are, then regrouped below.
  std::vector<double> mmd_fields(n_bra);
  std::vector<int32_t> mmd_off_a(n_bra_rec);
  std::vector<int32_t> mmd_off_b(n_bra_rec);
  size_t base = 0;
  for (const PrimitivePairClass &c : classes) {
    const PrimitivePairClassView v = c.view();
    const auto n_rec = static_cast<size_t>(v.n_rec);
    const std::array<const double *, device::kThreeCenterGradientBraFields> fields{
        v.zeta, v.p_x,  v.p_y,  v.p_z, v.pa_x, v.pa_y, v.pa_z,
        v.pb_x, v.pb_y, v.pb_z, v.s,   v.ea,   v.eb};
    for (size_t f = 0; f < fields.size(); ++f) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(mmd_fields.data() + (f * n_bra_rec) + base, fields[f],
                                             sizeof(double) * n_rec, wwr::wwrMemcpyDeviceToHost,
                                             stream)));
    }
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(mmd_off_a.data() + base, v.off_a, sizeof(int32_t) * n_rec,
                                       wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(mmd_off_b.data() + base, v.off_b, sizeof(int32_t) * n_rec,
                                       wwr::wwrMemcpyDeviceToHost, stream)));
    base += n_rec;
  }
  FBF_TRY(res.sync());

  // The atom of each Cartesian AO, from its shell's.
  const std::vector<int32_t> shell_atom = shell_atoms(basis);
  std::vector<int32_t> ao_atom(n);
  for (uint32_t s = 0; s < basis.nshells(); ++s) {
    std::fill_n(ao_atom.begin() + basis.ao_offset_cart(s), shells[s].function_count_cart(),
                shell_atom[s]);
  }
  const std::vector<int32_t> aux_atom = shell_atoms(aux);

  // Each class's records grouped by shell pair, (off_a, off_b) ascending, mmd's
  // order kept within a pair, as three_center.cpp groups them.
  std::vector<size_t> order(n_bra_rec);
  std::iota(order.begin(), order.end(), size_t{0});
  std::array<std::vector<ThreeCenterGradientPair>, kThreeCenterMomenta * kThreeCenterMomenta> pairs;
  base = 0;
  for (const PrimitivePairClass &c : classes) {
    const auto first = order.begin() + static_cast<std::ptrdiff_t>(base);
    const auto last = first + c.n_rec;
    std::stable_sort(first, last, [&](const size_t x, const size_t y) {
      return std::pair{mmd_off_a[x], mmd_off_b[x]} < std::pair{mmd_off_a[y], mmd_off_b[y]};
    });
    const size_t width_a = function_count_cart(c.la);
    const size_t width_b = function_count_cart(c.lb);
    std::vector<ThreeCenterGradientPair> &out =
        pairs[(static_cast<size_t>(c.la) * kThreeCenterMomenta) + static_cast<size_t>(c.lb)];
    for (auto run = first; run != last;) {
      const int32_t oa = mmd_off_a[*run];
      const int32_t ob = mmd_off_b[*run];
      const auto end = std::find_if(
          run, last, [&](const size_t r) { return mmd_off_a[r] != oa || mmd_off_b[r] != ob; });
      if (oa < 0 || ob < 0 || static_cast<size_t>(oa) + width_a > n ||
          static_cast<size_t>(ob) + width_b > n) {
        return err_config("three_center_gradient: a ({}, {}) record's AOs ({}, {}) are not the "
                          "orbital basis's {} Cartesian functions",
                          c.la, c.lb, oa, ob, n);
      }
      out.push_back({oa, ob, ao_atom[static_cast<size_t>(oa)], ao_atom[static_cast<size_t>(ob)],
                     static_cast<int32_t>(run - order.begin()),
                     static_cast<int32_t>(end - order.begin())});
      run = end;
    }
    base += static_cast<size_t>(c.n_rec);
  }
  size_t found = 0;
  for (const std::vector<ThreeCenterGradientPair> &p : pairs) {
    found += p.size();
  }
  if (found != n_pairs) {
    return err_config("three_center_gradient: the primitive-pair classes hold {} shell pairs, and "
                      "the orbital basis's {} shells have {}",
                      found, shells.size(), n_pairs);
  }

  // The staging: the bra records in that order, the aux records, then U and
  // U_aux.
  std::vector<double> staging(n_staging);
  for (size_t f = 0; f < static_cast<size_t>(device::kThreeCenterGradientBraFields); ++f) {
    for (size_t k = 0; k < n_bra_rec; ++k) {
      staging[(f * n_bra_rec) + k] = mmd_fields[(f * n_bra_rec) + order[k]];
    }
  }
  stage_aux_records(aux, aux_rec_begin, staging.data() + n_bra);
  double *const u = staging.data() + n_bra + n_aux_fields;
  std::ranges::copy(spherical_transform_matrix(basis), u);
  std::ranges::copy(spherical_transform_matrix(aux), u + (n * n_sph));

  // The cells, one per (shell pair, aux shell), per class (la, lb, lP), the
  // pair outer and the aux shell inner, field-major within each class's run.
  std::array<std::vector<size_t>, kThreeCenterAuxMomenta> aux_of_l;
  for (size_t i = 0; i < aux_shells.size(); ++i) {
    aux_of_l[static_cast<size_t>(aux_shells[i].angular_momentum())].push_back(i);
  }
  std::array<size_t, kThreeCenterClasses> n_cell{};
  std::array<size_t, kThreeCenterClasses> cell_begin{};
  std::vector<int32_t> cells(n_cell_fields);
  size_t cell_end = 0;
  for (size_t ab = 0; ab < pairs.size(); ++ab) {
    for (size_t lp = 0; lp < kThreeCenterAuxMomenta; ++lp) {
      const size_t c = (ab * kThreeCenterAuxMomenta) + lp;
      n_cell[c] = pairs[ab].size() * aux_of_l[lp].size();
      cell_begin[c] = cell_end;
      int32_t *const run =
          cells.data() +
          (static_cast<size_t>(device::kThreeCenterGradientCellFields) * cell_begin[c]);
      size_t k = 0;
      for (const ThreeCenterGradientPair &pair : pairs[ab]) {
        for (const size_t s : aux_of_l[lp]) {
          const std::array<int32_t, device::kThreeCenterGradientCellFields> fields{
              pair.rec0,
              pair.rec1,
              aux_rec_begin[s],
              aux_rec_begin[s + 1],
              pair.off_a,
              pair.off_b,
              static_cast<int32_t>(aux.ao_offset_cart(static_cast<uint32_t>(s))),
              pair.atom_a,
              pair.atom_b,
              aux_atom[s]};
          for (size_t f = 0; f < fields.size(); ++f) {
            run[(f * n_cell[c]) + k] = fields[f];
          }
          ++k;
        }
      }
      cell_end += n_cell[c];
    }
  }

  const auto [d_staging, d_cells, cart, work, d_g] =
      res.device_regions<double, int32_t, double, double, double>(n_staging, n_cell_fields, n_cart,
                                                                  n_work, n_g);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_staging, staging.data(), sizeof(double) * n_staging,
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_cells, cells.data(), sizeof(int32_t) * n_cell_fields,
                                         wwr::wwrMemcpyHostToDevice, stream)));

  const double *const d_aux = d_staging + n_bra;
  const double *const d_u = d_aux + n_aux_fields;
  const double *const d_u_aux = d_u + (n * n_sph);
  const auto n32 = static_cast<int32_t>(n);
  const auto n_sph32 = static_cast<int32_t>(n_sph);
  const auto n_aux32 = static_cast<int32_t>(n_aux);
  // The aux index: cart = U_aux Gam, n_aux x n_sph^2, from the n_aux_sph x
  // n_sph^2 view of Gam, P fastest.
  FBF_TRY(gpu_status(gemm(res.blas(), n_aux32, n_sph32 * n_sph32, static_cast<int32_t>(n_aux_sph),
                          &kOne<double>, d_u_aux, gamma, &kZero<double>, cart)));
  // The n index: for each spherical m, its n_aux x n_sph block of cart times
  // U^T, into work as n_aux x n blocks, U shared by the batch.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n_aux32, n32, n_sph32, &kOne<double>,
      cart, n_aux32, int64_t{n_aux32} * n_sph32, d_u, n32, 0, &kZero<double>, work, n_aux32,
      int64_t{n_aux32} * n32, n_sph32)));
  // The m index: the (n_aux n) x n_sph view of work times U^T, into cart,
  // element (m, n, P) at (m n + n) n_aux + P, the kernels' layout.
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      res.blas(), n_aux32 * n32, n32, n_sph32, &kOne<double>, work, d_u, &kZero<double>, cart)));

  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(d_g, 0, sizeof(double) * n_g, stream)));
  for (size_t la = 0; la < kThreeCenterMomenta; ++la) {
    for (size_t lb = la; lb < kThreeCenterMomenta; ++lb) {
      for (size_t lp = 0; lp < kThreeCenterAuxMomenta; ++lp) {
        const size_t c = (((la * kThreeCenterMomenta) + lb) * kThreeCenterAuxMomenta) + lp;
        if (n_cell[c] == 0) {
          continue;
        }
        FBF_TRY(gpu_status(device::three_center_gradient_class(
            stream, static_cast<int32_t>(la), static_cast<int32_t>(lb), static_cast<int32_t>(lp),
            static_cast<int32_t>(n_bra_rec), d_staging, static_cast<int32_t>(n_aux_rec), d_aux,
            static_cast<int32_t>(n_cell[c]),
            d_cells + (static_cast<size_t>(device::kThreeCenterGradientCellFields) * cell_begin[c]),
            n32, n_aux32, static_cast<int32_t>(basis.natoms()), cart, d_g)));
      }
    }
  }
  std::vector<double> g(n_g);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(g.data(), d_g, sizeof(double) * n_g,
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  // Also what lets the host staging go (the metric's comment).
  FBF_TRY(res.sync());
  return g;
}

} // namespace free_breakfast
