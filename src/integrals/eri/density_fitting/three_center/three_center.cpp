/// @file
/// @brief The (mn|P) build: mmd's records regrouped by shell pair, the aux
/// records and the cells staged, one launch per class, the spherical transform
/// and the (m, n) symmetrization.

module;

// The kernel launchers, defined in three_center_launch.cu: declared in the GMF
// so they keep external linkage (see the header).
#include "three_center_bridge.h"

module free_breakfast.integrals.eri.density_fitting.three_center;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne
import free_breakfast.core.primitives;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemmStridedBatched, wwrMemGetInfo, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.blas;

#include "diagnostics.h"

namespace free_breakfast {

// One extent per momentum the kernels cover; a class (la, lb, lP) is entry
// (la * kThreeCenterMomenta + lb) * kThreeCenterAuxMomenta + lP of the
// per-class tables below.
constexpr auto kThreeCenterMomenta = static_cast<size_t>(device::kThreeCenterMaxMomentum + 1);
constexpr auto kThreeCenterAuxMomenta = static_cast<size_t>(device::kThreeCenterAuxMaxMomentum + 1);
constexpr size_t kThreeCenterClasses =
    kThreeCenterMomenta * kThreeCenterMomenta * kThreeCenterAuxMomenta;

// One AO shell pair, as mmd stores it (the lower momentum first): its two
// shells' first Cartesian AOs and its run of the regrouped bra records.
struct ThreeCenterPair {
  int32_t off_a;
  int32_t off_b;
  int32_t rec0;
  int32_t rec1;
};

Status three_center(DeviceResources &res, const BasisSet &basis,
                    const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                    double *b, const size_t max_bytes) {
  const std::vector<Shell> &shells = basis.shells();
  const std::vector<Shell> &aux_shells = aux.shells();
  for (size_t i = 0; i < shells.size(); ++i) {
    if (const int32_t l = shells[i].angular_momentum(); l > device::kThreeCenterMaxMomentum) {
      return err_unsupported("three_center: orbital shell {} has momentum {}; the kernels cover "
                             "orbital shells through l = {}",
                             i, l, device::kThreeCenterMaxMomentum);
    }
  }
  for (size_t i = 0; i < aux_shells.size(); ++i) {
    if (const int32_t l = aux_shells[i].angular_momentum();
        l > device::kThreeCenterAuxMaxMomentum) {
      return err_unsupported("three_center: aux shell {} has momentum {}; the kernels cover aux "
                             "shells through l = {}",
                             i, l, device::kThreeCenterAuxMaxMomentum);
    }
  }
  if (shells.empty() || aux_shells.empty()) {
    return {};
  }

  const size_t n = basis.nao_cart();
  const size_t n_sph = basis.nao_sphe();
  const size_t n_aux = aux.nao_cart();
  const size_t n_aux_sph = aux.nao_sphe();
  const size_t n_cart = n * n * n_aux;
  // The kernels index the tensor as a 32-bit (m n_ao + n) n_aux + P.
  if (n_cart > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_unsupported("three_center: the Cartesian tensor, {} x {} x {} = {} elements ({} "
                           "bytes), is past the kernels' 32-bit index",
                           n, n, n_aux, n_cart, sizeof(double) * n_cart);
  }

  // The counts, from the classes' sizes and the bases alone, so the refusal
  // below comes before anything is downloaded or allocated.
  size_t n_bra_rec = 0;
  for (const PrimitivePairClass &c : classes) {
    if (c.la < 0 || c.lb < c.la || c.lb > device::kThreeCenterMaxMomentum) {
      return err_config("three_center: a primitive-pair class ({}, {}) is not one of the orbital "
                        "basis's",
                        c.la, c.lb);
    }
    n_bra_rec += static_cast<size_t>(c.n_rec);
  }
  std::vector<int32_t> aux_rec_begin{0};
  aux_rec_begin.reserve(aux_shells.size() + 1);
  for (const Shell &shell : aux_shells) {
    aux_rec_begin.push_back(aux_rec_begin.back() + static_cast<int32_t>(shell.nprimitives()));
  }
  const auto n_aux_rec = static_cast<size_t>(aux_rec_begin.back());
  const size_t n_pairs = shells.size() * (shells.size() + 1) / 2;
  const size_t n_cells = n_pairs * aux_shells.size();
  const size_t n_staging = (static_cast<size_t>(device::kThreeCenterBraFields) * n_bra_rec) +
                           (static_cast<size_t>(device::kThreeCenterAuxFields) * n_aux_rec) +
                           (n * n_sph) + (n_aux * n_aux_sph);
  const size_t n_cell_fields = static_cast<size_t>(device::kThreeCenterCellFields) * n_cells;
  // The aux index's transform out of the Cartesian tensor; the AO indices'
  // two steps write back over the Cartesian tensor and then over this.
  const size_t n_work = n_aux_sph * n * n;
  const auto region = [](const size_t bytes) {
    return math::align_up(bytes, DeviceResources::kWorkspaceAlignment);
  };
  const size_t need = region(sizeof(double) * n_staging) + region(sizeof(int32_t) * n_cell_fields) +
                      region(sizeof(double) * n_cart) + region(sizeof(double) * n_work);

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // Past the device's memory the workspace's growth would abort
  // (CreatePolicy), not return. A growth allocates the new block before it
  // frees the old, so it needs the free memory alone.
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  const size_t held = res.device_workspace(0).num_elements();
  const size_t available = std::min(max_bytes, std::max(held, free_bytes));
  if (need > available) {
    return err_unsupported("three_center: (mn|P) over {} Cartesian orbital and {} Cartesian "
                           "fitting functions needs {} bytes of device scratch (the Cartesian "
                           "tensor alone {} bytes), and {} bytes are available",
                           n, n_aux, need, sizeof(double) * n_cart, available);
  }

  // mmd's records, every class's in turn, field-major over all of them, in
  // mmd's order: copied back as they are, then regrouped below.
  std::vector<double> mmd_fields(static_cast<size_t>(device::kThreeCenterBraFields) * n_bra_rec);
  std::vector<int32_t> mmd_off_a(n_bra_rec);
  std::vector<int32_t> mmd_off_b(n_bra_rec);
  size_t base = 0;
  for (const PrimitivePairClass &c : classes) {
    const PrimitivePairClassView v = c.view();
    const auto n_rec = static_cast<size_t>(v.n_rec);
    const std::array<const double *, device::kThreeCenterBraFields> fields{
        v.zeta, v.p_x, v.p_y, v.p_z, v.pa_x, v.pa_y, v.pa_z, v.pb_x, v.pb_y, v.pb_z, v.s};
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

  // Each class's records grouped by shell pair, (off_a, off_b) ascending, mmd's
  // order kept within a pair: a kernel walks a cell's pair as one run.
  std::vector<size_t> order(n_bra_rec);
  std::iota(order.begin(), order.end(), size_t{0});
  std::array<std::vector<ThreeCenterPair>, kThreeCenterMomenta * kThreeCenterMomenta> pairs;
  base = 0;
  for (const PrimitivePairClass &c : classes) {
    const auto first = order.begin() + static_cast<std::ptrdiff_t>(base);
    const auto last = first + c.n_rec;
    std::stable_sort(first, last, [&](const size_t x, const size_t y) {
      return std::pair{mmd_off_a[x], mmd_off_b[x]} < std::pair{mmd_off_a[y], mmd_off_b[y]};
    });
    const size_t width_a = function_count_cart(c.la);
    const size_t width_b = function_count_cart(c.lb);
    std::vector<ThreeCenterPair> &out =
        pairs[(static_cast<size_t>(c.la) * kThreeCenterMomenta) + static_cast<size_t>(c.lb)];
    for (auto run = first; run != last;) {
      const int32_t oa = mmd_off_a[*run];
      const int32_t ob = mmd_off_b[*run];
      const auto end = std::find_if(
          run, last, [&](const size_t r) { return mmd_off_a[r] != oa || mmd_off_b[r] != ob; });
      if (oa < 0 || ob < 0 || static_cast<size_t>(oa) + width_a > n ||
          static_cast<size_t>(ob) + width_b > n) {
        return err_config("three_center: a ({}, {}) record's AOs ({}, {}) are not the orbital "
                          "basis's {} Cartesian functions",
                          c.la, c.lb, oa, ob, n);
      }
      out.push_back({oa, ob, static_cast<int32_t>(run - order.begin()),
                     static_cast<int32_t>(end - order.begin())});
      run = end;
    }
    base += static_cast<size_t>(c.n_rec);
  }
  size_t found = 0;
  for (const std::vector<ThreeCenterPair> &p : pairs) {
    found += p.size();
  }
  if (found != n_pairs) {
    return err_config("three_center: the primitive-pair classes hold {} shell pairs, and the "
                      "orbital basis's {} shells have {}",
                      found, shells.size(), n_pairs);
  }

  // The staging: the bra records in that order, the aux records (one per
  // primitive of every aux shell, aux_rec_begin's ranges), then U and U_aux.
  std::vector<double> staging(n_staging);
  for (size_t f = 0; f < static_cast<size_t>(device::kThreeCenterBraFields); ++f) {
    for (size_t k = 0; k < n_bra_rec; ++k) {
      staging[(f * n_bra_rec) + k] = mmd_fields[(f * n_bra_rec) + order[k]];
    }
  }
  double *const aux_records =
      staging.data() + (static_cast<size_t>(device::kThreeCenterBraFields) * n_bra_rec);
  for (size_t i = 0, r = 0; i < aux_shells.size(); ++i) {
    const Vec<double, 3> &centre = aux_shells[i].center();
    for (const Primitive &prim : aux_shells[i].primitives()) {
      // The metric's record (its README): c carries every normalisation.
      const double weight = prim.coefficient * std::pow(std::numbers::pi / prim.exponent, 1.5);
      const std::array<double, device::kThreeCenterAuxFields> fields{centre.x, centre.y, centre.z,
                                                                     prim.exponent, weight};
      for (size_t f = 0; f < fields.size(); ++f) {
        aux_records[(f * n_aux_rec) + r] = fields[f];
      }
      ++r;
    }
  }
  double *const u = aux_records + (static_cast<size_t>(device::kThreeCenterAuxFields) * n_aux_rec);
  std::ranges::copy(spherical_transform_matrix(basis), u);
  std::ranges::copy(spherical_transform_matrix(aux), u + (n * n_sph));

  // The cells, one per (shell pair, aux shell), per class (la, lb, lP), the
  // pair outer and the aux shell inner, so consecutive threads write
  // neighbouring aux columns; field-major within each class's run.
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
          cells.data() + (static_cast<size_t>(device::kThreeCenterCellFields) * cell_begin[c]);
      size_t k = 0;
      for (const ThreeCenterPair &pair : pairs[ab]) {
        for (const size_t s : aux_of_l[lp]) {
          const std::array<int32_t, device::kThreeCenterCellFields> fields{
              pair.rec0,
              pair.rec1,
              aux_rec_begin[s],
              aux_rec_begin[s + 1],
              pair.off_a,
              pair.off_b,
              static_cast<int32_t>(aux.ao_offset_cart(static_cast<uint32_t>(s)))};
          for (size_t f = 0; f < fields.size(); ++f) {
            run[(f * n_cell[c]) + k] = fields[f];
          }
          ++k;
        }
      }
      cell_end += n_cell[c];
    }
  }

  const auto [d_staging, d_cells, cart, work] =
      res.device_regions<double, int32_t, double, double>(n_staging, n_cell_fields, n_cart, n_work);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_staging, staging.data(), sizeof(double) * n_staging,
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_cells, cells.data(), sizeof(int32_t) * n_cell_fields,
                                         wwr::wwrMemcpyHostToDevice, stream)));
  // The staging vectors die at return; a pageable source does not promise the
  // copy has finished reading it until the stream does.
  FBF_TRY(res.sync());

  const double *const d_aux =
      d_staging + (static_cast<size_t>(device::kThreeCenterBraFields) * n_bra_rec);
  const double *const d_u =
      d_aux + (static_cast<size_t>(device::kThreeCenterAuxFields) * n_aux_rec);
  const double *const d_u_aux = d_u + (n * n_sph);
  for (size_t la = 0; la < kThreeCenterMomenta; ++la) {
    for (size_t lb = la; lb < kThreeCenterMomenta; ++lb) {
      for (size_t lp = 0; lp < kThreeCenterAuxMomenta; ++lp) {
        const size_t c = (((la * kThreeCenterMomenta) + lb) * kThreeCenterAuxMomenta) + lp;
        if (n_cell[c] == 0) {
          continue;
        }
        FBF_TRY(gpu_status(device::three_center_class(
            stream, static_cast<int32_t>(la), static_cast<int32_t>(lb), static_cast<int32_t>(lp),
            static_cast<int32_t>(n_bra_rec), d_staging, static_cast<int32_t>(n_aux_rec), d_aux,
            static_cast<int32_t>(n_cell[c]),
            d_cells + (static_cast<size_t>(device::kThreeCenterCellFields) * cell_begin[c]),
            static_cast<int32_t>(n), static_cast<int32_t>(n_aux), cart)));
      }
    }
  }

  // The aux index: work = U_aux^T cart, n_aux_sph x n^2, from the
  // n_aux x n^2 view of the Cartesian tensor, P fastest.
  const auto n32 = static_cast<int32_t>(n);
  const auto n_sph32 = static_cast<int32_t>(n_sph);
  const auto n_aux_sph32 = static_cast<int32_t>(n_aux_sph);
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), n_aux_sph32, n32 * n32,
                                             static_cast<int32_t>(n_aux), &kOne<double>, d_u_aux,
                                             cart, &kZero<double>, work)));
  // The n index: for each Cartesian m, its n_aux_sph x n block of work times
  // U, into cart as n_aux_sph x n_sph blocks, U shared by the batch.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_aux_sph32, n_sph32, n32,
      &kOne<double>, work, n_aux_sph32, int64_t{n_aux_sph32} * n32, d_u, n32, 0, &kZero<double>,
      cart, n_aux_sph32, int64_t{n_aux_sph32} * n_sph32, n32)));
  // The m index: the (n_aux_sph n_sph) x n view of cart times U, into work.
  FBF_TRY(gpu_status(gemm(res.blas(), n_aux_sph32 * n_sph32, n_sph32, n32, &kOne<double>, cart, d_u,
                          &kZero<double>, work)));
  // The two gemms round (m, n) and (n, m) in different orders; half of each
  // plus half of its mirror is exactly symmetric, written to b.
  return gpu_status(device::three_center_symmetrize(stream, n_sph32, n_aux_sph32, work, b));
}

} // namespace free_breakfast
