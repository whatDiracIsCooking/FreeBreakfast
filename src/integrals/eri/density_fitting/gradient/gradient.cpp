/// @file
/// @brief The DF gradients: the spherical matrices made Cartesian, the
/// energy builds' staging with each pair's or cell's atoms (and each bra
/// record's exponents), one launch per class into a zeroed G, and G
/// downloaded. The derivative integrals over the same staging, made
/// spherical; and over those, the DF J and K differentiated per coordinate.

module;

// The kernel launchers, defined in gradient_launch.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "gradient_bridge.h"

module free_breakfast.integrals.eri.density_fitting.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kTwo, kNegOne, kNegTwo
import free_breakfast.core.primitives;
import free_breakfast.core.error_handling;
// gemmStridedBatched, gemm, gemv, trsv, trsm, geam, axpy, wwrMemGetInfo,
// wwrMemsetAsync, wwrMemcpy2DAsync, gpu_status
import free_breakfast.core.wwr;
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
constexpr size_t kMetricClasses = kMetricMomenta * kMetricMomenta;
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

// The AOs atom @p atom's shells own in a placed basis, [first, last), in the
// spherical layout or the Cartesian one: an atom's shells are contiguous.
std::pair<size_t, size_t> atom_aos(const BasisSet &basis, const uint32_t atom,
                                   const bool spherical) {
  const uint32_t count = basis.atom_shell_count(atom);
  if (count == 0) {
    return {0, 0};
  }
  const uint32_t first = basis.atom_shell_offset(atom);
  const uint32_t last = first + count - 1;
  const Shell &shell = basis.shells()[last];
  if (spherical) {
    return {basis.ao_offset_sphe(first), basis.ao_offset_sphe(last) + shell.function_count_sphe()};
  }
  return {basis.ao_offset_cart(first), basis.ao_offset_cart(last) + shell.function_count_cart()};
}

// Refuses an aux basis the metric kernels cannot take, each message as @p who.
Status check_metric_basis(const BasisSet &aux, const std::string_view who) {
  if (aux.natoms() == 0) {
    return err_config("{}: the aux basis has no atoms; make it with place()", who);
  }
  for (const Shell &shell : aux.shells()) {
    if (shell.angular_momentum() > device::kMetricMaxMomentum) {
      return err_config("{}: no kernel for an aux shell of momentum {}; the kernels cover l <= {}",
                        who, shell.angular_momentum(), device::kMetricMaxMomentum);
    }
  }
  return {};
}

// The metric's aux shell pairs, P <= Q by shell index with the lower-momentum
// shell first, as metric.cpp stages them, each with its two atoms in the
// staged order: one run per class c = lP kMetricMomenta + lQ, count[c] pairs
// from pair begin[c], kMetricGradientPairFields segments of count[c] each.
struct MetricPairs {
  std::vector<int32_t> fields;
  std::array<size_t, kMetricClasses> count{};
  std::array<size_t, kMetricClasses> begin{};
};

MetricPairs stage_metric_pairs(const BasisSet &aux, const std::vector<int32_t> &rec_begin) {
  constexpr auto kFields = static_cast<size_t>(device::kMetricGradientPairFields);
  const std::vector<Shell> &shells = aux.shells();
  const std::vector<int32_t> atom_of = shell_atoms(aux);
  const auto class_of = [&](const size_t i, const size_t j) {
    const auto li = static_cast<size_t>(shells[i].angular_momentum());
    const auto lj = static_cast<size_t>(shells[j].angular_momentum());
    return (std::min(li, lj) * kMetricMomenta) + std::max(li, lj);
  };
  MetricPairs out;
  for (size_t i = 0; i < shells.size(); ++i) {
    for (size_t j = i; j < shells.size(); ++j) {
      ++out.count[class_of(i, j)];
    }
  }
  size_t n_pairs = 0;
  for (size_t c = 0; c < kMetricClasses; ++c) {
    out.begin[c] = n_pairs;
    n_pairs += out.count[c];
  }
  out.fields.resize(kFields * n_pairs);
  std::array<size_t, kMetricClasses> filled{};
  for (size_t i = 0; i < shells.size(); ++i) {
    for (size_t j = i; j < shells.size(); ++j) {
      const bool swapped = shells[i].angular_momentum() > shells[j].angular_momentum();
      const size_t p = swapped ? j : i;
      const size_t q = swapped ? i : j;
      const size_t c = class_of(i, j);
      int32_t *const run = out.fields.data() + (kFields * out.begin[c]);
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
        run[(f * out.count[c]) + filled[c]] = fields[f];
      }
      ++filled[c];
    }
  }
  return out;
}

Result<std::vector<double>> metric_gradient(DeviceResources &res, const BasisSet &aux,
                                            const double *w) {
  FBF_TRY(check_metric_basis(aux, "metric_gradient"));
  const size_t n = aux.nao_cart();
  const size_t n_sph = aux.nao_sphe();
  // The kernels index W as a 32-bit a * n_aux + b.
  if (n * n > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("metric_gradient: {} Cartesian fitting functions is too many for the "
                      "kernels' 32-bit index",
                      n);
  }
  const size_t n_g = 3 * static_cast<size_t>(aux.natoms());
  if (aux.shells().empty()) {
    return std::vector<double>(n_g, 0.0);
  }

  const std::vector<int32_t> rec_begin = record_begins(aux);
  const auto n_rec = static_cast<size_t>(rec_begin.back());
  const MetricPairs pairs = stage_metric_pairs(aux, rec_begin);

  // The staging: the aux records, then U_aux.
  const size_t n_records = static_cast<size_t>(device::kMetricRecordFields) * n_rec;
  std::vector<double> staging(n_records + (n * n_sph));
  stage_aux_records(aux, rec_begin, staging.data());
  std::ranges::copy(spherical_transform_matrix(aux), staging.data() + n_records);

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // t = U_aux W, then the Cartesian W = t U_aux^T.
  const auto [d_staging, d_pairs, t, w_cart, d_g] =
      res.device_regions<double, int32_t, double, double, double>(
          staging.size(), pairs.fields.size(), n * n_sph, n * n, n_g);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_staging, staging.data(), sizeof(double) * staging.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_pairs, pairs.fields.data(),
                                         sizeof(int32_t) * pairs.fields.size(),
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
      if (pairs.count[c] == 0) {
        continue;
      }
      FBF_TRY(gpu_status(device::metric_gradient_class(
          stream, static_cast<int32_t>(lp), static_cast<int32_t>(lq), static_cast<int32_t>(n_rec),
          d_staging, static_cast<int32_t>(pairs.count[c]),
          d_pairs + (static_cast<size_t>(device::kMetricGradientPairFields) * pairs.begin[c]), n32,
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

// The metric derivative over staged records and pairs: one ipmetric launch per
// class into the zeroed Cartesian m1c (3 n x n, row-major (k, P, Q)), then
// each slab made spherical, m1[k] = U_aux^T m1c[k] U_aux through t (n x
// n_sph). Read column-major, slab k of either is the matrix whose (Q, P)
// element is M1[k][P][Q], and the two-sided transform keeps that.
Status metric_derivative_device(DeviceResources &res, const size_t n, const size_t n_sph,
                                const MetricPairs &pairs, const int32_t n_rec,
                                const double *d_records, const int32_t *d_pairs,
                                const double *d_u_aux, double *m1c, double *t, double *m1) {
  const wwr::wwrStream_t stream = res.stream();
  const auto n32 = static_cast<int32_t>(n);
  const auto n_sph32 = static_cast<int32_t>(n_sph);
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(m1c, 0, sizeof(double) * 3 * n * n, stream)));
  for (size_t lp = 0; lp < kMetricMomenta; ++lp) {
    for (size_t lq = lp; lq < kMetricMomenta; ++lq) {
      const size_t c = (lp * kMetricMomenta) + lq;
      if (pairs.count[c] == 0) {
        continue;
      }
      FBF_TRY(gpu_status(device::metric_derivative_class(
          stream, static_cast<int32_t>(lp), static_cast<int32_t>(lq), n_rec, d_records,
          static_cast<int32_t>(pairs.count[c]),
          d_pairs + (static_cast<size_t>(device::kMetricGradientPairFields) * pairs.begin[c]), n32,
          m1c)));
    }
  }
  for (size_t k = 0; k < 3; ++k) {
    FBF_TRY(gpu_status(gemm(res.blas(), n32, n_sph32, n32, &kOne<double>, m1c + (k * n * n),
                            d_u_aux, &kZero<double>, t)));
    FBF_TRY(
        gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), n_sph32, n_sph32, n32, &kOne<double>,
                                           d_u_aux, t, &kZero<double>, m1 + (k * n_sph * n_sph))));
  }
  return {};
}

Status metric_derivative(DeviceResources &res, const BasisSet &aux, double *m1) {
  FBF_TRY(check_metric_basis(aux, "metric_derivative"));
  const size_t n = aux.nao_cart();
  const size_t n_sph = aux.nao_sphe();
  // The kernel indexes M1 as a 32-bit (k n_aux + P) n_aux + Q.
  if (3 * n * n > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_unsupported("metric_derivative: {} Cartesian fitting functions is too many for "
                           "the kernels' 32-bit index",
                           n);
  }
  if (aux.shells().empty()) {
    return {};
  }
  // The counts first, so the refusal below comes before anything is staged.
  const std::vector<int32_t> rec_begin = record_begins(aux);
  const auto n_rec = static_cast<size_t>(rec_begin.back());
  const size_t n_records = static_cast<size_t>(device::kMetricRecordFields) * n_rec;
  const size_t n_staging = n_records + (n * n_sph);
  const size_t n_pair_fields = static_cast<size_t>(device::kMetricGradientPairFields) *
                               aux.nshells() * (aux.nshells() + size_t{1}) / 2;
  const auto region = [](const size_t bytes) {
    return math::align_up(bytes, DeviceResources::kWorkspaceAlignment);
  };
  const size_t need = region(sizeof(double) * n_staging) + region(sizeof(int32_t) * n_pair_fields) +
                      region(sizeof(double) * 3 * n * n) + region(sizeof(double) * n * n_sph);
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (const size_t available = std::max(res.device_workspace(0).num_elements(), free_bytes);
      need > available) {
    return err_unsupported("metric_derivative: over {} Cartesian fitting functions it needs {} "
                           "bytes of device scratch, and {} bytes are available",
                           n, need, available);
  }

  const MetricPairs pairs = stage_metric_pairs(aux, rec_begin);
  std::vector<double> staging(n_staging);
  stage_aux_records(aux, rec_begin, staging.data());
  std::ranges::copy(spherical_transform_matrix(aux), staging.data() + n_records);
  const auto [d_staging, d_pairs, m1c, t] = res.device_regions<double, int32_t, double, double>(
      n_staging, pairs.fields.size(), 3 * n * n, n * n_sph);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_staging, staging.data(), sizeof(double) * staging.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_pairs, pairs.fields.data(),
                                         sizeof(int32_t) * pairs.fields.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(metric_derivative_device(res, n, n_sph, pairs, static_cast<int32_t>(n_rec), d_staging,
                                   d_pairs, d_staging + n_records, m1c, t, m1));
  // Also what lets the host staging go (metric_gradient's comment).
  return res.sync();
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

// mmd's bra records regrouped by shell pair: n_rec records,
// kThreeCenterGradientBraFields segments of n_rec in fields, and each pair
// class's (la, lb) shell pairs at la kThreeCenterMomenta + lb of pairs.
struct ThreeCenterBra {
  size_t n_rec = 0;
  std::vector<double> fields;
  std::array<std::vector<ThreeCenterGradientPair>, kThreeCenterMomenta * kThreeCenterMomenta> pairs;
};

// Refuses bases the three-center kernels cannot take, each message as @p who.
Status check_three_center_bases(const BasisSet &basis, const BasisSet &aux,
                                const std::string_view who) {
  if (basis.natoms() == 0 || aux.natoms() != basis.natoms()) {
    return err_config("{}: the bases have {} and {} atoms; make both with place() over one "
                      "molecule",
                      who, basis.natoms(), aux.natoms());
  }
  const std::vector<Shell> &shells = basis.shells();
  const std::vector<Shell> &aux_shells = aux.shells();
  for (size_t i = 0; i < shells.size(); ++i) {
    if (const int32_t l = shells[i].angular_momentum(); l > device::kThreeCenterMaxMomentum) {
      return err_unsupported("{}: orbital shell {} has momentum {}; the kernels cover orbital "
                             "shells through l = {}",
                             who, i, l, device::kThreeCenterMaxMomentum);
    }
  }
  for (size_t i = 0; i < aux_shells.size(); ++i) {
    if (const int32_t l = aux_shells[i].angular_momentum();
        l > device::kThreeCenterAuxMaxMomentum) {
      return err_unsupported("{}: aux shell {} has momentum {}; the kernels cover aux shells "
                             "through l = {}",
                             who, i, l, device::kThreeCenterAuxMaxMomentum);
    }
  }
  return {};
}

// The bra records @p classes hold, from their sizes alone (so a refusal on
// memory can come before anything is downloaded), refusing a class that is
// not one of an orbital basis's, as @p who.
Result<size_t> bra_record_count(const std::span<const PrimitivePairClass> classes,
                                const std::string_view who) {
  size_t n_bra_rec = 0;
  for (const PrimitivePairClass &c : classes) {
    if (c.la < 0 || c.lb < c.la || c.lb > device::kThreeCenterMaxMomentum) {
      return err_config("{}: a primitive-pair class ({}, {}) is not one of the orbital basis's",
                        who, c.la, c.lb);
    }
    n_bra_rec += static_cast<size_t>(c.n_rec);
  }
  return n_bra_rec;
}

// mmd's records, every class's in turn, copied back as they are (one
// synchronization), then grouped by shell pair, (off_a, off_b) ascending,
// mmd's order kept within a pair, as three_center.cpp groups them, each pair
// with its two atoms; as @p who.
Result<ThreeCenterBra> stage_three_center_bra(DeviceResources &res, const BasisSet &basis,
                                              const std::span<const PrimitivePairClass> classes,
                                              const size_t n_bra_rec, const std::string_view who) {
  const wwr::wwrStream_t stream = res.stream();
  const size_t n = basis.nao_cart();
  const size_t n_bra = static_cast<size_t>(device::kThreeCenterGradientBraFields) * n_bra_rec;
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
  const std::vector<Shell> &shells = basis.shells();
  const std::vector<int32_t> shell_atom = shell_atoms(basis);
  std::vector<int32_t> ao_atom(n);
  for (uint32_t s = 0; s < basis.nshells(); ++s) {
    std::fill_n(ao_atom.begin() + basis.ao_offset_cart(s), shells[s].function_count_cart(),
                shell_atom[s]);
  }

  std::vector<size_t> order(n_bra_rec);
  std::iota(order.begin(), order.end(), size_t{0});
  ThreeCenterBra out;
  base = 0;
  for (const PrimitivePairClass &c : classes) {
    const auto first = order.begin() + static_cast<std::ptrdiff_t>(base);
    const auto last = first + c.n_rec;
    std::stable_sort(first, last, [&](const size_t x, const size_t y) {
      return std::pair{mmd_off_a[x], mmd_off_b[x]} < std::pair{mmd_off_a[y], mmd_off_b[y]};
    });
    const size_t width_a = function_count_cart(c.la);
    const size_t width_b = function_count_cart(c.lb);
    std::vector<ThreeCenterGradientPair> &pairs =
        out.pairs[(static_cast<size_t>(c.la) * kThreeCenterMomenta) + static_cast<size_t>(c.lb)];
    for (auto run = first; run != last;) {
      const int32_t oa = mmd_off_a[*run];
      const int32_t ob = mmd_off_b[*run];
      const auto end = std::find_if(
          run, last, [&](const size_t r) { return mmd_off_a[r] != oa || mmd_off_b[r] != ob; });
      if (oa < 0 || ob < 0 || static_cast<size_t>(oa) + width_a > n ||
          static_cast<size_t>(ob) + width_b > n) {
        return err_config("{}: a ({}, {}) record's AOs ({}, {}) are not the orbital basis's {} "
                          "Cartesian functions",
                          who, c.la, c.lb, oa, ob, n);
      }
      pairs.push_back({oa, ob, ao_atom[static_cast<size_t>(oa)], ao_atom[static_cast<size_t>(ob)],
                       static_cast<int32_t>(run - order.begin()),
                       static_cast<int32_t>(end - order.begin())});
      run = end;
    }
    base += static_cast<size_t>(c.n_rec);
  }
  size_t found = 0;
  for (const std::vector<ThreeCenterGradientPair> &p : out.pairs) {
    found += p.size();
  }
  const size_t n_pairs = shells.size() * (shells.size() + 1) / 2;
  if (found != n_pairs) {
    return err_config("{}: the primitive-pair classes hold {} shell pairs, and the orbital "
                      "basis's {} shells have {}",
                      who, found, shells.size(), n_pairs);
  }

  out.n_rec = n_bra_rec;
  out.fields.resize(n_bra);
  for (size_t f = 0; f < static_cast<size_t>(device::kThreeCenterGradientBraFields); ++f) {
    for (size_t k = 0; k < n_bra_rec; ++k) {
      out.fields[(f * n_bra_rec) + k] = mmd_fields[(f * n_bra_rec) + order[k]];
    }
  }
  return out;
}

// The cells of every shell pair of @p bra crossed with the aux shells
// [first, first + count), one per (pair, aux shell), appended to @p cells
// one run per class (la, lb, lP), the pair outer and the aux shell inner,
// kThreeCenterGradientCellFields segments of each run's count; class c's run
// holds n_cell[c] cells from cell cell_begin[c] of the whole of @p cells.
void stage_three_center_cells(const ThreeCenterBra &bra, const BasisSet &aux,
                              const std::vector<int32_t> &aux_rec_begin, const uint32_t first,
                              const uint32_t count, std::vector<int32_t> &cells,
                              std::array<size_t, kThreeCenterClasses> &n_cell,
                              std::array<size_t, kThreeCenterClasses> &cell_begin) {
  constexpr auto kFields = static_cast<size_t>(device::kThreeCenterGradientCellFields);
  const std::vector<int32_t> aux_atom = shell_atoms(aux);
  std::array<std::vector<size_t>, kThreeCenterAuxMomenta> aux_of_l;
  for (size_t i = first; i < static_cast<size_t>(first) + count; ++i) {
    aux_of_l[static_cast<size_t>(aux.shells()[i].angular_momentum())].push_back(i);
  }
  size_t cell_end = cells.size() / kFields;
  for (size_t ab = 0; ab < bra.pairs.size(); ++ab) {
    for (size_t lp = 0; lp < kThreeCenterAuxMomenta; ++lp) {
      const size_t c = (ab * kThreeCenterAuxMomenta) + lp;
      n_cell[c] = bra.pairs[ab].size() * aux_of_l[lp].size();
      cell_begin[c] = cell_end;
      cells.resize(kFields * (cell_end + n_cell[c]));
      int32_t *const run = cells.data() + (kFields * cell_begin[c]);
      size_t k = 0;
      for (const ThreeCenterGradientPair &pair : bra.pairs[ab]) {
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
}

Result<std::vector<double>> three_center_gradient(DeviceResources &res, const BasisSet &basis,
                                                  const std::span<const PrimitivePairClass> classes,
                                                  const BasisSet &aux, const double *gamma) {
  constexpr std::string_view kWho = "three_center_gradient";
  FBF_TRY(check_three_center_bases(basis, aux, kWho));
  const std::vector<Shell> &shells = basis.shells();
  const std::vector<Shell> &aux_shells = aux.shells();
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
  const size_t n_bra_rec = FBF_TRY(bra_record_count(classes, kWho));
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

  ThreeCenterBra bra = FBF_TRY(stage_three_center_bra(res, basis, classes, n_bra_rec, kWho));

  // The staging: the bra records in that order, the aux records, then U and
  // U_aux.
  std::vector<double> staging = std::move(bra.fields);
  staging.resize(n_staging);
  stage_aux_records(aux, aux_rec_begin, staging.data() + n_bra);
  double *const u = staging.data() + n_bra + n_aux_fields;
  std::ranges::copy(spherical_transform_matrix(basis), u);
  std::ranges::copy(spherical_transform_matrix(aux), u + (n * n_sph));

  std::array<size_t, kThreeCenterClasses> n_cell{};
  std::array<size_t, kThreeCenterClasses> cell_begin{};
  std::vector<int32_t> cells;
  cells.reserve(n_cell_fields);
  stage_three_center_cells(bra, aux, aux_rec_begin, 0, aux.nshells(), cells, n_cell, cell_begin);

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

// One aux atom's block of the three-center derivative, its aux functions
// [cart0, cart0 + nblk_cart) in the Cartesian layout and [sph0, sph0 + nblk)
// in the spherical one, over staged records and cells (its own runs,
// n_cell[c] from cell_begin[c]): the xeri3 launches into the Cartesian xc,
// element (k, m, n, P) at P + nblk_cart (n + n_cart (m + n_cart k)), then
// each k made spherical on its three indices (t1 = U_aux's block^T xc_k,
// then U on n per Cartesian m into t2, then U on m) into x, at
// P + nblk (n + nao (m + nao k)), and, when xt is not null, the same last
// product with m and n's places swapped into xt, at P + nblk (m + nao (n +
// nao k)). t1 (nblk n_cart^2) and t2 (nblk nao n_cart) are scratch.
Status three_center_derivative_block(
    DeviceResources &res, const size_t n_cart, const size_t nao, const size_t naux_cart,
    const size_t cart0, const size_t nblk_cart, const size_t sph0, const size_t nblk,
    const size_t n_bra_rec, const double *d_bra, const size_t n_aux_rec, const double *d_aux,
    const int32_t *d_cells, const std::array<size_t, kThreeCenterClasses> &n_cell,
    const std::array<size_t, kThreeCenterClasses> &cell_begin, const double *d_u,
    const double *d_u_aux, double *xc, double *t1, double *t2, double *x, double *xt) {
  const wwr::wwrStream_t stream = res.stream();
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto nc = static_cast<int32_t>(n_cart);
  const auto n = static_cast<int32_t>(nao);
  const auto qc = static_cast<int32_t>(nblk_cart);
  const auto q = static_cast<int32_t>(nblk);
  for (size_t la = 0; la < kThreeCenterMomenta; ++la) {
    for (size_t lb = la; lb < kThreeCenterMomenta; ++lb) {
      for (size_t lp = 0; lp < kThreeCenterAuxMomenta; ++lp) {
        const size_t c = (((la * kThreeCenterMomenta) + lb) * kThreeCenterAuxMomenta) + lp;
        if (n_cell[c] == 0) {
          continue;
        }
        FBF_TRY(gpu_status(device::three_center_derivative_class(
            stream, static_cast<int32_t>(la), static_cast<int32_t>(lb), static_cast<int32_t>(lp),
            static_cast<int32_t>(n_bra_rec), d_bra, static_cast<int32_t>(n_aux_rec), d_aux,
            static_cast<int32_t>(n_cell[c]),
            d_cells + (static_cast<size_t>(device::kThreeCenterGradientCellFields) * cell_begin[c]),
            nc, qc, static_cast<int32_t>(cart0), xc)));
      }
    }
  }
  // U_aux is block-diagonal by shell, so the atom's aux functions are its
  // block, [cart0, cart0 + nblk_cart) x [sph0, sph0 + nblk).
  const double *const u_block = d_u_aux + cart0 + (naux_cart * sph0);
  for (size_t k = 0; k < 3; ++k) {
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, q, nc * nc, qc, &kOne<double>, u_block,
        static_cast<int32_t>(naux_cart), xc + (k * n_cart * n_cart * nblk_cart), qc, &kZero<double>,
        t1, q)));
    // For each Cartesian m, its nblk x n_cart block of t1 times U, into t2 as
    // nblk x nao blocks, (P, n, m_cart) at P + nblk (n + nao m_cart).
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, n, nc, &kOne<double>, t1, q, int64_t{q} * nc,
        d_u, nc, 0, &kZero<double>, t2, q, int64_t{q} * n, nc)));
    // m: the (nblk nao) x n_cart view of t2 times U.
    FBF_TRY(gpu_status(gemm(blas, q * n, n, nc, &kOne<double>, t2, d_u, &kZero<double>,
                            x + (k * nao * nao * nblk))));
    if (xt != nullptr) {
      // m again, one n at a time: t2's nblk x n_cart block at that n (leading
      // dimension nblk nao) times U, landing with m second.
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, n, nc, &kOne<double>, t2, q * n,
          int64_t{q}, d_u, nc, 0, &kZero<double>, xt + (k * nao * nao * nblk), q, int64_t{q} * n,
          n)));
    }
  }
  return {};
}

Status three_center_derivative(DeviceResources &res, const BasisSet &basis,
                               const std::span<const PrimitivePairClass> classes,
                               const BasisSet &aux, const uint32_t atom, double *x) {
  constexpr std::string_view kWho = "three_center_derivative";
  FBF_TRY(check_three_center_bases(basis, aux, kWho));
  if (atom >= aux.natoms()) {
    return err_config("three_center_derivative: atom {} is not one of the bases' {}", atom,
                      aux.natoms());
  }
  const auto [cart0, cart1] = atom_aos(aux, atom, false);
  const auto [sph0, sph1] = atom_aos(aux, atom, true);
  if (basis.shells().empty() || cart1 == cart0) {
    return {};
  }
  const size_t n_cart = basis.nao_cart();
  const size_t nao = basis.nao_sphe();
  const size_t naux_cart = aux.nao_cart();
  const size_t nblk_cart = cart1 - cart0;
  const size_t nblk = sph1 - sph0;
  const size_t n_xc = 3 * n_cart * n_cart * nblk_cart;
  // The kernels index the block as a 32-bit ((k n_ao + m) n_ao + n) n_blk + P.
  if (n_xc > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_unsupported("three_center_derivative: the Cartesian block, 3 x {} x {} x {} = {} "
                           "elements, is past the kernels' 32-bit index",
                           n_cart, n_cart, nblk_cart, n_xc);
  }

  const size_t n_bra_rec = FBF_TRY(bra_record_count(classes, kWho));
  const std::vector<int32_t> aux_rec_begin = record_begins(aux);
  const auto n_aux_rec = static_cast<size_t>(aux_rec_begin.back());
  const size_t n_pairs = basis.nshells() * (basis.nshells() + size_t{1}) / 2;
  const size_t n_cell_fields = static_cast<size_t>(device::kThreeCenterGradientCellFields) *
                               n_pairs * aux.atom_shell_count(atom);
  const size_t n_bra = static_cast<size_t>(device::kThreeCenterGradientBraFields) * n_bra_rec;
  const size_t n_aux_fields = static_cast<size_t>(device::kMetricRecordFields) * n_aux_rec;
  const size_t n_staging = n_bra + n_aux_fields + (n_cart * nao) + (naux_cart * aux.nao_sphe());
  const size_t n_t1 = nblk * n_cart * n_cart;
  const size_t n_t2 = nblk * nao * n_cart;
  const auto region = [](const size_t bytes) {
    return math::align_up(bytes, DeviceResources::kWorkspaceAlignment);
  };
  const size_t need = region(sizeof(double) * n_staging) + region(sizeof(int32_t) * n_cell_fields) +
                      region(sizeof(double) * n_xc) + region(sizeof(double) * n_t1) +
                      region(sizeof(double) * n_t2);

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (const size_t available = std::max(res.device_workspace(0).num_elements(), free_bytes);
      need > available) {
    return err_unsupported("three_center_derivative: over {} Cartesian orbital functions and "
                           "atom {}'s {} Cartesian fitting functions it needs {} bytes of device "
                           "scratch, and {} bytes are available",
                           n_cart, atom, nblk_cart, need, available);
  }

  ThreeCenterBra bra = FBF_TRY(stage_three_center_bra(res, basis, classes, n_bra_rec, kWho));
  std::vector<double> staging = std::move(bra.fields);
  staging.resize(n_staging);
  stage_aux_records(aux, aux_rec_begin, staging.data() + n_bra);
  double *const u = staging.data() + n_bra + n_aux_fields;
  std::ranges::copy(spherical_transform_matrix(basis), u);
  std::ranges::copy(spherical_transform_matrix(aux), u + (n_cart * nao));
  std::array<size_t, kThreeCenterClasses> n_cell{};
  std::array<size_t, kThreeCenterClasses> cell_begin{};
  std::vector<int32_t> cells;
  cells.reserve(n_cell_fields);
  stage_three_center_cells(bra, aux, aux_rec_begin, aux.atom_shell_offset(atom),
                           aux.atom_shell_count(atom), cells, n_cell, cell_begin);

  const auto [d_staging, d_cells, xc, t1, t2] =
      res.device_regions<double, int32_t, double, double, double>(n_staging, cells.size(), n_xc,
                                                                  n_t1, n_t2);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_staging, staging.data(), sizeof(double) * n_staging,
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_cells, cells.data(), sizeof(int32_t) * cells.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  const double *const d_aux = d_staging + n_bra;
  const double *const d_u = d_aux + n_aux_fields;
  FBF_TRY(three_center_derivative_block(
      res, n_cart, nao, naux_cart, cart0, nblk_cart, sph0, nblk, n_bra_rec, d_staging, n_aux_rec,
      d_aux, d_cells, n_cell, cell_begin, d_u, d_u + (n_cart * nao), xc, t1, t2, x, nullptr));
  // Also what lets the host staging go (the metric's comment).
  return res.sync();
}

Status df_jk_derivative(DeviceResources &res, const BasisSet &basis,
                        const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                        const double *b, const double *l, const double *c, const int32_t nocc,
                        double *jx, double *kx, double *g) {
  constexpr std::string_view kWho = "df_jk_derivative";
  FBF_TRY(check_three_center_bases(basis, aux, kWho));
  FBF_TRY(check_metric_basis(aux, kWho));
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const size_t n_cart = basis.nao_cart();
  const size_t naux_cart = aux.nao_cart();
  const size_t n_atom = basis.natoms();
  const size_t n_x = 3 * n_atom;
  if (nocc < 0 || static_cast<size_t>(nocc) > nao) {
    return err_config("df_jk_derivative: nocc {} is outside [0, {}]", nocc, nao);
  }
  const auto occ = static_cast<size_t>(nocc);

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const size_t slabs = n_x * nao * nao;
  if (nocc == 0 || basis.shells().empty() || aux.shells().empty()) {
    // No density, or nothing to fit it with: no two-electron Fock matrix.
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(jx, 0, sizeof(double) * slabs, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(kx, 0, sizeof(double) * slabs, stream)));
    if (g != nullptr) {
      FBF_TRY(gpu_status(wwr::wwrMemsetAsync(g, 0, sizeof(double) * naux * n_x, stream)));
    }
    return res.sync();
  }

  // Each aux atom's block, and the largest, which sizes the per-block scratch.
  size_t max_blk = 0;
  size_t max_blk_cart = 0;
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [cart0, cart1] = atom_aos(aux, a, false);
    const auto [sph0, sph1] = atom_aos(aux, a, true);
    max_blk = std::max(max_blk, sph1 - sph0);
    max_blk_cart = std::max(max_blk_cart, cart1 - cart0);
  }
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  // The kernels' 32-bit indices (ipmetric's (k n_aux + P) n_aux + Q, xeri3's
  // ((k n_ao + m) n_ao + n) n_blk + P), and B and the slabs as the BLAS calls'
  // int32_t dimensions see them.
  if (3 * naux_cart * naux_cart > kInt32Max || 3 * n_cart * n_cart * max_blk_cart > kInt32Max ||
      naux * nao * nao > kInt32Max || slabs > kInt32Max) {
    return err_unsupported("df_jk_derivative: {} orbital and {} fitting functions ({} and {} "
                           "Cartesian, up to {} on one atom) pass the kernels' or the BLAS calls' "
                           "32-bit indices",
                           nao, naux, n_cart, naux_cart, max_blk_cart);
  }

  // The counts, from the classes' sizes and the bases alone, so the refusal
  // below comes before anything is downloaded or allocated.
  const size_t n_bra_rec = FBF_TRY(bra_record_count(classes, kWho));
  const std::vector<int32_t> aux_rec_begin = record_begins(aux);
  const auto n_aux_rec = static_cast<size_t>(aux_rec_begin.back());
  const size_t n_pairs = basis.nshells() * (basis.nshells() + size_t{1}) / 2;
  const size_t n_cell_fields =
      static_cast<size_t>(device::kThreeCenterGradientCellFields) * n_pairs * aux.nshells();
  const size_t n_metric_pair_fields = static_cast<size_t>(device::kMetricGradientPairFields) *
                                      aux.nshells() * (aux.nshells() + size_t{1}) / 2;
  const size_t n_bra = static_cast<size_t>(device::kThreeCenterGradientBraFields) * n_bra_rec;
  const size_t n_aux_fields = static_cast<size_t>(device::kMetricRecordFields) * n_aux_rec;
  const size_t n_staging = n_bra + n_aux_fields + (n_cart * nao) + (naux_cart * naux);
  const size_t nn = nao * nao;
  // The regions, in device_regions' order below: the staging, the metric's
  // pairs, the cells; D, d, the fitted half-transformed pairs c; the Cartesian
  // M1, its transform's intermediate, M1; u^x - M^x d; the accumulated Y and
  // V1, one symmetrization's slab; then one aux atom's blocks: the Cartesian
  // derivative, t1, t2, X and its swap, Z, c's rows on the atom and F.
  const std::array<std::pair<size_t, size_t>, 21> regions{{
      {n_staging, sizeof(double)},
      {n_metric_pair_fields, sizeof(int32_t)},
      {n_cell_fields, sizeof(int32_t)},
      {nn, sizeof(double)},
      {naux, sizeof(double)},
      {naux * occ * nao, sizeof(double)},
      {3 * naux_cart * naux_cart, sizeof(double)},
      {naux_cart * naux, sizeof(double)},
      {3 * naux * naux, sizeof(double)},
      {naux * n_x, sizeof(double)},
      {3 * nn, sizeof(double)},
      {3 * nn, sizeof(double)},
      {nn, sizeof(double)},
      {3 * n_cart * n_cart * max_blk_cart, sizeof(double)},
      {max_blk * n_cart * n_cart, sizeof(double)},
      {max_blk * nao * n_cart, sizeof(double)},
      {3 * max_blk * nn, sizeof(double)},
      {3 * max_blk * nn, sizeof(double)},
      {max_blk * nn, sizeof(double)},
      {max_blk * occ * nao, sizeof(double)},
      {max_blk * occ * nao, sizeof(double)},
  }};
  size_t need = 0;
  for (const auto &[count, size] : regions) {
    need += math::align_up(count * size, DeviceResources::kWorkspaceAlignment);
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (const size_t available = std::max(res.device_workspace(0).num_elements(), free_bytes);
      need > available) {
    return err_unsupported("df_jk_derivative: over {} orbital and {} fitting functions ({} on "
                           "the largest aux atom) with {} occupied orbitals it needs {} bytes of "
                           "device scratch, and {} bytes are available",
                           nao, naux, max_blk, occ, need, available);
  }

  ThreeCenterBra bra = FBF_TRY(stage_three_center_bra(res, basis, classes, n_bra_rec, kWho));
  std::vector<double> staging = std::move(bra.fields);
  staging.resize(n_staging);
  stage_aux_records(aux, aux_rec_begin, staging.data() + n_bra);
  double *const u = staging.data() + n_bra + n_aux_fields;
  std::ranges::copy(spherical_transform_matrix(basis), u);
  std::ranges::copy(spherical_transform_matrix(aux), u + (n_cart * nao));
  const MetricPairs metric_pairs = stage_metric_pairs(aux, aux_rec_begin);
  // Every aux atom's cells, one atom after another in one array.
  std::vector<std::array<size_t, kThreeCenterClasses>> n_cell(n_atom);
  std::vector<std::array<size_t, kThreeCenterClasses>> cell_begin(n_atom);
  std::vector<int32_t> cells;
  cells.reserve(n_cell_fields);
  for (uint32_t a = 0; a < n_atom; ++a) {
    stage_three_center_cells(bra, aux, aux_rec_begin, aux.atom_shell_offset(a),
                             aux.atom_shell_count(a), cells, n_cell[a], cell_begin[a]);
  }

  const auto [d_staging, d_pairs, d_cells, dd, fit, cf, m1c, mt, m1, gx, y_acc, v1_acc, sym, xc, t1,
              t2, xs, xts, zb, cfb, fb] =
      res.device_regions<double, int32_t, int32_t, double, double, double, double, double, double,
                         double, double, double, double, double, double, double, double, double,
                         double, double, double>(
          regions[0].first, regions[1].first, regions[2].first, regions[3].first, regions[4].first,
          regions[5].first, regions[6].first, regions[7].first, regions[8].first, regions[9].first,
          regions[10].first, regions[11].first, regions[12].first, regions[13].first,
          regions[14].first, regions[15].first, regions[16].first, regions[17].first,
          regions[18].first, regions[19].first, regions[20].first);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_staging, staging.data(), sizeof(double) * n_staging,
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_pairs, metric_pairs.fields.data(),
                                         sizeof(int32_t) * metric_pairs.fields.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_cells, cells.data(), sizeof(int32_t) * cells.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  const double *const d_aux = d_staging + n_bra;
  const double *const d_u = d_aux + n_aux_fields;
  const double *const d_u_aux = d_u + (n_cart * nao);

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(nao);
  const auto q = static_cast<int32_t>(naux);
  const auto o = static_cast<int32_t>(occ);
  const auto nx = static_cast<int32_t>(n_x);

  // D = 2 C_occ C_occ^T; d = L^-T (B vec D) = M^-1 u.
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, n, n, o, &kTwo<double>, c, c,
                                                                &kZero<double>, dd)));
  FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, n * n, &kOne<double>, b,
                                                q, dd, 1, &kZero<double>, fit, 1)));
  FBF_TRY(
      gpu_status(wwr::trsv<double, int32_t>(blas, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
                                            wwr::WWRBLAS_DIAG_NON_UNIT, q, l, q, fit, 1)));
  // The half transform B_P,ls C_li for each s, at P + naux (i + nocc s), then
  // c = L^-T of it: c_P,is = (M^-1 T_i)_P,s, the fitted pairs of K.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, o, n, &kOne<double>, b, q, int64_t{q} * n, c,
      n, 0, &kZero<double>, cf, q, int64_t{q} * o, n)));
  FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
      blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
      wwr::WWRBLAS_DIAG_NON_UNIT, q, o * n, &kOne<double>, l, q, cf, q)));
  // M1 = d(P|Q)/dA_k, column-major element (Q, P) of slab k.
  FBF_TRY(metric_derivative_device(res, naux_cart, naux, metric_pairs,
                                   static_cast<int32_t>(n_aux_rec), d_aux, d_pairs, d_u_aux, m1c,
                                   mt, m1));

  // jx and kx accumulate S, the half whose S + S^T is the matrix, stored
  // transposed (S_mn at n + nao m): every term below lands that way round.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(jx, 0, sizeof(double) * slabs, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(kx, 0, sizeof(double) * slabs, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(gx, 0, sizeof(double) * naux * n_x, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(y_acc, 0, sizeof(double) * 3 * nn, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(v1_acc, 0, sizeof(double) * 3 * nn, stream)));

  // -M^x d, into column x = 3 A + k of g: with (P|Q)^x = [P on A] M1[k][P][Q]
  // + [Q on A] M1[k][Q][P], the rows on A of M1[k] d, and SUM_{Q on A}
  // M1[k][Q][P] d_Q on every row.
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [p0, p1] = atom_aos(aux, a, true);
    if (p1 == p0) {
      continue;
    }
    const auto qa = static_cast<int32_t>(p1 - p0);
    for (size_t k = 0; k < 3; ++k) {
      const size_t x = (3 * a) + k;
      const double *const m1_atom = m1 + (k * naux * naux) + (naux * p0);
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, q, qa,
                                                    &kNegOne<double>, m1_atom, q, fit, 1,
                                                    &kOne<double>, gx + p0 + (naux * x), 1)));
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, qa,
                                                    &kNegOne<double>, m1_atom, q, fit + p0, 1,
                                                    &kOne<double>, gx + (naux * x), 1)));
    }
  }

  // One aux atom B at a time: its P carry (mn|P)^x through X = (nabla m n|P),
  // (mn|P)^{A_k} = -[m on A] X_mn - [n on A] X_nm + [P on A] (X_mn + X_nm).
  for (uint32_t blk = 0; blk < n_atom; ++blk) {
    const auto [cart0, cart1] = atom_aos(aux, blk, false);
    const auto [p0, p1] = atom_aos(aux, blk, true);
    if (p1 == p0) {
      continue;
    }
    const size_t nb = p1 - p0;
    const auto qb = static_cast<int32_t>(nb);
    FBF_TRY(three_center_derivative_block(
        res, n_cart, nao, naux_cart, cart0, cart1 - cart0, p0, nb, n_bra_rec, d_staging, n_aux_rec,
        d_aux, d_cells, n_cell[blk], cell_begin[blk], d_u, d_u_aux, xc, t1, t2, xs, xts));
    // Z_P,ls = SUM_i c_P,is C_li over the atom's P, at P + nb (l + nao s); and
    // c's rows on the atom, packed, at P + nb (i + nocc s).
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, qb, n, o, &kOne<double>, cf + p0, q,
        int64_t{q} * o, c, n, 0, &kZero<double>, zb, qb, int64_t{qb} * n, n)));
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(cfb, sizeof(double) * nb, cf + p0,
                                             sizeof(double) * naux, sizeof(double) * nb, occ * nao,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
    for (size_t k = 0; k < 3; ++k) {
      const size_t x = (3 * blk) + k;
      const double *const xs_k = xs + (k * nb * nn);
      const double *const xt_k = xts + (k * nb * nn);
      double *const jx_x = jx + (x * nn);
      double *const kx_x = kx + (x * nn);
      // F_P,is = SUM_Q M1[k][P][Q] c_Q,is over the atom's P.
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
          blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, qb, o * n, q, &kOne<double>,
          m1 + (k * naux * naux) + (naux * p0), q, cf, q, &kZero<double>, fb, qb)));

      // J: SUM_P X_mn,P d_P, the [P on B] term here and every atom's [m on A]
      // term once the walk is done (Y).
      FBF_TRY(
          gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, qb, n * n, &kOne<double>,
                                                xs_k, qb, fit + p0, 1, &kOne<double>, jx_x, 1)));
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, qb, n * n,
                                                    &kOne<double>, xs_k, qb, fit + p0, 1,
                                                    &kOne<double>, y_acc + (k * nn), 1)));
      // u^x on the atom's rows: -2 SUM_{m on A, n} X_mn,P D_mn into column
      // 3 A + k, and 2 SUM_mn X_mn,P D_mn into column x.
      for (uint32_t a = 0; a < n_atom; ++a) {
        const auto [m0, m_end] = atom_aos(basis, a, true);
        if (m_end == m0) {
          continue;
        }
        FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, qb, n * static_cast<int32_t>(m_end - m0), &kNegTwo<double>,
            xs_k + (nb * nao * m0), qb, dd + (nao * m0), 1, &kOne<double>,
            gx + p0 + (naux * ((3 * a) + k)), 1)));
      }
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, qb, n * n,
                                                    &kTwo<double>, xs_k, qb, dd, 1, &kOne<double>,
                                                    gx + p0 + (naux * x), 1)));

      // K: SUM_lP (ml|P)^x Z_P,ln. The X_ml term over the atom's P (the
      // [P on B] term, and V1 for every atom's [m on A] one), the X_lm term
      // over them ([P on B]), and per atom A the X_lm term over l on A.
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, n, qb * n, &kOne<double>, zb, xs_k,
                                                 &kOne<double>, kx_x)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, n, qb * n, &kOne<double>, zb, xs_k,
                                                 &kOne<double>, v1_acc + (k * nn))));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, n, qb * n, &kOne<double>, zb, xt_k,
                                                 &kOne<double>, kx_x)));
      for (uint32_t a = 0; a < n_atom; ++a) {
        const auto [l0, l1] = atom_aos(basis, a, true);
        if (l1 == l0) {
          continue;
        }
        FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
            blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n, n, qb * static_cast<int32_t>(l1 - l0),
            &kNegOne<double>, zb + (nb * l0), qb * n, xt_k + (nb * l0), qb * n, &kOne<double>,
            kx + (((3 * a) + k) * nn), n)));
      }
      // The metric's: -SUM_{P on B, i} c_P,im F_P,in.
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, n, qb * o, &kNegOne<double>, fb, cfb,
                                                 &kOne<double>, kx_x)));
    }
  }

  // Every atom's [m on A] terms: -Y and -V1 over the columns m on A of the
  // transposed S.
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [m0, m_end] = atom_aos(basis, a, true);
    if (m_end == m0) {
      continue;
    }
    const auto count = n * static_cast<int32_t>(m_end - m0);
    for (size_t k = 0; k < 3; ++k) {
      const size_t at = (((3 * a) + k) * nn) + (nao * m0);
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, count, &kNegOne<double>,
                                                    y_acc + (k * nn) + (nao * m0), 1, jx + at, 1)));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(
          blas, count, &kNegOne<double>, v1_acc + (k * nn) + (nao * m0), 1, kx + at, 1)));
    }
  }

  if (g != nullptr) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(g, gx, sizeof(double) * naux * n_x,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  // L^-1 (u^x - M^x d), in place: T M^-1 (u^x - M^x d) = B^T L^-1 (u^x - M^x d).
  FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
      blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
      wwr::WWRBLAS_DIAG_NON_UNIT, q, nx, &kOne<double>, l, q, gx, q)));
  // J = S + S^T and K = 2 (S + S^T), one slab at a time through sym (geam may
  // not write its own transposed operand).
  for (size_t x = 0; x < n_x; ++x) {
    for (const auto &[slab, scale] :
         {std::pair{jx + (x * nn), &kOne<double>}, std::pair{kx + (x * nn), &kTwo<double>}}) {
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n,
                                                    n, scale, slab, n, scale, slab, n, sym, n)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(slab, sym, sizeof(double) * nn,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
    }
  }
  // J += B^T L^-1 (u^x - M^x d), every slab at once: symmetric already, B
  // being symmetric in (m, n).
  FBF_TRY(gpu_status(
      gemm<wwr::WWRBLAS_OP_T>(blas, n * n, nx, q, &kOne<double>, b, gx, &kOne<double>, jx)));
  return res.sync();
}

} // namespace free_breakfast
