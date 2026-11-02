/// @file
/// @brief The DF derivative kernels' shared staging (staging.cppm).

module;

// The gradient's and the Hessian's record layouts, which are one: declared in
// the GMF, as their modules declare them (see the headers).
#include "gradient_bridge.h"
#include "hessian_bridge.h"

module free_breakfast.integrals.eri.density_fitting.staging;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemmStridedBatched, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.blas;

#include "diagnostics.h"

namespace free_breakfast {

static_assert(kMetricMomenta == static_cast<size_t>(device::kMetricMaxMomentum + 1));
static_assert(kThreeCenterMomenta == static_cast<size_t>(device::kThreeCenterMaxMomentum + 1));
static_assert(kThreeCenterAuxMomenta ==
              static_cast<size_t>(device::kThreeCenterAuxMaxMomentum + 1));
static_assert(device::kMetricHessianPairFields == device::kMetricGradientPairFields &&
                  device::kThreeCenterHessianBraFields == device::kThreeCenterGradientBraFields &&
                  device::kThreeCenterHessianCellFields == device::kThreeCenterGradientCellFields,
              "the Hessian kernels read the gradient kernels' parameter block");

std::vector<int32_t> shell_atoms(const BasisSet &basis) {
  std::vector<int32_t> atom_of(basis.nshells());
  for (uint32_t a = 0; a < basis.natoms(); ++a) {
    const uint32_t first = basis.atom_shell_offset(a);
    std::fill_n(atom_of.begin() + first, basis.atom_shell_count(a), static_cast<int32_t>(a));
  }
  return atom_of;
}

std::vector<int32_t> record_begins(const BasisSet &basis) {
  std::vector<int32_t> rec_begin{0};
  rec_begin.reserve(basis.shells().size() + 1);
  for (const Shell &shell : basis.shells()) {
    rec_begin.push_back(rec_begin.back() + static_cast<int32_t>(shell.nprimitives()));
  }
  return rec_begin;
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

std::vector<double> stage_metric_records(const BasisSet &aux,
                                         const std::vector<int32_t> &rec_begin) {
  const size_t n_records =
      static_cast<size_t>(device::kMetricRecordFields) * static_cast<size_t>(rec_begin.back());
  std::vector<double> staging(n_records + (aux.nao_cart() * aux.nao_sphe()));
  stage_aux_records(aux, rec_begin, staging.data());
  std::ranges::copy(spherical_transform_matrix(aux), staging.data() + n_records);
  return staging;
}

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

Status metric_cartesian(DeviceResources &res, const size_t n, const size_t n_sph,
                        const double *d_u_aux, const double *w, double *t, double *w_cart) {
  const auto n32 = static_cast<int32_t>(n);
  const auto n_sph32 = static_cast<int32_t>(n_sph);
  // t = U_aux W, then the Cartesian W = t U_aux^T.
  FBF_TRY(gpu_status(
      gemm(res.blas(), n32, n_sph32, n_sph32, &kOne<double>, d_u_aux, w, &kZero<double>, t)));
  return gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      res.blas(), n32, n32, n_sph32, &kOne<double>, t, d_u_aux, &kZero<double>, w_cart));
}

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
    std::vector<ThreeCenterPair> &pairs =
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
  for (const std::vector<ThreeCenterPair> &p : out.pairs) {
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

std::vector<double> stage_three_center_records(std::vector<double> bra_fields,
                                               const BasisSet &basis, const BasisSet &aux,
                                               const std::vector<int32_t> &aux_rec_begin) {
  const size_t n_bra = bra_fields.size();
  const size_t n_aux_fields =
      static_cast<size_t>(device::kMetricRecordFields) * static_cast<size_t>(aux_rec_begin.back());
  const size_t n_u = basis.nao_cart() * basis.nao_sphe();
  std::vector<double> staging = std::move(bra_fields);
  staging.resize(n_bra + n_aux_fields + n_u + (aux.nao_cart() * aux.nao_sphe()));
  stage_aux_records(aux, aux_rec_begin, staging.data() + n_bra);
  double *const u = staging.data() + n_bra + n_aux_fields;
  std::ranges::copy(spherical_transform_matrix(basis), u);
  std::ranges::copy(spherical_transform_matrix(aux), u + n_u);
  return staging;
}

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
      for (const ThreeCenterPair &pair : bra.pairs[ab]) {
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

Status three_center_cartesian(DeviceResources &res, const size_t n, const size_t n_sph,
                              const size_t n_aux, const size_t n_aux_sph, const double *d_u,
                              const double *d_u_aux, const double *gamma, double *cart,
                              double *work) {
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
  return gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      res.blas(), n_aux32 * n32, n32, n_sph32, &kOne<double>, work, d_u, &kZero<double>, cart));
}

} // namespace free_breakfast
