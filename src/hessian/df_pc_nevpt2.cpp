/// @file
/// @brief df_pc_nevpt2: the PC-NEVPT2 analytic nuclear Hessian on a
/// single-state DF-CASSCF (docs/nevpt2/nevpt2.tex eq. hess), df_sc_nevpt2's
/// assembly over PC's adjoint, gradient multipliers and Hessian-vector
/// product. A degenerate inactive pair, which the tangents leave unrotated,
/// is taken through its off-diagonal f in rotated bases (#1268), and a
/// near-degenerate cluster whole, every pair within it (#1314). Refused
/// where one analytic Hessian does not represent the PC surface: a class
/// kept below the HVP's conditioning floor, and a coordinate whose Schur
/// lift reaches the cut within 1e-3 bohr; Unsupported for a cluster of more
/// than six orbitals joined by pairs under 1e-6 Eh. The degenerate pairs'
/// step, degenerate_pair_blocks and degenerate_pair_part, is defined here
/// and shared with df_qd_pc_nevpt2 (#1277).

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kTwo, kNegOne, kPi
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, axpy, gemm, gemv, gemmStridedBatched, wwrMemcpyAsync,
                                // wwrMemsetAsync, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.mo; // b_mo, OrbitalRange
import free_breakfast.recipes.blas;                     // gemm, gemmwich
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.response.orbital_pairs; // unpack_orbital_pairs
import free_breakfast.mcscf.solve;
import free_breakfast.response.zvector;
import free_breakfast.nevpt2.gradient; // pc_adjoint, zc_multipliers, zc_adjoint,
                                       // lagrangian_gradient, pc_hessian_product
import :state_specific;                // state_specific_electrons

#include "diagnostics.h"

namespace free_breakfast::hessian {

// The coordinates' rows are orthonormal to this (Q Q^T = 1), so that
// Q H Q^T is the Hessian in them.
constexpr double kPcNevpt2OrthonormalTolerance = 1e-10;
// A class's smallest kept metric eigenvalue, over its largest, below which
// its second derivative is not attainable (#917; nevpt2::pc_class_energies_hvp
// refuses there too). EnergyInputs::pc_floor raises the cut past it.
constexpr double kPcNevpt2Kept = 1e-10;
// The step, in bohr along a coordinate, within which a class's Schur lift
// must not reach the cut (#817's convention): closer, the PC surface switches
// branch inside a vibrational or finite-difference step, and one analytic
// Hessian does not represent it.
constexpr double kPcNevpt2LiftStep = 1e-3;
// The rotations that read a degenerate pair's off-diagonal f as a split
// (README "A degenerate inactive pair"): by pi/4 the split is f_pq's change,
// by pi/8 the sum of the two over sqrt 2.
constexpr double kPcNevpt2PairTurn = kPi<double> / 4.0;
constexpr double kPcNevpt2PairHalfTurn = kPi<double> / 8.0;
// PySCF's class order, PcClassEnergies'.
constexpr std::array<std::string_view, 8> kPcNevpt2Classes{"Sr",   "Si",  "Sijrs", "Sijr",
                                                           "Srsi", "Srs", "Sij",   "Sir"};

Result<std::vector<std::array<int32_t, 2>>>
degenerate_pair_blocks(const std::string_view who,
                       const std::span<const std::array<int32_t, 2>> records) {
  std::vector<std::array<int32_t, 2>> blocks;
  for (const std::array<int32_t, 2> &pq : records) {
    if (std::ranges::find(blocks, pq) == blocks.end()) {
      blocks.push_back(pq);
    }
  }
  // Each cluster: the orbitals the pairs join, grown from each pair.
  std::vector<bool> seen(blocks.size(), false);
  for (size_t start = 0; start < blocks.size(); ++start) {
    if (seen[start]) {
      continue;
    }
    std::vector<int32_t> orbitals{blocks[start][0], blocks[start][1]};
    seen[start] = true;
    for (bool grew = true; grew;) {
      grew = false;
      for (size_t at = 0; at < blocks.size(); ++at) {
        if (!seen[at] && (std::ranges::find(orbitals, blocks[at][0]) != orbitals.end() ||
                          std::ranges::find(orbitals, blocks[at][1]) != orbitals.end())) {
          seen[at] = true;
          grew = true;
          for (const int32_t o : blocks[at]) {
            if (std::ranges::find(orbitals, o) == orbitals.end()) {
              orbitals.push_back(o);
            }
          }
        }
      }
    }
    if (orbitals.size() > kPcClusterLimit) {
      std::ranges::sort(orbitals);
      std::string named;
      for (const int32_t o : orbitals) {
        named += std::format("{}{}", named.empty() ? "" : ", ", o);
      }
      return err_unsupported("{}: a cluster of {} near-degenerate inactive orbitals ({}) is past "
                             "the {} the pairs' step takes: its pairs' cross terms grow as the "
                             "fourth power of its size (README \"A near-degenerate cluster\")",
                             who, orbitals.size(), named, kPcClusterLimit);
    }
  }
  return blocks;
}

// Step 4b (README "A degenerate inactive pair", "A near-degenerate
// cluster"). Along the path that leaves pair P = (p, q) unrotated its
// f_pq = delta_P moves at first order, which the functional (f's diagonal
// alone) does not read; PC is invariant to the rotations within the core and
// within the virtuals, so the energy is the functional plus delta's Taylor
// terms:
//
//   H_ij += sum_P (phi^P_i K^P_j + phi^P_j K^P_i) + sum_PP' M_PP' phi^P_i phi^P'_j,
//
// phi^P_i = delta_P's change along t_i, K^P_j the change of dE/d delta_P along
// t_j at fixed delta, M_PP' = d2E/d delta_P d delta_P'. The pairs may share
// orbitals (a cluster takes every pair within it). Each term is read in a
// basis C U whose turn makes the deltas it needs diagonal: by pi/4 within P
// alone delta_P is the split d' of (p', q'), so the product along it (an h
// direction, +1 at p'p', -1 at q'q') gives M_PP from its own split and G^P_j
// paired with t_j; G^P_j = K^P_j - M^dd_P d^P_j + M_PP phi^P_j, d^P_j the
// change of the pair's own split, which that basis does not read, and
// M^dd_P = d2E/d delta_P d split_P follows from the pi/8 basis (split
// (d + delta) / sqrt 2) and the unturned one. M_PP' of two disjoint pairs is
// the split of P' along P's in a basis turning both by pi/4; of two that share
// an orbital s, P = (a, s) and P' = (s, b), it is the polarization
// (Y.Y - M_PP - M_P'P') / 2 of Y = delta_P + delta_P', read along Y in the
// basis that diagonalizes it within (a, s, b). Exact at a degenerate
// cluster, O(gap) off one.
Result<DegeneratePairPart> degenerate_pair_part(
    DeviceResources &res, const int32_t nmo32, const int32_t naux32, const size_t k,
    const double *c, const double energy, const std::span<const std::array<int32_t, 2>> blocks,
    const std::span<const double> phi, const std::span<const double> split,
    const std::function<Status(const double *, const double *,
                               const std::function<Status(const double *, const double *,
                                                          const double *, double)> &)> &product,
    const std::function<Status(const double *, const double *, const double *, double *)>
        &pair_with) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto nmo = static_cast<size_t>(nmo32);
  const auto naux = static_cast<size_t>(naux32);
  const size_t nn = nmo * nmo;
  const size_t np = blocks.size();
  const auto copy = [&res](double *to, const double *from, const size_t count,
                           const wwr::wwrMemcpyKind kind) -> Status {
    return gpu_status(wwr::wwrMemcpyAsync(to, from, count * sizeof(double), kind, res.stream()));
  };
  const auto download = [&](const double *const from,
                            const size_t count) -> Result<std::vector<double>> {
    std::vector<double> to(count);
    FBF_TRY(copy(to.data(), from, count, wwr::wwrMemcpyDeviceToHost));
    FBF_TRY(res.sync());
    return to;
  };
  DegeneratePairPart out{.part = std::vector<double>(k * k, 0.0), .rotation_shift = 0.0};
  Buffers::device<double> direction(nn, res.device());
  Buffers::device<double> u(nn, res.device());
  Buffers::device<double> c_turned(nn, res.device());
  Buffers::device<double> h_back(nn, res.device());
  Buffers::device<double> work(naux * nn, res.device());
  Buffers::device<double> l_back(naux * nn, res.device());
  Buffers::device<double> column(k, res.device());
  const auto identity = [&] {
    std::vector<double> host(nn, 0.0);
    for (size_t r = 0; r < nmo; ++r) {
      host[r * (nmo + 1)] = 1.0;
    }
    return host;
  };
  // Turns block @p at of @p host (column-major U) by @p angle: column
  // p' = cos p + sin q, column q' = -sin p + cos q.
  const auto turn = [&](std::vector<double> &host, const size_t at, const double angle) {
    const auto p = static_cast<size_t>(blocks[at][0]);
    const auto q = static_cast<size_t>(blocks[at][1]);
    host[p + (nmo * p)] = std::cos(angle);
    host[q + (nmo * p)] = std::sin(angle);
    host[p + (nmo * q)] = -std::sin(angle);
    host[q + (nmo * q)] = std::cos(angle);
  };
  // The h direction +1 at p, -1 at q of block @p at, in the turned basis.
  const auto split_of = [&](const size_t at) {
    std::vector<double> along(nmo, 0.0);
    along[static_cast<size_t>(blocks[at][0])] = 1.0;
    along[static_cast<size_t>(blocks[at][1])] = -1.0;
    return along;
  };
  // The product in the basis C U (@p turned, host, column-major) along the h
  // direction diag(@p along) there: h_bar's diagonal in that basis, and with
  // @p paired <t_j, U P U^T> over every j, the product turned back to C's
  // basis (h_bar's and each Lambda_Q's two indices) and paired, into *@p paired.
  const auto product_in = [&](const std::vector<double> &turned, const std::vector<double> &along,
                              std::vector<double> *paired) -> Result<std::vector<double>> {
    FBF_TRY(copy(u.data(), turned.data(), nn, wwr::wwrMemcpyHostToDevice));
    std::vector<double> host(nn, 0.0);
    for (size_t r = 0; r < nmo; ++r) {
      host[r * (nmo + 1)] = along[r];
    }
    FBF_TRY(copy(direction.data(), host.data(), nn, wwr::wwrMemcpyHostToDevice));
    FBF_TRY(
        gpu_status(gemm(blas, nmo32, &kOne<double>, c, u.data(), &kZero<double>, c_turned.data())));
    FBF_TRY(res.sync());
    std::vector<double> diagonal(nmo);
    FBF_TRY(product(
        c_turned.data(), direction.data(),
        [&](const double *h_bar, const double *lambda, const double *y,
            const double turned_energy) -> Status {
          out.rotation_shift = std::max(out.rotation_shift, std::abs(turned_energy - energy));
          const std::vector<double> hb = FBF_TRY(download(h_bar, nn));
          for (size_t r = 0; r < nmo; ++r) {
            diagonal[r] = hb[r * (nmo + 1)];
          }
          if (paired == nullptr) {
            return {};
          }
          FBF_TRY(gpu_status(
              gemm(blas, nmo32, &kOne<double>, u.data(), h_bar, &kZero<double>, work.data())));
          FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
              blas, nmo32, nmo32, nmo32, &kOne<double>, work.data(), u.data(), &kZero<double>,
              h_back.data())));
          FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
              blas, naux32 * nmo32, nmo32, nmo32, &kOne<double>, lambda, u.data(), &kZero<double>,
              work.data())));
          FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
              blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, naux32, nmo32, nmo32, &kOne<double>,
              work.data(), naux32, static_cast<int64_t>(naux * nmo), u.data(), nmo32, 0,
              &kZero<double>, l_back.data(), naux32, static_cast<int64_t>(naux * nmo), nmo32)));
          FBF_TRY(pair_with(h_back.data(), l_back.data(), y, column.data()));
          *paired = FBF_TRY(download(column.data(), k));
          return {};
        }));
    return diagonal;
  };
  // Block @p at's split in a product's diagonal.
  const auto split_in = [&](const std::vector<double> &diagonal, const size_t at) {
    return diagonal[static_cast<size_t>(blocks[at][0])] -
           diagonal[static_cast<size_t>(blocks[at][1])];
  };
  std::vector<double> m(np * np, 0.0); // M_PP', [P + np P']
  std::vector<double> mixed(np, 0.0);  // d2E/d delta_P d split_P
  std::vector<double> g(np * k, 0.0);  // G^P_j, [P k + j]
  for (size_t at = 0; at < np; ++at) {
    const std::vector<double> along = split_of(at);
    std::vector<double> turned = identity();
    const double dd = split_in(FBF_TRY(product_in(turned, along, nullptr)), at);
    turn(turned, at, kPcNevpt2PairTurn);
    std::vector<double> paired;
    m[at * (np + 1)] = split_in(FBF_TRY(product_in(turned, along, &paired)), at);
    std::ranges::copy(paired, g.begin() + static_cast<std::ptrdiff_t>(at * k));
    turn(turned, at, kPcNevpt2PairHalfTurn);
    const double half = split_in(FBF_TRY(product_in(turned, along, nullptr)), at);
    mixed[at] = half - (0.5 * (dd + m[at * (np + 1)]));
  }
  // The pair shared by blocks a and b, or -1 when they are disjoint.
  const auto shared = [&](const size_t a, const size_t b) -> int32_t {
    for (const int32_t o : blocks[a]) {
      if (o == blocks[b][0] || o == blocks[b][1]) {
        return o;
      }
    }
    return -1;
  };
  // M_PP' of disjoint pairs: for each P, its partners not yet read, as many
  // mutually disjoint ones at once as fit one basis turning them all with P.
  std::vector<bool> known(np * np, false);
  for (size_t at = 0; at < np; ++at) {
    for (;;) {
      std::vector<size_t> group;
      for (size_t other = 0; other < np; ++other) {
        if (other == at || known[at + (np * other)] || shared(at, other) >= 0 ||
            std::ranges::any_of(group, [&](const size_t g0) { return shared(g0, other) >= 0; })) {
          continue;
        }
        group.push_back(other);
      }
      if (group.empty()) {
        break;
      }
      std::vector<double> turned = identity();
      turn(turned, at, kPcNevpt2PairTurn);
      for (const size_t other : group) {
        turn(turned, other, kPcNevpt2PairTurn);
      }
      const std::vector<double> diagonal = FBF_TRY(product_in(turned, split_of(at), nullptr));
      for (const size_t other : group) {
        const double s = split_in(diagonal, other);
        m[at + (np * other)] = s;
        m[other + (np * at)] = s;
        known[at + (np * other)] = true;
        known[other + (np * at)] = true;
      }
    }
  }
  // M_PP' of pairs sharing orbital s, P = (a, s) and P' = (s, b): Y = delta_P
  // + delta_P' is [[0, 1, 0], [1, 0, 1], [0, 1, 0]] over (a, s, b), whose
  // eigenvectors (1, sqrt 2, 1) / 2, (1, 0, -1) / sqrt 2 and (1, -sqrt 2, 1) / 2
  // (eigenvalues sqrt 2, 0, -sqrt 2) are the columns a, s and b of U; along
  // diag(sqrt 2, 0, -sqrt 2) there, Y.Y = sqrt 2 (h_bar_aa - h_bar_bb).
  for (size_t at = 0; at < np; ++at) {
    for (size_t other = 0; other < at; ++other) {
      const int32_t s32 = shared(at, other);
      if (s32 < 0) {
        continue;
      }
      const auto s = static_cast<size_t>(s32);
      const auto a = static_cast<size_t>(blocks[at][0] == s32 ? blocks[at][1] : blocks[at][0]);
      const auto bo =
          static_cast<size_t>(blocks[other][0] == s32 ? blocks[other][1] : blocks[other][0]);
      std::vector<double> turned = identity();
      const double half_root = 0.5 * kSqrt2<double>;
      for (const auto &[col, va, vs, vb] :
           {std::tuple{a, 0.5, half_root, 0.5}, std::tuple{s, half_root, 0.0, -half_root},
            std::tuple{bo, 0.5, -half_root, 0.5}}) {
        turned[a + (nmo * col)] = va;
        turned[s + (nmo * col)] = vs;
        turned[bo + (nmo * col)] = vb;
      }
      std::vector<double> along(nmo, 0.0);
      along[a] = kSqrt2<double>;
      along[bo] = -kSqrt2<double>;
      const std::vector<double> diagonal = FBF_TRY(product_in(turned, along, nullptr));
      const double yy = kSqrt2<double> * (diagonal[a] - diagonal[bo]);
      const double value = 0.5 * (yy - m[at * (np + 1)] - m[other * (np + 1)]);
      m[at + (np * other)] = value;
      m[other + (np * at)] = value;
    }
  }
  // K^P_j = G^P_j + M^dd_P d^P_j - M_PP phi^P_j, then the terms.
  std::vector<double> kk(np * k);
  for (size_t at = 0; at < np; ++at) {
    for (size_t j = 0; j < k; ++j) {
      kk[(at * k) + j] = g[(at * k) + j] + (mixed[at] * split[(at * k) + j]) -
                         (m[at * (np + 1)] * phi[(at * k) + j]);
    }
  }
  for (size_t j = 0; j < k; ++j) {
    for (size_t i = 0; i < k; ++i) {
      double sum = 0.0;
      for (size_t at = 0; at < np; ++at) {
        sum += (phi[(at * k) + i] * kk[(at * k) + j]) + (phi[(at * k) + j] * kk[(at * k) + i]);
        for (size_t other = 0; other < np; ++other) {
          sum += m[at + (np * other)] * phi[(at * k) + i] * phi[(other * k) + j];
        }
      }
      out.part[i + (k * j)] = sum;
    }
  }
  return out;
}

Result<PcNevpt2Terms>
df_pc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
             const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
             const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
             const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
             const int32_t batch, const std::span<const double> coordinates,
             const double pc_floor) {
  constexpr std::string_view kWho = "hessian::df_pc_nevpt2";
  const auto [nalpha, nbeta] =
      FBF_TRY(state_specific_electrons(kWho, res, molecule, basis, aux, b, solution, ncore, ncas));
  const size_t n_x = static_cast<size_t>(3) * basis.natoms();
  const size_t nmo = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const size_t nn = nmo * nmo;
  const size_t nc = static_cast<size_t>(ncore);
  const size_t n_act = static_cast<size_t>(ncas);
  const size_t pairs = (nc * (nmo - nc)) + (n_act * (nmo - nc - n_act));

  // The coordinates: k orthonormal rows of n_x, the identity when none.
  if (coordinates.size() % n_x != 0 || (!coordinates.empty() && coordinates.size() < n_x)) {
    return err_config("{}: {} coordinate doubles are not whole rows of {}", kWho,
                      coordinates.size(), n_x);
  }
  const size_t k = coordinates.empty() ? n_x : coordinates.size() / n_x;
  std::vector<double> q(k * n_x, 0.0); // row i at i n_x, as given
  if (coordinates.empty()) {
    for (size_t i = 0; i < n_x; ++i) {
      q[i * (n_x + 1)] = 1.0;
    }
  } else {
    std::ranges::copy(coordinates, q.begin());
  }
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      double s = 0.0;
      for (size_t x = 0; x < n_x; ++x) {
        s += q[(i * n_x) + x] * q[(j * n_x) + x];
      }
      if (!(std::abs(s - (i == j ? 1.0 : 0.0)) <= kPcNevpt2OrthonormalTolerance)) {
        return err_config("{}: coordinates {} and {} have overlap {:.3e}: not orthonormal rows",
                          kWho, i, j, s);
      }
    }
  }
  if (naux * nn > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_unsupported("{}: naux {} times nmo^2 {} is past int32_t", kWho, naux, nn);
  }
  const auto k32 = static_cast<int32_t>(k);
  const auto n_x32 = static_cast<int32_t>(n_x);
  const auto nmo32 = static_cast<int32_t>(nmo);
  const auto naux32 = static_cast<int32_t>(naux);
  const auto nn32 = static_cast<int32_t>(nn);
  const auto bn32 = static_cast<int32_t>(naux * nn);
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();

  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  const auto ndet = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  const auto ndet32 = static_cast<int32_t>(ndet);
  if (solution.roots.vectors.num_elements() != ndet) {
    return err_config("{}: a root of {} doubles over {} determinants", kWho,
                      solution.roots.vectors.num_elements(), ndet);
  }

  const auto copy = [&res](double *to, const double *from, const size_t count,
                           const wwr::wwrMemcpyKind kind) -> Status {
    return gpu_status(wwr::wwrMemcpyAsync(to, from, count * sizeof(double), kind, res.stream()));
  };
  const auto download = [&](const double *const from,
                            const size_t count) -> Result<std::vector<double>> {
    std::vector<double> to(count);
    FBF_TRY(copy(to.data(), from, count, wwr::wwrMemcpyDeviceToHost));
    FBF_TRY(res.sync());
    return to;
  };
  const auto zeroed = [&res](const size_t count) -> Result<Buffers::device<double>> {
    Buffers::device<double> d(count, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(d.data(), 0, count * sizeof(double), res.stream())));
    return d;
  };
  // (i, j) = <a_i, b_j> over two stacks of k columns of @p rows, on the host.
  const auto gram = [&](const int32_t rows, const double *a,
                        const double *bb) -> Result<std::vector<double>> {
    Buffers::device<double> out(k * k, res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, k32, k32, rows, &kOne<double>, a, bb,
                                               &kZero<double>, out.data())));
    return download(out.data(), k * k);
  };
  // Q P Q^T of a Cartesian n_x x n_x matrix (column-major), k x k.
  const auto to_coordinates = [&](std::span<const double> p) {
    std::vector<double> qp(k * n_x, 0.0); // (i, y) at i + k y
    for (size_t y = 0; y < n_x; ++y) {
      for (size_t x = 0; x < n_x; ++x) {
        const double pxy = p[x + (n_x * y)];
        for (size_t i = 0; i < k; ++i) {
          qp[i + (k * y)] += q[(i * n_x) + x] * pxy;
        }
      }
    }
    std::vector<double> out(k * k, 0.0);
    for (size_t j = 0; j < k; ++j) {
      for (size_t y = 0; y < n_x; ++y) {
        const double qjy = q[(j * n_x) + y];
        for (size_t i = 0; i < k; ++i) {
          out[i + (k * j)] += qp[i + (k * y)] * qjy;
        }
      }
    }
    return out;
  };

  // 1. E_cas, and lambda^x (kappa_nr^x, c^x) with it.
  CasscfTerms casscf =
      FBF_TRY(df_casscf(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas));

  // 2. t_x in the coordinates: t_i = sum_x Q_ix t_x, accumulated a chunk of
  // Cartesian coordinates at a time, with kappa (the total) and kappa_nr.
  Buffers::device<double> qt(n_x * k, res.device()); // Q^T, n_x x k column-major
  FBF_TRY(copy(qt.data(), q.data(), n_x * k, wwr::wwrMemcpyHostToDevice));
  Buffers::device<double> t_h = FBF_TRY(zeroed(nn * k));
  Buffers::device<double> t_b = FBF_TRY(zeroed(naux * nn * k));
  Buffers::device<double> t_c = FBF_TRY(zeroed(ndet * k));
  Buffers::device<double> t_kappa = FBF_TRY(zeroed(nn * k));
  // S_coords += S_chunk Q^T[x0 .. x0 + count), each stack of @p rows.
  const auto accumulate = [&](const int32_t rows, const int32_t x0, const int32_t count,
                              const double *chunk, double *stack) -> Status {
    return gpu_status(wwr::gemm<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, rows, k32, count, &kOne<double>, chunk, rows,
        qt.data() + x0, n_x32, &kOne<double>, stack, rows));
  };
  const std::vector<DegeneratePair> degenerate = FBF_TRY(df_nevpt2_tangents(
      res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, casscf.cp, true,
      [&](const int32_t x0, const int32_t count, const double *h_dot, const double *b_dot,
          const double *c_dot, const double *kappa) -> Status {
        FBF_TRY(accumulate(nn32, x0, count, h_dot, t_h.data()));
        FBF_TRY(accumulate(bn32, x0, count, b_dot, t_b.data()));
        FBF_TRY(accumulate(ndet32, x0, count, c_dot, t_c.data()));
        return accumulate(nn32, x0, count, kappa, t_kappa.data());
      }));
  // The degenerate pairs: kappa_oo^x_pq is not a division the state
  // determines, and the tangents left it 0. PC is invariant to the pair's
  // rotation, so step 4b takes the pair's off-diagonal f instead (README "A
  // degenerate inactive pair"), and every pair of a cluster ("A
  // near-degenerate cluster").
  std::vector<std::array<int32_t, 2>> records;
  std::ranges::transform(degenerate, std::back_inserter(records),
                         [](const DegeneratePair &d) { return std::array{d.p, d.q}; });
  std::vector<std::array<int32_t, 2>> blocks = FBF_TRY(degenerate_pair_blocks(kWho, records));
  Buffers::device<double> t_nr = FBF_TRY(zeroed(nn * k));
  {
    Buffers::device<double> kappa_nr(nn * n_x, res.device());
    for (size_t x = 0; x < n_x; ++x) {
      FBF_TRY(response::unpack_orbital_pairs(res, nmo32, ncore, ncas,
                                             casscf.cp.orbital.data() + (x * pairs),
                                             kappa_nr.data() + (x * nn)));
    }
    FBF_TRY(accumulate(nn32, 0, n_x32, kappa_nr.data(), t_nr.data()));
    FBF_TRY(res.sync());
  }

  // 3. E_corr + Phi's adjoint, kept (h_bar, Lambda), then the gradient's
  // Lagrangian: its multipliers (Z, z) and z^c, pc_gradient's calls. A class
  // kept below the HVP's floor is refused before the Lagrangian is solved.
  const double *const c = solution.c.data();
  const double *const vector = solution.roots.vectors.data();
  Buffers::device<double> h_bar(nn, res.device());
  Buffers::device<double> lambda(naux * nn, res.device());
  std::optional<nevpt2::ScGradient> lagrangian;
  std::array<double, 8> kept{};
  std::array<int64_t, 8> dropped{};
  double e_corr = 0.0;
  {
    nevpt2::PcAdjoint corr =
        FBF_TRY(nevpt2::pc_adjoint(res, alpha, beta, naux32, nmo32, nmo32, b.data(), c, h.data(),
                                   ncore, ncas, vector, batch, 0, pc_floor));
    kept = corr.energies.kept;
    dropped = corr.energies.dropped;
    e_corr = corr.energies.total;
    std::string thin;
    for (size_t cls = 0; cls < kept.size(); ++cls) {
      // kept is 0 for Sijrs and a class with no tuple: no solve.
      if (kept[cls] > 0.0 && kept[cls] < kPcNevpt2Kept) {
        thin += std::format("{}{} keeps {:.2e}", thin.empty() ? "" : ", ", kPcNevpt2Classes[cls],
                            kept[cls]);
      }
    }
    if (!thin.empty()) {
      return err_numerical("{}: {} of its metric's largest eigenvalue, below {:g}, where PC's "
                           "second derivative is not attainable (#917); a pc_floor of {:g} or "
                           "more cuts it",
                           kWho, thin, kPcNevpt2Kept, kPcNevpt2Kept);
    }
    nevpt2::ZcMultipliers zc = FBF_TRY(
        nevpt2::zc_multipliers(res, nmo32, ncore, ncas, corr.a.data(), corr.epsilon.data()));
    const nevpt2::ZcAdjoint phi =
        FBF_TRY(nevpt2::zc_adjoint(res, alpha, beta, naux32, nmo32, nmo32, b.data(), c, h.data(),
                                   ncore, ncas, std::array{1.0}, vector, zc.z.data()));
    FBF_TRY(copy(h_bar.data(), corr.h_bar.data(), nn, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(copy(lambda.data(), corr.lambda.data(), naux * nn, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, nn32, &kOne<double>, phi.h_bar.data(), 1,
                                                  h_bar.data(), 1)));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, bn32, &kOne<double>, phi.lambda.data(), 1,
                                                  lambda.data(), 1)));
    FBF_TRY(res.sync());
    lagrangian.emplace(FBF_TRY(nevpt2::lagrangian_gradient(
        res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
        nevpt2::ScAdjoint{.energies = {},
                          .h_bar = std::move(corr.h_bar),
                          .lambda = std::move(corr.lambda),
                          .y = std::move(corr.y),
                          .x = std::move(corr.x),
                          .a = std::move(corr.a),
                          .epsilon = std::move(corr.epsilon)},
        std::move(zc), phi)));
  }
  const nevpt2::ScGradient &gradient = *lagrangian;

  // 4. Sum <t_i, HVP t_j>: the product along every coordinate's t (PC has no
  // empty perturbers and no partners), with each direction's Schur lift and
  // its step t*. The directions are chunked as widely as the memory budget
  // holds the product's footprint.
  // <t_i, (h_bar, Lambda, y)> over every i, into k doubles.
  const auto pair_with = [&](const double *h_bar_dot, const double *lambda_dot, const double *y_dot,
                             double *into) -> Status {
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, nn32, k32, &kOne<double>,
                                                  t_h.data(), nn32, h_bar_dot, 1, &kZero<double>,
                                                  into, 1)));
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, bn32, k32, &kOne<double>,
                                                  t_b.data(), bn32, lambda_dot, 1, &kOne<double>,
                                                  into, 1)));
    return gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, ndet32, k32,
                                                 &kOne<double>, t_c.data(), ndet32, y_dot, 1,
                                                 &kOne<double>, into, 1));
  };
  std::vector<double> hvp(k * k, 0.0);
  std::vector<std::array<double, 8>> lift_steps;
  lift_steps.reserve(k);
  {
    Buffers::device<double> paired(k * k, res.device());
    // As many directions at once as the memory budget holds the product of,
    // at the classes' kept ranks (syevd's lwork is the workspace's alone).
    const std::array<std::array<int64_t, 8>, 1> ranks{dropped};
    const size_t chunk = FBF_TRY(widest_fitting(res, kWho, k, [&](const size_t count) {
      const MemoryFootprint f = nevpt2::pc_hessian_product_footprint(
          naux32, nmo32, nmo32, ncore, ncas, nalpha, nbeta, batch, {}, static_cast<int32_t>(count),
          false, ranks);
      return f.device.persistent + f.device.scratch;
    }));
    for (size_t j0 = 0; j0 < k; j0 += chunk) {
      const size_t count = std::min(chunk, k - j0);
      const nevpt2::PcHessianProduct p = FBF_TRY(
          nevpt2::pc_hessian_product(res, alpha, beta, naux32, nmo32, nmo32, b.data(), c, h.data(),
                                     ncore, ncas, vector, gradient.zc.z.data(), batch,
                                     {.count = static_cast<int32_t>(count),
                                      .h = t_h.data() + (j0 * nn),
                                      .b = t_b.data() + (j0 * naux * nn),
                                      .c = t_c.data() + (j0 * ndet)},
                                     false, pc_floor));
      for (size_t u = 0; u < count; ++u) {
        FBF_TRY(pair_with(p.products[u].h_bar.data(), p.products[u].lambda.data(),
                          p.products[u].y.data(), paired.data() + ((j0 + u) * k)));
        lift_steps.push_back(p.lift_steps[u]);
      }
      FBF_TRY(res.sync());
    }
    hvp = FBF_TRY(download(paired.data(), k * k));
  }
  // The Schur lift: a coordinate along which a class's dropped metric mode
  // reaches the cut within kPcNevpt2LiftStep is refused, every one named.
  {
    std::string named;
    for (size_t i = 0; i < k; ++i) {
      for (size_t cls = 0; cls < lift_steps[i].size(); ++cls) {
        if (lift_steps[i][cls] < kPcNevpt2LiftStep) {
          named += std::format("{}coordinate {} class {} t* {:.3e} bohr", named.empty() ? "" : ", ",
                               i, kPcNevpt2Classes[cls], lift_steps[i][cls]);
        }
      }
    }
    if (!named.empty()) {
      return err_numerical("{}: the PC surface switches branch within {:g} bohr (a Schur lift "
                           "reaches the cut, README \"The PC-NEVPT2 Hessian\"): {}",
                           kWho, kPcNevpt2LiftStep, named);
    }
  }

  // 4b. The degenerate pairs' off-diagonal f (README "A degenerate inactive
  // pair"), degenerate_pair_part's: each read by the product at z = 0 in a
  // basis turned within the pairs.
  std::vector<double> degenerate_part(k * k, 0.0);
  double rotation_shift = 0.0;
  if (!blocks.empty()) {
    const size_t np = blocks.size();
    // phi^P_i and d^P_i, [P k + i], from the Cartesian records.
    std::vector<double> phi(np * k, 0.0);
    std::vector<double> split(np * k, 0.0);
    for (const DegeneratePair &d : degenerate) {
      const auto at =
          static_cast<size_t>(std::ranges::find(blocks, std::array{d.p, d.q}) - blocks.begin());
      const auto x = static_cast<size_t>(d.coordinate);
      for (size_t i = 0; i < k; ++i) {
        phi[(at * k) + i] += q[(i * n_x) + x] * d.f;
        split[(at * k) + i] += q[(i * n_x) + x] * d.split;
      }
    }
    Buffers::device<double> z_none = FBF_TRY(zeroed(nn));
    Buffers::device<double> b_none = FBF_TRY(zeroed(naux * nn));
    Buffers::device<double> c_none = FBF_TRY(zeroed(ndet));
    DegeneratePairPart part = FBF_TRY(degenerate_pair_part(
        res, nmo32, naux32, k, c, e_corr, blocks, phi, split,
        [&](const double *c_turned, const double *direction, const auto &consume) -> Status {
          const nevpt2::PcHessianProduct p = FBF_TRY(nevpt2::pc_hessian_product(
              res, alpha, beta, naux32, nmo32, nmo32, b.data(), c_turned, h.data(), ncore, ncas,
              vector, z_none.data(), batch,
              {.count = 1, .h = direction, .b = b_none.data(), .c = c_none.data()}, false,
              pc_floor));
          const nevpt2::AdjointTangent &t = p.products[0];
          return consume(t.h_bar.data(), t.lambda.data(), t.y.data(), p.energies.total);
        },
        pair_with));
    degenerate_part = std::move(part.part);
    rotation_shift = part.rotation_shift;
  }

  // 5. The Z terms at (Z, z): Z . d2g/dRdR, the L_Ax pair and the third
  // derivative over Lambda = lambda^x in ThirdDerivative's layout.
  const size_t size = pairs + ndet;
  const auto size32 = static_cast<int32_t>(size);
  std::optional<CasscfZTerms> z_terms;
  std::vector<double> lx_cart;
  {
    Buffers::device<double> z(size, res.device());
    FBF_TRY(
        copy(z.data(), gradient.multipliers.orbital.data(), pairs, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(
        copy(z.data() + pairs, gradient.multipliers.ci.data(), ndet, wwr::wwrMemcpyDeviceToDevice));
    Buffers::device<double> lambda_x(size * n_x, res.device());
    for (size_t x = 0; x < n_x; ++x) {
      FBF_TRY(copy(lambda_x.data() + (x * size), casscf.cp.orbital.data() + (x * pairs), pairs,
                   wwr::wwrMemcpyDeviceToDevice));
      FBF_TRY(copy(lambda_x.data() + (x * size) + pairs, casscf.cp.ci.data() + (x * ndet), ndet,
                   wwr::wwrMemcpyDeviceToDevice));
    }
    FBF_TRY(res.sync());
    z_terms.emplace(FBF_TRY(df_casscf_z(res, molecule, basis, classes, aux, h, b, l, solution,
                                        ncore, ncas, z, n_x32, lambda_x.data())));
    Buffers::device<double> out(n_x * n_x, res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_x32, n_x32, size32, &kOne<double>,
                                               z_terms->lxlambda.data(), lambda_x.data(),
                                               &kZero<double>, out.data())));
    lx_cart = FBF_TRY(download(out.data(), n_x * n_x));
  }

  // 6. The skeleton at E_corr + Phi's adjoint.
  const std::vector<double> skeleton_cart = FBF_TRY(df_adjoint_xy(
      res, molecule, basis, classes, aux, h, b, l, solution.c, h_bar.data(), lambda.data()));

  // 7. The rotation cross terms and the Z terms' kappa_oo rows, from
  // per-coordinate nmo x nmo stacks: X(t_i), Y_i = X(O(I, kappa_i)) and
  // V_i = [X_N, kappa_nr_i], X(M) = 2 (M_h h_bar + sum_Q M_Q Lambda_Q) and
  // O(M, kappa) = kappa^T M + M kappa (README "The SC-NEVPT2 Hessian").
  std::vector<double> rotation_tau(k * k);
  std::vector<double> rotation_kappa(k * k);
  std::vector<double> z_koo(k * k);
  {
    Buffers::device<double> h_mo(nn, res.device());
    Buffers::device<double> l_mo(naux * nn, res.device());
    {
      Buffers::device<double> work(nn, res.device());
      FBF_TRY(gpu_status(gemmwich(blas, nmo32, nmo32, c, h.data(), work.data(), h_mo.data())));
      FBF_TRY(res.sync());
    }
    const OrbitalRange all{.first = 0, .count = nmo32};
    FBF_TRY(b_mo(res, naux32, nmo32, nmo32, b.data(), c, all, all, l_mo.data()));
    // out = 2 (m_h h_bar + sum_Q m_Q Lambda_Q), m_b read as the
    // (naux nmo) x nmo matrix of rows (Q, r).
    const auto x_of = [&](const double *m_h, const double *m_b, double *out) -> Status {
      FBF_TRY(gpu_status(gemm(blas, nmo32, &kTwo<double>, m_h, h_bar.data(), &kZero<double>, out)));
      return gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nmo32, nmo32, naux32 * nmo32, &kTwo<double>,
                                                m_b, lambda.data(), &kOne<double>, out));
    };
    Buffers::device<double> x_n(nn, res.device());
    FBF_TRY(x_of(h_mo.data(), l_mo.data(), x_n.data()));
    Buffers::device<double> x_t(nn * k, res.device());
    Buffers::device<double> y(nn * k, res.device());
    Buffers::device<double> v(nn * k, res.device());
    Buffers::device<double> work(nn, res.device());
    Buffers::device<double> kl(naux * nn, res.device());
    for (size_t i = 0; i < k; ++i) {
      const double *const kappa = t_kappa.data() + (i * nn);
      const double *const kappa_nr = t_nr.data() + (i * nn);
      double *const yi = y.data() + (i * nn);
      double *const vi = v.data() + (i * nn);
      FBF_TRY(x_of(t_h.data() + (i * nn), t_b.data() + (i * naux * nn), x_t.data() + (i * nn)));
      // Y_i = kappa^T X_N + 2 h kappa h_bar + 2 sum_Q B_Q kappa Lambda_Q,
      // (kappa Lambda_Q)_rs per slab s as Lambda's [Q, t] slab times kappa^T.
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nmo32, &kOne<double>, kappa, x_n.data(),
                                                 &kZero<double>, yi)));
      FBF_TRY(gpu_status(
          gemm(blas, nmo32, &kOne<double>, h_mo.data(), kappa, &kZero<double>, work.data())));
      FBF_TRY(gpu_status(
          gemm(blas, nmo32, &kTwo<double>, work.data(), h_bar.data(), &kOne<double>, yi)));
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, naux32, nmo32, nmo32, &kOne<double>,
          lambda.data(), naux32, static_cast<int64_t>(naux * nmo), kappa, nmo32, 0, &kZero<double>,
          kl.data(), naux32, static_cast<int64_t>(naux * nmo), nmo32)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nmo32, nmo32, naux32 * nmo32, &kTwo<double>,
                                                 l_mo.data(), kl.data(), &kOne<double>, yi)));
      // V_i = X_N kappa_nr - kappa_nr X_N.
      FBF_TRY(
          gpu_status(gemm(blas, nmo32, &kOne<double>, x_n.data(), kappa_nr, &kZero<double>, vi)));
      FBF_TRY(
          gpu_status(gemm(blas, nmo32, &kNegOne<double>, kappa_nr, x_n.data(), &kOne<double>, vi)));
    }
    // X(I^i) = X(t_i) - Y_i, and kappa_oo = kappa - kappa_nr.
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, k32 * nn32, &kNegOne<double>, y.data(), 1,
                                                  x_t.data(), 1)));
    Buffers::device<double> t_oo(nn * k, res.device());
    FBF_TRY(copy(t_oo.data(), t_kappa.data(), nn * k, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, k32 * nn32, &kNegOne<double>, t_nr.data(),
                                                  1, t_oo.data(), 1)));
    const std::vector<double> g_tau = FBF_TRY(gram(nn32, t_kappa.data(), x_t.data()));
    const std::vector<double> g_kappa = FBF_TRY(gram(nn32, t_kappa.data(), y.data()));
    const std::vector<double> g_oo = FBF_TRY(gram(nn32, t_oo.data(), v.data()));
    for (size_t j = 0; j < k; ++j) {
      for (size_t i = 0; i < k; ++i) {
        const size_t at = i + (k * j);
        const size_t ta = j + (k * i);
        rotation_tau[at] = g_tau[at] + g_tau[ta];
        rotation_kappa[at] = 0.5 * (g_kappa[at] + g_kappa[ta]);
        z_koo[at] = 0.5 * (g_oo[at] + g_oo[ta]);
      }
    }
  }

  // 8. The parts in the coordinates, summed and symmetrized.
  std::vector<double> e_cas = to_coordinates(casscf.hessian);
  std::vector<double> z_lxy = to_coordinates(z_terms->lxy.exy);
  std::vector<double> z_lx = to_coordinates(lx_cart);
  for (size_t j = 0; j < k; ++j) {
    for (size_t i = 0; i < j; ++i) {
      const double s = z_lx[i + (k * j)] + z_lx[j + (k * i)];
      z_lx[i + (k * j)] = s;
      z_lx[j + (k * i)] = s;
    }
    z_lx[j * (k + 1)] *= 2.0;
  }
  std::vector<double> z_third = to_coordinates(z_terms->third);
  std::vector<double> skeleton = to_coordinates(skeleton_cart);
  std::vector<double> total(k * k);
  double asymmetry = 0.0;
  const auto element = [&](const size_t at) {
    return e_cas[at] + z_lxy[at] + z_lx[at] + z_third[at] + z_koo[at] + skeleton[at] + hvp[at] +
           rotation_tau[at] + rotation_kappa[at] + degenerate_part[at];
  };
  for (size_t j = 0; j < k; ++j) {
    for (size_t i = 0; i < k; ++i) {
      const double upper = element(i + (k * j));
      const double lower = element(j + (k * i));
      asymmetry = std::max(asymmetry, std::abs(upper - lower));
      total[i + (k * j)] = 0.5 * (upper + lower);
    }
  }
  // The translational sum rule, on a full set of coordinates only: the
  // Cartesian Hessian Q^T H Q against each rigid translation.
  double translation = std::numeric_limits<double>::quiet_NaN();
  if (k == n_x) {
    translation = 0.0;
    for (size_t axis = 0; axis < 3; ++axis) {
      std::vector<double> qt_axis(k, 0.0); // Q T_axis
      for (size_t i = 0; i < k; ++i) {
        for (size_t a = 0; a < n_x / 3; ++a) {
          qt_axis[i] += q[(i * n_x) + (3 * a) + axis];
        }
      }
      std::vector<double> hq(k, 0.0); // H_q Q T_axis
      for (size_t j = 0; j < k; ++j) {
        for (size_t i = 0; i < k; ++i) {
          hq[i] += total[i + (k * j)] * qt_axis[j];
        }
      }
      for (size_t x = 0; x < n_x; ++x) { // Q^T of it
        double s = 0.0;
        for (size_t i = 0; i < k; ++i) {
          s += q[(i * n_x) + x] * hq[i];
        }
        translation = std::max(translation, std::abs(s));
      }
    }
  }
  return PcNevpt2Terms{.hessian = std::move(total),
                       .asymmetry = asymmetry,
                       .translation = translation,
                       .e_cas = std::move(e_cas),
                       .z_lxy = std::move(z_lxy),
                       .z_lx = std::move(z_lx),
                       .z_third = std::move(z_third),
                       .z_koo = std::move(z_koo),
                       .skeleton = std::move(skeleton),
                       .hvp = std::move(hvp),
                       .rotation_tau = std::move(rotation_tau),
                       .rotation_kappa = std::move(rotation_kappa),
                       .degenerate = std::move(degenerate_part),
                       .casscf = std::move(casscf),
                       .e_corr = e_corr,
                       .kept = kept,
                       .max_zc = gradient.zc.max_z,
                       .min_gap = gradient.zc.min_gap,
                       .zvector_residual = gradient.multipliers.residual,
                       .lift_steps = std::move(lift_steps),
                       .degenerate_pairs = std::move(blocks),
                       .rotation_shift = rotation_shift};
}

MemoryFootprint df_pc_nevpt2_footprint(const BasisSet &basis, const BasisSet &aux,
                                       const int32_t ncore, const int32_t ncas,
                                       const int32_t nalpha, const int32_t nbeta, const int32_t k,
                                       const int32_t batch,
                                       const std::span<const size_t> syevd_lwork,
                                       const std::span<const std::array<int64_t, 8>> dropped) {
  const size_t nao = basis.nao_sphe();
  if (basis.natoms() == 0 || aux.natoms() != basis.natoms() || aux.nao_sphe() == 0 || ncore < 0 ||
      ncas < 1 || static_cast<size_t>(ncore) + static_cast<size_t>(ncas) > nao || nalpha < 0 ||
      nbeta < 0 || nalpha > ncas || nbeta > ncas || k < 1 ||
      syevd_lwork.size() <= static_cast<size_t>(ncas)) {
    return {};
  }
  const auto nmo = static_cast<int32_t>(nao);
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const MemoryFootprint gradient = without_strings(
      nevpt2::pc_gradient_footprint(basis, aux, ncore, ncas, nalpha + nbeta, nalpha - nbeta + 1,
                                    batch, syevd_lwork, 0, dropped),
      ncas, nalpha, nbeta);
  return nevpt2_hessian_footprint(
      basis, aux, ncore, ncas, nalpha, nbeta, k, syevd_lwork, gradient,
      [&](const int32_t count, const int32_t /*partners*/) {
        return nevpt2::pc_hessian_product_footprint(naux, nmo, nmo, ncore, ncas, nalpha, nbeta,
                                                    batch, syevd_lwork, count, false, dropped);
      },
      false, false, true);
}

} // namespace free_breakfast::hessian
