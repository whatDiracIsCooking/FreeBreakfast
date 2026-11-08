/// @file
/// @brief df_qd_sc_nevpt2 and df_qd_pc_nevpt2: the SC- and PC-QD-NEVPT2
/// analytic nuclear Hessians on a DF-SA-CASSCF (docs/nevpt2/nevpt2.tex Eqs.
/// qd-hess and qd-hessW, Algorithm qdhess), one assembly over each method's
/// pieces, as df_sc_nevpt2 is, part for part, from merged ones: E_rho's at the
/// rho-weighted RDMs; the multipliers' z terms (df_sacasscf_z's body with
/// ThirdDerivative's action, every rotation among the roots a parameter); the
/// zeta term (df_qd_zeta); E^corr[W] + Phi^SA as df_adjoint_xy at its
/// adjoint, the Hessian-vector product between the tangents t_x of
/// df_qd_nevpt2_tangents and the rotation cross terms; the kappa_oo rows in
/// closed form; and for a state, term (ii) (df_qd_eigenvector_response).
/// Every gap of Table qd-divisions is checked first and returned. SC's
/// Lagrangian is nevpt2::qd_sc_lagrangian's and its product
/// qd_sc_hessian_product's; PC's are qd_pc_gradient's and
/// qd_pc_hessian_product's, every one at the same pc_floor, under
/// df_pc_nevpt2's refusals on every model state.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kTwo, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, axpy, gemm, gemv, gemmStridedBatched, wwrMemcpyAsync,
                                // wwrMemsetAsync, wwrMemGetInfo, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.active_space;           // casscf_integrals
import free_breakfast.integrals.eri.density_fitting.mo; // b_mo, OrbitalRange
import free_breakfast.recipes.blas;                     // gemm, gemmwich
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.ci.solver; // solve: the ensemble's edge
import free_breakfast.response.fock;
import free_breakfast.response.hessian;
import free_breakfast.response.orbital_pairs; // pack_orbital_pairs, unpack_orbital_pairs
import free_breakfast.response.relaxed_hessian;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.derivative;
import free_breakfast.mcscf.gradient;      // sa_gradient_weights
import free_breakfast.mcscf.root_rotation; // kRootGapFloor
import free_breakfast.response.zvector;
import free_breakfast.nevpt2.density_fitting; // MoBlocks
import free_breakfast.nevpt2.energy;   // sc_effective_hamiltonian, ScAmplitudes, PcClassEnergies
import free_breakfast.nevpt2.gradient; // qd_inputs, qd_sc_lagrangian, qd_sc_hessian_product,
                                       // qd_pc_energy, qd_pc_gradient, qd_pc_hessian_product
import :state_specific;                // state_averaged_electrons, qd_response_eigen

#include "diagnostics.h"

namespace free_breakfast::hessian {

// The coordinates' rows are orthonormal to this (df_sc_nevpt2's).
constexpr double kQdNevpt2OrthonormalTolerance = 1e-10;
// df_sc_nevpt2's kScNevpt2MixStep: a coordinate that mixes a canonical
// inactive pair within this (bohr) is refused (SC).
constexpr double kQdScNevpt2MixStep = 1e-3;
// A coordinate whose total tangent is below this fraction of the largest
// over the coordinates moves nothing: a rigid translation or rotation, its
// tangent the CP solve's residual (1e-10) and round-off (SC).
constexpr double kQdScNevpt2RigidTangent = 1e-6;
// W symmetric, and rho = diag W a mixture, to this.
constexpr double kQdNevpt2WeightTolerance = 1e-12;
// df_pc_nevpt2's kPcNevpt2Kept: a class's smallest kept metric eigenvalue,
// over its largest, below which its second derivative is not attainable
// (#917), on any model state (PC).
constexpr double kQdPcNevpt2Kept = 1e-10;
// df_pc_nevpt2's kPcNevpt2LiftStep: the step (bohr along a coordinate)
// within which no model state's Schur lift may reach the cut (#817's
// convention; PC).
constexpr double kQdPcNevpt2LiftStep = 1e-3;
// PC's Lagrangian's Z-vector is solved to this, SC's own (qd_sc_lagrangian
// takes the nevpt2 module's 1e-10).
constexpr double kQdNevpt2ZvectorTolerance = 1e-10;
// PySCF's class order, PcClassEnergies'.
constexpr std::array<std::string_view, 8> kQdPcNevpt2Classes{"Sr",   "Si",  "Sijrs", "Sijr",
                                                             "Srsi", "Srs", "Sij",   "Sir"};

// H_eff's correlation elements over qd_inputs, the smallest kept
// |Delta_t + H_t / N_t| over every state (1 / max |c_t|, the amplitudes'),
// and H_eff's eigenpairs (qd_response_eigen).
struct QdScForward {
  std::vector<double> eigenvalues;
  std::vector<double> eigenvectors;
  double min_denominator = std::numeric_limits<double>::infinity();
};

Result<QdScForward> qd_sc_forward(DeviceResources &res, const Molecule &molecule,
                                  const BasisSet &basis, const BasisSet &aux,
                                  const SphericalMatrixAO &h, const Buffers::device<double> &b,
                                  const mcscf::Solution &solution, const int32_t ncore,
                                  const int32_t ncas, const int32_t batch) {
  std::vector<double> totals;
  double largest = 0.0;
  FBF_TRY(nevpt2::qd_inputs(
      res, molecule, basis, aux, h, b, solution, ncore, ncas,
      [&](const nevpt2::MoBlocks &blocks,
          const std::span<const nevpt2::EnergyInputs> pairs) -> Status {
        nevpt2::ScAmplitudes amplitudes;
        const std::vector<nevpt2::ClassEnergies> e =
            FBF_TRY(nevpt2::sc_effective_hamiltonian(res, blocks, pairs, batch, &amplitudes));
        std::ranges::transform(e, std::back_inserter(totals), &nevpt2::ClassEnergies::total);
        for (const auto *klass :
             {&amplitudes.sr, &amplitudes.si, &amplitudes.sijr, &amplitudes.srsi, &amplitudes.srs,
              &amplitudes.sij, &amplitudes.sir}) {
          for (const std::vector<double> &state : *klass) {
            for (const double c : state) {
              largest = std::max(largest, std::abs(c));
            }
          }
        }
        return {};
      }));
  auto [values, vectors] = FBF_TRY(qd_response_eigen(res, solution.roots.energies, totals));
  return QdScForward{.eigenvalues = std::move(values),
                     .eigenvectors = std::move(vectors),
                     .min_denominator =
                         largest > 0.0 ? 1.0 / largest : std::numeric_limits<double>::infinity()};
}

// The ensemble's edge: E_{n+1} - E_n, the n + 1-th root of the active space
// at the solution's orbitals over its n-th; infinite when there is none.
Result<double> qd_sc_edge_gap(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                              const SphericalMatrixAO &h, const Buffers::device<double> &b,
                              const int32_t naux, const mcscf::Solution &solution,
                              const int32_t ncore, const int32_t ncas,
                              const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta) {
  const size_t nroots = solution.roots.energies.size();
  const size_t dim = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  if (nroots + 1 > dim) {
    return std::numeric_limits<double>::infinity();
  }
  const auto n_act = static_cast<size_t>(ncas);
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux, solution.c, ncore, ncas));
  std::vector<double> h_eff(n_act * n_act);
  std::vector<double> eri(n_act * n_act * n_act * n_act);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h_eff.data(), ints.hamiltonian.h_eff.data(),
                                         h_eff.size() * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                         res.stream())));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(eri.data(), ints.hamiltonian.eri.data(),
                                         eri.size() * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                         res.stream())));
  FBF_TRY(res.sync());
  // Both are symmetric as their column-major layout is read row-major.
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  FBF_TRY(sigma.set_integrals(res, ints.hamiltonian.e_core, h_eff, eri));
  Buffers::device<double> diagonal(dim, res.device());
  FBF_TRY(sigma.diagonal(res, diagonal.data()));
  const ci::Roots more = FBF_TRY(
      ci::solve(res, sigma, alpha, beta, diagonal.data(), static_cast<int32_t>(nroots + 1)));
  return more.energies[nroots] - solution.roots.energies[nroots - 1];
}

// The body of all four forms: term (i) at W (@p state's W^P when given, else
// @p w_given), then for a state term (ii); SC's when @p pc_floor is empty,
// else PC's at that floor (its reports beside the terms).
Result<QdPcNevpt2Terms>
qd_nevpt2_terms(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                const std::optional<int32_t> state, const std::span<const double> w_given,
                const int32_t batch, const std::span<const double> coordinates,
                const int32_t nfrozen, const std::optional<double> pc_floor) {
  const std::string_view kWho = pc_floor ? "hessian::df_qd_pc_nevpt2" : "hessian::df_qd_sc_nevpt2";
  const bool pc = pc_floor.has_value();
  const double floor = pc_floor.value_or(0.0);
  if (nfrozen != 0) {
    return err_unsupported("{}: the analytic QD Hessian is all-electron, and {} core orbitals are "
                           "frozen",
                           kWho, nfrozen);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const std::vector<double> weights =
      FBF_TRY(mcscf::sa_gradient_weights(solution, static_cast<size_t>(ncas)));
  // The CASSCF root gaps refused here, before anything runs.
  const auto [nalpha, nbeta] = FBF_TRY(
      state_averaged_electrons(kWho, res, molecule, basis, aux, b, solution, weights, ncore, ncas));
  const size_t nroots = weights.size();
  const size_t n_x = static_cast<size_t>(3) * basis.natoms();
  const size_t nmo = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const size_t nn = nmo * nmo;
  const size_t nc = static_cast<size_t>(ncore);
  const size_t n_act = static_cast<size_t>(ncas);
  const size_t pairs = (nc * (nmo - nc)) + (n_act * (nmo - nc - n_act));
  const size_t all_pairs = nroots * (nroots - 1) / 2;
  if (state && (*state < 0 || std::cmp_greater_equal(*state, nroots))) {
    return err_config("{}: state {} is not one of the {} QD states", kWho, *state, nroots);
  }
  if (!state) {
    if (w_given.size() != nroots * nroots) {
      return err_config("{}: W of {} doubles for {} roots", kWho, w_given.size(), nroots);
    }
    double trace = 0.0;
    for (size_t r = 0; r < nroots; ++r) {
      for (size_t s = 0; s < r; ++s) {
        if (!(std::abs(w_given[(r * nroots) + s] - w_given[(s * nroots) + r]) <=
              kQdNevpt2WeightTolerance)) {
          return err_config("{}: W is not symmetric at ({}, {})", kWho, r, s);
        }
      }
      const double rho = w_given[r * (nroots + 1)];
      if (!(rho >= -kQdNevpt2WeightTolerance)) {
        return err_config("{}: W's diagonal is no mixture: rho_{} = {:g}", kWho, r, rho);
      }
      trace += rho;
    }
    if (!(std::abs(trace - 1.0) <= kQdNevpt2WeightTolerance)) {
      return err_config("{}: W's diagonal is no mixture: it sums to {:.15g}", kWho, trace);
    }
  }

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
      if (!(std::abs(s - (i == j ? 1.0 : 0.0)) <= kQdNevpt2OrthonormalTolerance)) {
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
  const wwr::wwrblasHandle_t blas = res.blas().get();

  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  const auto ndet = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  const size_t n_ci = ndet * nroots;
  const auto n_ci32 = static_cast<int32_t>(n_ci);
  if (solution.roots.vectors.num_elements() != n_ci) {
    return err_config("{}: {} roots of {} doubles over {} determinants", kWho, nroots,
                      solution.roots.vectors.num_elements(), ndet);
  }
  // x's layout, ThirdDerivative's with every rotation: the packed pairs, d_r
  // per root, Theta_sr for every r < s.
  const size_t size = pairs + n_ci + all_pairs;
  const auto size32 = static_cast<int32_t>(size);

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
  // P + P^T of a k x k host matrix.
  const auto both_orders = [k](std::vector<double> p) {
    for (size_t j = 0; j < k; ++j) {
      for (size_t i = 0; i < j; ++i) {
        const double s = p[i + (k * j)] + p[j + (k * i)];
        p[i + (k * j)] = s;
        p[j + (k * i)] = s;
      }
      p[j * (k + 1)] *= 2.0;
    }
    return p;
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
  Buffers::device<double> qt(n_x * k, res.device()); // Q^T, n_x x k column-major
  FBF_TRY(copy(qt.data(), q.data(), n_x * k, wwr::wwrMemcpyHostToDevice));
  // S_coords += S_chunk Q^T[x0 .. x0 + count), each stack of @p rows.
  const auto accumulate = [&](const int32_t rows, const int32_t x0, const int32_t count,
                              const double *chunk, double *stack) -> Status {
    return gpu_status(wwr::gemm<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, rows, k32, count, &kOne<double>, chunk, rows,
        qt.data() + x0, n_x32, &kOne<double>, stack, rows));
  };

  // 1. The gaps refused before anything runs: the ensemble's edge, and H_eff
  // (its eigenpairs; SC's smallest denominator, PC's kept classes, a class
  // kept below kQdPcNevpt2Kept refused) with the QD gap.
  QdPcNevpt2Terms result;
  QdScNevpt2Terms &out = result.terms;
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = r + 1; s < nroots; ++s) {
      out.root_gap =
          std::min(out.root_gap, std::abs(solution.roots.energies[r] - solution.roots.energies[s]));
    }
  }
  out.edge_gap = FBF_TRY(
      qd_sc_edge_gap(res, molecule, basis, h, b, naux32, solution, ncore, ncas, alpha, beta));
  if (!(out.edge_gap >= mcscf::kRootGapFloor)) {
    return err_numerical("{}: the ensemble's edge E_{} - E_{} is {:.3e} Eh, under {:g}: the first "
                         "excluded root enters the average there",
                         kWho, nroots, nroots - 1, out.edge_gap, mcscf::kRootGapFloor);
  }
  {
    QdScForward forward;
    if (pc) {
      const std::vector<nevpt2::PcClassEnergies> elements = FBF_TRY(nevpt2::qd_pc_energy(
          res, alpha, beta, naux32, nmo32, nmo32, b.data(), solution.c.data(), h.data(), ncore,
          ncas, weights, solution.roots.vectors.data(), batch, 0, floor));
      std::vector<double> totals;
      std::ranges::transform(elements, std::back_inserter(totals), &nevpt2::PcClassEnergies::total);
      // Ket N's conditioning is the same at every [M n + N]: its diagonal's.
      std::string thin;
      for (size_t n = 0; n < nroots; ++n) {
        const std::array<double, 8> &kept = elements[n * (nroots + 1)].kept;
        result.kept.push_back(kept);
        for (size_t cls = 0; cls < kept.size(); ++cls) {
          // kept is 0 for Sijrs and a class with no tuple: no solve.
          if (kept[cls] > 0.0 && kept[cls] < kQdPcNevpt2Kept) {
            thin += std::format("{}state {}'s {} keeps {:.2e}", thin.empty() ? "" : ", ", n,
                                kQdPcNevpt2Classes[cls], kept[cls]);
          }
        }
      }
      if (!thin.empty()) {
        return err_numerical("{}: {} of its metric's largest eigenvalue at a pc_floor of {:g}, "
                             "below {:g}, where PC's second derivative is not attainable (#917); "
                             "a pc_floor of {:g} or more cuts it",
                             kWho, thin, floor, kQdPcNevpt2Kept, kQdPcNevpt2Kept);
      }
      auto [values, vectors] = FBF_TRY(qd_response_eigen(res, solution.roots.energies, totals));
      forward.eigenvalues = std::move(values);
      forward.eigenvectors = std::move(vectors);
    } else {
      forward =
          FBF_TRY(qd_sc_forward(res, molecule, basis, aux, h, b, solution, ncore, ncas, batch));
      out.min_denominator = forward.min_denominator;
    }
    out.eigenvalues = std::move(forward.eigenvalues);
    if (state) {
      const auto p = static_cast<size_t>(*state);
      for (size_t other = 0; other < nroots; ++other) {
        if (other == p) {
          continue;
        }
        const double gap = std::abs(out.eigenvalues[other] - out.eigenvalues[p]);
        if (!(gap >= mcscf::kRootGapFloor)) {
          return err_numerical("{}: QD states {} and {} are {:.3e} Eh apart, under {:g}: term (ii) "
                               "divides by their gap, and a degenerate state has no Hessian of "
                               "this form",
                               kWho, std::min(p, other), std::max(p, other), gap,
                               mcscf::kRootGapFloor);
        }
        out.qd_gap = std::min(out.qd_gap, gap);
      }
      const double *const u = forward.eigenvectors.data() + (p * nroots);
      out.w.resize(nroots * nroots);
      for (size_t i = 0; i < nroots; ++i) {
        for (size_t j = 0; j < nroots; ++j) {
          out.w[(i * nroots) + j] = u[i] * u[j];
        }
      }
    } else {
      out.w.assign(w_given.begin(), w_given.end());
    }
  }

  // 2. x_A: lambda^x of the CP-SA-CASSCF (kappa_nr, xi_r, the unequal pairs'
  // Theta), then the tangents (the equal pairs' theta, Eq. qd-theta, and
  // kappa_oo at f^SA, the degenerate-pair rule), t_i in the coordinates.
  const SaCasscfCp cp = FBF_TRY(
      df_sacasscf_cp(res, molecule, basis, classes, aux, h, b, l, solution, weights, ncore, ncas));
  out.cp_residual = std::ranges::max(cp.residuals);
  Buffers::device<double> t_h = FBF_TRY(zeroed(nn * k));
  Buffers::device<double> t_b = FBF_TRY(zeroed(naux * nn * k));
  Buffers::device<double> t_c = FBF_TRY(zeroed(n_ci * k));
  Buffers::device<double> t_kappa = FBF_TRY(zeroed(nn * k));
  const QdNevpt2Tangents tangents = FBF_TRY(df_qd_nevpt2_tangents(
      res, molecule, basis, classes, aux, h, b, l, solution, weights, ncore, ncas, cp.cp, pc,
      [&](const int32_t x0, const int32_t count, const double *h_dot, const double *b_dot,
          const double *c_dot, const double *kappa) -> Status {
        FBF_TRY(accumulate(nn32, x0, count, h_dot, t_h.data()));
        FBF_TRY(accumulate(bn32, x0, count, b_dot, t_b.data()));
        FBF_TRY(accumulate(n_ci32, x0, count, c_dot, t_c.data()));
        return accumulate(nn32, x0, count, kappa, t_kappa.data());
      }));
  out.max_theta = tangents.max_theta;
  out.max_kappa_oo = tangents.max_kappa_oo;
  // PC's degenerate pair the displacement splits (SC's the tangents refuse):
  // kappa_oo^x_pq is not a division the state determines, and the canonical
  // orbitals PC reads are not differentiable there (df_pc_nevpt2's refusal).
  if (!tangents.recorded.empty()) {
    std::string named;
    for (const DegeneratePair &d : tangents.recorded) {
      named += std::format("{}(coordinate {}, orbitals {} and {}, gap {:.2e}, f {:.2e})",
                           named.empty() ? "" : ", ", d.coordinate, d.p, d.q, d.gap, d.f);
    }
    return err_numerical("{}: the canonical inactive orbitals are not differentiable at a "
                         "degenerate pair the displacement splits: {}",
                         kWho, named);
  }
  Buffers::device<double> t_nr = FBF_TRY(zeroed(nn * k));
  // x_A in the coordinates, size x k column-major: lambda_i = sum_x Q_ix x_x.
  Buffers::device<double> lambda_q = FBF_TRY(zeroed(size * k));
  {
    Buffers::device<double> kappa_nr(nn * n_x, res.device());
    Buffers::device<double> lambda_x(size * n_x, res.device());
    for (size_t x = 0; x < n_x; ++x) {
      FBF_TRY(response::unpack_orbital_pairs(
          res, nmo32, ncore, ncas, cp.cp.orbital.data() + (x * pairs), kappa_nr.data() + (x * nn)));
      double *const column = lambda_x.data() + (x * size);
      FBF_TRY(
          copy(column, cp.cp.orbital.data() + (x * pairs), pairs, wwr::wwrMemcpyDeviceToDevice));
      FBF_TRY(
          copy(column + pairs, cp.cp.ci.data() + (x * n_ci), n_ci, wwr::wwrMemcpyDeviceToDevice));
      FBF_TRY(copy(column + pairs + n_ci, tangents.theta.data() + (x * all_pairs), all_pairs,
                   wwr::wwrMemcpyHostToDevice));
    }
    FBF_TRY(accumulate(nn32, 0, n_x32, kappa_nr.data(), t_nr.data()));
    FBF_TRY(accumulate(size32, 0, n_x32, lambda_x.data(), lambda_q.data()));
    FBF_TRY(res.sync());
  }
  // A covector stack C (size x n_x, Cartesian) against x_B in the
  // coordinates: (i, j) = <(C Q^T)_i, lambda_j>.
  const auto against_x = [&](const double *covectors) -> Result<std::vector<double>> {
    Buffers::device<double> cq = FBF_TRY(zeroed(size * k));
    FBF_TRY(accumulate(size32, 0, n_x32, covectors, cq.data()));
    return gram(size32, cq.data(), lambda_q.data());
  };

  // 3. The Lagrangian at W: E^corr[W] + Phi^SA's adjoint (h_bar, Lambda), z^c,
  // zeta and the multipliers (Z, z, Theta), PC's from qd_pc_gradient's W form
  // at the floor (its derivative passes unread). Then, for SC, the inactive
  // pairs' mixing (df_sc_nevpt2's t_mix rule, every coordinate named): PC is
  // invariant to that rotation.
  nevpt2::QdScLagrangian lagrangian = FBF_TRY([&]() -> Result<nevpt2::QdScLagrangian> {
    if (pc) {
      nevpt2::QdPcGradient g = FBF_TRY(
          nevpt2::qd_pc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                                 out.w, batch, kQdNevpt2ZvectorTolerance, 0, std::nullopt, floor));
      return std::move(g.lagrangian);
    }
    return nevpt2::qd_sc_lagrangian(res, molecule, basis, aux, h, b, solution, ncore, ncas, out.w,
                                    batch);
  }());
  out.rho = lagrangian.rho;
  out.min_gap = lagrangian.zc.min_gap;
  out.max_zc = lagrangian.zc.max_z;
  out.zvector_residual = lagrangian.multipliers.residual;
  for (const double zeta : lagrangian.pairs.zeta) {
    out.max_zeta = std::max(out.max_zeta, std::abs(zeta));
  }
  if (!pc) {
    const std::vector<double> epsilon = FBF_TRY(download(lagrangian.corr.epsilon.data(), nmo));
    const std::vector<double> kappa = FBF_TRY(download(t_kappa.data(), nn * k));
    const std::vector<double> kappa_nr = FBF_TRY(download(t_nr.data(), nn * k));
    // 0 core, 1 active, 2 virtual.
    const auto space = [&](const size_t p) -> int32_t {
      return p < nc ? 0 : (p < nc + n_act ? 1 : 2);
    };
    std::string named;
    std::string why;
    for (size_t i = 0; i < k; ++i) {
      double worst = 0.0;
      size_t wp = 0;
      size_t wq = 0;
      for (size_t qq = 0; qq < nmo; ++qq) {
        for (size_t p = 0; p < qq; ++p) {
          if (space(p) != space(qq) || space(p) == 1) {
            continue;
          }
          const size_t at = (i * nn) + p + (nmo * qq);
          const double oo = std::abs(kappa[at] - kappa_nr[at]);
          if (oo > worst) {
            worst = oo;
            wp = p;
            wq = qq;
          }
        }
      }
      const double t_mix = 0.5 / worst;
      out.mix_step = std::min(out.mix_step, t_mix);
      if (t_mix < kQdScNevpt2MixStep) {
        named += std::format("{}{}", named.empty() ? "" : ", ", i);
        why += std::format("; coordinate {}: orbitals ({}, {}), gap {:.3e} Eh, t_mix {:.3e} bohr",
                           i, wp, wq, std::abs(epsilon[wp] - epsilon[wq]), t_mix);
      }
    }
    if (!named.empty()) {
      return err_numerical(
          "{}: SC-QD-NEVPT2 has no Hessian along coordinates {}: each mixes a canonical inactive "
          "pair within {:g} bohr (t_mix = gap / 2 |F^i_pq|), and SC is not invariant to that "
          "rotation, so its surface switches branch inside a vibrational or finite-difference "
          "step and one analytic Hessian does not represent it; use PC-QD-NEVPT2{}",
          kWho, named, kQdScNevpt2MixStep, why);
    }
  }
  const double *const c = solution.c.data();
  const double *const vectors = solution.roots.vectors.data();
  const Buffers::device<double> &h_bar = lagrangian.corr.h_bar;
  const Buffers::device<double> &lambda = lagrangian.corr.lambda;

  // 4. Sum <t_i, HVP t_j>: the product along every coordinate, then each pair
  // of coordinates that both populate an empty tuple of some model state
  // again, each as the other's partner (K6 at the pair's mean eta, and the
  // per-block refusal on every ket). Chunked by an estimate of the product's
  // per-direction memory.
  out.hvp.assign(k * k, 0.0);
  std::vector<std::string> refused;
  {
    Buffers::device<double> paired(k * k, res.device());
    // Column j of @p into (k long) = <t_i, P> over every i.
    const auto pair_with = [&](const nevpt2::AdjointTangent &p, double *into) -> Status {
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, nn32, k32,
                                                    &kOne<double>, t_h.data(), nn32, p.h_bar.data(),
                                                    1, &kZero<double>, into, 1)));
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, bn32, k32,
                                                    &kOne<double>, t_b.data(), bn32,
                                                    p.lambda.data(), 1, &kOne<double>, into, 1)));
      return gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, n_ci32, k32,
                                                   &kOne<double>, t_c.data(), n_ci32, p.y.data(), 1,
                                                   &kOne<double>, into, 1));
    };
    const auto product = [&](const int32_t count, const double *dh, const double *db,
                             const double *dc, const nevpt2::MoDirections &partners) {
      return nevpt2::qd_sc_hessian_product(res, alpha, beta, naux32, nmo32, nmo32, b.data(), c,
                                           h.data(), ncore, ncas, weights, vectors, out.w,
                                           lagrangian.zc.z.data(), batch,
                                           {.count = count, .h = dh, .b = db, .c = dc}, partners);
    };
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
    const size_t act6 = n_act * n_act * n_act * n_act * n_act * n_act;
    const size_t per_direction =
        sizeof(double) * ((16 * naux * nn) + (8 * nroots * nroots * act6) + (8 * n_ci) + (16 * nn));
    const size_t chunk = std::clamp<size_t>(free_bytes / 2 / per_direction, 1, k);
    if (pc) {
      // PC has no empty tuples and its product no partners: the product along
      // every coordinate at the floor, with each direction's Schur lift on
      // every model state. A chunk the product refuses (a ket whose class's
      // null modes reach another model state at first order along a
      // direction, the block (i, i)) is taken a direction at a time, each
      // refused one named. Block (i, j) needs that overlap along both, so
      // only pairs of such coordinates are taken again, j as i's partner.
      const auto pc_product = [&](const size_t j0, const size_t count,
                                  const nevpt2::MoDirections &partners) {
        return nevpt2::qd_pc_hessian_product(res, alpha, beta, naux32, nmo32, nmo32, b.data(), c,
                                             h.data(), ncore, ncas, weights, vectors, out.w,
                                             lagrangian.zc.z.data(), batch,
                                             {.count = static_cast<int32_t>(count),
                                              .h = t_h.data() + (j0 * nn),
                                              .b = t_b.data() + (j0 * naux * nn),
                                              .c = t_c.data() + (j0 * n_ci)},
                                             partners, false, floor);
      };
      result.lift_steps.assign(k, {});
      const auto take = [&](const size_t j0, const nevpt2::QdPcHessianProduct &p) -> Status {
        for (size_t u = 0; u < p.products.size(); ++u) {
          FBF_TRY(pair_with(p.products[u], paired.data() + ((j0 + u) * k)));
          result.lift_steps[j0 + u] = p.lift_steps[u];
        }
        return res.sync();
      };
      std::vector<size_t> own_refused;
      std::vector<std::pair<std::pair<size_t, size_t>, std::string>> blocks;
      for (size_t j0 = 0; j0 < k; j0 += chunk) {
        const size_t count = std::min(chunk, k - j0);
        auto p = pc_product(j0, count, {});
        if (p.has_value()) {
          FBF_TRY(take(j0, *p));
          continue;
        }
        if (p.error().kind != ErrorKind::Numerical) {
          return std::unexpected(std::move(p.error()));
        }
        for (size_t u = 0; u < count; ++u) {
          auto one = count == 1 ? std::move(p) : pc_product(j0 + u, 1, {});
          if (one.has_value()) {
            FBF_TRY(take(j0 + u, *one));
            continue;
          }
          if (one.error().kind != ErrorKind::Numerical) {
            return std::unexpected(std::move(one.error()));
          }
          own_refused.push_back(j0 + u);
          blocks.push_back({{j0 + u, j0 + u}, std::move(one.error().message)});
        }
      }
      out.hvp = FBF_TRY(download(paired.data(), k * k));
      for (size_t a = 0; a < own_refused.size(); ++a) {
        for (size_t bb = a + 1; bb < own_refused.size(); ++bb) {
          const size_t j = own_refused[bb];
          auto p = pc_product(own_refused[a], 1,
                              {.count = 1,
                               .h = t_h.data() + (j * nn),
                               .b = t_b.data() + (j * naux * nn),
                               .c = t_c.data() + (j * n_ci)});
          if (p.has_value()) {
            continue;
          }
          if (p.error().kind != ErrorKind::Numerical) {
            return std::unexpected(std::move(p.error()));
          }
          blocks.push_back({{own_refused[a], j}, std::move(p.error().message)});
        }
      }
      std::ranges::sort(blocks, {}, [](const auto &block) { return block.first; });
      // df_pc_nevpt2's Schur-lift rule on every model state, along each
      // coordinate the product took: a class's dropped mode that reaches the
      // cut within kQdPcNevpt2LiftStep with no overlap to another state.
      std::string branch;
      for (size_t i = 0; i < k; ++i) {
        for (size_t n = 0; n < result.lift_steps[i].size(); ++n) {
          for (size_t cls = 0; cls < kQdPcNevpt2Classes.size(); ++cls) {
            const double step = result.lift_steps[i][n][cls];
            if (step < kQdPcNevpt2LiftStep) {
              branch +=
                  std::format("{}coordinate {} state {} class {} t* {:.3e} bohr",
                              branch.empty() ? "" : ", ", i, n, kQdPcNevpt2Classes[cls], step);
            }
          }
        }
      }
      std::string why;
      if (!blocks.empty()) {
        std::string named;
        std::string each;
        for (const auto &[at, message] : blocks) {
          named += std::format("{}({}, {})", named.empty() ? "" : ", ", at.first, at.second);
          each += std::format("; ({}, {}): {}", at.first, at.second, message);
        }
        why = std::format("PC-QD-NEVPT2 has no Hessian on the coordinate pairs {}: a model "
                          "state's cut metric modes lift along both with a first-order overlap "
                          "to another state, so their coupling misses a first-order piece "
                          "(Eq. qd-pcfail){}",
                          named, each);
      }
      if (!branch.empty()) {
        why += std::format("{}the PC surface switches branch within {:g} bohr (a Schur lift "
                           "reaches the cut, README \"The PC-NEVPT2 Hessian\"): {}",
                           why.empty() ? "" : "; and ", kQdPcNevpt2LiftStep, branch);
      }
      if (!why.empty()) {
        return err_numerical("{}: {}", kWho, why);
      }
    } else {
      std::vector<bool> breaking(k, false);
      for (size_t j0 = 0; j0 < k; j0 += chunk) {
        const size_t count = std::min(chunk, k - j0);
        const nevpt2::QdScHessianProduct p =
            FBF_TRY(product(static_cast<int32_t>(count), t_h.data() + (j0 * nn),
                            t_b.data() + (j0 * naux * nn), t_c.data() + (j0 * n_ci), {}));
        for (size_t u = 0; u < count; ++u) {
          FBF_TRY(pair_with(p.products[u], paired.data() + ((j0 + u) * k)));
          for (const std::array<int64_t, 8> &ket : p.populated[u]) {
            breaking[j0 + u] =
                breaking[j0 + u] || std::ranges::any_of(ket, [](const int64_t n) { return n > 0; });
          }
        }
        FBF_TRY(res.sync());
      }
      out.hvp = FBF_TRY(download(paired.data(), k * k));
      // A coordinate that moves nothing (a rigid translation or rotation, whose
      // tangent is the CP solve's residual and round-off) breaks no symmetry:
      // the product's population floor is against its own scale, and against
      // that round-off populates everything (df_sc_nevpt2's #952 rule, there
      // against the largest scale; here by the tangent's size).
      {
        std::vector<double> size_of(k, 0.0);
        for (size_t i = 0; i < k; ++i) {
          double norm_h = 0.0;
          double norm_b = 0.0;
          double norm_c = 0.0;
          FBF_TRY(gpu_status(
              wwr::nrm2<double, int32_t>(blas, nn32, t_h.data() + (i * nn), 1, &norm_h)));
          FBF_TRY(gpu_status(
              wwr::nrm2<double, int32_t>(blas, bn32, t_b.data() + (i * naux * nn), 1, &norm_b)));
          FBF_TRY(gpu_status(
              wwr::nrm2<double, int32_t>(blas, n_ci32, t_c.data() + (i * n_ci), 1, &norm_c)));
          size_of[i] = std::hypot(norm_h, norm_b, norm_c);
        }
        const double largest = std::ranges::max(size_of);
        for (size_t i = 0; i < k; ++i) {
          breaking[i] = breaking[i] && size_of[i] > kQdScNevpt2RigidTangent * largest;
        }
      }

      // The pairs: directions (t_j, t_i), partners (t_i, t_j).
      Buffers::device<double> two_h(2 * nn, res.device());
      Buffers::device<double> two_b(2 * naux * nn, res.device());
      Buffers::device<double> two_c(2 * n_ci, res.device());
      Buffers::device<double> swap_h(2 * nn, res.device());
      Buffers::device<double> swap_b(2 * naux * nn, res.device());
      Buffers::device<double> swap_c(2 * n_ci, res.device());
      for (size_t j = 0; j < k; ++j) {
        for (size_t i = 0; i < j; ++i) {
          if (!breaking[i] || !breaking[j]) {
            continue;
          }
          for (const auto &[stack, rows, two, swap] :
               {std::tuple{t_h.data(), nn, two_h.data(), swap_h.data()},
                std::tuple{t_b.data(), naux * nn, two_b.data(), swap_b.data()},
                std::tuple{t_c.data(), n_ci, two_c.data(), swap_c.data()}}) {
            FBF_TRY(copy(two, stack + (j * rows), rows, wwr::wwrMemcpyDeviceToDevice));
            FBF_TRY(copy(two + rows, stack + (i * rows), rows, wwr::wwrMemcpyDeviceToDevice));
            FBF_TRY(copy(swap, stack + (i * rows), rows, wwr::wwrMemcpyDeviceToDevice));
            FBF_TRY(copy(swap + rows, stack + (j * rows), rows, wwr::wwrMemcpyDeviceToDevice));
          }
          auto p =
              product(2, two_h.data(), two_b.data(), two_c.data(),
                      {.count = 2, .h = swap_h.data(), .b = swap_b.data(), .c = swap_c.data()});
          if (!p.has_value()) {
            if (p.error().kind != ErrorKind::Numerical) {
              return std::unexpected(std::move(p.error()));
            }
            refused.push_back(std::format("({}, {}): {}", i, j, p.error().message));
            continue;
          }
          out.defect = std::max({out.defect, p->defect[0], p->defect[1], p->coupling_defect[0],
                                 p->coupling_defect[1]});
          FBF_TRY(pair_with(p->products[0], paired.data()));
          FBF_TRY(pair_with(p->products[1], paired.data() + k));
          const std::vector<double> two_columns = FBF_TRY(download(paired.data(), 2 * k));
          out.hvp[i + (k * j)] = two_columns[i];
          out.hvp[j + (k * i)] = two_columns[k + j];
        }
      }
    }
  }
  if (!refused.empty()) {
    std::string named;
    std::string why;
    for (const std::string &r : refused) {
      named += std::format("{}{}", named.empty() ? "" : ", ", r.substr(0, r.find(':')));
      why += std::format("; {}", r);
    }
    return err_numerical("{}: SC-QD-NEVPT2 has no Hessian on the coordinate pairs {}: an empty "
                         "tuple's energy is not a quadratic form on the pair{}",
                         kWho, named, why);
  }

  // 5. E_rho's share at rho = diag W: E_xy at the rho-weighted RDMs (F^[x]
  // from df_derivatives at rho), the CP right-hand side at rho against x_B
  // in both orders, and x_A (d^2 E_rho / dx dx) x_B, the unreduced
  // CI-relaxed Hessian at rho (H at E_rho's Fock), its CI rows times
  // 2 rho_r and its rotation rows times 2 (rho_r - rho_s), back to the
  // energy's.
  const std::span<const double> rho = out.rho;
  for (const double r : rho) {
    if (!(r >= -kQdNevpt2WeightTolerance)) {
      return err_config("{}: rho = diag W is no mixture ({})", kWho, rho);
    }
  }
  std::vector<double> e_rho_xy;
  std::vector<double> e_rho_lx;
  std::vector<double> e_rho_xx;
  {
    ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
    std::vector<double> gamma(n_act * n_act);
    std::vector<double> gamma2(n_act * n_act * n_act * n_act);
    {
      Buffers::device<double> g1(gamma.size(), res.device());
      Buffers::device<double> g2(gamma2.size(), res.device());
      FBF_TRY(rdm.rdm12(res, rho, vectors, g1.data(), g2.data()));
      gamma = FBF_TRY(download(g1.data(), gamma.size()));
      gamma2 = FBF_TRY(download(g2.data(), gamma2.size()));
    }
    {
      const mcscf::Derivatives own =
          FBF_TRY(mcscf::df_derivatives(res, molecule, basis, classes, aux, h, b, l, solution.c,
                                        ncore, ncas, nalpha, nbeta, vectors, rho));
      Buffers::device<double> rhs(size * n_x, res.device());
      std::vector<double> theta(all_pairs);
      for (size_t x = 0; x < n_x; ++x) {
        double *const column = rhs.data() + (x * size);
        FBF_TRY(response::pack_orbital_pairs(res, nmo32, ncore, ncas,
                                             own.rhs_orbital.data() + (x * nn), column));
        FBF_TRY(copy(column + pairs, own.rhs_ci.data() + (x * n_ci), n_ci,
                     wwr::wwrMemcpyDeviceToDevice));
        size_t at = 0;
        for (size_t r = 0; r < nroots; ++r) {
          for (size_t s = r + 1; s < nroots; ++s) {
            theta[at++] = own.rhs_theta[s + (nroots * (r + (nroots * x)))];
          }
        }
        FBF_TRY(copy(column + pairs + n_ci, theta.data(), all_pairs, wwr::wwrMemcpyHostToDevice));
        FBF_TRY(res.sync());
      }
      e_rho_lx = both_orders(FBF_TRY(against_x(rhs.data())));
      Buffers::device<double> c_columns(nn, res.device());
      FBF_TRY(copy(c_columns.data(), c, nn, wwr::wwrMemcpyDeviceToDevice));
      const CasscfExy exy = FBF_TRY(df_casscf_exy(res, molecule, basis, classes, aux, h, b, l,
                                                  c_columns, ncore, ncas, gamma, gamma2, own.fock));
      e_rho_xy = to_coordinates(exy.exy);
    }
    const CasscfIntegrals ints =
        FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux32, solution.c, ncore, ncas));
    const response::Fock fock_rho = FBF_TRY(response::fock(
        res, molecule, h, b, naux32, solution.c, ncore, ncas, ints.ppaa, ints.papa, gamma, gamma2));
    ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
    FBF_TRY(sigma.set_integrals(
        res, ints.hamiltonian.e_core,
        FBF_TRY(download(ints.hamiltonian.h_eff.data(), n_act * n_act)),
        FBF_TRY(download(ints.hamiltonian.eri.data(), n_act * n_act * n_act * n_act))));
    response::Hessian hessian_rho = FBF_TRY(response::Hessian::create(
        res, h, b, naux32, solution.c, ncore, ncas, ints.ppaa, ints.papa, fock_rho));
    response::RelaxedHessian relaxed = FBF_TRY(response::RelaxedHessian::create(
        res, hessian_rho, sigma, rdm, alpha, beta, solution.roots, rho,
        response::ResponseIntegrals{nmo, nc, n_act,
                                    FBF_TRY(download(fock_rho.fock_inactive.data(), nn)),
                                    FBF_TRY(download(ints.ppaa.data(), nn * n_act * n_act)),
                                    FBF_TRY(download(ints.papa.data(), nn * n_act * n_act))}));
    Buffers::device<double> x_o(pairs * k, res.device());
    Buffers::device<double> x_c(n_ci * k, res.device());
    std::vector<double> theta(all_pairs * k);
    {
      const std::vector<double> host = FBF_TRY(download(lambda_q.data(), size * k));
      for (size_t j = 0; j < k; ++j) {
        FBF_TRY(copy(x_o.data() + (j * pairs), lambda_q.data() + (j * size), pairs,
                     wwr::wwrMemcpyDeviceToDevice));
        FBF_TRY(copy(x_c.data() + (j * n_ci), lambda_q.data() + (j * size) + pairs, n_ci,
                     wwr::wwrMemcpyDeviceToDevice));
        std::copy_n(host.begin() + static_cast<std::ptrdiff_t>((j * size) + pairs + n_ci),
                    all_pairs, theta.begin() + static_cast<std::ptrdiff_t>(j * all_pairs));
      }
    }
    Buffers::device<double> y_o(pairs * k, res.device());
    Buffers::device<double> y_c(n_ci * k, res.device());
    std::vector<double> y_theta(all_pairs * k);
    FBF_TRY(relaxed.apply_coupled(res, k32, x_o.data(), x_c.data(), y_o.data(), y_c.data(), theta,
                                  y_theta));
    Buffers::device<double> y(size * k, res.device());
    const auto ndet32 = static_cast<int32_t>(ndet);
    for (size_t j = 0; j < k; ++j) {
      double *const column = y.data() + (j * size);
      FBF_TRY(copy(column, y_o.data() + (j * pairs), pairs, wwr::wwrMemcpyDeviceToDevice));
      FBF_TRY(copy(column + pairs, y_c.data() + (j * n_ci), n_ci, wwr::wwrMemcpyDeviceToDevice));
      for (size_t r = 0; r < nroots; ++r) {
        const double scale = 2.0 * rho[r];
        FBF_TRY(gpu_status(
            wwr::scal<double, int32_t>(blas, ndet32, &scale, column + pairs + (r * ndet), 1)));
      }
      size_t at = j * all_pairs;
      for (size_t r = 0; r < nroots; ++r) {
        for (size_t s = r + 1; s < nroots; ++s, ++at) {
          y_theta[at] *= 2.0 * (rho[r] - rho[s]);
        }
      }
      FBF_TRY(copy(column + pairs + n_ci, y_theta.data() + (j * all_pairs), all_pairs,
                   wwr::wwrMemcpyHostToDevice));
      FBF_TRY(res.sync());
    }
    e_rho_xx = FBF_TRY(gram(size32, lambda_q.data(), y.data()));
  }

  // 6. The multipliers' z terms at (Z, z, Theta), every rotation a parameter:
  // L_xy's z part, I^x z against x_B in both orders, and the third
  // derivative's action over x.
  {
    Buffers::device<double> z(size, res.device());
    const response::ZVector &m = lagrangian.multipliers;
    if (m.theta.size() != all_pairs) {
      return err_config("{}: the multipliers carry {} rotations for {} roots", kWho, m.theta.size(),
                        nroots);
    }
    FBF_TRY(copy(z.data(), m.orbital.data(), pairs, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(copy(z.data() + pairs, m.ci.data(), n_ci, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(copy(z.data() + pairs + n_ci, m.theta.data(), all_pairs, wwr::wwrMemcpyHostToDevice));
    FBF_TRY(res.sync());
    const SaCasscfZTerms terms = FBF_TRY(
        df_sacasscf_z(res, molecule, basis, classes, aux, h, b, l, solution.c, ncore, ncas, nalpha,
                      nbeta, solution.roots.vectors, weights, {}, z, k32, lambda_q.data()));
    out.z_lxy = to_coordinates(terms.lxy.exy);
    out.z_lx = both_orders(FBF_TRY(against_x(terms.lxlambda.data())));
    out.z_third = terms.third;
  }

  // 7. The zeta term's three shares.
  {
    const QdZetaTerms terms =
        FBF_TRY(df_qd_zeta(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                           lagrangian.pairs.zeta, k32, lambda_q.data()));
    out.zeta = to_coordinates(terms.lxy.exy);
    const std::vector<double> lx = both_orders(FBF_TRY(against_x(terms.lxlambda.data())));
    for (size_t e = 0; e < k * k; ++e) {
      out.zeta[e] += lx[e] + terms.lxx[e];
    }
  }

  // 8. The skeleton at E^corr[W] + Phi^SA's adjoint.
  out.skeleton = to_coordinates(FBF_TRY(df_adjoint_xy(res, molecule, basis, classes, aux, h, b, l,
                                                      solution.c, h_bar.data(), lambda.data())));

  // 9. The rotation cross terms and the kappa_oo rows, df_sc_nevpt2's step 7
  // at E^corr[W] + Phi^SA's adjoint: X(t_i), Y_i = X(O(I, kappa_i)) and
  // V_i = [X_N, kappa_nr_i], X(M) = 2 (M_h h_bar + sum_Q M_Q Lambda_Q).
  out.rotation_tau.resize(k * k);
  out.rotation_kappa.resize(k * k);
  out.z_koo.resize(k * k);
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
    const auto x_of = [&](const double *m_h, const double *m_b, double *into) -> Status {
      FBF_TRY(
          gpu_status(gemm(blas, nmo32, &kTwo<double>, m_h, h_bar.data(), &kZero<double>, into)));
      return gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nmo32, nmo32, naux32 * nmo32, &kTwo<double>,
                                                m_b, lambda.data(), &kOne<double>, into));
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
      FBF_TRY(
          gpu_status(gemm(blas, nmo32, &kOne<double>, x_n.data(), kappa_nr, &kZero<double>, vi)));
      FBF_TRY(
          gpu_status(gemm(blas, nmo32, &kNegOne<double>, kappa_nr, x_n.data(), &kOne<double>, vi)));
    }
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
        out.rotation_tau[at] = g_tau[at] + g_tau[ta];
        out.rotation_kappa[at] = 0.5 * (g_kappa[at] + g_kappa[ta]);
        out.z_koo[at] = 0.5 * (g_oo[at] + g_oo[ta]);
      }
    }
  }

  // 10. Term (ii) for a state, in the coordinates.
  if (state) {
    const QdEigenvectorResponse response = FBF_TRY(
        df_qd_eigenvector_response(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                   ncas, pc, batch, kQdNevpt2ZvectorTolerance, floor));
    out.term_ii = to_coordinates(std::span<const double>(response.term)
                                     .subspan(static_cast<size_t>(*state) * n_x * n_x, n_x * n_x));
  }

  // 11. The parts summed and symmetrized.
  out.e_sa.resize(k * k);
  for (size_t e = 0; e < k * k; ++e) {
    out.e_sa[e] = e_rho_xy[e] + e_rho_lx[e] + e_rho_xx[e];
  }
  out.hessian.resize(k * k);
  const auto element = [&](const size_t at) {
    return out.e_sa[at] + out.z_lxy[at] + out.z_lx[at] + out.z_third[at] + out.z_koo[at] +
           out.zeta[at] + out.skeleton[at] + out.hvp[at] + out.rotation_tau[at] +
           out.rotation_kappa[at] + (out.term_ii.empty() ? 0.0 : out.term_ii[at]);
  };
  for (size_t j = 0; j < k; ++j) {
    for (size_t i = 0; i < k; ++i) {
      const double upper = element(i + (k * j));
      const double lower = element(j + (k * i));
      out.asymmetry = std::max(out.asymmetry, std::abs(upper - lower));
      out.hessian[i + (k * j)] = 0.5 * (upper + lower);
    }
  }
  if (!out.term_ii.empty()) {
    double largest = 0.0;
    double share = 0.0;
    for (size_t e = 0; e < k * k; ++e) {
      largest = std::max(largest, std::abs(out.hessian[e]));
      share = std::max(share, std::abs(out.term_ii[e]));
    }
    out.term_ii_share = largest > 0.0 ? share / largest : 0.0;
  }
  // The translational sum rule, on a full set of coordinates only.
  out.translation = std::numeric_limits<double>::quiet_NaN();
  if (k == n_x) {
    out.translation = 0.0;
    for (size_t axis = 0; axis < 3; ++axis) {
      std::vector<double> qt_axis(k, 0.0); // Q T_axis
      for (size_t i = 0; i < k; ++i) {
        for (size_t a = 0; a < n_x / 3; ++a) {
          qt_axis[i] += q[(i * n_x) + (3 * a) + axis];
        }
      }
      std::vector<double> hq(k, 0.0);
      for (size_t j = 0; j < k; ++j) {
        for (size_t i = 0; i < k; ++i) {
          hq[i] += out.hessian[i + (k * j)] * qt_axis[j];
        }
      }
      for (size_t x = 0; x < n_x; ++x) {
        double s = 0.0;
        for (size_t i = 0; i < k; ++i) {
          s += q[(i * n_x) + x] * hq[i];
        }
        out.translation = std::max(out.translation, std::abs(s));
      }
    }
  }
  return result;
}

Result<QdScNevpt2Terms>
df_qd_sc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                const int32_t state, const int32_t batch, const std::span<const double> coordinates,
                const int32_t nfrozen) {
  QdPcNevpt2Terms t =
      FBF_TRY(qd_nevpt2_terms(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                              state, {}, batch, coordinates, nfrozen, std::nullopt));
  return std::move(t.terms);
}

Result<QdScNevpt2Terms>
df_qd_sc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                const std::span<const double> w, const int32_t batch,
                const std::span<const double> coordinates, const int32_t nfrozen) {
  QdPcNevpt2Terms t =
      FBF_TRY(qd_nevpt2_terms(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                              std::nullopt, w, batch, coordinates, nfrozen, std::nullopt));
  return std::move(t.terms);
}

Result<QdPcNevpt2Terms>
df_qd_pc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                const int32_t state, const int32_t batch, const std::span<const double> coordinates,
                const double pc_floor, const int32_t nfrozen) {
  return qd_nevpt2_terms(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, state,
                         {}, batch, coordinates, nfrozen, pc_floor);
}

Result<QdPcNevpt2Terms>
df_qd_pc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                const std::span<const double> w, const int32_t batch,
                const std::span<const double> coordinates, const double pc_floor,
                const int32_t nfrozen) {
  return qd_nevpt2_terms(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                         std::nullopt, w, batch, coordinates, nfrozen, pc_floor);
}

} // namespace free_breakfast::hessian
