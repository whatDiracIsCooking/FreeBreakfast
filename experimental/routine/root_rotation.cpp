/// @file
/// @brief The root_rotation port's check, footprint and invoke, and the
/// root's run: mcscf::root_rotation's body with its entry checks moved to
/// check and its five device buffers keyed.

module free_breakfast.experimental.routine;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr; // gemv, geam
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.core.primitives; // math::binomial
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.active_space;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.rdm;
import free_breakfast.response.fock;
import free_breakfast.response.orbital_pairs;
import free_breakfast.mcscf.derivative;
import free_breakfast.mcscf.root_rotation; // kRootGapFloor, the result
import free_breakfast.response.relaxed_hessian;

#include "diagnostics.h"

// #1226's FBF_PRE, which has not landed: on false, log the condition's text
// and the call site at critical, then abort, in every build. No message, as
// #1226 specifies, so a later migration is a rename.
#define FBF_ROUTINE_PRE(cond)                                                                      \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      ::free_breakfast::log::critical("precondition failed: " #cond,                               \
                                      std::source_location::current());                            \
      std::abort();                                                                                \
    }                                                                                              \
  } while (false)

namespace free_breakfast::routine {

// mcscf::root_rotation's: weights this close (relative to the largest) are
// equal, being recovered from a solution by a normalization.
constexpr double kRoutineEqualWeightTolerance = 1e-10;

bool routine_weights_equal(const std::span<const double> weights) {
  const double w_max = std::ranges::max(weights);
  return std::ranges::all_of(weights, [w_max](const double w) {
    return std::abs(w - w_max) <= kRoutineEqualWeightTolerance * w_max;
  });
}

void RootRotationRoutine::check(const In &in) {
  const size_t nao = in.basis.nao_sphe();
  const size_t n_x = 3 * static_cast<size_t>(in.basis.natoms());
  const size_t nroots = in.weights.size();
  FBF_ROUTINE_PRE(in.c.n() == nao && in.h.n() == nao);
  FBF_ROUTINE_PRE(in.ncore >= 0 && in.ncas >= 1 &&
                  static_cast<size_t>(in.ncore) + static_cast<size_t>(in.ncas) <= nao);
  FBF_ROUTINE_PRE(nao != 0 && in.b.num_elements() != 0 && in.b.num_elements() % (nao * nao) == 0);
  FBF_ROUTINE_PRE(nroots != 0 && in.energies.size() == nroots);
  FBF_ROUTINE_PRE(routine_weights_equal(in.weights));
  FBF_ROUTINE_PRE(in.derivatives.hamiltonian.size() == nroots * nroots * n_x);
  FBF_ROUTINE_PRE(in.max_rotation > 0.0);
}

MemoryFootprint RootRotationRoutine::footprint(const In &in) {
  const size_t nao = in.basis.nao_sphe();
  const auto nroots = static_cast<int32_t>(in.weights.size());
  if (nroots < 2 || in.ncore < 0 || in.ncas < 1 ||
      static_cast<size_t>(in.ncore) + static_cast<size_t>(in.ncas) > nao || nao == 0 ||
      in.b.num_elements() < nao * nao) {
    return {};
  }
  const auto nmo = static_cast<int32_t>(nao);
  const auto naux = static_cast<int32_t>(in.b.num_elements() / (nao * nao));
  const auto n_core = static_cast<size_t>(in.ncore);
  const auto n_act = static_cast<size_t>(in.ncas);
  const size_t na2 = n_act * n_act;
  const size_t na4 = na2 * na2;
  const size_t n_x = 3 * static_cast<size_t>(in.basis.natoms());
  const size_t n_pairs = (n_core * (nao - n_core)) + (n_act * (nao - n_core - n_act));
  const auto strings = [&in](const int32_t electrons) {
    return static_cast<size_t>(std::round(math::binomial(in.ncas, electrons)));
  };
  const size_t dim = strings(in.nalpha) * strings(in.nbeta);
  MemoryFootprint own{
      .device = {.persistent = sizeof(double) * ((2 * dim) + (2 * (na2 + na4)) + (nao * nao) +
                                                 std::max<size_t>(n_pairs, 1) + n_x)}};
  own.device.workspace = ci::rdm12_footprint(in.ncas, static_cast<int32_t>(dim)).device.workspace;
  const MemoryFootprint chain =
      then(then(ci::device_string_set_footprint(in.ncas, in.nalpha),
                ci::device_string_set_footprint(in.ncas, in.nbeta)),
           then(own, then(casscf_integrals_footprint(nmo, naux, in.ncore, in.ncas),
                          response::fock_footprint(nmo, naux, in.ncore, in.ncas))));
  return drop(chain);
}

Result<mcscf::RootRotation> RootRotationRoutine::invoke(const AllocKey &key, const In &in) {
  DeviceResources &res = in.res;
  const size_t nao = in.basis.nao_sphe();
  const size_t n_x = 3 * static_cast<size_t>(in.basis.natoms());
  const size_t nroots = in.weights.size();
  const double *const roots = in.roots;
  const std::span<const double> energies = in.energies;
  const int32_t ncore = in.ncore;
  const int32_t ncas = in.ncas;
  mcscf::RootRotation out{std::vector<double>(nroots * nroots * n_x, 0.0),
                          std::vector<double>(nroots * nroots * n_x, 0.0),
                          std::vector<double>(n_x * n_x * nroots, 0.0)};
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = 0; s < r; ++s) {
      const double gap = energies[r] - energies[s];
      if (std::abs(gap) < mcscf::kRootGapFloor) {
        return err_numerical("routine::root_rotation: roots {} and {} are {:.3e} hartree apart, "
                             "under {:.0e}: at a degeneracy the root has no Hessian",
                             s, r, std::abs(gap), mcscf::kRootGapFloor);
      }
    }
  }
  if (nroots == 1) {
    return out;
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto n_core = static_cast<size_t>(ncore);
  const auto n_act = static_cast<size_t>(ncas);
  const size_t na2 = n_act * n_act;
  const size_t na4 = na2 * na2;
  const size_t nn = nao * nao;
  const auto naux = static_cast<int32_t>(in.b.num_elements() / nn);
  const auto n_pairs =
      static_cast<int32_t>((n_core * (nao - n_core)) + (n_act * (nao - n_core - n_act)));

  // The integrals response_gradient contracts: F^I (any RDMs give it; the
  // average's are at hand) and ppaa, papa, on the host. The strings, the
  // integrals and the Fock allocate outside the key (README.md "Callees").
  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, in.nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, in.nbeta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  const auto dim32 = static_cast<int32_t>(dim);
  // Held across rdm12, casscf_integrals and fock, which take the workspace
  // (docs/style.md "Scratch memory"): c_r + c_s and c_r - c_s, the RDMs of
  // each, the gradient's antisymmetric matrix, its packed pairs and the
  // contraction with kappa. The routine's own allocations: keyed.
  KeyedBuffer<double> pm(key, 2 * dim, res.device());
  KeyedBuffer<double> rdms(key, 2 * (na2 + na4), res.device());
  KeyedBuffer<double> a_full(key, nn, res.device());
  KeyedBuffer<double> a_packed(key, std::max<size_t>(static_cast<size_t>(n_pairs), 1),
                               res.device());
  KeyedBuffer<double> orbital(key, n_x, res.device());

  std::vector<double> gamma(na2);
  std::vector<double> gamma2(na4);
  const auto download = [&](double *to, const double *from, const size_t count) {
    return gpu_status(
        wwr::wwrMemcpyAsync(to, from, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost, stream));
  };
  FBF_TRY(rdm.rdm12(res, in.weights, roots, rdms.data(), rdms.data() + na2));
  FBF_TRY(download(gamma.data(), rdms.data(), na2));
  FBF_TRY(download(gamma2.data(), rdms.data() + na2, na4));
  FBF_TRY(res.sync());
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, in.basis, in.molecule, in.h, in.b, naux, in.c, ncore, ncas));
  const response::Fock f0 = FBF_TRY(response::fock(res, in.molecule, in.h, in.b, naux, in.c, ncore,
                                                   ncas, ints.ppaa, ints.papa, gamma, gamma2));
  response::ResponseIntegrals response_ints{nao,
                                            n_core,
                                            n_act,
                                            std::vector<double>(nn),
                                            std::vector<double>(nn * na2),
                                            std::vector<double>(nn * na2)};
  FBF_TRY(download(response_ints.fock_inactive.data(), f0.fock_inactive.data(), nn));
  FBF_TRY(download(response_ints.ppaa.data(), ints.ppaa.data(), nn * na2));
  FBF_TRY(download(response_ints.papa.data(), ints.papa.data(), nn * na2));
  FBF_TRY(res.sync());

  // H^x_rs per pair r > s, as mcscf::root_rotation forms it: H~^x_rs plus
  // <c_r|H'(kappa^x)|c_s> from the symmetrized transition RDMs by
  // polarization, its packed gradient against every column of kappa by one
  // gemv.
  const std::array<double, 1> one{1.0};
  std::vector<double> gp(2 * (na2 + na4));
  std::vector<double> a(nn);
  std::vector<double> coupling_orb(n_x);
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = 0; s < r; ++s) {
      const double *const c_r = roots + (r * dim);
      const double *const c_s = roots + (s * dim);
      for (const auto &[to, sign] :
           {std::pair{pm.data(), &kOne<double>}, std::pair{pm.data() + dim, &kNegOne<double>}}) {
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                                      dim32, 1, &kOne<double>, c_r, dim32, sign,
                                                      c_s, dim32, to, dim32)));
      }
      double *const plus = rdms.data();
      double *const minus = plus + na2 + na4;
      FBF_TRY(rdm.rdm12(res, one, pm.data(), plus, plus + na2));
      FBF_TRY(rdm.rdm12(res, one, pm.data() + dim, minus, minus + na2));
      FBF_TRY(download(gp.data(), rdms.data(), gp.size()));
      FBF_TRY(res.sync());
      const size_t off = na2 + na4;
      for (size_t i = 0; i < na2; ++i) {
        gamma[i] = 0.25 * (gp[i] - gp[off + i]);
      }
      for (size_t i = 0; i < na4; ++i) {
        gamma2[i] = 0.125 * (gp[na2 + i] - gp[off + na2 + i]);
      }
      response::response_gradient(response_ints, gamma, gamma2, a);
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a_full.data(), a.data(), sizeof(double) * nn,
                                             wwr::wwrMemcpyHostToDevice, stream)));
      FBF_TRY(response::pack_orbital_pairs(res, static_cast<int32_t>(nao), ncore, ncas,
                                           a_full.data(), a_packed.data()));
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
          blas, wwr::WWRBLAS_OP_T, n_pairs, static_cast<int32_t>(n_x), &kOne<double>, in.kappa,
          n_pairs, a_packed.data(), 1, &kZero<double>, orbital.data(), 1)));
      FBF_TRY(download(coupling_orb.data(), orbital.data(), n_x));
      FBF_TRY(res.sync());
      for (size_t x = 0; x < n_x; ++x) {
        const size_t slab = nroots * nroots * x;
        const double value = in.derivatives.hamiltonian[r + (nroots * s) + slab] + coupling_orb[x];
        out.coupling[r + (nroots * s) + slab] = value;
        out.coupling[s + (nroots * r) + slab] = value;
      }
    }
  }

  // Eq. theta and the refusal, then eq. rot per root.
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = 0; s < nroots; ++s) {
      if (s == r) {
        continue;
      }
      const double gap = energies[r] - energies[s];
      double worst = 0.0;
      for (size_t x = 0; x < n_x; ++x) {
        const size_t slab = nroots * nroots * x;
        const double theta = out.coupling[r + (nroots * s) + slab] / gap;
        out.theta[s + (nroots * r) + slab] = theta;
        worst = std::max(worst, std::abs(theta));
      }
      if (worst > in.max_rotation) {
        return err_numerical("routine::root_rotation: roots {} and {} are {:.3e} hartree apart, "
                             "and their largest coupling over that gap is {:.3e} per bohr, past "
                             "{:.3e}: too near a degeneracy for root {}'s Hessian",
                             std::min(r, s), std::max(r, s), std::abs(gap), worst, in.max_rotation,
                             r);
      }
    }
  }
  for (size_t i = 0; i < nroots; ++i) {
    double *const term = out.hessian.data() + (n_x * n_x * i);
    for (size_t s = 0; s < nroots; ++s) {
      if (s == i) {
        continue;
      }
      const double scale = 2.0 / (energies[i] - energies[s]);
      for (size_t y = 0; y < n_x; ++y) {
        const double hy = out.coupling[i + (nroots * s) + (nroots * nroots * y)];
        for (size_t x = 0; x < n_x; ++x) {
          term[x + (n_x * y)] +=
              scale * out.coupling[i + (nroots * s) + (nroots * nroots * x)] * hy;
        }
      }
    }
  }
  return out;
}

Result<mcscf::RootRotation> RootRotationStep::run(const RootRotationRoutine::In &in) {
  const AllocKey key;
  return root_rotation(key, in);
}

} // namespace free_breakfast::routine
