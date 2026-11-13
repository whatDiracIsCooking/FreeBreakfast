// GPU cases for experimental.routine (#1246): does the root_rotation port,
// called as a Routine<..., Alloc::Yes> with a root-minted key, compute what
// mcscf::root_rotation does, and is its footprint(in) the pool's peak?
//
// Both run in one process over the same inputs, set up as
// test/gtest/mcscf_root_rotation_gpu.cpp sets them up: the equal-weight
// averages of test/gtest/data/casscf_hessian/, orbitals and roots loaded from
// the fixture, df_derivatives' H~ and the fixture's lambda^x orbital part.
// The port makes the same calls in the same order, so only the reduction
// order of atomics can tell the two apart: 1e-12 absolute.
//
// The must-not-compile entries beside this file (routine_reject_*.cpp) are
// ctest entries of their own, registered in this directory's CMakeLists.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.response;
import free_breakfast.mcscf;
import free_breakfast.experimental.routine;

#include "pool_footprint.h"
#include "test_workspaces.h"

// The fixture's minter, AllocKey's one friend outside the module: a
// global-module class, so this TU can define it (key.cppm says why, and
// README.md what it costs). Outside the anonymous namespace for that reason.
namespace free_breakfast::test {
struct RoutineKeyMinter {
  template <class Call>
  static decltype(auto) with_key(Call &&call) {
    const routine::AllocKey key;
    return std::forward<Call>(call)(key);
  }
};
} // namespace free_breakfast::test

namespace {

namespace fb = free_breakfast;
namespace fbt = free_breakfast::test;

// The port against the original, absolute: same calls, same order, so only
// the atomics' reduction order can differ (the issue's bar).
constexpr double kPortTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_hessian";

// <i|i> of a shell's (l, 0, 0) Cartesian (mcscf_root_rotation_gpu.cpp's).
double self_overlap(const fb::Shell &shell) {
  const std::int32_t l = shell.angular_momentum();
  double dfact = 1.0;
  for (std::int32_t k = (2 * l) - 1; k > 1; k -= 2) {
    dfact *= k;
  }
  double sum = 0.0;
  for (const fb::Primitive &p : shell.primitives()) {
    for (const fb::Primitive &q : shell.primitives()) {
      const double g = p.exponent + q.exponent;
      sum += p.coefficient * q.coefficient * dfact * std::pow(std::numbers::pi / g, 1.5) /
             std::pow(2.0 * g, l);
    }
  }
  return sum;
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *device) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(device, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
}

std::vector<double> read_array(const std::string &file, const std::vector<std::size_t> &shape) {
  const auto a = fb::read_npy(kDataDir / file);
  EXPECT_TRUE(a) << file;
  if (!a) {
    return {};
  }
  EXPECT_EQ(a->shape, shape) << file;
  return a->data;
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

class RoutineRootRotation : public ::testing::Test {
protected:
  // One case set up as mcscf_root_rotation_gpu.cpp's load() does it: the
  // bases placed, h, the fit B, the fixture's orbitals made ours and its
  // roots, then df_derivatives and the fixture's lambda^x packed.
  void load(const std::string &name) {
    const auto doc = fb::read_json(kDataDir / "casscf_hessian.json");
    ASSERT_TRUE(doc) << doc.error().message;
    for (const fb::json &entry : (*doc)["cases"]) {
      if (entry["name"].get<std::string>() == name) {
        case_ = entry;
      }
    }
    ASSERT_FALSE(case_.is_null()) << "no case " << name;
    res_.emplace(0, fbt::kWorkspaces);
    std::vector<fb::Atom> atoms;
    for (const auto &atom : case_["atoms"]) {
      const auto xyz = atom["coord"].get<std::vector<double>>();
      atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                               static_cast<fb::Element>(atom["charge"].get<std::int32_t>())});
    }
    molecule_.emplace(*res_, std::move(atoms));
    const auto stamp = fb::resolve_basis(case_["basis"].get<std::string>());
    ASSERT_TRUE(stamp) << stamp.error().message;
    auto basis = fb::place(stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(basis) << basis.error().message;
    basis_.emplace(std::move(*basis));
    const auto aux_stamp = fb::resolve_aux_basis(case_["aux_basis"].get<std::string>());
    ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
    auto aux = fb::place(aux_stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(aux) << aux.error().message;
    aux_.emplace(std::move(*aux));
    nao_ = case_["nao"].get<std::size_t>();
    naux_ = case_["naux"].get<std::size_t>();
    ASSERT_EQ(basis_->nao_sphe(), nao_);
    ASSERT_EQ(aux_->nao_sphe(), naux_);
    n_x_ = 3 * basis_->natoms();
    ncore_ = case_["ncore"].get<std::int32_t>();
    ncas_ = case_["ncas"].get<std::int32_t>();
    const auto nelecas = case_["nelecas"].get<std::vector<std::int32_t>>();
    nalpha_ = nelecas[0];
    nbeta_ = nelecas[1];
    weights_ = case_["weights"].get<std::vector<double>>();
    energies_ = case_["energies"].get<std::vector<double>>();
    const std::size_t n_det = case_["na"].get<std::size_t>() * case_["nb"].get<std::size_t>();

    auto classes = fb::primitive_pair_classes(*res_, *basis_);
    ASSERT_TRUE(classes) << classes.error().message;
    classes_ = std::move(*classes);
    auto su = fb::upload_spherical_transform_matrix(*res_, *basis_);
    ASSERT_TRUE(su) << su.error().message;
    h_.emplace(*res_, *basis_, std::move(*su));
    ASSERT_TRUE(fb::core_hamiltonian(*res_, classes_, *molecule_, h_->cartesian()));
    ASSERT_TRUE(h_->cartesian_to_spherical(*res_));
    auto u = fb::upload_spherical_transform_matrix(*res_, *aux_);
    ASSERT_TRUE(u) << u.error().message;
    metric_.emplace(*res_, *aux_, std::move(*u));
    b_.emplace(naux_ * nao_ * nao_, res_->device());
    ASSERT_TRUE(fb::metric(*res_, *aux_, *metric_));
    ASSERT_TRUE(fb::three_center(*res_, *basis_, classes_, *aux_, b_->data()));
    ASSERT_TRUE(fb::density_fit(*res_, static_cast<std::int32_t>(nao_),
                                static_cast<std::int32_t>(naux_), metric_->spherical().data(),
                                b_->data()));

    std::vector<double> norms;
    for (const fb::Shell &shell : basis_->shells()) {
      norms.insert(norms.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
    }
    // PySCF orders each atom's shells by l (mcscf_root_rotation_gpu.cpp says
    // why this matters for 6-31G): row k of the fixture's C is order[k].
    std::vector<std::size_t> order;
    for (std::uint32_t atom = 0; atom < basis_->natoms(); ++atom) {
      std::vector<std::uint32_t> shells(basis_->atom_shell_count(atom));
      std::iota(shells.begin(), shells.end(), basis_->atom_shell_offset(atom));
      std::ranges::stable_sort(shells, {}, [&](const std::uint32_t i) {
        return basis_->shells()[i].angular_momentum();
      });
      for (const std::uint32_t i : shells) {
        const std::size_t first = basis_->ao_offset_sphe(i);
        for (std::size_t f = 0; f < basis_->shells()[i].function_count_sphe(); ++f) {
          order.push_back(first + f);
        }
      }
    }
    ASSERT_EQ(order.size(), nao_);
    const std::vector<double> c_pyscf =
        read_array(case_["arrays"]["c"].get<std::string>(), {nao_, nao_});
    std::vector<double> c(nao_ * nao_);
    for (std::size_t j = 0; j < nao_; ++j) {
      for (std::size_t k = 0; k < nao_; ++k) {
        c[order[k] + (nao_ * j)] = c_pyscf[k + (nao_ * j)] / norms[order[k]];
      }
    }
    c_.emplace(*res_, *basis_);
    upload(*res_, c, c_->data());
    const std::vector<double> ci =
        read_array(case_["arrays"]["ci"].get<std::string>(), {n_det, weights_.size()});
    roots_.emplace(ci.size(), res_->device());
    upload(*res_, ci, roots_->data());

    auto d = fb::mcscf::df_derivatives(*res_, *molecule_, *basis_, classes_, *aux_, h_->spherical(),
                                       *b_, metric_->spherical().data(), *c_, ncore_, ncas_,
                                       nalpha_, nbeta_, roots_->data(), weights_);
    ASSERT_TRUE(d) << d.error().message;
    derivatives_.emplace(std::move(*d));

    const std::vector<double> full =
        read_array(case_["arrays"]["lambda_orb"].get<std::string>(), {nao_, nao_, n_x_});
    fb::Buffers::device<double> device(full.size(), res_->device());
    upload(*res_, full, device.data());
    const auto core = static_cast<std::size_t>(ncore_);
    const auto act = static_cast<std::size_t>(ncas_);
    const std::size_t pairs = (core * (nao_ - core)) + (act * (nao_ - core - act));
    kappa_.emplace(pairs * n_x_, res_->device());
    for (std::size_t x = 0; x < n_x_; ++x) {
      ASSERT_TRUE(fb::response::pack_orbital_pairs(*res_, static_cast<std::int32_t>(nao_), ncore_,
                                                   ncas_, device.data() + (nao_ * nao_ * x),
                                                   kappa_->data() + (pairs * x)));
    }
    ASSERT_TRUE(res_->sync());
  }

  // The port's In over the loaded case, on @p res. The call-site shape
  // README.md "Call-site diff" quotes.
  [[nodiscard]] fb::routine::RootRotationRoutine::In in(fb::DeviceResources &res) const {
    return {.res = res,
            .molecule = *molecule_,
            .basis = *basis_,
            .h = h_->spherical(),
            .b = *b_,
            .c = *c_,
            .ncore = ncore_,
            .ncas = ncas_,
            .nalpha = nalpha_,
            .nbeta = nbeta_,
            .roots = roots_->data(),
            .weights = weights_,
            .energies = energies_,
            .derivatives = *derivatives_,
            .kappa = kappa_->data()};
  }

  fb::json case_;
  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Molecule> molecule_;
  std::optional<fb::BasisSet> basis_;
  std::optional<fb::BasisSet> aux_;
  std::vector<fb::PrimitivePairClass> classes_;
  std::optional<fb::MatrixAOPair> h_;
  std::optional<fb::MatrixAOPair> metric_;
  std::optional<fb::Buffers::device<double>> b_;
  std::optional<fb::SphericalMatrixAO> c_;
  std::optional<fb::Buffers::device<double>> roots_;
  std::optional<fb::mcscf::Derivatives> derivatives_;
  std::optional<fb::Buffers::device<double>> kappa_;
  std::vector<double> weights_;
  std::vector<double> energies_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::int32_t nalpha_ = 0;
  std::int32_t nbeta_ = 0;
};

// The port through its root (RootRotationStep::run, the key minted there)
// against mcscf::root_rotation, two roots and three, in one process.
TEST_F(RoutineRootRotation, PortMatchesOriginal) {
  FBF_REQUIRE_GPU();
  for (const std::string name : {"h2o_631g_sa2", "h2o_631g_sa3"}) {
    SCOPED_TRACE(name);
    load(name);
    ASSERT_FALSE(HasFatalFailure());
    auto want = fb::mcscf::root_rotation(*res_, *molecule_, *basis_, h_->spherical(), *b_, *c_,
                                         ncore_, ncas_, nalpha_, nbeta_, roots_->data(), weights_,
                                         energies_, *derivatives_, kappa_->data());
    ASSERT_TRUE(want) << want.error().message;
    auto got = fb::routine::RootRotationStep::run(in(*res_));
    ASSERT_TRUE(got) << got.error().message;
    const double coupling = max_diff(got->coupling, want->coupling);
    const double theta = max_diff(got->theta, want->theta);
    const double hessian = max_diff(got->hessian, want->hessian);
    std::cout << std::format("[ port     ] {}: coupling {:.3e} (max {:.3e}), theta {:.3e} (max "
                             "{:.3e}), rot {:.3e} (max {:.3e})\n",
                             name, coupling, max_abs(want->coupling), theta, max_abs(want->theta),
                             hessian, max_abs(want->hessian));
    EXPECT_LE(coupling, kPortTol);
    EXPECT_LE(theta, kPortTol);
    EXPECT_LE(hessian, kPortTol);
    // Not vacuous: the three-root term is as large as the Hessian itself.
    EXPECT_GT(max_abs(want->hessian), 1e-2);
  }
}

// footprint(in) is root_rotation_footprint's number over the same case, and
// the pool's peak for one call (test/gtest/df_casscf_hessian_gpu.cpp's
// expect_fixed_footprint): on a fresh DeviceResources, run once to grow the
// workspace to the footprint's, then measured. The call goes straight to the
// routine with the fixture's key, not through the root.
TEST_F(RoutineRootRotation, FootprintIsSoundAndTight) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa3");
  ASSERT_FALSE(HasFatalFailure());
  fb::DeviceResources res{0, fbt::kWorkspaces};
  const fb::MemoryFootprint fp = fb::routine::RootRotationRoutine::footprint(in(res));
  const fb::MemoryFootprint want = fb::mcscf::root_rotation_footprint(
      *basis_, *aux_, ncore_, ncas_, nalpha_, nbeta_, static_cast<std::int32_t>(weights_.size()));
  EXPECT_EQ(fp.device.persistent, want.device.persistent);
  EXPECT_EQ(fp.device.scratch, want.device.scratch);
  EXPECT_EQ(fp.device.workspace, want.device.workspace);
  EXPECT_GT(fp.device.scratch, 0U);
  const auto call = [&] {
    auto got = fbt::RoutineKeyMinter::with_key(
        [&](const fb::routine::AllocKey &key) { return fb::routine::root_rotation(key, in(res)); });
    ASSERT_TRUE(got) << got.error().message;
  };
  call();
  EXPECT_EQ(res.workspace_high_water().device.workspace, fp.device.workspace);
  const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), call);
  std::cout << std::format("[ footprint ] measured {}, footprint {} + workspace {}\n", peak.used,
                           fp.device.persistent + fp.device.scratch, fp.device.workspace);
  // At the peak, in response::fock: the strings' four, the five keyed
  // buffers, the integrals' four, the Fock's six and its three of scratch.
  fbt::expect_footprint(peak, {.persistent = fp.device.persistent, .scratch = fp.device.scratch},
                        22);
}

} // namespace
