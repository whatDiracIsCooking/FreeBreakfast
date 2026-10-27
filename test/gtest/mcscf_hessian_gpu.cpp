// OrbitalHessian on the device, at each case's CASCI orbitals, against PySCF's
// mc1step.gen_g_hop (data/casscf/casscf.json, tools/emit_casscf_reference.py,
// doubled there as here): the sign-invariant gᵀHg, |Hg| and the diagonal
// summed per pair of orbital sets; H x against a central finite difference of
// casscf_fock's gradient along random kappa, the RDMs fixed; and symmetry.
// On H2O STO-3G, H e_i for every unit vector against the diagonal, at a
// two-root CAS(2,2) and at ncas = 0; and the refusals.
//
// The orbitals are the setup's own, as in mcscf_fock_gpu.cpp: the SCF (run
// with J unscreened), the active space picked by pipeline.active_space as the
// Casci step picks it (by index, or by AVAS), and the RDMs from the CasciOutput
// step's roots and ci.rdm's ci::Rdm. So this executable links pipeline; mcscf
// itself does not. Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.recipes.expm;
import free_breakfast.integrals;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;
import free_breakfast.core.nontransferrable;

#include "mcscf_setup.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bars.
constexpr double kInvariantTolerance = 1e-7; // gᵀHg and |Hg|, relative
constexpr double kDiagonalTolerance = 1e-7;
constexpr double kFiniteDifferenceTolerance = 1e-6; // relative to |H x|
// OrbitalHessian's default tau_j and tau_k, which every operator here is built
// at.
constexpr double kTau = 1e-14;
// |x^T H y - y^T H x| <= this max |H| |x| |y|: 100 tau, src/mcscf/hessian's
// README says why (seen: 1e-13 at most, of max |diag H|).
constexpr double kSymmetryTolerance = 100.0 * kTau;
// The orbital energies only pair our orbitals with PySCF's by index.
constexpr double kOrbitalEnergyTolerance = 1e-6;
// The finite difference: Richardson's extrapolation of two central
// differences, at steps kStep and kStep / 2, which cancels the step^2 term.
constexpr double kStep = 2e-3;
constexpr std::int32_t kDirections = 2;

const fb::json &reference() {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf" / "casscf.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return doc;
}

const fb::json &reference_case(std::string_view name) {
  for (const fb::json &entry : reference().at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf/casscf.json";
  static const fb::json kNone;
  return kNone;
}

// Copies in res.stream()'s order, waited for.
std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  if (count == 0) {
    return host;
  }
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *data) {
  if (host.empty()) {
    return;
  }
  EXPECT_EQ(wwr::wwrMemcpyAsync(data, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

// The packed order, as src/mcscf/README.md documents it: (p, q) per packed
// element.
std::vector<std::pair<std::size_t, std::size_t>> packed_pairs(std::size_t nmo, std::size_t ncore,
                                                              std::size_t ncas) {
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  for (std::size_t q = 0; q < ncore; ++q) {
    for (std::size_t p = ncore; p < nmo; ++p) {
      pairs.emplace_back(p, q);
    }
  }
  for (std::size_t q = ncore; q < ncore + ncas; ++q) {
    for (std::size_t p = ncore + ncas; p < nmo; ++p) {
      pairs.emplace_back(p, q);
    }
  }
  return pairs;
}

// The reference's molecule through the real registry to the Scf step's inputs. The
// Scf step itself is run by hand: tau_j is not a TOML key.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("mcscf_hessian");
  // Bohr in the reference, Angstrom in an XYZ file.
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
                  "[orthogonalizer]\n[core_hamiltonian]\n[guess]\n",
                  entry.at("charge").get<std::int32_t>(), entry.at("basis").get<std::string>());
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  if (!calc.has_value()) {
    ADD_FAILURE() << calc.error().message;
    return std::nullopt;
  }
  if (const fb::Status ran = calc->run(node); !ran.has_value()) {
    ADD_FAILURE() << ran.error().message;
    return std::nullopt;
  }
  return std::move(*calc);
}

// The entry's active space as the Casci step's selection.
fb::CasciStep casci_step(const fb::json &entry) {
  fb::CasciStep step;
  if (!entry.at("avas").is_null()) {
    step.avas = entry.at("avas").at("labels").get<std::vector<std::string>>();
    step.avas_threshold = entry.at("avas").at("threshold").get<double>();
  } else {
    step.ncas = entry.at("ncas").get<std::int32_t>();
    step.nelecas = entry.at("nelecas").get<std::int32_t>();
    if (!entry.at("active").is_null()) {
      step.active = entry.at("active").get<std::vector<std::int32_t>>();
    }
  }
  step.nroots = static_cast<std::int32_t>(entry.at("weights").size());
  return step;
}

// Everything casscf_fock and OrbitalHessian read, from one converged SCF,
// and with a CASCI, its orbitals and the weighted RDMs of its roots.
struct Setup {
  fb::NodeResources node;
  std::optional<fb::Calculation> calc;
  const fb::Molecule *molecule = nullptr;
  const fb::BasisSet *basis = nullptr;
  const fb::SphericalTransformOutput *transform = nullptr;
  const fb::CoreHamiltonianOutput *h = nullptr;
  std::optional<fb::ScfOutput> scf;
  std::vector<fb::CoulombPairClass> j_classes;
  std::vector<fb::OrderedPairClass> k_classes;
  std::optional<fb::CasciOutput> casci;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;
  std::vector<double> gamma;
  std::vector<double> gamma2;
  std::size_t nmo = 0;

  [[nodiscard]] fb::DeviceResources &res() { return node.device(0); }

  // ppaa and papa over orbitals, or two empty buffers for ncas = 0, which
  // casscf_integrals refuses.
  [[nodiscard]] fb::Result<std::pair<fb::Buffers::device<double>, fb::Buffers::device<double>>>
  integrals(const fb::SphericalMatrixAO &orbitals) {
    if (ncas == 0) {
      return std::pair{fb::Buffers::device<double>(0, res().device()),
                       fb::Buffers::device<double>(0, res().device())};
    }
    auto built = fb::casscf_integrals(res(), *basis, *molecule, h->h, transform->u, j_classes,
                                      k_classes, orbitals, ncore, ncas);
    if (!built.has_value()) {
      return std::unexpected(std::move(built.error()));
    }
    return std::pair{std::move(built->ppaa), std::move(built->papa)};
  }

  [[nodiscard]] fb::Result<fb::CasscfFock> fock(const fb::SphericalMatrixAO &orbitals,
                                                const fb::Buffers::device<double> &ppaa) {
    return fb::casscf_fock(res(), *basis, *molecule, h->h, transform->u, j_classes, k_classes,
                           orbitals, ncore, ncas, ppaa, gamma, gamma2);
  }

  // casscf_fock over orbitals, its integrals built over them too.
  [[nodiscard]] fb::Result<fb::CasscfFock> fock(const fb::SphericalMatrixAO &orbitals) {
    auto built = integrals(orbitals);
    if (!built.has_value()) {
      return std::unexpected(std::move(built.error()));
    }
    return fock(orbitals, built->first);
  }
};

// The setup over the entry's molecule, with the CASCI and its RDMs when
// with_casci. The CASCI orbitals are paired with PySCF's by energy when the
// entry carries a "casci".
std::unique_ptr<Setup> make_setup(const fb::json &entry, bool with_casci) {
  auto setup = std::make_unique<Setup>();
  auto calc = run_setup_steps(entry, setup->node);
  if (!calc.has_value()) {
    return nullptr;
  }
  setup->calc.emplace(std::move(*calc)); // a Pipeline moves, but is not move-assigned
  const auto &molecule = setup->calc->get<fb::MoleculeOutput>();
  const auto &basis_set = setup->calc->get<fb::BasisSetOutput>();
  const auto &transform = setup->calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = setup->calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = setup->calc->get<fb::OverlapOutput>();
  const auto &x = setup->calc->get<fb::OrthogonalizerOutput>();
  const auto &h = setup->calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = setup->calc->get<fb::GuessOutput>();
  if (!molecule.has_value() || !basis_set.has_value() || !transform.has_value() ||
      !pairs.has_value() || !overlap.has_value() || !x.has_value() || !h.has_value() ||
      !guess.has_value()) {
    ADD_FAILURE() << "a product of the setup steps is missing";
    return nullptr;
  }
  setup->molecule = &molecule->molecule;
  setup->basis = &basis_set->basis;
  setup->transform = &*transform;
  setup->h = &*h;
  setup->nmo = basis_set->basis.nao_sphe();
  auto scf = fb::test::kExactScf.run(fb::StepInput<fb::ScfStep>{*molecule, *basis_set, *transform,
                                                                *pairs, *overlap, *x, *h, *guess},
                                     setup->res());
  if (!scf.has_value()) {
    ADD_FAILURE() << scf.error().message;
    return nullptr;
  }
  setup->scf.emplace(std::move(*scf));

  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto j_classes = fb::coulomb_pair_classes(res, pairs->classes);
  auto k_classes = fb::ordered_pair_classes(res, basis_set->basis, pairs->classes);
  if (!j_classes.has_value() || !k_classes.has_value()) {
    ADD_FAILURE() << "the J or K classes failed";
    return nullptr;
  }
  setup->j_classes = std::move(*j_classes);
  setup->k_classes = std::move(*k_classes);
  if (!with_casci) {
    return setup;
  }

  const fb::CasciStep step = casci_step(entry);
  auto space = step.select(res, molecule->molecule, basis_set->basis, *setup->scf);
  if (!space.has_value()) {
    ADD_FAILURE() << space.error().message;
    return nullptr;
  }
  auto casci = step.run(
      fb::StepInput<fb::CasciStep>{*molecule, *basis_set, *transform, *pairs, *h, *setup->scf},
      res);
  if (!casci.has_value()) {
    ADD_FAILURE() << casci.error().message;
    return nullptr;
  }
  setup->casci.emplace(std::move(*casci));
  setup->ncore = setup->casci->ncore;
  setup->ncas = setup->casci->ncas;
  EXPECT_EQ(setup->ncore, entry.at("ncore").get<std::int32_t>());
  EXPECT_EQ(setup->ncas, entry.at("ncas").get<std::int32_t>());
  EXPECT_EQ(setup->casci->nelecas, entry.at("nelecas").get<std::int32_t>());
  if (entry.contains("casci")) {
    // The selection's rotation R gives each column's energy in the ScfOutput's
    // Fock matrix: sum_k R_kj^2 eps_k.
    const auto expected = entry.at("casci").at("orbital_energies").get<std::vector<double>>();
    EXPECT_EQ(expected.size(), setup->nmo);
    for (std::size_t col = 0; col < std::min(expected.size(), setup->nmo); ++col) {
      double energy = 0.0;
      for (std::size_t k = 0; k < setup->nmo; ++k) {
        const double r = space->r[k + (setup->nmo * col)];
        energy += r * r * setup->scf->mo_energies[k];
      }
      EXPECT_NEAR(energy, expected[col], kOrbitalEnergyTolerance) << "column " << col;
    }
  }

  const auto weights = entry.at("weights").get<std::vector<double>>();
  const std::int32_t n_beta = setup->casci->nelecas / 2;
  auto alpha = fb::ci::StringSet::create(setup->ncas, setup->casci->nelecas - n_beta);
  auto beta = fb::ci::StringSet::create(setup->ncas, n_beta);
  if (!alpha.has_value() || !beta.has_value()) {
    ADD_FAILURE() << "the string sets failed";
    return nullptr;
  }
  const auto n = static_cast<std::size_t>(setup->ncas);
  auto ci_rdm = fb::ci::Rdm::create(res, *alpha, *beta);
  if (!ci_rdm.has_value()) {
    ADD_FAILURE() << ci_rdm.error().message;
    return nullptr;
  }
  fb::Buffers::device<double> vectors(setup->casci->vectors.size(), res.device());
  fb::Buffers::device<double> gamma(n * n, res.device());
  fb::Buffers::device<double> gamma2(n * n * n * n, res.device());
  upload(res, setup->casci->vectors, vectors.data());
  const fb::Status rdm = ci_rdm->rdm12(res, weights, vectors.data(), gamma.data(), gamma2.data());
  if (!rdm.has_value()) {
    ADD_FAILURE() << rdm.error().message;
    return nullptr;
  }
  setup->gamma = download(res, gamma.data(), n * n);
  setup->gamma2 = download(res, gamma2.data(), n * n * n * n);
  return setup;
}

// C exp(t kappa(x)) for the packed x: kappa unpacked on the device, then
// skew_eig and skew_expm.
std::optional<fb::SphericalMatrixAO> rotate(Setup &setup, const fb::SphericalMatrixAO &c,
                                            const std::vector<double> &x, double t) {
  fb::DeviceResources &res = setup.res();
  const auto nmo = static_cast<std::int32_t>(setup.nmo);
  const std::size_t nn = setup.nmo * setup.nmo;
  fb::Buffers::device<double> packed(x.size(), res.device());
  fb::Buffers::device<double> kappa(nn, res.device());
  fb::Buffers::device<wwr::wwrDoubleComplex> u(nn, res.device());
  fb::Buffers::device<double> w(setup.nmo, res.device());
  fb::Buffers::device<double> e(nn, res.device());
  upload(res, x, packed.data());
  if (!fb::unpack_orbital_pairs(res, nmo, setup.ncore, setup.ncas, packed.data(), kappa.data())
           .has_value() ||
      !fb::skew_eig<double>(res, nmo, kappa.data(), u.data(), w.data()).has_value() ||
      !fb::skew_expm<double>(res, nmo, u.data(), w.data(), t, e.data()).has_value()) {
    ADD_FAILURE() << "the rotation failed";
    return std::nullopt;
  }
  std::optional<fb::SphericalMatrixAO> rotated;
  rotated.emplace(res, *setup.basis);
  constexpr double kOne = 1.0;
  constexpr double kZero = 0.0;
  const auto status = wwr::gemm<double, std::int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, nmo, nmo, &kOne, c.data(), nmo,
      e.data(), nmo, &kZero, rotated->data(), nmo);
  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return rotated;
}

// Unit random directions in the packed space, from a fixed seed.
std::vector<std::vector<double>> directions(std::size_t count, std::size_t n_pairs) {
  std::mt19937_64 engine{250};
  std::normal_distribution<double> normal;
  std::vector<std::vector<double>> out(count, std::vector<double>(n_pairs));
  for (auto &x : out) {
    double norm = 0.0;
    for (double &value : x) {
      value = normal(engine);
      norm += value * value;
    }
    for (double &value : x) {
      value /= std::sqrt(norm);
    }
  }
  return out;
}

double dot(const std::vector<double> &a, const std::vector<double> &b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double norm(const std::vector<double> &a) {
  return std::sqrt(dot(a, a));
}

// H X for the columns of x, applied as one block.
std::vector<std::vector<double>> apply_block(fb::DeviceResources &res, fb::OrbitalHessian &hessian,
                                             const std::vector<std::vector<double>> &x) {
  const auto n = static_cast<std::size_t>(hessian.size());
  std::vector<double> block;
  for (const auto &column : x) {
    block.insert(block.end(), column.begin(), column.end());
  }
  fb::Buffers::device<double> in(block.size(), res.device());
  fb::Buffers::device<double> out(block.size(), res.device());
  upload(res, block, in.data());
  const fb::Status applied =
      hessian.apply(res, static_cast<std::int32_t>(x.size()), in.data(), out.data());
  EXPECT_TRUE(applied.has_value()) << applied.error().message;
  const std::vector<double> got = download(res, out.data(), block.size());
  std::vector<std::vector<double>> columns;
  for (std::size_t i = 0; i < x.size(); ++i) {
    columns.emplace_back(got.begin() + static_cast<std::ptrdiff_t>(i * n),
                         got.begin() + static_cast<std::ptrdiff_t>((i + 1) * n));
  }
  return columns;
}

// The gradient of f(K) = E(C exp(K)) at K = t kappa(x), from casscf_fock at
// C' = C exp(t kappa): exp(K + Y) = exp(K) exp(Y - [K, Y] / 2 + O(K^2 Y)), so
// df/dY = G' + t [kappa, G'] / 2 packed, G' = 2 (F' - F'^T) over every pair.
// casscf_fock's gradient alone is G' packed: its t derivative differs from
// H x by [kappa, G] / 2, which is not small off a stationary point.
std::optional<std::vector<double>> reference_gradient(Setup &setup, const fb::CasscfFock &fock,
                                                      const std::vector<double> &kappa, double t) {
  const std::size_t nmo = setup.nmo;
  const std::vector<double> f = download(setup.res(), fock.fock.data(), nmo * nmo);
  std::vector<double> g(nmo * nmo);
  for (std::size_t q = 0; q < nmo; ++q) {
    for (std::size_t p = 0; p < nmo; ++p) {
      g[p + (nmo * q)] = 2.0 * (f[p + (nmo * q)] - f[q + (nmo * p)]);
    }
  }
  std::vector<double> corrected = g;
  for (std::size_t q = 0; q < nmo; ++q) {
    for (std::size_t p = 0; p < nmo; ++p) {
      double commutator = 0.0;
      for (std::size_t r = 0; r < nmo; ++r) {
        commutator +=
            (kappa[p + (nmo * r)] * g[r + (nmo * q)]) - (g[p + (nmo * r)] * kappa[r + (nmo * q)]);
      }
      corrected[p + (nmo * q)] += 0.5 * t * commutator;
    }
  }
  std::vector<double> packed;
  for (const auto &[p, q] : packed_pairs(nmo, static_cast<std::size_t>(setup.ncore),
                                         static_cast<std::size_t>(setup.ncas))) {
    packed.push_back(corrected[p + (nmo * q)]);
  }
  return packed;
}

// d/dt of reference_gradient along x at t = 0, by Richardson's extrapolation
// of two central differences: H x, with fock(orbitals) casscf_fock's result
// at the setup's RDMs.
template <typename Fock>
std::optional<std::vector<double>> gradient_slope(Setup &setup, const fb::SphericalMatrixAO &c,
                                                  const std::vector<double> &x, Fock &&fock) {
  const std::size_t nmo = setup.nmo;
  std::vector<double> kappa(nmo * nmo, 0.0);
  const auto order = packed_pairs(nmo, static_cast<std::size_t>(setup.ncore),
                                  static_cast<std::size_t>(setup.ncas));
  for (std::size_t i = 0; i < order.size(); ++i) {
    const auto [p, q] = order[i];
    kappa[p + (nmo * q)] = x[i];
    kappa[q + (nmo * p)] = -x[i];
  }
  std::array<std::vector<double>, 2> central;
  for (std::size_t i = 0; i < 2; ++i) {
    const double step = kStep / static_cast<double>(1U << i);
    std::array<std::vector<double>, 2> ends;
    for (std::size_t side = 0; side < 2; ++side) {
      const double t = side == 0 ? step : -step;
      auto rotated = rotate(setup, c, x, t);
      if (!rotated.has_value()) {
        return std::nullopt;
      }
      auto at = fock(*rotated);
      if (!at.has_value()) {
        ADD_FAILURE() << at.error().message;
        return std::nullopt;
      }
      auto g = reference_gradient(setup, *at, kappa, t);
      if (!g.has_value()) {
        return std::nullopt;
      }
      ends[side] = std::move(*g);
    }
    central[i].resize(order.size());
    for (std::size_t j = 0; j < order.size(); ++j) {
      central[i][j] = (ends[0][j] - ends[1][j]) / (2.0 * step);
    }
  }
  std::vector<double> slope(order.size());
  for (std::size_t j = 0; j < order.size(); ++j) {
    slope[j] = ((4.0 * central[1][j]) - central[0][j]) / 3.0;
  }
  return slope;
}

// H x against the gradient's finite difference along each x, for the
// columns of x and their products hx.
template <typename Fock>
void expect_derivative(Setup &setup, const fb::SphericalMatrixAO &c,
                       const std::vector<std::vector<double>> &x,
                       const std::vector<std::vector<double>> &hx, Fock &&fock) {
  for (std::size_t a = 0; a < x.size(); ++a) {
    const auto fd = gradient_slope(setup, c, x[a], fock);
    ASSERT_TRUE(fd.has_value());
    double error = 0.0;
    for (std::size_t j = 0; j < fd->size(); ++j) {
      error += ((*fd)[j] - hx[a][j]) * ((*fd)[j] - hx[a][j]);
    }
    EXPECT_LE(std::sqrt(error), kFiniteDifferenceTolerance * norm(hx[a]))
        << "direction " << a << ": |H x| " << norm(hx[a]);
  }
}

// x_a^T H x_b = x_b^T H x_a over every pair of the columns of x, hx their
// products, to kSymmetryTolerance of scale |x_a| |x_b|, scale H's.
void expect_symmetric(const std::vector<std::vector<double>> &x,
                      const std::vector<std::vector<double>> &hx, double scale) {
  for (std::size_t a = 0; a < x.size(); ++a) {
    for (std::size_t b = a + 1; b < x.size(); ++b) {
      const double ab = dot(x[a], hx[b]);
      const double ba = dot(x[b], hx[a]);
      EXPECT_LE(std::abs(ab - ba), kSymmetryTolerance * scale * norm(x[a]) * norm(x[b]))
          << "directions " << a << ", " << b << ": " << ab << " and " << ba;
    }
  }
}

// The operator at the setup's orbitals; tau is both J's and K's screen.
fb::Result<fb::OrbitalHessian> build_hessian(Setup &setup, const fb::SphericalMatrixAO &c,
                                             const fb::Buffers::device<double> &ppaa,
                                             const fb::Buffers::device<double> &papa,
                                             const fb::CasscfFock &fock, double tau = kTau) {
  return fb::OrbitalHessian::create(setup.res(), *setup.basis, setup.h->h, setup.transform->u,
                                    setup.j_classes, setup.k_classes, c, setup.ncore, setup.ncas,
                                    ppaa, papa, fock, tau, tau);
}

struct Case {
  std::string_view name;
  // Under slow_test.h: every case but H2O cc-pVDZ, the cheapest.
  bool slow;
};

// ctest names a discovered case by this, so `ctest -R n2_ccpvdz` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

class McscfHessian : public ::testing::TestWithParam<Case> {};

TEST_P(McscfHessian, MatchesPyscfAtTheCasciOrbitals) {
  FBF_REQUIRE_GPU();
  if (GetParam().slow) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &entry = reference_case(GetParam().name);
  const fb::json &casci = entry.at("casci");
  auto setup = make_setup(entry, true);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const fb::SphericalMatrixAO &c = setup->casci->c;
  const auto order = packed_pairs(setup->nmo, static_cast<std::size_t>(setup->ncore),
                                  static_cast<std::size_t>(setup->ncas));

  auto integrals = setup->integrals(c);
  ASSERT_TRUE(integrals.has_value()) << integrals.error().message;
  auto fock = setup->fock(c, integrals->first);
  ASSERT_TRUE(fock.has_value()) << fock.error().message;
  auto hessian = build_hessian(*setup, c, integrals->first, integrals->second, *fock);
  ASSERT_TRUE(hessian.has_value()) << hessian.error().message;
  ASSERT_EQ(static_cast<std::size_t>(hessian->size()), order.size());

  // The diagonal, summed over each pair of orbital sets.
  const std::vector<double> diagonal = download(res, hessian->diagonal().data(), order.size());
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> index;
  for (std::size_t i = 0; i < order.size(); ++i) {
    index[order[i]] = i;
  }
  std::size_t covered = 0;
  for (const fb::json &pair : casci.at("pairs")) {
    const auto ps = pair.at("p").get<std::vector<std::size_t>>();
    const auto qs = pair.at("q").get<std::vector<std::size_t>>();
    double sum = 0.0;
    for (const std::size_t p : ps) {
      for (const std::size_t q : qs) {
        const auto at = index.find({p, q});
        ASSERT_NE(at, index.end()) << "(" << p << ", " << q << ") is not a packed pair";
        sum += diagonal[at->second];
        ++covered;
      }
    }
    EXPECT_NEAR(sum, pair.at("hessian_diagonal").get<double>(), kDiagonalTolerance)
        << "p " << ps.front() << ", q " << qs.front();
  }
  EXPECT_EQ(covered, order.size());

  // gᵀ H g and |H g|, and H against the gradient's finite difference along
  // random kappa; both, and the random ones, in one block.
  std::vector<std::vector<double>> x{download(res, fock->gradient.data(), order.size())};
  for (auto &d : directions(kDirections, order.size())) {
    x.push_back(std::move(d));
  }
  const auto hx = apply_block(res, *hessian, x);
  const double ghg = dot(x[0], hx[0]);
  const double want_ghg = casci.at("g_hessian_g").get<double>();
  const double want_hg = casci.at("hessian_g_norm").get<double>();
  EXPECT_LE(std::abs(ghg - want_ghg), kInvariantTolerance * std::abs(want_ghg))
      << ghg << " against " << want_ghg;
  EXPECT_LE(std::abs(norm(hx[0]) - want_hg), kInvariantTolerance * want_hg)
      << norm(hx[0]) << " against " << want_hg;

  const std::vector<std::vector<double>> random(x.begin() + 1, x.end());
  const std::vector<std::vector<double>> h_random(hx.begin() + 1, hx.end());
  expect_derivative(*setup, c, random, h_random,
                    [&](const fb::SphericalMatrixAO &orbitals) { return setup->fock(orbitals); });

  // Symmetry over the gradient and the random directions, at the default
  // taus, against max |diag H|: H's scale without forming H, and no more than
  // max |H|, so the bar is no looser for it.
  double top = 0.0;
  for (const double d : diagonal) {
    top = std::max(top, std::abs(d));
  }
  expect_symmetric(x, hx, top);
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfHessian,
                         ::testing::Values(Case{.name = "h2o_ccpvdz", .slow = false},
                                           Case{.name = "n2_ccpvdz", .slow = true},
                                           Case{.name = "h2co_ccpvdz_sa2", .slow = true},
                                           Case{.name = "h2co_ccpvdz_sa2_weighted", .slow = true},
                                           Case{.name = "h2o_ccpvdz_avas", .slow = true}),
                         [](const auto &info) { return std::string(info.param.name); });

// The whole H on H2O STO-3G, one column per unit vector in one block: its
// diagonal is diagonal(), it is symmetric, and two of its products are the
// gradient's finite differences.
void expect_whole_hessian(Setup &setup, const fb::SphericalMatrixAO &c,
                          const fb::Buffers::device<double> &ppaa,
                          const fb::Buffers::device<double> &papa) {
  fb::DeviceResources &res = setup.res();
  auto fock = setup.fock(c, ppaa);
  ASSERT_TRUE(fock.has_value()) << fock.error().message;
  auto hessian = build_hessian(setup, c, ppaa, papa, *fock);
  ASSERT_TRUE(hessian.has_value()) << hessian.error().message;
  const auto n = static_cast<std::size_t>(hessian->size());
  ASSERT_EQ(n, packed_pairs(setup.nmo, static_cast<std::size_t>(setup.ncore),
                            static_cast<std::size_t>(setup.ncas))
                   .size());
  ASSERT_GT(n, 0U);
  std::vector<std::vector<double>> unit(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    unit[i][i] = 1.0;
  }
  const auto columns = apply_block(res, *hessian, unit);
  const std::vector<double> diagonal = download(res, hessian->diagonal().data(), n);
  double top = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      top = std::max(top, std::abs(columns[j][i]));
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(diagonal[i], columns[i][i], kDiagonalTolerance) << "pair " << i;
    for (std::size_t j = 0; j < i; ++j) {
      EXPECT_LE(std::abs(columns[j][i] - columns[i][j]), kSymmetryTolerance * top)
          << "(" << i << ", " << j << ")";
    }
  }

  const auto x = directions(kDirections, n);
  expect_derivative(setup, c, x, apply_block(res, *hessian, x),
                    [&](const fb::SphericalMatrixAO &orbitals) { return setup.fock(orbitals); });
}

// H2O STO-3G, frontier CAS(2,2) at two singlets weighted [0.6, 0.4]: every
// kind of pair (core-active, core-virtual, active-virtual) and the weighted
// RDMs.
TEST(McscfHessianUnitVectors, DiagonalIsHOnEveryUnitVector) {
  FBF_REQUIRE_GPU();
  fb::json entry = reference_case("h2o_ccpvdz");
  entry["name"] = "h2o_sto3g_cas22";
  entry["basis"] = "sto-3g";
  entry["active"] = nullptr;
  entry["ncas"] = 2;
  entry["nelecas"] = 2;
  entry["ncore"] = 4;
  entry["weights"] = std::vector<double>{0.6, 0.4};
  entry.erase("casci");
  auto setup = make_setup(entry, true);
  ASSERT_NE(setup, nullptr);
  const fb::ScopedDeviceIndex on{setup->res().dev_idx()};
  const fb::SphericalMatrixAO &c = setup->casci->c;
  auto integrals = setup->integrals(c);
  ASSERT_TRUE(integrals.has_value()) << integrals.error().message;
  expect_whole_hessian(*setup, c, integrals->first, integrals->second);
}

// ncas = 0, the RHF orbital Hessian, at and off the SCF: with no active space
// there is no H_act, and every pair is core-virtual.
TEST(McscfHessianRhfLimit, DiagonalIsHOnEveryUnitVector) {
  FBF_REQUIRE_GPU();
  fb::json entry = reference_case("h2o_ccpvdz");
  entry["basis"] = "sto-3g";
  auto setup = make_setup(entry, false);
  ASSERT_NE(setup, nullptr);
  const fb::ScopedDeviceIndex on{setup->res().dev_idx()};
  setup->ncore = setup->molecule->electron_count() / 2;
  setup->ncas = 0;
  auto integrals = setup->integrals(setup->scf->c);
  ASSERT_TRUE(integrals.has_value()) << integrals.error().message;
  expect_whole_hessian(*setup, setup->scf->c, integrals->first, integrals->second);
  const auto n = packed_pairs(setup->nmo, static_cast<std::size_t>(setup->ncore), 0).size();
  auto off = rotate(*setup, setup->scf->c, directions(1, n).front(), 0.05);
  ASSERT_TRUE(off.has_value());
  expect_whole_hessian(*setup, *off, integrals->first, integrals->second);
}

// Everything create refuses, and apply's.
TEST(McscfHessianRefusals, AreInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::json entry = reference_case("h2o_ccpvdz");
  entry["basis"] = "sto-3g";
  auto setup = make_setup(entry, false);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const std::size_t nmo = setup->nmo;
  setup->ncore = 4;
  setup->ncas = 2;
  setup->gamma = {1.0, 0.0, 0.0, 1.0};
  setup->gamma2.assign(16, 0.0);
  auto integrals = setup->integrals(setup->scf->c);
  ASSERT_TRUE(integrals.has_value()) << integrals.error().message;
  const fb::Buffers::device<double> &ppaa = integrals->first;
  const fb::Buffers::device<double> &papa = integrals->second;
  auto fock = setup->fock(setup->scf->c, ppaa);
  ASSERT_TRUE(fock.has_value()) << fock.error().message;
  const fb::Buffers::device<double> short_ppaa(nmo * nmo, res.device());
  const auto call = [&](std::int32_t ncore, std::int32_t ncas, const fb::Buffers::device<double> &p,
                        double tau = kTau) {
    return fb::OrbitalHessian::create(res, *setup->basis, setup->h->h, setup->transform->u,
                                      setup->j_classes, setup->k_classes, setup->scf->c, ncore,
                                      ncas, p, papa, *fock, tau, tau);
  };
  const auto refused = [](const fb::Result<fb::OrbitalHessian> &r, std::string_view what) {
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(r.error().message.find(what), std::string::npos) << r.error().message;
  };
  auto built = call(4, 2, ppaa);
  ASSERT_TRUE(built.has_value()) << built.error().message;
  refused(call(-1, 2, ppaa), "must not be negative");
  refused(call(6, 2, ppaa), "exceeds the 7 orbitals");
  refused(call(0, 0, ppaa), "nothing is occupied");
  refused(call(4, 2, ppaa, -1.0), "must be 0 or above");
  refused(call(4, 2, ppaa, std::nan("")), "must be 0 or above");
  refused(call(4, 2, short_ppaa), "ppaa holds");
  refused(call(3, 2, ppaa), "CasscfFock is not of");
  const fb::Status negative = built->apply(res, -1, nullptr, nullptr);
  ASSERT_FALSE(negative.has_value());
  EXPECT_EQ(negative.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_TRUE(built->apply(res, 0, nullptr, nullptr).has_value());
}

} // namespace
