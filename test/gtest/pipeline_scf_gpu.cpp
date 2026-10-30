// The Scf step against PySCF's RHF and ROHF (data/scf/rhf.json, rohf.json and
// rohf_localized_hole.json, tools/emit/scf_reference.py), integral-direct:
// the energy and orbital energies, and the invariants of the wavefunction,
// over the setup steps and the guess run by hand; then through the real
// registry from a config naming only [scf]. Runs the device chain, so needs a
// GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

using ScfInput = fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                          fb::EriOutput, fb::OverlapOutput, fb::OrthogonalizerOutput,
                          fb::CoreHamiltonianOutput, fb::GuessOutput>;

// Against PySCF converged to conv_tol 1e-12. What is left is J's and K's
// screening and the step's own tolerances.
constexpr double kEnergyTolerance = 1e-8;
constexpr double kOrbitalEnergyTolerance = 1e-6;

fb::json read_reference(std::string_view file) {
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "scf" / file);
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return read.has_value() ? std::move(*read) : fb::json{};
}

const fb::json &find_case(const fb::json &doc, std::string_view file, std::string_view name) {
  for (const fb::json &entry : doc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in scf/" << file;
  static const fb::json kNone;
  return kNone;
}

const fb::json &reference_case(std::string_view name) {
  static const fb::json doc = read_reference("rhf.json");
  return find_case(doc, "rhf.json", name);
}

// rohf.json's cases and rohf_localized_hole.json's, as one list: the second
// file's are minima PySCF's own ROHF iteration does not converge to, with
// their orbital energies by occupation class, as this step's columns are.
const fb::json &rohf_case(std::string_view name) {
  static const fb::json doc = [] {
    fb::json merged = read_reference("rohf.json");
    for (const fb::json &entry : read_reference("rohf_localized_hole.json").at("cases")) {
      merged.at("cases").push_back(entry);
    }
    return merged;
  }();
  return find_case(doc, "rohf.json or rohf_localized_hole.json", name);
}

std::vector<fb::Atom> atoms_of(const fb::json &entry) {
  std::vector<fb::Atom> atoms;
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    atoms.push_back(fb::Atom{{coord[0], coord[1], coord[2]},
                             static_cast<fb::Element>(atom.at("z").get<std::int32_t>())});
  }
  return atoms;
}

std::vector<double> download(fb::DeviceResources &res, const fb::SphericalMatrixAO &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// Column-major n x n products on the host: small enough to check exactly.
std::vector<double> matmul(const std::vector<double> &a, const std::vector<double> &b,
                           std::size_t n, bool transpose_a = false) {
  std::vector<double> c(n * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < n; ++k) {
      const double bkj = b[k + (n * j)];
      for (std::size_t i = 0; i < n; ++i) {
        c[i + (n * j)] += (transpose_a ? a[k + (n * i)] : a[i + (n * k)]) * bkj;
      }
    }
  }
  return c;
}

// Every product the Scf step reads, made by the steps called by hand in the
// order the planner would choose; the ERIs integral-direct.
struct Inputs {
  fb::MoleculeOutput molecule;
  fb::BasisSetOutput basis;
  fb::SphericalTransformOutput transform;
  fb::EriOutput eri;
  fb::OverlapOutput overlap;
  fb::OrthogonalizerOutput x;
  fb::CoreHamiltonianOutput h;
  fb::GuessOutput guess;

  [[nodiscard]] ScfInput view() const {
    return ScfInput{molecule, basis, transform, eri, overlap, x, h, guess};
  }
};

std::optional<Inputs> make_inputs(fb::DeviceResources &res, std::vector<fb::Atom> atoms,
                                  std::string_view basis_name, std::int32_t charge,
                                  std::int32_t multiplicity = 1) {
  const fb::EriStep eri_step{};
  fb::MoleculeOutput molecule{fb::Molecule(res, std::move(atoms), charge, multiplicity)};
  auto basis = fb::BasisSetStep{.basis = std::string(basis_name)}.run(
      fb::View<fb::MoleculeOutput>{molecule});
  if (!basis.has_value()) {
    ADD_FAILURE() << basis.error().message;
    return std::nullopt;
  }
  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  auto pairs = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  if (!transform.has_value() || !pairs.has_value()) {
    ADD_FAILURE() << "spherical transform or pair classes failed";
    return std::nullopt;
  }
  auto eri = eri_step.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
          molecule, *basis, *pairs},
      res);
  if (!eri.has_value()) {
    ADD_FAILURE() << eri.error().message;
    return std::nullopt;
  }
  auto overlap = fb::OverlapStep{}.run(
      fb::View<fb::BasisSetOutput, fb::SphericalTransformOutput, fb::PrimitivePairClassesOutput>{
          *basis, *transform, *pairs},
      res);
  if (!overlap.has_value()) {
    ADD_FAILURE() << overlap.error().message;
    return std::nullopt;
  }
  auto x = fb::OrthogonalizerStep{}.run(
      fb::View<fb::BasisSetOutput, fb::OverlapOutput>{*basis, *overlap}, res);
  auto h = fb::CoreHamiltonianStep{}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
               fb::PrimitivePairClassesOutput>{molecule, *basis, *transform, *pairs},
      res);
  if (!x.has_value() || !h.has_value()) {
    ADD_FAILURE() << "orthogonalizer or core Hamiltonian failed";
    return std::nullopt;
  }
  fb::tf::Executor executor(4);
  fb::tf::Taskflow flow;
  fb::Result<fb::GuessOutput> guess = fb::err_config("the guess task did not run");
  flow.emplace([&](fb::tf::Subflow &subflow) {
    guess = fb::GuessStep{}.run(fb::View<fb::MoleculeOutput, fb::BasisSetOutput>{molecule, *basis},
                                subflow);
  });
  executor.run(flow).wait();
  if (!guess.has_value()) {
    ADD_FAILURE() << guess.error().message;
    return std::nullopt;
  }
  return Inputs{std::move(molecule), std::move(*basis), std::move(*transform), std::move(*eri),
                std::move(*overlap), std::move(*x),     std::move(*h),         std::move(*guess)};
}

class PipelineScf : public ::testing::TestWithParam<std::string_view> {};

TEST_P(PipelineScf, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  // Under slow_test.h: CO cc-pVTZ, the dearest case.
  if (GetParam() == "co_ccpvtz") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &expected = reference_case(GetParam());
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(expected), expected.at("basis").get<std::string>(),
                            expected.at("charge").get<std::int32_t>());
  ASSERT_TRUE(inputs.has_value());
  const std::size_t n = inputs->basis.basis.nao_sphe();
  ASSERT_EQ(n, expected.at("nao").get<std::size_t>());

  auto scf = fb::ScfStep{}.run(inputs->view(), res);
  ASSERT_TRUE(scf.has_value()) << scf.error().message;
  EXPECT_NEAR(fb::nuclear_repulsion_energy(inputs->molecule.molecule),
              expected.at("nuclear_repulsion").get<double>(), 1e-10);
  EXPECT_NEAR(scf->energy, expected.at("energy").get<double>(), kEnergyTolerance);
  const auto mo_energies = expected.at("mo_energies").get<std::vector<double>>();
  ASSERT_EQ(scf->mo_energies.size(), mo_energies.size());
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(scf->mo_energies[i], mo_energies[i], kOrbitalEnergyTolerance) << i;
  }
  EXPECT_LT(scf->iterations, fb::ScfStep{}.max_iterations);

  // The invariants of the closed-shell wavefunction it returned, whatever the
  // AO normalisation: C^T S C = I, tr(D S) = N and D S D = 2 D.
  const std::vector<double> s = download(res, inputs->overlap.s);
  const std::vector<double> c = download(res, scf->c);
  const std::vector<double> d = download(res, scf->d);
  const std::vector<double> ctsc = matmul(c, matmul(s, c, n), n, /*transpose_a=*/true);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(ctsc[i + (n * j)], i == j ? 1.0 : 0.0, 1e-10) << i << "," << j;
    }
  }
  double trace = 0.0;
  for (std::size_t k = 0; k < n * n; ++k) {
    trace += d[k] * s[k]; // both symmetric
  }
  EXPECT_NEAR(trace, expected.at("electrons").get<double>(), 1e-10);
  const std::vector<double> dsd = matmul(d, matmul(s, d, n), n);
  for (std::size_t k = 0; k < n * n; ++k) {
    EXPECT_NEAR(dsd[k], 2.0 * d[k], 1e-10) << k;
  }
}

INSTANTIATE_TEST_SUITE_P(Rhf, PipelineScf,
                         ::testing::Values("h2o_sto3g", "h2o_ccpvdz", "nh4_ccpvdz", "co_ccpvtz"),
                         [](const auto &info) { return std::string(info.param); });

// Incremental Fock builds (the default) against a full build every iteration:
// G is linear in D, so the two differ only by J's and K's screening of the
// density differences, and the converged result is a full build's either way.
TEST(PipelineScfIncremental, MatchesFullBuilds) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case("h2o_ccpvdz");
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(expected), expected.at("basis").get<std::string>(),
                            expected.at("charge").get<std::int32_t>());
  ASSERT_TRUE(inputs.has_value());
  ASSERT_GT(fb::ScfStep{}.fock_rebuild_interval, 1);

  auto incremental = fb::ScfStep{}.run(inputs->view(), res);
  auto full = fb::ScfStep{.fock_rebuild_interval = 1}.run(inputs->view(), res);
  ASSERT_TRUE(incremental.has_value()) << incremental.error().message;
  ASSERT_TRUE(full.has_value()) << full.error().message;
  EXPECT_NEAR(incremental->energy, full->energy, 1e-9);
  ASSERT_EQ(incremental->mo_energies.size(), full->mo_energies.size());
  for (std::size_t i = 0; i < full->mo_energies.size(); ++i) {
    EXPECT_NEAR(incremental->mo_energies[i], full->mo_energies[i], 1e-7) << i;
  }
}

class PipelineScfRohf : public ::testing::TestWithParam<std::string_view> {};

// The density of the first @p nocc columns of @p c, C_occ C_occ^T, on the host.
std::vector<double> spin_density(const std::vector<double> &c, std::size_t n, std::size_t nocc) {
  std::vector<double> d(n * n, 0.0);
  for (std::size_t k = 0; k < nocc; ++k) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t i = 0; i < n; ++i) {
        d[i + (n * j)] += c[i + (n * k)] * c[j + (n * k)];
      }
    }
  }
  return d;
}

TEST_P(PipelineScfRohf, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  // Under slow_test.h: CH2 cc-pVTZ, the dearest case.
  if (GetParam() == "ch2_ccpvtz") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &expected = rohf_case(GetParam());
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(expected), expected.at("basis").get<std::string>(),
                            expected.at("charge").get<std::int32_t>(),
                            expected.at("multiplicity").get<std::int32_t>());
  ASSERT_TRUE(inputs.has_value());
  const std::size_t n = inputs->basis.basis.nao_sphe();
  ASSERT_EQ(n, expected.at("nao").get<std::size_t>());

  auto scf = fb::ScfStep{}.run(inputs->view(), res);
  ASSERT_TRUE(scf.has_value()) << scf.error().message;
  EXPECT_NEAR(fb::nuclear_repulsion_energy(inputs->molecule.molecule),
              expected.at("nuclear_repulsion").get<double>(), 1e-10);
  const auto n_alpha = expected.at("n_alpha").get<std::int32_t>();
  const auto n_beta = expected.at("n_beta").get<std::int32_t>();
  EXPECT_EQ(scf->n_alpha, n_alpha);
  EXPECT_EQ(scf->n_beta, n_beta);
  EXPECT_NEAR(scf->energy, expected.at("energy").get<double>(), kEnergyTolerance);
  const auto mo_energies = expected.at("mo_energies").get<std::vector<double>>();
  ASSERT_EQ(scf->mo_energies.size(), mo_energies.size());
  double orbital_error = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(scf->mo_energies[i], mo_energies[i], kOrbitalEnergyTolerance) << i;
    orbital_error = std::max(orbital_error, std::abs(scf->mo_energies[i] - mo_energies[i]));
  }
  EXPECT_LT(scf->iterations, fb::ScfStep{}.max_iterations);
  // The errors themselves, for the record a PR reports.
  std::cout << std::format(
      "{}: {} iterations, |dE| = {:.2e} Eh, max |d eps| = {:.2e} Eh\n", GetParam(), scf->iterations,
      std::abs(scf->energy - expected.at("energy").get<double>()), orbital_error);

  // The invariants of the open-shell wavefunction: C^T S C = I; each spin
  // density, Da = C[:, :n_alpha] C[:, :n_alpha]^T and Db over n_beta, holds
  // its own electrons, tr(Ds S) = n_s, and is idempotent in S, Ds S Ds = Ds;
  // and the published D is their sum.
  const std::vector<double> s = download(res, inputs->overlap.s);
  const std::vector<double> c = download(res, scf->c);
  const std::vector<double> d = download(res, scf->d);
  const std::vector<double> ctsc = matmul(c, matmul(s, c, n), n, /*transpose_a=*/true);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(ctsc[i + (n * j)], i == j ? 1.0 : 0.0, 1e-10) << i << "," << j;
    }
  }
  const std::vector<double> da = spin_density(c, n, static_cast<std::size_t>(n_alpha));
  const std::vector<double> db = spin_density(c, n, static_cast<std::size_t>(n_beta));
  for (const auto &[density, electrons] : {std::pair{&da, n_alpha}, std::pair{&db, n_beta}}) {
    double trace = 0.0;
    for (std::size_t k = 0; k < n * n; ++k) {
      trace += (*density)[k] * s[k]; // both symmetric
    }
    EXPECT_NEAR(trace, electrons, 1e-10);
    const std::vector<double> dsd = matmul(*density, matmul(s, *density, n), n);
    for (std::size_t k = 0; k < n * n; ++k) {
      EXPECT_NEAR(dsd[k], (*density)[k], 1e-10) << k;
    }
  }
  for (std::size_t k = 0; k < n * n; ++k) {
    EXPECT_NEAR(d[k], da[k] + db[k], 1e-12) << k;
  }
}

// Doublets OH (STO-3G and cc-pVDZ), NH2 and H2O+ (charged, so the guess is
// rescaled before it is split) and triplet CH2 (off every axis) in cc-pVDZ,
// and CH2 in cc-pVTZ: f shells. And glycylglycine+ in STO-3G, whose hole sits
// on one carbonyl lone pair: on PySCF's Roothaan matrix its open orbital lies
// below three doubly occupied ones, so the loop ran 150 iterations without
// converging (issue #476) until it iterated on an open block of its own. Its
// published open orbital energy, PySCF's, stays in its column below three
// higher core ones.
INSTANTIATE_TEST_SUITE_P(Rohf, PipelineScfRohf,
                         ::testing::Values("oh_sto3g", "oh_ccpvdz", "nh2_ccpvdz", "ch2_ccpvdz",
                                           "h2o_cation_ccpvdz", "ch2_ccpvtz",
                                           "glygly_cation_sto3g"),
                         [](const auto &info) { return std::string(info.param); });

// The open shell's incremental builds, J of D - D_built and each spin's K of
// Ds - Ds_built, against a full build every iteration, on a doublet.
TEST(PipelineScfIncremental, RohfMatchesFullBuilds) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = rohf_case("oh_ccpvdz");
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(expected), expected.at("basis").get<std::string>(),
                            expected.at("charge").get<std::int32_t>(),
                            expected.at("multiplicity").get<std::int32_t>());
  ASSERT_TRUE(inputs.has_value());
  ASSERT_GT(fb::ScfStep{}.fock_rebuild_interval, 1);

  auto incremental = fb::ScfStep{}.run(inputs->view(), res);
  auto full = fb::ScfStep{.fock_rebuild_interval = 1}.run(inputs->view(), res);
  ASSERT_TRUE(incremental.has_value()) << incremental.error().message;
  ASSERT_TRUE(full.has_value()) << full.error().message;
  EXPECT_NEAR(incremental->energy, full->energy, 1e-9);
}

// molden set: the converged orbitals land in a Molden file, every MO, the
// occupied ones at 2. The writer's own conversions are io_molden_host's.
TEST(PipelineScfMolden, WritesTheConvergedOrbitals) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case("h2o_sto3g");
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(expected), expected.at("basis").get<std::string>(),
                            expected.at("charge").get<std::int32_t>());
  ASSERT_TRUE(inputs.has_value());
  const fb::test::ScratchDir dir("pipeline_scf_molden");
  auto scf = fb::ScfStep{.molden = dir / "h2o.molden"}.run(inputs->view(), res);
  ASSERT_TRUE(scf.has_value()) << scf.error().message;

  std::ifstream file(dir / "h2o.molden");
  ASSERT_TRUE(file.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(file, line));
  EXPECT_EQ(line, "[Molden Format]");
  std::size_t orbitals = 0;
  std::size_t occupied = 0;
  while (std::getline(file, line)) {
    orbitals += line.starts_with(" Ene=") ? 1 : 0;
    occupied += line == " Occup= 2.000000" ? 1 : 0;
  }
  EXPECT_EQ(orbitals, inputs->basis.basis.nao_sphe());
  EXPECT_EQ(occupied, expected.at("electrons").get<std::size_t>() / 2);
}

// Ten electrons cannot be a doublet, nor H2O+'s nine a singlet: n_alpha -
// n_beta = multiplicity - 1 has N's parity.
TEST(PipelineScfErrors, AMultiplicityOfTheWrongParityIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  for (const auto [charge, multiplicity] :
       {std::pair{0, 2}, std::pair{1, 1}, std::pair{0, 0}, std::pair{0, 13}}) {
    auto inputs =
        make_inputs(res, atoms_of(reference_case("h2o_sto3g")), "sto-3g", charge, multiplicity);
    ASSERT_TRUE(inputs.has_value());
    auto scf = fb::ScfStep{}.run(inputs->view(), res);
    ASSERT_FALSE(scf.has_value()) << charge << " " << multiplicity;
    EXPECT_EQ(scf.error().kind, fb::ErrorKind::InvalidConfig) << charge << " " << multiplicity;
  }
}

TEST(PipelineScfErrors, TooFewIterationsIsMaxIterationsExceeded) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(reference_case("h2o_sto3g")), "sto-3g", 0);
  ASSERT_TRUE(inputs.has_value());
  auto scf = fb::ScfStep{.max_iterations = 2}.run(inputs->view(), res);
  ASSERT_FALSE(scf.has_value());
  EXPECT_EQ(scf.error().kind, fb::ErrorKind::MaxIterationsExceeded);
}

TEST(PipelineScfErrors, AnOptionOutOfRangeIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(reference_case("h2o_sto3g")), "sto-3g", 0);
  ASSERT_TRUE(inputs.has_value());
  for (const fb::ScfStep step :
       {fb::ScfStep{.max_iterations = 0}, fb::ScfStep{.diis_history = 0},
        fb::ScfStep{.fock_rebuild_interval = 0}, fb::ScfStep{.energy_tolerance = -1.0},
        fb::ScfStep{.commutator_tolerance = std::nan("")}, fb::ScfStep{.tau_j = -1.0},
        fb::ScfStep{.tau_j = std::nan("")}}) {
    auto scf = step.run(inputs->view(), res);
    ASSERT_FALSE(scf.has_value());
    EXPECT_EQ(scf.error().kind, fb::ErrorKind::InvalidConfig) << scf.error().message;
  }
}

// h2o_ccpvdz, and co_ccpvtz for f shells: a config names the basis by its
// builtin name and the molecule by an XYZ file, as a user would.
class PipelineScfCalculation : public ::testing::TestWithParam<std::string_view> {};

TEST_P(PipelineScfCalculation, AConfigNamingOnlyScfPlansAndRunsTheWholeChain) {
  FBF_REQUIRE_GPU();
  // Under slow_test.h: CO cc-pVTZ, the dearest case.
  if (GetParam() == "co_ccpvtz") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &expected = reference_case(GetParam());
  // The reference geometry is in bohr; XYZ is in Angstrom, read back through
  // the same constant, so the molecule is the reference's to rounding.
  std::string xyz = std::format("{}\n{}\n", expected.at("atoms").size(), GetParam());
  for (const fb::json &atom : expected.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  const fb::test::ScratchDir dir("pipeline_scf");
  ASSERT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  ASSERT_TRUE(fb::write_file(dir / "run.toml",
                             std::format("[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\n"
                                         "basis = \"{}\"\n[scf]\n",
                                         expected.at("basis").get<std::string>()))
                  .has_value());

  fb::NodeResources node;
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &scf = calc->get<fb::ScfOutput>();
  ASSERT_TRUE(scf.has_value());
  const std::int32_t n_occ = calc->get<fb::MoleculeOutput>()->molecule.electron_count() / 2;
  EXPECT_EQ(scf->n_alpha, n_occ);
  EXPECT_EQ(scf->n_beta, n_occ);
  EXPECT_NEAR(scf->energy, expected.at("energy").get<double>(), kEnergyTolerance);
  const auto mo_energies = expected.at("mo_energies").get<std::vector<double>>();
  ASSERT_EQ(scf->mo_energies.size(), mo_energies.size());
  for (std::size_t i = 0; i < mo_energies.size(); ++i) {
    EXPECT_NEAR(scf->mo_energies[i], mo_energies[i], kOrbitalEnergyTolerance) << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Rhf, PipelineScfCalculation, ::testing::Values("h2o_ccpvdz", "co_ccpvtz"),
                         [](const auto &info) { return std::string(info.param); });

} // namespace
