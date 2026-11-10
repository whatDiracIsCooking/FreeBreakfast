// The footprints (#1044) of the library routines the RHF chain calls outside
// src/recipes (#1048): the Molecule's upload, mmd's classes, J's and K's
// classes and builds, the one-electron integrals, and the density-fitting
// metric, three-center tensor, fit, J, K and MO transform; and those CASCI and
// FCI call (#1049): the mixed classes and overlap AVAS builds, the active-space
// Hamiltonian, and the CI's strings, sigma and solve; and those CASSCF's loop
// calls (#1151): its integrals, the RDMs, the Fock, the orbital Hessian and the
// CI-relaxed one, and the packed pairs' copies; and those the RHF gradient
// calls (#1050): the one-electron, J's and K's derivative contractions, the
// fitted metric's and three-center tensor's, and the two gradients. Each is
// held to the
// pool peak of one call by test/support/pool_footprint.h's two bars, sound
// (the measured peak is no more than the footprint's) and tight (within
// G x A), so an undercount fails at the routine that owns the allocation.
// Water in cc-pVDZ (s, p, d: six mmd classes, and U is not the identity) and
// cc-pVDZ-JKFIT. test/recipes/footprint_gpu.cpp has the recipes'.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.recipes.diagonalize; // syevd_lwork
import free_breakfast.ci;
import free_breakfast.response;
import free_breakfast.gradient;

#include "pool_footprint.h"

namespace {

namespace fb = free_breakfast;
namespace fbt = free_breakfast::test;

// Water, in bohr.
std::vector<fb::Atom> water() {
  return {fb::Atom{{0.0, 0.0, -0.1294}, fb::Element::O},
          fb::Atom{{0.0, -1.4941, 1.0274}, fb::Element::H},
          fb::Atom{{0.0, 1.4941, 1.0274}, fb::Element::H}};
}

// @p name, an orbital or a fitting set, placed over @p molecule with its
// primitives normalized, as the BasisSet and Eri steps place theirs.
fb::BasisSet placed(const std::string &name, const fb::Molecule &molecule) {
  auto stamp = fb::is_builtin_basis(name) ? fb::resolve_basis(name) : fb::resolve_aux_basis(name);
  EXPECT_TRUE(stamp) << name;
  auto basis = fb::place(stamp->primitive_normalized(), molecule);
  EXPECT_TRUE(basis) << name;
  return std::move(*basis);
}

// @p call held to @p fp on @p res. It runs once first, which grows each
// workspace to the most the call asks of it: on a DeviceResources whose
// workspaces nothing has grown yet, exactly the footprint's, and otherwise no
// further than it. Then again, measured, its workspaces already grown, as the
// pipeline grows them before the first step: the pool's peak is then the
// call's persistent and scratch, with @p live allocations live at it. No
// routine here keeps a CountedPinned, so the pinned total is the workspace.
// With @p workspace_bound the device workspace is held to at most the
// footprint's, and the gap logged: a Davidson's syevd is counted at its
// subspace's largest order, which a solve that converges first never reaches.
template <typename Call>
void expect_routine_footprint(fb::DeviceResources &res, const fb::MemoryFootprint &fp,
                              const std::size_t live, Call &&call, bool workspace_bound = false) {
  const std::size_t device_before = res.device_workspace(0).num_elements();
  const std::size_t pinned_before = res.pinned_bytes();
  call();
  const std::size_t grown = res.device_workspace(0).num_elements();
  if (workspace_bound) {
    EXPECT_LE(grown, std::max(device_before, fp.device.workspace));
    std::cout << std::format("[ footprint ] workspace {} of the bound {}\n", grown,
                             fp.device.workspace);
  } else {
    EXPECT_EQ(grown, std::max(device_before, fp.device.workspace));
  }
  EXPECT_EQ(res.pinned_bytes(), std::max(pinned_before, fp.pinned.peak()));
  const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), call);
  fbt::expect_footprint(peak, {.persistent = fp.device.persistent, .scratch = fp.device.scratch},
                        live);
}

// mmd's record counts of @p classes, indexed [la][lb], as
// FootprintDimensions::mmd_records holds them.
std::vector<std::vector<std::size_t>>
record_counts(const std::vector<fb::PrimitivePairClass> &classes) {
  std::vector<std::vector<std::size_t>> out(fb::kMaxAngularMomentum + 1,
                                            std::vector<std::size_t>(fb::kMaxAngularMomentum + 1));
  for (const fb::PrimitivePairClass &cls : classes) {
    out[static_cast<std::size_t>(cls.la)][static_cast<std::size_t>(cls.lb)] =
        static_cast<std::size_t>(cls.n_rec);
  }
  return out;
}

// Water's molecule, orbital basis and mmd classes, on their own
// DeviceResources, so a routine's res starts with nothing allocated or grown.
struct Water {
  fb::DeviceResources inputs{0};
  fb::Molecule molecule{inputs, water()};
  fb::BasisSet basis = placed("cc-pvdz", molecule);
  std::vector<fb::PrimitivePairClass> classes = fb::primitive_pair_classes(inputs, basis).value();
};

TEST(LibraryFootprint, Molecule) {
  FBF_REQUIRE_GPU();
  const fb::MemoryFootprint fp = fb::molecule_footprint(3);
  fb::DeviceResources res{0};
  // Z and the positions.
  expect_routine_footprint(res, fp, 2, [&] { const fb::Molecule molecule(res, water()); });
}

TEST(LibraryFootprint, PrimitivePairClasses) {
  FBF_REQUIRE_GPU();
  const Water w;
  const fb::MemoryFootprint fp = fb::primitive_pair_classes_footprint(w.basis);
  ASSERT_EQ(w.classes.size(), 6U);
  fb::DeviceResources res{0};
  // At the peak: the classes' 20 fields each and the 11 tables; the classes
  // sort in the device workspace.
  expect_routine_footprint(res, fp, (6 * 20) + 11, [&] {
    const auto classes = fb::primitive_pair_classes(res, w.basis);
    ASSERT_TRUE(classes);
  });
}

// At tau = 0, where every record is kept and the footprint is exact.
TEST(LibraryFootprint, CoulombPairClasses) {
  FBF_REQUIRE_GPU();
  const Water w;
  const fb::MemoryFootprint fp = fb::coulomb_pair_classes_footprint(record_counts(w.classes));
  fb::DeviceResources res{0};
  // J's classes' 31 fields each, the maxima, the counts and a destination per
  // class.
  expect_routine_footprint(res, fp, (6 * 31) + 2 + 6, [&] {
    const auto j_classes = fb::coulomb_pair_classes(res, w.classes, 0.0);
    ASSERT_TRUE(j_classes);
    ASSERT_EQ(j_classes->size(), 6U);
  });
}

TEST(LibraryFootprint, OrderedPairClasses) {
  FBF_REQUIRE_GPU();
  const Water w;
  const fb::MemoryFootprint fp = fb::ordered_pair_classes_footprint(w.basis);
  fb::DeviceResources res{0};
  // At most the 9 ordered classes' 30 fields each (an empty field allocates
  // nothing); the gathers sort in the device workspace.
  expect_routine_footprint(res, fp, 9 * 30, [&] {
    const auto k_classes = fb::ordered_pair_classes(res, w.basis, w.classes);
    ASSERT_TRUE(k_classes);
    ASSERT_EQ(k_classes->size(), 9U);
  });
}

// The one-electron builds and J's and K's write only their caller's matrices
// and their classes' own scratch: each allocates nothing.
TEST(LibraryFootprint, OneElectronAndDirectJkAllocateNothing) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule(res, water());
  const fb::BasisSet basis = placed("cc-pvdz", molecule);
  const auto classes = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(classes);
  auto j_classes = fb::coulomb_pair_classes(res, *classes, 0.0);
  ASSERT_TRUE(j_classes);
  auto k_classes = fb::ordered_pair_classes(res, basis, *classes);
  ASSERT_TRUE(k_classes);
  fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO out(res, basis);
  ASSERT_TRUE(res.sync());
  expect_routine_footprint(res, {}, 0, [&] { EXPECT_TRUE(fb::overlap(res, *classes, out)); });
  expect_routine_footprint(
      res, {}, 0, [&] { EXPECT_TRUE(fb::core_hamiltonian(res, *classes, molecule, out)); });
  expect_routine_footprint(res, {}, 0,
                           [&] { EXPECT_TRUE(fb::coulomb(res, *j_classes, density, out, 0.0)); });
  expect_routine_footprint(res, {}, 0,
                           [&] { EXPECT_TRUE(fb::exchange(res, *k_classes, density, out)); });
}

TEST(LibraryFootprint, Metric) {
  FBF_REQUIRE_GPU();
  const Water w;
  const fb::BasisSet aux = placed("cc-pvdz-jkfit", w.molecule);
  const fb::MemoryFootprint fp = fb::metric_footprint(aux);
  ASSERT_GT(fp.device.workspace, 0U);
  fb::DeviceResources res{0};
  const auto u = fb::upload_spherical_transform_matrix(res, aux);
  ASSERT_TRUE(u);
  fb::MatrixAOPair m(res, aux, *u);
  expect_routine_footprint(res, fp, 0, [&] { EXPECT_TRUE(fb::metric(res, aux, m)); });
}

TEST(LibraryFootprint, ThreeCenter) {
  FBF_REQUIRE_GPU();
  const Water w;
  const fb::BasisSet aux = placed("cc-pvdz-jkfit", w.molecule);
  const fb::MemoryFootprint fp = fb::three_center_footprint(w.basis, aux);
  fb::DeviceResources res{0};
  fb::Buffers::device<double> b(
      std::size_t{aux.nao_sphe()} * w.basis.nao_sphe() * w.basis.nao_sphe(), res.device());
  // The staging, the cells, the Cartesian tensor and the work.
  expect_routine_footprint(
      res, fp, 4, [&] { EXPECT_TRUE(fb::three_center(res, w.basis, w.classes, aux, b.data())); });
}

// The metric and (P|mn) a fit factors, built once on their own res and copied
// in before every fit, which overwrites both.
struct FitInputs {
  Water w;
  fb::BasisSet aux = placed("cc-pvdz-jkfit", w.molecule);
  std::int32_t nao = static_cast<std::int32_t>(w.basis.nao_sphe());
  std::int32_t naux = static_cast<std::int32_t>(aux.nao_sphe());
  std::size_t nn = static_cast<std::size_t>(naux) * static_cast<std::size_t>(naux);
  std::size_t nb = static_cast<std::size_t>(naux) * static_cast<std::size_t>(nao) *
                   static_cast<std::size_t>(nao);
  fb::Buffers::device<double> metric{nn, w.inputs.device()};
  fb::Buffers::device<double> b{nb, w.inputs.device()};

  FitInputs() {
    fb::MatrixAOPair m(w.inputs, aux, fb::upload_spherical_transform_matrix(w.inputs, aux).value());
    EXPECT_TRUE(fb::metric(w.inputs, aux, m));
    EXPECT_EQ(wwr::wwrMemcpyAsync(metric.data(), m.spherical().data(), sizeof(double) * nn,
                                  wwr::wwrMemcpyDeviceToDevice, w.inputs.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(fb::three_center(w.inputs, w.basis, w.classes, aux, b.data()));
    EXPECT_TRUE(w.inputs.sync());
  }
};

// Both paths: Cholesky, which allocates nothing, and lindep's M^-1/2.
TEST(LibraryFootprint, DensityFit) {
  FBF_REQUIRE_GPU();
  const FitInputs in;
  for (const double lindep : {0.0, 1e-10}) {
    SCOPED_TRACE(lindep);
    fb::DeviceResources res{0};
    const auto lwork = fb::syevd_lwork<double>(res, in.naux);
    ASSERT_TRUE(lwork);
    const fb::MemoryFootprint fp = fb::density_fit_footprint(in.nao, in.naux, *lwork, lindep);
    EXPECT_EQ(fp.device.peak() == 0, lindep == 0.0);
    fb::Buffers::device<double> metric(in.nn, res.device());
    fb::Buffers::device<double> b(in.nb, res.device());
    // M's copy, U and the eigenvalues.
    expect_routine_footprint(res, fp, lindep == 0.0 ? 0 : 3, [&] {
      EXPECT_EQ(wwr::wwrMemcpyAsync(metric.data(), in.metric.data(), sizeof(double) * in.nn,
                                    wwr::wwrMemcpyDeviceToDevice, res.stream()),
                wwr::wwrSuccess);
      EXPECT_EQ(wwr::wwrMemcpyAsync(b.data(), in.b.data(), sizeof(double) * in.nb,
                                    wwr::wwrMemcpyDeviceToDevice, res.stream()),
                wwr::wwrSuccess);
      EXPECT_TRUE(fb::density_fit(res, in.nao, in.naux, metric.data(), b.data(), lindep));
    });
  }
}

// DF's J and K of the SCF (one density, one factor of five occupied
// orbitals) and B in the MO basis, over a B whose values do not matter: each
// takes only the device workspace.
TEST(LibraryFootprint, DfJkAndMo) {
  FBF_REQUIRE_GPU();
  const FitInputs in;
  const std::int32_t nocc = 5;
  const std::size_t nn = static_cast<std::size_t>(in.nao) * static_cast<std::size_t>(in.nao);
  {
    fb::DeviceResources res{0};
    fb::Buffers::device<double> d(nn, res.device());
    fb::Buffers::device<double> j(nn, res.device());
    expect_routine_footprint(res, fb::df_coulomb_footprint(in.nao, in.naux), 0, [&] {
      EXPECT_TRUE(fb::df_coulomb(res, in.nao, in.naux, in.b.data(), d.data(), j.data()));
    });
  }
  {
    fb::DeviceResources res{0};
    fb::Buffers::device<double> c(nn, res.device());
    fb::Buffers::device<double> k(nn, res.device());
    expect_routine_footprint(res, fb::df_exchange_footprint(in.nao, in.naux, nocc), 0, [&] {
      EXPECT_TRUE(fb::df_exchange(res, in.nao, in.naux, nocc, in.b.data(), c.data(), k.data()));
    });
  }
  {
    const fb::OrbitalRange rows{.first = 0, .count = nocc};
    const fb::OrbitalRange cols{.first = nocc, .count = in.nao - nocc};
    fb::DeviceResources res{0};
    fb::Buffers::device<double> c(nn, res.device());
    fb::Buffers::device<double> out(static_cast<std::size_t>(in.naux) * nocc * (in.nao - nocc),
                                    res.device());
    expect_routine_footprint(res, fb::b_mo_footprint(in.naux, in.nao, rows, cols), 0, [&] {
      EXPECT_TRUE(
          fb::b_mo(res, in.naux, in.nao, in.nao, in.b.data(), c.data(), rows, cols, out.data()));
    });
  }
}

// The block forms (#1174): what grows with the count of densities is one
// scratch allocation, and the symmetrized K's X, the same at any count, the
// workspace. Three densities of rank five.
TEST(LibraryFootprint, DfJkBlocks) {
  FBF_REQUIRE_GPU();
  const FitInputs in;
  constexpr std::int32_t kRank = 5;
  constexpr std::int32_t kCount = 3;
  const std::size_t nn = static_cast<std::size_t>(in.nao) * static_cast<std::size_t>(in.nao);
  const std::size_t factors = static_cast<std::size_t>(in.nao) * kRank * kCount;
  {
    fb::DeviceResources res{0};
    fb::Buffers::device<double> d(kCount * nn, res.device());
    fb::Buffers::device<double> j(kCount * nn, res.device());
    expect_routine_footprint(res, fb::df_coulomb_footprint(in.nao, in.naux, kCount), 1, [&] {
      EXPECT_TRUE(fb::df_coulomb(res, in.nao, in.naux, kCount, in.b.data(), d.data(), j.data()));
    });
  }
  {
    fb::DeviceResources res{0};
    fb::Buffers::device<double> left(factors, res.device());
    fb::Buffers::device<double> right(factors, res.device());
    fb::Buffers::device<double> k(kCount * nn, res.device());
    expect_routine_footprint(
        res, fb::df_exchange_footprint(in.nao, in.naux, kRank, kCount), 1, [&] {
          EXPECT_TRUE(fb::df_exchange(res, in.nao, in.naux, kRank, kCount, in.b.data(), left.data(),
                                      right.data(), k.data()));
        });
  }
  {
    fb::DeviceResources res{0};
    fb::Buffers::device<double> left(factors / kCount, res.device());
    fb::Buffers::device<double> right(factors, res.device());
    fb::Buffers::device<double> k(kCount * nn, res.device());
    expect_routine_footprint(
        res, fb::df_exchange_symmetrized_footprint(in.nao, in.naux, kRank, kCount), 1, [&] {
          EXPECT_TRUE(fb::df_exchange_symmetrized(res, in.nao, in.naux, kRank, kCount, in.b.data(),
                                                  left.data(), right.data(), k.data()));
        });
  }
}

// MINAO against cc-pVDZ, as AVAS builds <MINAO AO | basis AO>, and MINAO
// against itself.
TEST(LibraryFootprint, MixedClassesAndOverlap) {
  FBF_REQUIRE_GPU();
  Water w;
  const fb::BasisSet minao = placed("minao", w.molecule);
  for (const fb::BasisSet *cols : std::array<const fb::BasisSet *, 2>{&w.basis, &minao}) {
    SCOPED_TRACE(cols == &minao ? "minao" : "cc-pvdz");
    const std::size_t n_classes =
        fb::mixed_primitive_pair_classes(w.inputs, minao, *cols).value().size();
    {
      fb::DeviceResources res{0};
      // The classes' 20 fields each and the 11 tables.
      expect_routine_footprint(res, fb::mixed_primitive_pair_classes_footprint(minao, *cols),
                               (n_classes * 20) + 11, [&] {
                                 const auto classes =
                                     fb::mixed_primitive_pair_classes(res, minao, *cols);
                                 ASSERT_TRUE(classes);
                               });
    }
    fb::DeviceResources res{0};
    fb::Buffers::device<double> s(std::size_t{minao.nao_sphe()} * cols->nao_sphe(), res.device());
    // The Cartesian block beside the records' build: their fields and tables.
    expect_routine_footprint(
        res, fb::overlap_mixed_spherical_footprint(minao, *cols), 1 + (n_classes * 20) + 11,
        [&] { EXPECT_TRUE(fb::overlap_mixed_spherical(res, minao, *cols, s)); });
  }
}

// Water in cc-pVDZ, 2 core and 4 active orbitals over C = I: the allocation
// does not depend on C's values.
TEST(LibraryFootprint, ActiveSpaceHamiltonian) {
  FBF_REQUIRE_GPU();
  const Water w;
  fb::DeviceResources res{0};
  const auto u = fb::upload_spherical_transform_matrix(res, w.basis);
  ASSERT_TRUE(u);
  auto j_classes = fb::coulomb_pair_classes(res, w.classes);
  ASSERT_TRUE(j_classes);
  auto k_classes = fb::ordered_pair_classes(res, w.basis, w.classes);
  ASSERT_TRUE(k_classes);
  const fb::SphericalMatrixAO h(res, w.basis);
  fb::SphericalMatrixAO c(res, w.basis);
  const std::size_t nao = w.basis.nao_sphe();
  std::vector<double> identity(nao * nao, 0.0);
  for (std::size_t i = 0; i < nao; ++i) {
    identity[i * (nao + 1)] = 1.0;
  }
  EXPECT_EQ(wwr::wwrMemcpyAsync(c.data(), identity.data(), sizeof(double) * identity.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync());
  // A fresh workspace: the classes' sorts grew it.
  fb::DeviceResources own{0};
  const fb::MemoryFootprint fp = fb::active_space_hamiltonian_footprint(w.basis, 2, 4);
  ASSERT_GT(fp.device.workspace, 0U);
  // h_eff, (tu|vw), the density, J and K.
  expect_routine_footprint(own, fp, 5, [&] {
    const auto ham = fb::active_space_hamiltonian(own, w.basis, w.molecule, h, *u, *j_classes,
                                                  *k_classes, c, 2, 4);
    ASSERT_TRUE(ham) << ham.error().message;
  });
}

TEST(LibraryFootprint, DeviceStringSet) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // The strings and their links.
  expect_routine_footprint(res, fb::ci::device_string_set_footprint(8, 3), 2,
                           [&] { EXPECT_TRUE(fb::ci::DeviceStringSet::create(res, 8, 3)); });
  EXPECT_EQ(fb::ci::device_string_set_footprint(8, 3).device.persistent,
            (56 * sizeof(std::uint64_t)) + (56 * 3 * 6 * sizeof(fb::ci::Replacement)));
}

// Integrals of an aufbau-dominated H over n orbitals: h_tt = t, small
// symmetric couplings and two-electron terms, so the lowest root of each spin
// is near the lowest determinants and the Davidson converges. eri 8-fold,
// row-major (t, u, v, w) as Sigma::set_integrals takes it.
struct CiIntegrals {
  std::vector<double> h;
  std::vector<double> eri;
};

CiIntegrals ci_integrals(std::int32_t n) {
  std::mt19937_64 engine{20261109U};
  std::uniform_real_distribution<double> uniform{-1.0, 1.0};
  const auto m = static_cast<std::size_t>(n);
  CiIntegrals out{std::vector<double>(m * m), std::vector<double>(m * m * m * m)};
  for (std::size_t t = 0; t < m; ++t) {
    for (std::size_t u = 0; u <= t; ++u) {
      out.h[(t * m) + u] = out.h[(u * m) + t] =
          t == u ? static_cast<double>(t) : 1e-2 * uniform(engine);
    }
  }
  for (std::size_t t = 0; t < m; ++t) {
    for (std::size_t u = 0; u <= t; ++u) {
      for (std::size_t v = 0; v < m; ++v) {
        for (std::size_t x = 0; x <= v; ++x) {
          if ((t * (t + 1) / 2) + u < (v * (v + 1) / 2) + x) {
            continue;
          }
          const double value = 1e-2 * uniform(engine);
          for (const auto [a, b, c, d] :
               {std::array{t, u, v, x}, std::array{u, t, v, x}, std::array{t, u, x, v},
                std::array{u, t, x, v}, std::array{v, x, t, u}, std::array{x, v, t, u},
                std::array{v, x, u, t}, std::array{x, v, u, t}}) {
            out.eri[(((((a * m) + b) * m) + c) * m) + d] = value;
          }
        }
      }
    }
  }
  return out;
}

// Over one string set, or two: a Sigma's tables and apply scratch (whole, and
// at the least under a limit of one byte), with rdm1's workspace; then
// ci::solve over it, a singlet in the symmetric layout and a triplet in the
// full one, with its two stages and the roots unpacked.
TEST(LibraryFootprint, SigmaAndSolve) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kOrbitals = 6;
  constexpr std::int32_t kRoots = 2;
  const CiIntegrals ints = ci_integrals(kOrbitals);
  struct Sector {
    std::int32_t n_alpha;
    std::int32_t n_beta;
  };
  for (const Sector sector : {Sector{3, 3}, Sector{4, 2}}) {
    SCOPED_TRACE(std::format("({}, {})", sector.n_alpha, sector.n_beta));
    const bool one_set = sector.n_alpha == sector.n_beta;
    const fb::ci::Layout layout = one_set ? fb::ci::Layout::symmetric : fb::ci::Layout::full;
    for (const std::size_t limit : {std::numeric_limits<std::size_t>::max(), std::size_t{1}}) {
      SCOPED_TRACE(limit);
      fb::DeviceResources res{0};
      const auto alpha = fb::ci::DeviceStringSet::create(res, kOrbitals, sector.n_alpha);
      const auto beta = fb::ci::DeviceStringSet::create(res, kOrbitals, sector.n_beta);
      ASSERT_TRUE(alpha && beta);
      const fb::ci::DeviceStringSet &b = one_set ? *alpha : *beta;
      const auto dim = static_cast<std::size_t>(alpha->size()) * b.size();
      const fb::MemoryFootprint tables =
          fb::ci::sigma_footprint(kOrbitals, alpha->size(), b.size(), layout);
      const auto held = [](std::size_t bytes) {
        return fb::MemoryFootprint{.device = {.persistent = bytes}};
      };
      // The Sigma, its scratch from one apply, and a 1-RDM.
      const std::size_t apply =
          fb::ci::sigma_apply_bytes(kOrbitals, alpha->size(), b.size(), layout, limit);
      if (limit == 1) {
        EXPECT_LT(apply, fb::ci::sigma_apply_bytes(kOrbitals, alpha->size(), b.size(), layout));
      }
      const std::size_t length =
          layout == fb::ci::Layout::symmetric
              ? static_cast<std::size_t>(alpha->size()) * (alpha->size() + 1) / 2
              : dim;
      fb::Buffers::device<double> x(length, res.device());
      fb::Buffers::device<double> y(length, res.device());
      fb::Buffers::device<double> gamma(kOrbitals * kOrbitals, res.device());
      EXPECT_EQ(wwr::wwrMemsetAsync(x.data(), 0, sizeof(double) * length, res.stream()),
                wwr::wwrSuccess);
      // The three tables and the scratch.
      expect_routine_footprint(res, then(tables, held(apply)), 4, [&] {
        auto sigma = fb::ci::Sigma::create(res, *alpha, b);
        ASSERT_TRUE(sigma);
        sigma->set_scratch_limit(limit);
        ASSERT_TRUE(sigma->set_integrals(res, 0.0, ints.h, ints.eri));
        EXPECT_TRUE(sigma->apply(res, 1, x.data(), y.data(), layout));
        EXPECT_TRUE(sigma->rdm1(res, x.data(), gamma.data(), layout));
      });

      // The solve, over a Sigma and diagonal of its own each run, as a step
      // makes them.
      const auto lwork = [&] {
        std::vector<std::size_t> out(16 * kRoots + 1);
        for (std::size_t order = 0; order < out.size(); ++order) {
          out[order] = fb::syevd_lwork<double>(res, static_cast<std::int32_t>(order)).value();
        }
        return out;
      }();
      const fb::ci::SolverOptions options;
      const fb::MemoryFootprint solve = fb::ci::solve_footprint(
          kOrbitals, sector.n_alpha, sector.n_beta, kRoots, options, false, lwork, limit);
      // The sigma's tables, the diagonal, the solve's guesses, shifted
      // diagonal, values, vectors, picks, P-space U and values (empty) and
      // S^2's table; the Davidson's 11; the sigma's scratch; the unpacked
      // roots.
      fb::DeviceResources own{0};
      expect_routine_footprint(
          own, then(then(held(tables.device.persistent), held(sizeof(double) * dim)), solve),
          3 + 1 + 8 + 11 + 1 + 1,
          [&] {
            auto sigma = fb::ci::Sigma::create(own, *alpha, b);
            ASSERT_TRUE(sigma);
            sigma->set_scratch_limit(limit);
            ASSERT_TRUE(sigma->set_integrals(own, 0.0, ints.h, ints.eri));
            fb::Buffers::device<double> diagonal(dim, own.device());
            ASSERT_TRUE(sigma->diagonal(own, diagonal.data()));
            const auto roots =
                fb::ci::solve(own, *sigma, *alpha, b, diagonal.data(), kRoots, options);
            ASSERT_TRUE(roots) << roots.error().message;
          },
          true);
    }
  }
}

// CASSCF's routines over water in cc-pVDZ, 2 core and 4 active orbitals, from
// C = I and h = 0, and B unfitted (the three-center tensor) under density
// fitting: the allocations depend on none of their values. The inputs live on
// fit.w.inputs; each routine runs on a fresh DeviceResources of its own.
constexpr std::int32_t kCore = 2;
constexpr std::int32_t kActive = 4;

struct CasscfInputs {
  FitInputs fit;
  fb::DeviceResources &inputs = fit.w.inputs;
  const fb::BasisSet &basis = fit.w.basis;
  std::shared_ptr<const fb::Buffers::device<double>> u =
      fb::upload_spherical_transform_matrix(inputs, basis).value();
  std::vector<fb::CoulombPairClass> j_classes =
      fb::coulomb_pair_classes(inputs, fit.w.classes).value();
  std::vector<fb::OrderedPairClass> k_classes =
      fb::ordered_pair_classes(inputs, basis, fit.w.classes).value();
  fb::SphericalMatrixAO h{inputs, basis};
  fb::SphericalMatrixAO c{inputs, basis};
  // gamma = 1 on the active diagonal, Gamma = 0, row-major as rdm12 gives them.
  std::vector<double> gamma = std::vector<double>(kActive * kActive, 0.0);
  std::vector<double> gamma2 = std::vector<double>(kActive * kActive * kActive * kActive, 0.0);

  CasscfInputs() {
    const std::size_t nao = basis.nao_sphe();
    std::vector<double> identity(nao * nao, 0.0);
    for (std::size_t i = 0; i < nao; ++i) {
      identity[i * (nao + 1)] = 1.0;
    }
    for (std::size_t t = 0; t < kActive; ++t) {
      gamma[t * (kActive + 1)] = 1.0;
    }
    EXPECT_EQ(wwr::wwrMemcpyAsync(c.data(), identity.data(), sizeof(double) * identity.size(),
                                  wwr::wwrMemcpyHostToDevice, inputs.stream()),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrMemsetAsync(h.data(), 0, sizeof(double) * nao * nao, inputs.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(inputs.sync());
  }

  [[nodiscard]] std::int32_t nmo() const { return static_cast<std::int32_t>(basis.nao_sphe()); }
  [[nodiscard]] std::int32_t pairs() const {
    return (kCore * (nmo() - kCore)) + (kActive * (nmo() - kCore - kActive));
  }
  [[nodiscard]] fb::CasscfIntegrals direct_integrals(fb::DeviceResources &res) {
    auto ints = fb::casscf_integrals(res, basis, fit.w.molecule, h, u, j_classes, k_classes, c,
                                     kCore, kActive);
    EXPECT_TRUE(ints) << ints.error().message;
    return std::move(*ints);
  }
  [[nodiscard]] fb::CasscfIntegrals fitted_integrals(fb::DeviceResources &res) {
    auto ints =
        fb::casscf_integrals(res, basis, fit.w.molecule, h, fit.b, fit.naux, c, kCore, kActive);
    EXPECT_TRUE(ints) << ints.error().message;
    return std::move(*ints);
  }
  [[nodiscard]] fb::response::Fock fock(fb::DeviceResources &res, const fb::CasscfIntegrals &ints,
                                        bool fitted) {
    auto out = fitted ? fb::response::fock(res, fit.w.molecule, h, fit.b, fit.naux, c, kCore,
                                           kActive, ints.ppaa, ints.papa, gamma, gamma2)
                      : fb::response::fock(res, basis, fit.w.molecule, h, u, j_classes, k_classes,
                                           c, kCore, kActive, ints.ppaa, ints.papa, gamma, gamma2);
    EXPECT_TRUE(out) << out.error().message;
    return std::move(*out);
  }
  [[nodiscard]] fb::response::Hessian hessian(fb::DeviceResources &res,
                                              const fb::CasscfIntegrals &ints,
                                              const fb::response::Fock &f, bool fitted) {
    auto out = fitted ? fb::response::Hessian::create(res, h, fit.b, fit.naux, c, kCore, kActive,
                                                      ints.ppaa, ints.papa, f)
                      : fb::response::Hessian::create(res, basis, h, u, j_classes, k_classes, c,
                                                      kCore, kActive, ints.ppaa, ints.papa, f);
    EXPECT_TRUE(out) << out.error().message;
    return std::move(*out);
  }
};

TEST(LibraryFootprint, CasscfIntegrals) {
  FBF_REQUIRE_GPU();
  CasscfInputs in;
  {
    fb::DeviceResources res{0};
    // ppaa, papa, h_eff, (tu|vw), and the density, J and K.
    expect_routine_footprint(res, fb::casscf_integrals_footprint(in.basis, kCore, kActive), 7,
                             [&] { (void)in.direct_integrals(res); });
  }
  {
    fb::DeviceResources res{0};
    // The four, and the core's P, J and K in one allocation.
    expect_routine_footprint(res,
                             fb::casscf_integrals_footprint(in.nmo(), in.fit.naux, kCore, kActive),
                             5, [&] { (void)in.fitted_integrals(res); });
  }
}

// Rdm::create allocates nothing, and rdm12 and trans_rdm12 take the device
// workspace alone; so do the packed pairs' copies.
TEST(LibraryFootprint, RdmAndOrbitalPairs) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources inputs{0};
  const auto strings = fb::ci::DeviceStringSet::create(inputs, kActive, 2);
  ASSERT_TRUE(strings);
  const auto dim = static_cast<std::size_t>(strings->size()) * strings->size();
  fb::Buffers::device<double> vectors(2 * dim, inputs.device());
  fb::Buffers::device<double> gamma(kActive * kActive, inputs.device());
  fb::Buffers::device<double> gamma2(kActive * kActive * kActive * kActive, inputs.device());
  EXPECT_EQ(wwr::wwrMemsetAsync(vectors.data(), 0, sizeof(double) * 2 * dim, inputs.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(inputs.sync());
  const std::vector<double> weights{0.5, 0.5};
  {
    fb::DeviceResources res{0};
    expect_routine_footprint(res, {}, 0,
                             [&] { EXPECT_TRUE(fb::ci::Rdm::create(res, *strings, *strings)); });
  }
  {
    fb::DeviceResources res{0};
    auto rdm = fb::ci::Rdm::create(res, *strings, *strings);
    ASSERT_TRUE(rdm);
    expect_routine_footprint(
        res, fb::ci::rdm12_footprint(kActive, static_cast<std::int32_t>(dim)), 0, [&] {
          EXPECT_TRUE(rdm->rdm12(res, weights, vectors.data(), gamma.data(), gamma2.data()));
        });
  }
  {
    fb::DeviceResources res{0};
    auto rdm = fb::ci::Rdm::create(res, *strings, *strings);
    ASSERT_TRUE(rdm);
    expect_routine_footprint(
        res, fb::ci::trans_rdm12_footprint(kActive, static_cast<std::int32_t>(dim)), 0, [&] {
          EXPECT_TRUE(rdm->trans_rdm12(res, vectors.data(), vectors.data() + dim, gamma.data(),
                                       gamma2.data()));
        });
  }
  {
    constexpr std::int32_t kOrbitals = 10;
    const std::int32_t pairs =
        (kCore * (kOrbitals - kCore)) + (kActive * (kOrbitals - kCore - kActive));
    fb::DeviceResources res{0};
    fb::Buffers::device<double> square(kOrbitals * kOrbitals, res.device());
    fb::Buffers::device<double> packed(static_cast<std::size_t>(pairs), res.device());
    expect_routine_footprint(res, {}, 0, [&] {
      EXPECT_TRUE(fb::response::pack_orbital_pairs(res, kOrbitals, kCore, kActive, square.data(),
                                                   packed.data()));
      EXPECT_TRUE(fb::response::unpack_orbital_pairs(res, kOrbitals, kCore, kActive, packed.data(),
                                                     square.data()));
    });
  }
}

TEST(LibraryFootprint, Fock) {
  FBF_REQUIRE_GPU();
  CasscfInputs in;
  for (const bool fitted : {false, true}) {
    SCOPED_TRACE(fitted);
    const fb::CasscfIntegrals ints =
        fitted ? in.fitted_integrals(in.inputs) : in.direct_integrals(in.inputs);
    fb::DeviceResources res{0};
    // F^I, F^A, F, the gradient, gamma and Gamma; and P_c, J and K.
    expect_routine_footprint(
        res,
        fitted ? fb::response::fock_footprint(in.nmo(), in.fit.naux, kCore, kActive)
               : fb::response::fock_footprint(in.basis, kCore, kActive),
        6 + 3, [&] { (void)in.fock(res, ints, fitted); });
  }
}

// The orbital Hessian's create and one column's product, integral-direct and
// density-fitted.
TEST(LibraryFootprint, Hessian) {
  FBF_REQUIRE_GPU();
  CasscfInputs in;
  for (const bool fitted : {false, true}) {
    SCOPED_TRACE(fitted);
    const fb::CasscfIntegrals ints =
        fitted ? in.fitted_integrals(in.inputs) : in.direct_integrals(in.inputs);
    const fb::response::Fock f = in.fock(in.inputs, ints, fitted);
    {
      fb::DeviceResources res{0};
      // Its nine buffers, and J and K's two (fitted) or the Cartesian three.
      expect_routine_footprint(
          res,
          fitted ? fb::response::hessian_footprint(in.nmo(), in.fit.naux, kCore, kActive)
                 : fb::response::hessian_footprint(in.basis, kCore, kActive),
          fitted ? 9 + 2 : 9 + 3, [&] { (void)in.hessian(res, ints, f, fitted); });
    }
    fb::response::Hessian hessian = in.hessian(in.inputs, ints, f, fitted);
    const auto pairs = static_cast<std::size_t>(in.pairs());
    fb::Buffers::device<double> x(pairs, in.inputs.device());
    fb::Buffers::device<double> y(pairs, in.inputs.device());
    EXPECT_EQ(wwr::wwrMemsetAsync(x.data(), 0, sizeof(double) * pairs, in.inputs.stream()),
              wwr::wwrSuccess);
    ASSERT_TRUE(in.inputs.sync());
    fb::DeviceResources res{0};
    // The fitted product's one scratch allocation.
    expect_routine_footprint(
        res,
        fitted ? fb::response::hessian_apply_footprint(in.nmo(), in.fit.naux, kCore, kActive)
               : fb::response::hessian_apply_footprint(in.basis, kCore, kActive),
        fitted ? 1 : 0, [&] { EXPECT_TRUE(hessian.apply(res, 1, x.data(), y.data())); });
  }
}

// The pool cap that leaves @p room bytes of memory budget past the headroom
// once @p warm has run: what a fresh uncapped pool holds after the same call
// (its workspace, grown, and DeviceResources' own bytes), plus the headroom and
// @p room. The call allocates the same on either pool, so a capped one warmed
// by it holds the same, and its budget less the headroom is @p room.
template <typename Warm>
std::uint64_t cap_leaving(const std::size_t room, Warm &&warm) {
  fb::DeviceResources probe{0};
  warm(probe);
  EXPECT_TRUE(probe.sync());
  return probe.device()->pool_usage().used_current + fb::kPoolHeadroomBytes + room;
}

std::vector<double> download(fb::DeviceResources &res, const fb::Buffers::device<double> &d) {
  std::vector<double> host(d.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync());
  return host;
}

// The width only regroups the products' gemms, so Y and U at one column a
// time against the unconstrained width are a few roundings apart: 1e-10 of
// the largest element is a rounding bound, not a fit.
constexpr double kWidthTol = 1e-10;

// The density-fitted Hessian::apply of a block of three columns (#1174),
// whose width is sized from the memory budget: under a cap leaving exactly
// hessian_apply_footprint's scratch, one column at a time, held to it; under
// a cap with room to spare, wider than that and under the cap; and Y at one
// column a time equal to Y uncapped, to kWidthTol of max |Y|.
TEST(LibraryFootprint, HessianBlockWidthFromTheBudget) {
  FBF_REQUIRE_GPU();
  CasscfInputs in;
  const fb::CasscfIntegrals ints = in.fitted_integrals(in.inputs);
  const fb::response::Fock f = in.fock(in.inputs, ints, true);
  fb::response::Hessian hessian = in.hessian(in.inputs, ints, f, true);
  constexpr std::int32_t kBlock = 3;
  const std::size_t count = kBlock * static_cast<std::size_t>(in.pairs());
  std::vector<double> host_x(count);
  for (std::size_t i = 0; i < count; ++i) {
    host_x[i] = 1e-2 * std::sin(static_cast<double>(i + 1));
  }
  fb::Buffers::device<double> x(count, in.inputs.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(x.data(), host_x.data(), sizeof(double) * count,
                                wwr::wwrMemcpyHostToDevice, in.inputs.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(in.inputs.sync());
  // Y lives on the inputs' pool, so the pool measured holds only the call's.
  fb::Buffers::device<double> y_least(count, in.inputs.device());
  fb::Buffers::device<double> y_spare(count, in.inputs.device());
  fb::Buffers::device<double> y_whole(count, in.inputs.device());
  const auto apply = [&](fb::DeviceResources &res, fb::Buffers::device<double> &y) {
    EXPECT_TRUE(hessian.apply(res, kBlock, x.data(), y.data()));
  };
  const fb::MemoryFootprint least =
      fb::response::hessian_apply_footprint(in.nmo(), in.fit.naux, kCore, kActive);
  {
    fb::DeviceResources res{
        0, fb::DeviceResources::kDefaultPoolReleaseThreshold, 0,
        cap_leaving(least.device.scratch, [&](fb::DeviceResources &on) { apply(on, y_least); })};
    // One column's carve, the one allocation.
    expect_routine_footprint(res, least, 1, [&] { apply(res, y_least); });
  }
  {
    const std::uint64_t cap =
        cap_leaving(std::size_t{64} << 20, [&](fb::DeviceResources &on) { apply(on, y_spare); });
    fb::DeviceResources res{0, fb::DeviceResources::kDefaultPoolReleaseThreshold, 0, cap};
    apply(res, y_spare);
    const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), [&] { apply(res, y_spare); });
    std::cout << std::format("[ footprint ] block of {}: measured {}, the least {}\n", kBlock,
                             peak.used, least.device.scratch);
    EXPECT_GT(peak.used, least.device.scratch);
    EXPECT_LE(res.device()->pool_usage().used_high, cap);
  }
  {
    fb::DeviceResources res{0};
    apply(res, y_whole);
    ASSERT_TRUE(res.sync());
  }
  const std::vector<double> narrow = download(in.inputs, y_least);
  const std::vector<double> whole = download(in.inputs, y_whole);
  double top = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    top = std::max(top, std::abs(whole[i]));
    worst = std::max(worst, std::abs(narrow[i] - whole[i]));
  }
  std::cout << std::format("[ footprint ] max |Y_1 - Y| / max |Y| = {:.2e} (max |Y| {:.3e})\n",
                           worst / top, top);
  ASSERT_GT(top, 0.0);
  EXPECT_LE(worst, kWidthTol * top);
}

// H~ over two singlet roots of CAS(4, 4)'s model H (the CI's integrals are
// ci_integrals', the response's the water's): create and two columns' products
// over the density-fitted H, whose product allocates its scratch; and a block
// product of three columns over the integral-direct H, whose takes only the
// workspace, which reserves the block's CI blocks and lockstep scratch.
TEST(LibraryFootprint, RelaxedHessian) {
  FBF_REQUIRE_GPU();
  CasscfInputs in;
  constexpr std::int32_t kRoots = 2;
  constexpr std::size_t kBlock = 3;
  const auto alpha = fb::ci::DeviceStringSet::create(in.inputs, kActive, 2);
  ASSERT_TRUE(alpha);
  auto sigma = fb::ci::Sigma::create(in.inputs, *alpha, *alpha);
  ASSERT_TRUE(sigma);
  const CiIntegrals model = ci_integrals(kActive);
  ASSERT_TRUE(sigma->set_integrals(in.inputs, 0.0, model.h, model.eri));
  const auto dim = static_cast<std::size_t>(sigma->size());
  fb::Buffers::device<double> diagonal(dim, in.inputs.device());
  ASSERT_TRUE(sigma->diagonal(in.inputs, diagonal.data()));
  const auto roots = fb::ci::solve(in.inputs, *sigma, *alpha, *alpha, diagonal.data(), kRoots);
  ASSERT_TRUE(roots) << roots.error().message;
  auto rdm = fb::ci::Rdm::create(in.inputs, *alpha, *alpha);
  ASSERT_TRUE(rdm);
  const std::vector<double> weights{0.5, 0.5};
  const auto pairs = static_cast<std::size_t>(in.pairs());
  fb::Buffers::device<double> x(kBlock * pairs, in.inputs.device());
  fb::Buffers::device<double> y(kBlock * pairs, in.inputs.device());
  std::vector<double> ones(kBlock * pairs, 1e-3);
  EXPECT_EQ(wwr::wwrMemcpyAsync(x.data(), ones.data(), sizeof(double) * ones.size(),
                                wwr::wwrMemcpyHostToDevice, in.inputs.stream()),
            wwr::wwrSuccess);
  for (const bool fitted : {true, false}) {
    SCOPED_TRACE(fitted);
    const fb::CasscfIntegrals ints =
        fitted ? in.fitted_integrals(in.inputs) : in.direct_integrals(in.inputs);
    const fb::response::Fock f = in.fock(in.inputs, ints, fitted);
    fb::response::Hessian hessian = in.hessian(in.inputs, ints, f, fitted);
    const auto nmo = static_cast<std::size_t>(in.nmo());
    const auto host = [&](const fb::Buffers::device<double> &from) {
      std::vector<double> to(from.num_elements());
      EXPECT_EQ(wwr::wwrMemcpyAsync(to.data(), from.data(), sizeof(double) * to.size(),
                                    wwr::wwrMemcpyDeviceToHost, in.inputs.stream()),
                wwr::wwrSuccess);
      EXPECT_TRUE(in.inputs.sync());
      return to;
    };
    const fb::response::ResponseIntegrals response{
        nmo, kCore, kActive, host(f.fock_inactive), host(ints.ppaa), host(ints.papa)};
    const std::size_t block = fitted ? 0 : kBlock;
    const fb::MemoryFootprint relaxed =
        fb::response::relaxed_hessian_footprint(kActive, 2, 2, kRoots, in.nmo(), kCore, block);
    const fb::MemoryFootprint product =
        fitted ? fb::response::hessian_apply_footprint(in.nmo(), in.fit.naux, kCore, kActive)
               : fb::response::hessian_apply_footprint(in.basis, kCore, kActive);
    fb::DeviceResources res{0};
    // H'(x)'s sigma's three tables and its scratch, the relaxed Hessian's nine
    // buffers, and the fitted product's scratch.
    expect_routine_footprint(res, then(relaxed, product), 3 + 1 + 9 + (fitted ? 1 : 0), [&] {
      auto made = fb::response::RelaxedHessian::create(res, hessian, *sigma, *rdm, *alpha, *alpha,
                                                       *roots, weights, response);
      ASSERT_TRUE(made) << made.error().message;
      if (fitted) {
        for (std::size_t column = 0; column < 2; ++column) {
          EXPECT_TRUE(made->apply(res, x.data() + (column * pairs), y.data() + (column * pairs)));
        }
      } else {
        EXPECT_TRUE(made->apply(res, static_cast<std::int32_t>(kBlock), x.data(), y.data()));
      }
    });
  }
}

// The gradients' derivative contractions over a density whose values do not
// matter: the one-electron three, J's (over J's classes at tau = 0, every
// record) and K's (over the ordered classes), and the fitted metric's and
// three-center tensor's, which take only the device workspace.
TEST(LibraryFootprint, GradientIntegrals) {
  FBF_REQUIRE_GPU();
  Water w;
  ASSERT_NE(w.basis.nao_cart(), w.basis.nao_sphe());
  const fb::BasisSet aux = placed("cc-pvdz-jkfit", w.molecule);
  auto k_classes = fb::ordered_pair_classes(w.inputs, w.basis, w.classes);
  ASSERT_TRUE(k_classes);
  ASSERT_TRUE(w.inputs.sync());
  fb::DeviceResources res{0};
  const fb::SphericalMatrixAO m(res, w.basis);
  // U, the pair's three matrices, each Cartesian AO's atom, the records'
  // atoms and the device gradient.
  const fb::MemoryFootprint one = fb::one_electron_gradient_footprint(w.basis);
  expect_routine_footprint(res, one, 7,
                           [&] { EXPECT_TRUE(fb::overlap_gradient(res, w.basis, w.classes, m)); });
  expect_routine_footprint(res, one, 7,
                           [&] { EXPECT_TRUE(fb::kinetic_gradient(res, w.basis, w.classes, m)); });
  expect_routine_footprint(res, one, 7, [&] {
    EXPECT_TRUE(fb::potential_gradient(res, w.basis, w.classes, w.molecule, m));
  });
  // U and the pair's three, and J's classes' 31 fields each, the maxima, the
  // counts and a destination per class; the rest in the workspace.
  expect_routine_footprint(res, fb::coulomb_gradient_footprint(w.basis), 4 + (6 * 31) + 2 + 6,
                           [&] { EXPECT_TRUE(fb::coulomb_gradient(res, w.basis, w.classes, m)); });
  expect_routine_footprint(res, fb::exchange_gradient_footprint(w.basis), 7, [&] {
    EXPECT_TRUE(fb::exchange_gradient(res, w.basis, *k_classes, m));
  });
  const std::size_t naux = aux.nao_sphe();
  const fb::Buffers::device<double> wpq(naux * naux, res.device());
  expect_routine_footprint(res, fb::metric_gradient_footprint(aux), 0,
                           [&] { EXPECT_TRUE(fb::metric_gradient(res, aux, wpq.data())); });
  const fb::Buffers::device<double> gamma(naux * w.basis.nao_sphe() * w.basis.nao_sphe(),
                                          res.device());
  expect_routine_footprint(res, fb::three_center_gradient_footprint(w.basis, aux), 0, [&] {
    EXPECT_TRUE(fb::three_center_gradient(res, w.basis, w.classes, aux, gamma.data()));
  });
}

// The RHF gradients over water's five occupied orbitals, over a C, B and L
// whose values do not matter: the allocation does not depend on them. D, -W, gmmwich's
// diagonal and work and a derivative call's seven are live at the fitted
// one's peak; D and the J gradient's at the integral-direct one's.
TEST(LibraryFootprint, RhfGradient) {
  FBF_REQUIRE_GPU();
  Water w;
  const fb::BasisSet aux = placed("cc-pvdz-jkfit", w.molecule);
  auto k_classes = fb::ordered_pair_classes(w.inputs, w.basis, w.classes);
  ASSERT_TRUE(k_classes);
  ASSERT_TRUE(w.inputs.sync());
  constexpr std::int32_t kOcc = 5;
  const std::size_t nao = w.basis.nao_sphe();
  const std::size_t naux = aux.nao_sphe();
  const std::vector<double> energies(nao, -1.0);
  {
    fb::DeviceResources res{0};
    const fb::SphericalMatrixAO c(res, w.basis);
    const fb::Buffers::device<double> b(naux * nao * nao, res.device());
    const fb::Buffers::device<double> l(naux * naux, res.device());
    expect_routine_footprint(res, fb::gradient::df_rhf_footprint(w.basis, aux, kOcc), 4 + 7, [&] {
      EXPECT_TRUE(fb::gradient::df_rhf(res, w.molecule, w.basis, w.classes, aux, b.data(), l.data(),
                                       c, energies, kOcc));
    });
  }
  fb::DeviceResources res{0};
  const fb::SphericalMatrixAO c(res, w.basis);
  expect_routine_footprint(res, fb::gradient::rhf_footprint(w.basis, kOcc), 1 + 4 + (6 * 31) + 8,
                           [&] {
                             EXPECT_TRUE(fb::gradient::rhf(res, w.molecule, w.basis, w.classes,
                                                           *k_classes, c, energies, kOcc));
                           });
}

} // namespace
