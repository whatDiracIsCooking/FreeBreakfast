// The footprints (#1044) of the library routines the RHF chain calls outside
// src/recipes (#1048): the Molecule's upload, mmd's classes, J's and K's
// classes and builds, the one-electron integrals, and the density-fitting
// metric, three-center tensor, fit, J, K and MO transform. Each is held to the
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
template <typename Call>
void expect_routine_footprint(fb::DeviceResources &res, const fb::MemoryFootprint &fp,
                              const std::size_t live, Call &&call) {
  const std::size_t device_before = res.device_workspace(0).num_elements();
  const std::size_t pinned_before = res.pinned_bytes();
  call();
  EXPECT_EQ(res.device_workspace(0).num_elements(), std::max(device_before, fp.device.workspace));
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
    const auto fp = fb::density_fit_footprint(res, in.nao, in.naux, lindep);
    ASSERT_TRUE(fp);
    EXPECT_EQ(fp->device.peak() == 0, lindep == 0.0);
    fb::Buffers::device<double> metric(in.nn, res.device());
    fb::Buffers::device<double> b(in.nb, res.device());
    // M's copy, U and the eigenvalues.
    expect_routine_footprint(res, *fp, lindep == 0.0 ? 0 : 3, [&] {
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

} // namespace
