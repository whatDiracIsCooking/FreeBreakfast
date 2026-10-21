// Host cases for experimental/atomic_scf's :scf: the atomic SCF against
// PySCF's AtomSphAverageRHF (data/atomic_scf.json, written by
// experimental/atomic_scf/emit_reference.py), the density's invariants, and
// the refusals.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.io.basis;
import free_breakfast.experimental.atomic_scf;

#include "atomic_scf_reference.h"

namespace {

namespace fb = free_breakfast;
using fb::test::kOrigin;
using fb::test::max_abs;
using fb::test::reference;
using fb::test::reference_shells;

// tr(A B) for two symmetric n x n matrices.
double trace_product(const std::vector<double> &a, const std::vector<double> &b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// C = A B, n x n column-major.
std::vector<double> matmul(const std::vector<double> &a, const std::vector<double> &b,
                           std::size_t n) {
  std::vector<double> c(n * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t i = 0; i < n; ++i) {
        c[i + (n * j)] += a[i + (n * k)] * b[k + (n * j)];
      }
    }
  }
  return c;
}

struct ReferenceAtom {
  std::string basis;
  std::string symbol;
  std::int32_t z;
  double energy;
  std::vector<std::vector<double>> orbital_energies;
};

std::vector<ReferenceAtom> reference_atoms() {
  std::vector<ReferenceAtom> out;
  for (const auto &a : reference()["atoms"]) {
    out.push_back({a["basis"].get<std::string>(), a["symbol"].get<std::string>(),
                   a["z"].get<std::int32_t>(), a["energy"].get<double>(),
                   a["orbital_energies"].get<std::vector<std::vector<double>>>()});
  }
  return out;
}

// --- Against PySCF ------------------------------------------------------------

TEST(AtomicScf, MatchesPyscfAtomSphAverageRhf) {
  const std::vector<ReferenceAtom> atoms = reference_atoms();
  ASSERT_FALSE(atoms.empty());
  for (const ReferenceAtom &ref : atoms) {
    SCOPED_TRACE(ref.symbol + "/" + ref.basis);
    const std::vector<fb::Shell> shells = reference_shells(ref.basis, ref.z);
    const auto scf = fb::atomic_scf(ref.z, shells);
    ASSERT_TRUE(scf) << scf.error().message;
    EXPECT_NEAR(scf->energy, ref.energy, 1e-8);
    ASSERT_EQ(scf->orbital_energies.size(), ref.orbital_energies.size());
    for (std::size_t l = 0; l < ref.orbital_energies.size(); ++l) {
      ASSERT_EQ(scf->orbital_energies[l].size(), ref.orbital_energies[l].size()) << "l=" << l;
      for (std::size_t k = 0; k < ref.orbital_energies[l].size(); ++k) {
        EXPECT_NEAR(scf->orbital_energies[l][k], ref.orbital_energies[l][k], 1e-6)
            << "l=" << l << " k=" << k;
      }
    }

    // The density holds Z electrons against S, and is symmetric.
    const auto ints = fb::atomic_integrals(ref.z, shells);
    ASSERT_TRUE(ints);
    EXPECT_NEAR(trace_product(scf->density, ints->overlap), ref.z, 1e-9);
    const std::size_t n = ints->nao_sphe;
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t i = 0; i < j; ++i) {
        EXPECT_EQ(scf->density[i + (n * j)], scf->density[j + (n * i)]);
      }
    }
  }
}

// --- The density ----------------------------------------------------------------

TEST(AtomicScf, ClosedShellDensityIsIdempotent) {
  // Every orbital of Ne doubly occupied: D S D = 2 D.
  const std::vector<fb::Shell> shells = reference_shells("cc-pvdz", 10);
  const auto scf = fb::atomic_scf(10, shells);
  const auto ints = fb::atomic_integrals(10, shells);
  ASSERT_TRUE(scf && ints);
  const std::size_t n = ints->nao_sphe;
  const std::vector<double> dsd = matmul(matmul(scf->density, ints->overlap, n), scf->density, n);
  for (std::size_t i = 0; i < n * n; ++i) {
    EXPECT_NEAR(dsd[i], 2.0 * scf->density[i], 1e-8);
  }
}

TEST(AtomicScf, GhostAtomIsEmpty) {
  const auto scf = fb::atomic_scf(0, reference_shells("sto-3g", 1));
  ASSERT_TRUE(scf) << scf.error().message;
  EXPECT_EQ(scf->energy, 0.0);
  EXPECT_EQ(max_abs(scf->density), 0.0);
}

// --- Refusals -------------------------------------------------------------------

TEST(AtomicScf, RefusesNoShellsAndSplitCenters) {
  EXPECT_EQ(fb::atomic_scf(1, {}).error().kind, fb::ErrorKind::InvalidConfig);
  const std::vector<fb::Shell> split{fb::Shell(0, {{1.0, 1.0}}, kOrigin),
                                     fb::Shell(0, {{0.5, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 1.0})};
  EXPECT_EQ(fb::atomic_scf(2, split).error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(AtomicScf, RefusesABasisThatCannotSeatTheAtom) {
  // O needs a p orbital; two s shells have none.
  const std::vector<fb::Shell> s_only{fb::Shell(0, {{10.0, 1.0}}, kOrigin),
                                      fb::Shell(0, {{1.0, 1.0}}, kOrigin)};
  EXPECT_EQ(fb::atomic_scf(8, s_only).error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(AtomicScf, ReportsAnUnconvergedRun) {
  const auto scf = fb::atomic_scf(8, reference_shells("cc-pvdz", 8), 1e-9, 1);
  ASSERT_FALSE(scf);
  EXPECT_EQ(scf.error().kind, fb::ErrorKind::MaxIterationsExceeded);
}

} // namespace
