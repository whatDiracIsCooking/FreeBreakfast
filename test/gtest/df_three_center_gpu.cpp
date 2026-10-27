// GPU cases for integrals.eri.density_fitting.three_center: the spherical
// (mn|P) of the compiled-in cc-pVDZ / cc-pVDZ-JKFIT (orbitals to d, aux to f)
// and cc-pVTZ / cc-pVTZ-JKFIT (orbitals to f, aux to g, so (ff|g)) over
// FelizJueves's water, against PySCF's int3c2e (data/density_fitting/);
// exactly (m, n)-symmetric, and bit-identical across two runs; and the
// refusals: a g orbital shell, a k aux shell, and a tensor past a forced
// small limit.

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
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.three_center;

namespace {

namespace fb = free_breakfast;

// The issue's bar, FelizJueves's own against libcint (three_center.md,
// "Tolerances"): 1e-12 of max|(mn|P)|.
constexpr double kRelTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "density_fitting";

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too: Integral x^2l exp(-g r^2) = (2l-1)!! pi^(3/2) / ((2g)^l
// g^(3/2)), over its coefficients as held.
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

// Each spherical function's norm N_i = sqrt(<i|i>): function i here is N_i
// times PySCF's unit-normalised one (test/gtest/data/README.md).
std::vector<double> norms(const fb::BasisSet &basis) {
  std::vector<double> n;
  for (const fb::Shell &shell : basis.shells()) {
    n.insert(n.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
  }
  return n;
}

// The case's two bases over its molecule, placed as the DF steps will, their
// shells checked against the manifest's momenta; then (mn|P) twice, against
// int3c2e over PySCF's functions.
void check_three_center(const std::string &name) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataDir / "density_fitting.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const fb::json *c = nullptr;
  for (const fb::json &entry : (*doc)["integrals"]) {
    if (entry["name"].get<std::string>() == name) {
      c = &entry;
    }
  }
  ASSERT_NE(c, nullptr) << "no integrals case " << name;

  fb::DeviceResources res{0};
  std::vector<fb::Atom> atoms;
  for (const auto &atom : (*c)["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  const fb::Molecule molecule(res, std::move(atoms));
  const auto stamp = fb::resolve_basis((*c)["basis"].get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto basis = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  const auto aux_stamp = fb::resolve_aux_basis((*c)["aux_basis"].get<std::string>());
  ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
  const auto aux = fb::place(aux_stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(aux) << aux.error().message;

  const auto nao = (*c)["nao"].get<std::size_t>();
  const auto naux = (*c)["naux"].get<std::size_t>();
  ASSERT_EQ(basis->nao_sphe(), nao);
  ASSERT_EQ(aux->nao_sphe(), naux);
  const auto lmax = [](const fb::BasisSet &b) {
    std::int32_t l = 0;
    for (const fb::Shell &shell : b.shells()) {
      l = std::max(l, shell.angular_momentum());
    }
    return l;
  };
  // DZ reaches (dd|f), TZ (ff|g), the family's top class.
  EXPECT_EQ(lmax(*basis), name == "h2o_ccpvdz" ? 2 : 3);
  EXPECT_EQ(lmax(*aux), name == "h2o_ccpvdz" ? 3 : 4);

  // Each piece is (nao, nao, end - begin), Fortran order: (mn|P) at
  // m + nao (n + nao (P - begin)). Gathered here into the call's layout,
  // P + naux (n + nao m).
  std::vector<double> golden(naux * nao * nao, std::numeric_limits<double>::quiet_NaN());
  std::size_t covered = 0;
  for (const auto &piece : (*c)["int3c2e"]) {
    const auto range = piece["aux"].get<std::vector<std::size_t>>();
    const auto slab = fb::read_npy(kDataDir / piece["file"].get<std::string>());
    ASSERT_TRUE(slab) << slab.error().message;
    ASSERT_EQ(slab->shape, (std::vector<std::size_t>{nao, nao, range[1] - range[0]}));
    ASSERT_EQ(range[0], covered);
    for (std::size_t p = range[0]; p < range[1]; ++p) {
      for (std::size_t n = 0; n < nao; ++n) {
        for (std::size_t m = 0; m < nao; ++m) {
          golden[p + (naux * (n + (nao * m)))] =
              slab->data[m + (nao * (n + (nao * (p - range[0]))))];
        }
      }
    }
    covered = range[1];
  }
  ASSERT_EQ(covered, naux);
  double scale = 0.0;
  for (const double v : golden) {
    scale = std::max(scale, std::abs(v));
  }

  const auto classes = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(classes) << classes.error().message;
  fb::Buffers::device<double> b(naux * nao * nao, res.device());
  std::array<std::vector<double>, 2> runs;
  for (std::vector<double> &run : runs) {
    const fb::Status status = fb::three_center(res, *basis, *classes, *aux, b.data());
    ASSERT_TRUE(status) << status.error().message;
    run = download(res, b.data(), b.num_elements());
  }

  const std::vector<double> &ours = runs[0];
  const std::vector<double> n_ao = norms(*basis);
  const std::vector<double> n_aux = norms(*aux);
  double worst = 0.0;
  std::array<std::size_t, 3> at{};
  double asymmetry = 0.0;
  for (std::size_t m = 0; m < nao; ++m) {
    for (std::size_t n = 0; n < nao; ++n) {
      for (std::size_t p = 0; p < naux; ++p) {
        const std::size_t i = p + (naux * (n + (nao * m)));
        const double d = std::abs((ours[i] / (n_ao[m] * n_ao[n] * n_aux[p])) - golden[i]);
        if (!(d <= worst)) {
          worst = d;
          at = {m, n, p};
        }
        asymmetry = std::max(asymmetry, std::abs(ours[i] - ours[p + (naux * (m + (nao * n)))]));
      }
    }
  }
  EXPECT_LE(worst, kRelTol * scale)
      << name << ": worst |(mn|P) - int3c2e| " << worst << " at (" << at[0] << ", " << at[1]
      << " | " << at[2] << "), max|(mn|P)| " << scale;
  std::cout << std::format("[ eri3     ] {}: nao {}, naux {}, max|(mn|P)| {:.4f}, worst "
                           "|(mn|P) - int3c2e| / max|(mn|P)| {:.3e}\n",
                           name, nao, naux, scale, worst / scale);
  EXPECT_EQ(asymmetry, 0.0) << "(mn|P) and (nm|P) differ";
  EXPECT_EQ(std::memcmp(runs[0].data(), runs[1].data(), sizeof(double) * runs[0].size()), 0)
      << "two runs differ";
}

TEST(DfThreeCenter, DzThroughDdFMatchesInt3c2e) {
  check_three_center("h2o_ccpvdz");
}

TEST(DfThreeCenter, TzThroughFfGMatchesInt3c2e) {
  check_three_center("h2o_ccpvtz");
}

// A g orbital shell and a k aux shell have no kernel, and a tensor past the
// limit is refused, naming its bytes; each before anything is written. The
// aux ladder reaches i (kThreeCenterAuxMaxMomentum), which is why the aux
// refusal is k and not h; df_high_aux_gpu.cpp runs the h and i classes.
TEST(DfThreeCenter, Refusals) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet s(std::vector<fb::Shell>{fb::Shell{0, {{1.0, 1.0}}, {0.0, 0.0, 0.0}}});
  const fb::BasisSet sg(std::vector<fb::Shell>{fb::Shell{0, {{1.0, 1.0}}, {0.0, 0.0, 0.0}},
                                               fb::Shell{4, {{1.0, 1.0}}, {0.0, 0.0, 1.0}}});
  const fb::BasisSet k(std::vector<fb::Shell>{fb::Shell{7, {{1.0, 1.0}}, {0.0, 0.0, 0.0}}});
  const auto refusal = [&](const fb::BasisSet &basis, const fb::BasisSet &aux,
                           const std::size_t max_bytes, const std::string &says) {
    const auto classes = fb::primitive_pair_classes(res, basis);
    ASSERT_TRUE(classes) << classes.error().message;
    fb::Buffers::device<double> b(basis.nao_sphe() * basis.nao_sphe() * aux.nao_sphe(),
                                  res.device());
    const fb::Status status = fb::three_center(res, basis, *classes, aux, b.data(), max_bytes);
    ASSERT_FALSE(status) << says;
    EXPECT_EQ(status.error().kind, fb::ErrorKind::Unsupported) << status.error().message;
    EXPECT_NE(status.error().message.find(says), std::string::npos) << status.error().message;
  };
  constexpr std::size_t kUnlimited = std::numeric_limits<std::size_t>::max();
  refusal(sg, s, kUnlimited, "three_center: orbital shell 1 has momentum 4");
  refusal(s, k, kUnlimited, "three_center: aux shell 0 has momentum 7");
  // One (ss|s) element, 8 bytes, but four 256-byte regions of scratch.
  refusal(s, s, 1023, "needs 1024 bytes of device scratch (the Cartesian tensor alone 8 bytes)");
  // At the limit it runs.
  const auto classes = fb::primitive_pair_classes(res, s);
  ASSERT_TRUE(classes) << classes.error().message;
  fb::Buffers::device<double> b(1, res.device());
  const fb::Status status = fb::three_center(res, s, *classes, s, b.data(), 1024);
  ASSERT_TRUE(status) << status.error().message;
  // (ss|s) of unit exponents and coefficients at one centre:
  // 2 pi^{5/2} / (p alpha sqrt(p + alpha)), p = 2, alpha = 1.
  EXPECT_NEAR(download(res, b.data(), 1)[0],
              2.0 * std::pow(std::numbers::pi, 2.5) / (2.0 * std::sqrt(3.0)), 1e-13);
}

} // namespace
