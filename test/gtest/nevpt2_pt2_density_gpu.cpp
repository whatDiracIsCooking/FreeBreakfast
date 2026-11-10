// GPU cases for nevpt2::sc_pt2_transition_density and
// nevpt2::pc_pt2_transition_density, the QD-NEVPT2 PT2 transition density
// D = Σ_MN T_MN ⟨Ψ_M⁽¹⁾|E_xy|N⟩, against #744's brute force
// (test/gtest/data/qd_nevpt2_nac/brute_force/, its README has the layout),
// which is written on test/gtest/data/qd_nevpt2/'s own state: so both sides
// start from that fixture's integrals, orbitals and golden transition RDMs,
// as the QD energy goldens do. On its three cases, SC and PC: every pair
// M < N through T = ½(E_MN − E_NM) against ½(D^(MN) − D^(NM)), and one seeded
// random antisymmetric T against Σ T_MN D^(MN), each to 1e-10 of the compared
// density's largest element (#746's "How results are compared"; 1e-8 for
// h8_ring_sto3g_sa3's SC, kRingScDensityTol); every
// element outside the (t,i), (a,t) and (a,i) blocks exactly zero. Then the
// effective Hamiltonians with their new amplitude output against the same
// calls without it, a PC refusal, and malformed T.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.npy;
import free_breakfast.nevpt2;

#include "test_workspaces.h"

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

// Path agreement and the brute force: 1e-10 of the compared quantity's
// largest element.
constexpr double kDensityTol = 1e-10;
// h8_ring_sto3g_sa3's SC alone: the conditioning bound, not 1e-10 (#746,
// https://github.com/whatDiracIsCooking/FreeBreakfast/issues/746#issuecomment-6600539219).
// Its Si class's own norms are N = 6e-9 to 1.5e-8 (every other class and case
// 1e-3 to 1e-1), so the SC coefficient c = -1/(Δ + H/N) amplifies the
// Rayleigh quotient's round-off by ~1/N: c is off by ~2e-8 relative, and the
// (t,i) element it scales by 1.4-2.3e-10 of max |D| (measured, gfx1200). The
// energy cannot see it (c N ~ 1e-8), and its walk is not changed.
constexpr double kRingScDensityTol = 1e-8;
// The effective Hamiltonian with and without its amplitude output.
constexpr double kSameTol = 1e-10;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR};

fb::NpyArray read_array(const std::filesystem::path &path) {
  auto a = fb::read_npy(path);
  EXPECT_TRUE(a) << path << ": " << (a ? "" : a.error().message);
  return a ? std::move(*a) : fb::NpyArray{};
}

// @p a, column-major as io.npy returns it, row-major (C order).
std::vector<double> row_major(const fb::NpyArray &a) {
  std::vector<std::size_t> stride(a.shape.size(), 1);
  for (std::size_t d = 1; d < a.shape.size(); ++d) {
    stride[d] = stride[d - 1] * a.shape[d - 1];
  }
  std::vector<double> out(a.data.size());
  for (std::size_t at = 0; at < out.size(); ++at) {
    std::size_t rest = at;
    std::size_t cm = 0;
    for (std::size_t d = a.shape.size(); d-- > 0;) {
      cm += (rest % a.shape[d]) * stride[d];
      rest /= a.shape[d];
    }
    out[at] = a.data[cm];
  }
  return out;
}

fb::Buffers::device<double> upload(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

std::vector<double> download(fb::DeviceResources &res, const fb::Buffers::device<double> &d,
                             const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// One qd_nevpt2/ case on the device, as nevpt2_qd_energy_gpu.cpp loads it:
// the MO blocks, the shared inputs and the golden transition RDMs per pair.
struct Case {
  std::size_t nroots = 0;
  std::size_t ncore = 0;
  std::size_t ncas = 0;
  std::size_t nvirt = 0;
  std::optional<nevpt2::MoBlocks> blocks;
  std::deque<fb::Buffers::device<double>> arrays;
  std::vector<nevpt2::EnergyInputs> pairs;
};

Case load_case(fb::DeviceResources &res, const std::filesystem::path &dir) {
  const fb::NpyArray aa = read_array(dir / "B_aa.npy");
  const fb::NpyArray ca = read_array(dir / "B_ca.npy");
  const fb::NpyArray va = read_array(dir / "B_va.npy");
  const fb::NpyArray cv = read_array(dir / "B_cv.npy");
  const fb::NpyArray energies = read_array(dir / "casci_energies.npy");
  Case c;
  if (aa.shape.size() != 3 || cv.shape.size() != 3 || energies.shape.size() != 1) {
    ADD_FAILURE() << dir << ": B blocks or casci_energies missing";
    return c;
  }
  const std::size_t naux = aa.shape[0];
  const std::size_t na = aa.shape[1];
  const std::size_t nc = cv.shape[1];
  const std::size_t nv = cv.shape[2];
  std::vector<double> vc(naux * nv * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    for (std::size_t r = 0; r < nv; ++r) {
      for (std::size_t q = 0; q < naux; ++q) {
        vc[q + naux * (r + nv * i)] = cv.data[q + naux * (i + nc * r)];
      }
    }
  }
  c.nroots = energies.shape[0];
  c.ncore = nc;
  c.ncas = na;
  c.nvirt = nv;
  c.blocks.emplace(nevpt2::MoBlocks{.naux = static_cast<std::int32_t>(naux),
                                    .ncore = static_cast<std::int32_t>(nc),
                                    .ncas = static_cast<std::int32_t>(na),
                                    .nvirt = static_cast<std::int32_t>(nv),
                                    .aa = upload(res, aa.data),
                                    .ca = upload(res, ca.data),
                                    .va = upload(res, va.data),
                                    .cv = upload(res, cv.data),
                                    .vc = upload(res, vc)});
  const auto put = [&](const char *name, const std::vector<std::size_t> &shape) {
    const fb::NpyArray a = read_array(dir / (std::string{name} + ".npy"));
    EXPECT_EQ(a.shape, shape) << name;
    c.arrays.push_back(upload(res, a.data));
    return static_cast<const double *>(c.arrays.back().data());
  };
  nevpt2::EnergyInputs shared;
  shared.h1e = put("h1e", {na, na});
  shared.h2e = put("h2e", {na, na, na, na});
  shared.e_core = put("e_core", {nc});
  shared.e_virt = put("e_virt", {nv});
  shared.h1e_v_sr = put("h1e_v_Sr", {nv, na});
  shared.h1e_v_si = put("h1e_v_Si", {na, nc});
  shared.h1e_v_sir = put("h1e_v_Sir", {nv, nc});
  c.pairs.assign(c.nroots * c.nroots, shared);
  for (std::size_t i = 0; i < c.nroots; ++i) {
    for (std::size_t j = 0; j < c.nroots; ++j) {
      const std::filesystem::path pair =
          dir / ("pair_" + std::to_string(i) + "_" + std::to_string(j));
      const auto rdm = [&](const char *name) {
        c.arrays.push_back(upload(res, row_major(read_array(pair / (std::string{name} + ".npy")))));
        return static_cast<const double *>(c.arrays.back().data());
      };
      nevpt2::EnergyInputs &in = c.pairs[(i * c.nroots) + j];
      in.dm1 = rdm("dm1");
      in.dm2 = rdm("dm2");
      in.dm3 = rdm("dm3");
      in.f3ca = rdm("f3ca");
      in.f3ac = rdm("f3ac");
    }
  }
  return c;
}

using Density = fb::Result<fb::Buffers::device<double>> (*)(fb::DeviceResources &,
                                                            const nevpt2::MoBlocks &,
                                                            std::span<const nevpt2::EnergyInputs>,
                                                            std::span<const double>, std::int32_t);

struct Scheme {
  std::string name;
  Density density;
  std::string file;
};

const std::array<Scheme, 2> kSchemes{
    Scheme{.name = "SC", .density = nevpt2::sc_pt2_transition_density, .file = "sc_pt2_density"},
    Scheme{.name = "PC", .density = nevpt2::pc_pt2_transition_density, .file = "pc_pt2_density"}};

// Whether (x, y) is in the (t,i), (a,t) or (a,i) block.
bool in_blocks(const std::size_t x, const std::size_t y, const Case &c) {
  const auto core = [&](const std::size_t p) { return p < c.ncore; };
  const auto active = [&](const std::size_t p) { return p >= c.ncore && p < c.ncore + c.ncas; };
  const auto virt = [&](const std::size_t p) { return p >= c.ncore + c.ncas; };
  return (active(x) && core(y)) || (virt(x) && active(y)) || (virt(x) && core(y));
}

// The density at T against @p ref (nmo² row-major), to @p tol of
// max |ref|, and exactly zero off the blocks. Returns the miss over max |ref|.
double expect_density(fb::DeviceResources &res, const Case &c, const Scheme &scheme,
                      const std::vector<double> &t, const std::vector<double> &ref,
                      const std::string &what, const double tol, const std::int32_t batch = 0) {
  const std::size_t nmo = c.ncore + c.ncas + c.nvirt;
  const auto got = scheme.density(res, *c.blocks, c.pairs, t, batch);
  EXPECT_TRUE(got) << what << ": " << (got ? "" : got.error().message);
  if (!got) {
    return std::numeric_limits<double>::infinity();
  }
  const std::vector<double> d = download(res, *got, nmo * nmo);
  double largest = 0.0;
  for (const double x : ref) {
    largest = std::max(largest, std::abs(x));
  }
  EXPECT_GT(largest, 0.0) << what;
  double miss = 0.0;
  std::size_t worst_at = 0;
  std::size_t off_nonzero = 0;
  for (std::size_t x = 0; x < nmo; ++x) {
    for (std::size_t y = 0; y < nmo; ++y) {
      const std::size_t at = (x * nmo) + y;
      if (!in_blocks(x, y, c) && d[at] != 0.0) {
        ++off_nonzero;
      }
      if (std::abs(d[at] - ref[at]) > miss) {
        miss = std::abs(d[at] - ref[at]);
        worst_at = at;
      }
    }
  }
  EXPECT_EQ(off_nonzero, 0U) << what << ": elements off the (t,i), (a,t), (a,i) blocks";
  EXPECT_LE(miss, tol * largest) << what << ": max |ref| " << largest << ", worst at ("
                                 << worst_at / nmo << ", " << worst_at % nmo << "): " << d[worst_at]
                                 << " against " << ref[worst_at];
  return miss / largest;
}

const std::array<std::string, 3> kCases{"h6_631g_sa2", "h8_sto3g_sa3", "h8_ring_sto3g_sa3"};

class Nevpt2Pt2Density : public ::testing::TestWithParam<std::string> {};

// Every pair through T = ½(E_MN − E_NM), and one random antisymmetric T,
// against the brute force, SC and PC; SC's random T again in batches of two
// (each core and virtual walk split), against the whole walk.
TEST_P(Nevpt2Pt2Density, MatchesBruteForce) {
  FBF_REQUIRE_GPU();
  const std::string name = GetParam();
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  Case c = load_case(res, kDataDir / "qd_nevpt2" / name);
  ASSERT_TRUE(c.blocks.has_value());
  const std::size_t n = c.nroots;
  const std::size_t nmo = c.ncore + c.ncas + c.nvirt;

  std::mt19937_64 rng(746);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> random_t(n * n, 0.0);
  for (std::size_t m = 0; m < n; ++m) {
    for (std::size_t k = m + 1; k < n; ++k) {
      random_t[(m * n) + k] = uniform(rng);
      random_t[(k * n) + m] = -random_t[(m * n) + k];
    }
  }

  for (const Scheme &scheme : kSchemes) {
    const fb::NpyArray brute =
        read_array(kDataDir / "qd_nevpt2_nac" / "brute_force" / name / (scheme.file + ".npy"));
    ASSERT_EQ(brute.shape, (std::vector<std::size_t>{n, n, nmo, nmo})) << scheme.file;
    // D^(MN)_xy, Fortran order.
    const auto dmn = [&](const std::size_t m, const std::size_t k, const std::size_t x,
                         const std::size_t y) {
      return brute.data[m + n * (k + n * (x + nmo * y))];
    };
    const double tol =
        name == "h8_ring_sto3g_sa3" && scheme.name == "SC" ? kRingScDensityTol : kDensityTol;
    double worst = 0.0;
    for (std::size_t m = 0; m < n; ++m) {
      for (std::size_t k = m + 1; k < n; ++k) {
        std::vector<double> t(n * n, 0.0);
        t[(m * n) + k] = 0.5;
        t[(k * n) + m] = -0.5;
        std::vector<double> ref(nmo * nmo);
        for (std::size_t x = 0; x < nmo; ++x) {
          for (std::size_t y = 0; y < nmo; ++y) {
            ref[(x * nmo) + y] = 0.5 * (dmn(m, k, x, y) - dmn(k, m, x, y));
          }
        }
        const std::string what = name + " " + scheme.name + " pair (" + std::to_string(m) + ", " +
                                 std::to_string(k) + ")";
        worst = std::max(worst, expect_density(res, c, scheme, t, ref, what, tol));
      }
    }
    std::vector<double> ref(nmo * nmo, 0.0);
    for (std::size_t m = 0; m < n; ++m) {
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t x = 0; x < nmo; ++x) {
          for (std::size_t y = 0; y < nmo; ++y) {
            ref[(x * nmo) + y] += random_t[(m * n) + k] * dmn(m, k, x, y);
          }
        }
      }
    }
    worst = std::max(worst, expect_density(res, c, scheme, random_t, ref,
                                           name + " " + scheme.name + " random T", tol));
    worst =
        std::max(worst, expect_density(res, c, scheme, random_t, ref,
                                       name + " " + scheme.name + " random T, batch 2", tol, 2));
    std::cout << "[ pt2 density ] " << name << " " << scheme.name
              << ": worst miss / max |ref| = " << std::scientific << std::setprecision(2) << worst
              << std::defaultfloat << "\n";
  }
}

// The effective Hamiltonians that hand their amplitudes over return what
// they return without the hand-over, to 1e-10, with PC's dropped counts
// equal, and hand over every state's amplitudes for every class.
TEST_P(Nevpt2Pt2Density, AmplitudeOutputLeavesHeffUnchanged) {
  FBF_REQUIRE_GPU();
  const std::string name = GetParam();
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  Case c = load_case(res, kDataDir / "qd_nevpt2" / name);
  ASSERT_TRUE(c.blocks.has_value());
  const std::size_t n = c.nroots;

  const auto sc = nevpt2::sc_effective_hamiltonian(res, *c.blocks, c.pairs, 0);
  ASSERT_TRUE(sc) << sc.error().message;
  nevpt2::ScAmplitudes sc_amps;
  const auto sc_out = nevpt2::sc_effective_hamiltonian(res, *c.blocks, c.pairs, 0, &sc_amps);
  ASSERT_TRUE(sc_out) << sc_out.error().message;
  ASSERT_EQ(sc_out->size(), sc->size());
  for (std::size_t p = 0; p < sc->size(); ++p) {
    for (std::size_t k = 0; k < 8; ++k) {
      EXPECT_NEAR((*sc_out)[p].energies[k], (*sc)[p].energies[k], kSameTol) << p << " " << k;
      EXPECT_NEAR((*sc_out)[p].norms[k], (*sc)[p].norms[k], kSameTol) << p << " " << k;
    }
    EXPECT_NEAR((*sc_out)[p].total, (*sc)[p].total, kSameTol) << p;
  }
  ASSERT_EQ(sc_amps.sr.size(), n);
  ASSERT_EQ(sc_amps.si.size(), n);
  ASSERT_EQ(sc_amps.sir.size(), n);
  for (std::size_t j = 0; j < n; ++j) {
    EXPECT_EQ(sc_amps.sr[j].size(), c.nvirt);
    EXPECT_EQ(sc_amps.si[j].size(), c.ncore);
    EXPECT_EQ(sc_amps.sir[j].size(), c.nvirt * c.ncore);
  }

  const auto pc = nevpt2::pc_effective_hamiltonian(res, *c.blocks, c.pairs, 0);
  ASSERT_TRUE(pc) << pc.error().message;
  nevpt2::PcAmplitudes pc_amps;
  const auto pc_out = nevpt2::pc_effective_hamiltonian(res, *c.blocks, c.pairs, 0, &pc_amps);
  ASSERT_TRUE(pc_out) << pc_out.error().message;
  ASSERT_EQ(pc_out->size(), pc->size());
  for (std::size_t p = 0; p < pc->size(); ++p) {
    for (std::size_t k = 0; k < 8; ++k) {
      EXPECT_NEAR((*pc_out)[p].energies[k], (*pc)[p].energies[k], kSameTol) << p << " " << k;
      EXPECT_EQ((*pc_out)[p].dropped[k], (*pc)[p].dropped[k]) << p << " " << k;
    }
    EXPECT_NEAR((*pc_out)[p].total, (*pc)[p].total, kSameTol) << p;
  }
  // Every class's g is tuples x m_J, m_J = d - J's dropped count.
  const std::size_t na = c.ncas;
  const auto expect_class = [&](const nevpt2::PcAmplitudes::Class &cls, const std::size_t k,
                                const std::size_t d, const std::size_t tuples) {
    ASSERT_EQ(cls.u.size(), n) << k;
    ASSERT_EQ(cls.g.size(), n) << k;
    for (std::size_t j = 0; j < n; ++j) {
      const auto m = d - static_cast<std::size_t>((*pc)[(j * n) + j].dropped[k]);
      EXPECT_EQ(cls.g[j].size(), tuples * m) << k << " state " << j;
    }
  };
  expect_class(pc_amps.sr, 0, na * na * na, c.nvirt);
  expect_class(pc_amps.si, 1, na * na * na, c.ncore);
  expect_class(pc_amps.sir, 7, 2 * na * na, c.nvirt * c.ncore);
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2Pt2Density, ::testing::ValuesIn(kCases),
                         [](const auto &info) { return info.param; });

// A PC refusal of any class of any state refuses the density with the
// energy's error: h8_sto3g_sa3 with state 1's own RDMs zero (Sr's metric has
// no positive eigenvalue), as Nevpt2PcQdEnergyRefusals has it. SC, which has
// no such refusal, still runs.
TEST(Nevpt2Pt2DensityRefusals, PcRefusalRefuses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  Case c = load_case(res, kDataDir / "qd_nevpt2" / "h8_sto3g_sa3");
  ASSERT_TRUE(c.blocks.has_value());
  const std::size_t n = c.nroots;
  ASSERT_EQ(n, 3U);
  c.arrays.push_back(
      upload(res, std::vector<double>(c.ncas * c.ncas * c.ncas * c.ncas * c.ncas * c.ncas, 0.0)));
  const double *const zeros = c.arrays.back().data();
  nevpt2::EnergyInputs &own1 = c.pairs[n + 1];
  own1.dm1 = own1.dm2 = own1.dm3 = own1.f3ca = own1.f3ac = zeros;
  std::vector<double> t(n * n, 0.0);
  t[1] = 0.5;
  t[n] = -0.5;

  const auto heff = nevpt2::pc_effective_hamiltonian(res, *c.blocks, c.pairs, 0);
  ASSERT_FALSE(heff);
  const auto got = nevpt2::pc_pt2_transition_density(res, *c.blocks, c.pairs, t, 0);
  ASSERT_FALSE(got);
  EXPECT_EQ(got.error().kind, heff.error().kind);
  EXPECT_EQ(got.error().message, heff.error().message);
  EXPECT_NE(got.error().message.find("state 1: Sr's metric"), std::string::npos)
      << got.error().message;
  const auto sc = nevpt2::sc_pt2_transition_density(res, *c.blocks, c.pairs, t, 0);
  EXPECT_TRUE(sc) << sc.error().message;
}

// T not n x n, or not antisymmetric, is InvalidConfig, SC and PC.
TEST(Nevpt2Pt2DensityRefusals, MalformedTIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  Case c = load_case(res, kDataDir / "qd_nevpt2" / "h6_631g_sa2");
  ASSERT_TRUE(c.blocks.has_value());
  ASSERT_EQ(c.pairs.size(), 4U);
  const std::vector<double> short_t{0.0, 0.5, -0.5};
  const std::vector<double> symmetric{0.0, 0.5, 0.5, 0.0};
  const std::vector<double> diagonal{1.0, 0.5, -0.5, 0.0};
  for (const Scheme &scheme : kSchemes) {
    for (const std::vector<double> *t : {&short_t, &symmetric, &diagonal}) {
      const auto got = scheme.density(res, *c.blocks, c.pairs, *t, 0);
      ASSERT_FALSE(got) << scheme.name;
      EXPECT_EQ(got.error().kind, fb::ErrorKind::InvalidConfig) << scheme.name;
    }
  }
}

} // namespace
