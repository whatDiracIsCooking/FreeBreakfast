// GPU cases for nevpt2::sc_effective_hamiltonian, the SC-QD-NEVPT2 effective
// Hamiltonian, against test/gtest/data/qd_nevpt2/ (its README has the
// conventions): from the golden transition RDMs and the slabs over the golden
// B blocks, every class's coupling ⟨I|V|Ψ_J⁽¹⁾⟩ and transition norm Σ_t T^t_IJ
// of the unsymmetrized matrix, and its total heff2, to 1e-10 on all six
// cases; the diagonal against class_energies on the same inputs, and a walk in
// batches of two against the whole one, to round-off. Then the chain: the
// transition RDMs from nevpt2::rdms over the golden CI vectors, whose phases
// the goldens fix, in place of the golden ones. Last, the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.npy;
import free_breakfast.ci;
import free_breakfast.nevpt2;

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

// Against the goldens: each element is a sum of a few hundred perturbers of
// at most 1e-2, and the brute force agrees with them to 2.7e-16.
constexpr double kGoldenTol = 1e-10;
// Between two walks of the same arithmetic: the einsums' atomics reassociate
// their sums, and a batched walk splits each class's sum by slab.
constexpr double kRoundOffTol = 1e-13;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2";

constexpr std::array<const char *, 8> kClassNames{"Sr",   "Si",  "Sijrs", "Sijr",
                                                  "Srsi", "Srs", "Sij",   "Sir"};

fb::NpyArray read_array(const std::filesystem::path &path) {
  auto a = fb::read_npy(path);
  EXPECT_TRUE(a) << path << ": " << (a ? "" : a.error().message);
  return a ? std::move(*a) : fb::NpyArray{};
}

// @p a, column-major as io.npy returns it, row-major (C order): the RDMs'
// layout, and the one the golden (n, n, 8) tables are indexed in here.
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
  // host is pageable: the copy has not finished reading it until the stream has.
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

// One case on the device: the MO blocks, the shared inputs, and the n x n
// pair inputs, each pair's RDMs left to the caller.
struct Case {
  std::size_t nroots = 0;
  std::size_t ncas = 0;
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
  // B_vc, B_cv with its orbital pair swapped.
  std::vector<double> vc(naux * nv * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    for (std::size_t r = 0; r < nv; ++r) {
      for (std::size_t q = 0; q < naux; ++q) {
        vc[q + naux * (r + nv * i)] = cv.data[q + naux * (i + nc * r)];
      }
    }
  }
  c.nroots = energies.shape[0];
  c.ncas = na;
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
  return c;
}

// Every pair's five RDMs from pair_I_J/, the golden transition RDMs.
void golden_rdms(fb::DeviceResources &res, const std::filesystem::path &dir, Case &c) {
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
}

// Every element's per-class coupling and transition norm against the golden
// couplings and t_norms, its total against heff2.
void expect_golden(const std::filesystem::path &dir, const Case &c,
                   const std::vector<nevpt2::ClassEnergies> &got, const std::string &what) {
  const std::vector<double> couplings = row_major(read_array(dir / "couplings.npy"));
  const std::vector<double> t_norms = row_major(read_array(dir / "t_norms.npy"));
  const std::vector<double> heff2 = row_major(read_array(dir / "heff2.npy"));
  const std::size_t n = c.nroots;
  ASSERT_EQ(got.size(), n * n) << what;
  ASSERT_EQ(couplings.size(), n * n * 8) << what;
  ASSERT_EQ(t_norms.size(), n * n * 8) << what;
  ASSERT_EQ(heff2.size(), n * n) << what;
  for (std::size_t p = 0; p < n * n; ++p) {
    const std::string at = what + " (" + std::to_string(p / n) + ", " + std::to_string(p % n) + ")";
    for (std::size_t k = 0; k < 8; ++k) {
      EXPECT_NEAR(got[p].energies[k], couplings[(p * 8) + k], kGoldenTol) << at << kClassNames[k];
      EXPECT_NEAR(got[p].norms[k], t_norms[(p * 8) + k], kGoldenTol) << at << kClassNames[k];
    }
    EXPECT_NEAR(got[p].total, heff2[p], kGoldenTol) << at;
  }
}

void expect_same(const nevpt2::ClassEnergies &a, const nevpt2::ClassEnergies &b,
                 const std::string &what) {
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(a.energies[k], b.energies[k], kRoundOffTol) << what << " " << kClassNames[k];
    EXPECT_NEAR(a.norms[k], b.norms[k], kRoundOffTol) << what << " " << kClassNames[k];
  }
  EXPECT_NEAR(a.total, b.total, kRoundOffTol) << what;
}

// std::string, not const char *: ctest names each case by printing its
// parameter, and a pointer prints as its address.
const std::array<std::string, 6> kCases{"h2co_ccpvdz_sa2",      "h2co_ccpvdz_sa2_weighted",
                                        "h2co_skew_ccpvdz_sa3", "h6_631g_sa2",
                                        "h8_sto3g_sa3",         "h8_ring_sto3g_sa3"};

class Nevpt2QdEnergy : public ::testing::TestWithParam<std::string> {};

// Batch 2 splits every core and virtual walk of the cc-pVDZ and 6-31G cases
// (6 and 2 core, 29 and 7 virtual orbitals), the odd ones ending partial.
TEST_P(Nevpt2QdEnergy, MatchesGoldenDiagonalAndBatches) {
  FBF_REQUIRE_GPU();
  const std::string name = GetParam();
  const std::filesystem::path dir = kDataDir / name;
  fb::DeviceResources res(0);
  Case c = load_case(res, dir);
  ASSERT_TRUE(c.blocks.has_value());
  golden_rdms(res, dir, c);

  const auto whole = nevpt2::sc_effective_hamiltonian(res, *c.blocks, c.pairs, 0);
  ASSERT_TRUE(whole) << whole.error().message;
  expect_golden(dir, c, *whole, name);

  const std::size_t n = c.nroots;
  for (std::size_t j = 0; j < n; ++j) {
    const auto own = nevpt2::class_energies(res, *c.blocks, c.pairs[(j * n) + j], 0);
    ASSERT_TRUE(own) << own.error().message;
    expect_same((*whole)[(j * n) + j], *own, name + " diagonal " + std::to_string(j));
  }

  const auto batched = nevpt2::sc_effective_hamiltonian(res, *c.blocks, c.pairs, 2);
  ASSERT_TRUE(batched) << batched.error().message;
  ASSERT_EQ(batched->size(), whole->size());
  for (std::size_t p = 0; p < n * n; ++p) {
    expect_same((*batched)[p], (*whole)[p],
                name + " batch 2 (" + std::to_string(p / n) + ", " + std::to_string(p % n) + ")");
  }
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2QdEnergy, ::testing::ValuesIn(kCases),
                         [](const auto &info) { return info.param; });

// The chain: nevpt2::rdms' transition dm1, dm2, dm3, f3ca and f3ac between the
// golden CI vectors, bra I and ket J, in place of the golden RDMs. The three
// three-root cases, whose couplings do not vanish by symmetry.
// The parameter is the case and its electrons per spin.
class Nevpt2QdEnergyDeviceRdms
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

TEST_P(Nevpt2QdEnergyDeviceRdms, MatchesGolden) {
  FBF_REQUIRE_GPU();
  const auto [name, per_spin] = GetParam();
  const std::filesystem::path dir = kDataDir / name;
  fb::DeviceResources res(0);
  Case c = load_case(res, dir);
  ASSERT_TRUE(c.blocks.has_value());
  const std::size_t na = c.ncas;
  const fb::NpyArray ci = read_array(dir / "ci.npy");
  ASSERT_EQ(ci.shape.size(), 3U);
  ASSERT_EQ(ci.shape[0], c.nroots);
  ASSERT_EQ(ci.shape[1], ci.shape[2]) << "n_alpha = n_beta";
  auto strings = fb::ci::DeviceStringSet::create(res, static_cast<std::int32_t>(na), per_spin);
  ASSERT_TRUE(strings) << strings.error().message;

  // rdms reads (tu|vw) at t + n (u + n (v + n w)); h2e[p, q, r, s] = (pr|qs)
  // at p + n (q + n (r + n s)), so (tu|vw) = h2e[t, v, u, w].
  const fb::NpyArray h2e = read_array(dir / "h2e.npy");
  std::vector<double> eri(na * na * na * na);
  for (std::size_t t = 0; t < na; ++t) {
    for (std::size_t u = 0; u < na; ++u) {
      for (std::size_t v = 0; v < na; ++v) {
        for (std::size_t w = 0; w < na; ++w) {
          eri[t + na * (u + na * (v + na * w))] = h2e.data[t + na * (v + na * (u + na * w))];
        }
      }
    }
  }
  const fb::Buffers::device<double> eri_d = upload(res, eri);
  // Row-major (root, alpha, beta): root I's vector is contiguous.
  const fb::Buffers::device<double> vectors = upload(res, row_major(ci));
  const std::size_t dim = ci.shape[1] * ci.shape[2];
  const std::size_t n2 = na * na;
  const std::size_t n6 = n2 * n2 * n2;
  for (std::size_t i = 0; i < c.nroots; ++i) {
    for (std::size_t j = 0; j < c.nroots; ++j) {
      fb::Buffers::device<double> &dm1 = c.arrays.emplace_back(n2, res.device());
      fb::Buffers::device<double> &dm2 = c.arrays.emplace_back(n2 * n2, res.device());
      fb::Buffers::device<double> &dm3 = c.arrays.emplace_back(n6, res.device());
      fb::Buffers::device<double> &f3ca = c.arrays.emplace_back(n6, res.device());
      fb::Buffers::device<double> &f3ac = c.arrays.emplace_back(n6, res.device());
      const auto tiles = nevpt2::rdms(res, *strings, *strings, vectors.data() + (i * dim),
                                      vectors.data() + (j * dim), eri_d.data(), dm1.data(),
                                      dm2.data(), dm3.data(), f3ca.data(), f3ac.data());
      ASSERT_TRUE(tiles) << tiles.error().message;
      nevpt2::EnergyInputs &in = c.pairs[(i * c.nroots) + j];
      in.dm1 = dm1.data();
      in.dm2 = dm2.data();
      in.dm3 = dm3.data();
      in.f3ca = f3ca.data();
      in.f3ac = f3ac.data();
    }
  }

  const auto got = nevpt2::sc_effective_hamiltonian(res, *c.blocks, c.pairs, 0);
  ASSERT_TRUE(got) << got.error().message;
  expect_golden(dir, c, *got, name + " from nevpt2::rdms");
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2QdEnergyDeviceRdms,
                         ::testing::Values(std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, 2},
                                           std::tuple{std::string{"h8_sto3g_sa3"}, 2},
                                           std::tuple{std::string{"h8_ring_sto3g_sa3"}, 3}),
                         [](const auto &info) { return std::get<0>(info.param); });

// Pair inputs that are not n x n, or that do not share their integrals, are
// InvalidConfig before anything is launched.
TEST(Nevpt2QdEnergyRefusals, MalformedPairsAreInvalidConfig) {
  FBF_REQUIRE_GPU();
  const std::filesystem::path dir = kDataDir / "h6_631g_sa2";
  fb::DeviceResources res(0);
  Case c = load_case(res, dir);
  ASSERT_TRUE(c.blocks.has_value());
  golden_rdms(res, dir, c);
  ASSERT_EQ(c.pairs.size(), 4U);

  const std::span<const nevpt2::EnergyInputs> all{c.pairs};
  const auto three = nevpt2::sc_effective_hamiltonian(res, *c.blocks, all.first(3), 0);
  ASSERT_FALSE(three);
  EXPECT_EQ(three.error().kind, fb::ErrorKind::InvalidConfig);
  const auto none = nevpt2::sc_effective_hamiltonian(res, *c.blocks, all.first(0), 0);
  ASSERT_FALSE(none);
  EXPECT_EQ(none.error().kind, fb::ErrorKind::InvalidConfig);

  std::vector<nevpt2::EnergyInputs> mixed = c.pairs;
  mixed[3].e_virt = mixed[3].e_core;
  const auto unshared = nevpt2::sc_effective_hamiltonian(res, *c.blocks, mixed, 0);
  ASSERT_FALSE(unshared);
  EXPECT_EQ(unshared.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
