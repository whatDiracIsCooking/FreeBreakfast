// GPU cases for the exchange build's hand-written device steps
// (src/integrals/exchange/helpers): max|D|, the Dcd gather and the block
// scatter, each against a host loop. max|D| and the gather move or compare
// doubles and nothing else, so they are checked bit for bit, on H2O/cc-pVDZ's
// recorded density and shells (s, p and d, interleaved by atom, so a shell's
// rank among its momentum differs from its index). The scatter adds, in the
// atomics' order, so it is held to 1e-15 relative on a synthetic block whose
// records repeat shells, as a bra class's primitive pairs do, mirrored and
// not.
//
// The launchers are reached through their bridge header, which brings in wwr's
// textual runtime surface (wwr/runtime_api.h). clang rejects that next to the
// same surface imported from wwr.runtime_api in a plain TU (an ODR mismatch on
// wwrMallocAsync), so this TU imports nothing that carries it --
// free_breakfast.core and core.resources among them -- and manages its device
// memory and stream through the textual wwr* calls.

#include <gtest/gtest.h>

// After every standard header: under HIP a libc++ header must not follow the
// vendor runtime header this brings in (see primitives.h).
#include "exchange_helpers_bridge.h"
#include "gpu_requirement.h"

import std;
import free_breakfast.basis;
import free_breakfast.io.json;

namespace {

namespace fb = free_breakfast;

constexpr std::int32_t kMomenta = fb::kMaxAngularMomentum + 1;

struct Golden {
  fb::BasisSet basis;
  std::vector<double> density; // nao x nao, row-major
};

Golden read_golden(const std::string &stem) {
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "coulomb" / (stem + ".json"));
  EXPECT_TRUE(doc) << stem;
  if (!doc) {
    return {};
  }
  std::vector<fb::Shell> shells;
  for (const auto &sh : (*doc)["shells"]) {
    const auto c = sh["centre"].get<std::vector<double>>();
    shells.emplace_back(sh["l"].get<std::int32_t>(), sh["exponents"].get<std::vector<double>>(),
                        sh["coefficients"].get<std::vector<double>>(),
                        fb::Vec<double, 3>{c[0], c[1], c[2]});
  }
  Golden g{fb::BasisSet(std::move(shells)),
           (*doc)["arrays"]["density"]["values"].get<std::vector<double>>()};
  EXPECT_EQ(g.density.size(), std::size_t{g.basis.nao_cart()} * g.basis.nao_cart()) << stem;
  return g;
}

std::int32_t cartesians(const std::int32_t l) {
  return (l + 1) * (l + 2) / 2;
}

// The AO offset of every shell of each momentum, by rank: a shell's index among
// the basis's shells of its momentum, in basis order.
std::array<std::vector<std::int32_t>, kMomenta> offsets_by_momentum(const fb::BasisSet &basis) {
  std::array<std::vector<std::int32_t>, kMomenta> out;
  for (std::uint32_t s = 0; s < basis.nshells(); ++s) {
    const std::int32_t l = basis.shells()[s].angular_momentum();
    out.at(static_cast<std::size_t>(l))
        .push_back(static_cast<std::int32_t>(basis.ao_offset_cart(s)));
  }
  return out;
}

// A device array of n T, freed on destruction.
template <typename T>
class DeviceArray {
public:
  explicit DeviceArray(const std::size_t n) {
    void *p = nullptr;
    EXPECT_EQ(wwr::wwrMalloc(&p, sizeof(T) * n), wwr::wwrSuccess);
    data_ = static_cast<T *>(p);
  }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  DeviceArray(DeviceArray &&other) noexcept : data_(std::exchange(other.data_, nullptr)) {}
  DeviceArray &operator=(DeviceArray &&) = delete;
  ~DeviceArray() {
    if (data_ != nullptr) {
      EXPECT_EQ(wwr::wwrFree(data_), wwr::wwrSuccess);
    }
  }
  [[nodiscard]] T *data() const { return data_; }

private:
  T *data_ = nullptr;
};

// The stream every launch goes on, made and destroyed by each case.
class Stream {
public:
  Stream() { EXPECT_EQ(wwr::wwrStreamCreate(&stream_), wwr::wwrSuccess); }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  ~Stream() { EXPECT_EQ(wwr::wwrStreamDestroy(stream_), wwr::wwrSuccess); }
  [[nodiscard]] wwr::wwrStream_t get() const { return stream_; }

private:
  wwr::wwrStream_t stream_ = nullptr;
};

template <typename T>
DeviceArray<T> upload(const Stream &stream, const std::vector<T> &host) {
  DeviceArray<T> dev(host.size());
  EXPECT_EQ(wwr::wwrMemcpyAsync(dev.data(), host.data(), sizeof(T) * host.size(),
                                wwr::wwrMemcpyHostToDevice, stream.get()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream.get()), wwr::wwrSuccess);
  return dev;
}

std::vector<double> download(const Stream &stream, const double *device, std::size_t n) {
  std::vector<double> host(n);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(double) * n, wwr::wwrMemcpyDeviceToHost,
                                stream.get()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream.get()), wwr::wwrSuccess);
  return host;
}

std::uint64_t bits(const double x) {
  return std::bit_cast<std::uint64_t>(x);
}

double host_max_abs(const std::vector<double> &d) {
  double m = 0.0;
  for (const double x : d) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

double device_max_abs(const Stream &stream, const std::vector<double> &d, std::int32_t n_ao) {
  const auto density = upload(stream, d);
  DeviceArray<double> dmax(1);
  EXPECT_EQ(fb::device::exchange_density_max(stream.get(), n_ao, density.data(), dmax.data()),
            wwr::wwrSuccess);
  return download(stream, dmax.data(), 1)[0];
}

TEST(ExchangeHelpers, DensityMaxIsTheHostMaximum) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  const Golden g = read_golden("h2o_ccpvdz");
  ASSERT_FALSE(g.density.empty());
  const auto n_ao = static_cast<std::int32_t>(g.basis.nao_cart());
  EXPECT_EQ(bits(device_max_abs(stream, g.density, n_ao)), bits(host_max_abs(g.density)));

  // The recorded density is positive throughout, so plant the largest element
  // as a negative one, off the diagonal and late in the matrix: the maximum is
  // of |D|, and reaches every element.
  std::vector<double> d = g.density;
  const std::size_t at = d.size() - 3;
  d[at] = -2.0 * host_max_abs(g.density);
  EXPECT_EQ(bits(device_max_abs(stream, d, n_ao)), bits(-d[at]));

  // A NaN is the result, not dropped.
  d[17] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(device_max_abs(stream, d, n_ao)));
}

TEST(ExchangeHelpers, GathersEveryDensityClassByRank) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  const Golden g = read_golden("h2o_ccpvdz");
  ASSERT_FALSE(g.density.empty());
  const auto n_ao = static_cast<std::int32_t>(g.basis.nao_cart());
  const auto by_l = offsets_by_momentum(g.basis);
  const auto density = upload(stream, g.density);
  for (std::int32_t lc = 0; lc < kMomenta; ++lc) {
    for (std::int32_t ld = 0; ld < kMomenta; ++ld) {
      const auto &off_c = by_l.at(static_cast<std::size_t>(lc));
      const auto &off_d = by_l.at(static_cast<std::size_t>(ld));
      ASSERT_FALSE(off_c.empty() || off_d.empty()) << "cc-pVDZ H2O has s, p and d shells";
      const auto n_c = static_cast<std::int32_t>(off_c.size());
      const auto n_d = static_cast<std::int32_t>(off_d.size());
      const std::int32_t nc = cartesians(lc);
      const std::int32_t nd = cartesians(ld);
      const auto n = static_cast<std::size_t>(n_c) * n_d * nc * nd;
      const auto off_c_d = upload(stream, off_c);
      const auto off_d_d = upload(stream, off_d);
      DeviceArray<double> dcd(n);
      ASSERT_EQ(fb::device::exchange_gather_dcd(stream.get(), lc, ld, n_ao, density.data(), n_c,
                                                off_c_d.data(), n_d, off_d_d.data(), dcd.data()),
                wwr::wwrSuccess);
      const auto got = download(stream, dcd.data(), n);
      for (std::int32_t rc = 0; rc < n_c; ++rc) {
        for (std::int32_t rd = 0; rd < n_d; ++rd) {
          const auto row = static_cast<std::size_t>(rc) * n_d + rd;
          for (std::int32_t i = 0; i < nc; ++i) {
            for (std::int32_t j = 0; j < nd; ++j) {
              const auto m = static_cast<std::size_t>(off_c[rc] + i);
              const auto q = static_cast<std::size_t>(off_d[rd] + j);
              const double want = g.density[m * n_ao + q];
              ASSERT_EQ(bits(got[row * nc * nd + i * nd + j]), bits(want))
                  << "(lc, ld) = (" << lc << ", " << ld << "), ranks (" << rc << ", " << rd
                  << "), (" << i << ", " << j << ")";
            }
          }
        }
      }
    }
  }
}

// Every (la, lb), mirrored and not: records cycling over the la shells, several
// to a shell, so atomics collide on one element; every lb shell its own block,
// scattered into a K that already holds values, which the scatter must add to.
// At la == lb the records include the b shell's own, which a mirrored scatter
// must add once, untransposed.
TEST(ExchangeHelpers, ScattersABlockIntoK) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  const Golden g = read_golden("h2o_ccpvdz");
  ASSERT_FALSE(g.density.empty());
  const auto n_ao = static_cast<std::int32_t>(g.basis.nao_cart());
  const auto nn = static_cast<std::size_t>(n_ao) * n_ao;
  const auto by_l = offsets_by_momentum(g.basis);
  constexpr std::int32_t kRecordsPerShell = 5;

  for (const bool mirror : {false, true}) {
    for (std::int32_t la = 0; la < kMomenta; ++la) {
      for (std::int32_t lb = 0; lb < kMomenta; ++lb) {
        const auto &shells_a = by_l.at(static_cast<std::size_t>(la));
        const auto &shells_b = by_l.at(static_cast<std::size_t>(lb));
        const std::int32_t na = cartesians(la);
        const std::int32_t nb = cartesians(lb);
        std::vector<std::int32_t> off_a;
        for (std::int32_t r = 0; r < kRecordsPerShell; ++r) {
          off_a.insert(off_a.end(), shells_a.begin(), shells_a.end());
        }
        const auto n_bra = static_cast<std::int32_t>(off_a.size());
        const auto width = static_cast<std::size_t>(na) * nb;

        // Positive, so the sums do not cancel and 1e-15 relative is a bar on the
        // adds alone.
        std::vector<double> want(nn);
        for (std::size_t e = 0; e < nn; ++e) {
          want[e] = 1.0 + 0.001 * static_cast<double>(e % 97);
        }
        DeviceArray<double> k = upload(stream, want);
        const auto off_a_d = upload(stream, off_a);
        std::vector<DeviceArray<double>> blocks;
        for (std::size_t b = 0; b < shells_b.size(); ++b) {
          std::vector<double> block(static_cast<std::size_t>(n_bra) * width);
          for (std::size_t e = 0; e < block.size(); ++e) {
            block[e] = std::sqrt(2.0 + static_cast<double>(e) + 7.0 * static_cast<double>(b));
          }
          for (std::int32_t r = 0; r < n_bra; ++r) {
            for (std::int32_t i = 0; i < na; ++i) {
              for (std::int32_t j = 0; j < nb; ++j) {
                const auto m = static_cast<std::size_t>(off_a[r] + i);
                const auto q = static_cast<std::size_t>(shells_b[b] + j);
                const double v = block[static_cast<std::size_t>(r) * width + i * nb + j];
                want[m * n_ao + q] += v;
                if (mirror && off_a[r] != shells_b[b]) {
                  want[q * n_ao + m] += v;
                }
              }
            }
          }
          // Kept alive to the download: the scatters are asynchronous.
          blocks.push_back(upload(stream, block));
          ASSERT_EQ(fb::device::exchange_scatter_block(stream.get(), la, lb, n_ao, n_bra,
                                                       off_a_d.data(), shells_b[b], mirror,
                                                       blocks.back().data(), k.data()),
                    wwr::wwrSuccess);
        }
        const auto got = download(stream, k.data(), nn);
        for (std::size_t e = 0; e < nn; ++e) {
          ASSERT_LE(std::abs(got[e] - want[e]), 1e-15 * std::abs(want[e]))
              << "mirror " << mirror << ", (la, lb) = (" << la << ", " << lb << "), element " << e;
        }
      }
    }
  }
}

TEST(ExchangeHelpers, RefusesWhatTheyCannotLaunch) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  DeviceArray<double> d(4);
  DeviceArray<std::int32_t> off(1);
  constexpr std::int32_t kAboveD = fb::kMaxAngularMomentum + 1;
  EXPECT_EQ(fb::device::exchange_density_max(stream.get(), 0, d.data(), d.data()),
            wwr::wwrErrorInvalidValue);
  for (const auto &[lc, ld, n_ao, n_c, n_d] : {std::array{kAboveD, 0, 2, 1, 1},
                                               {0, -1, 2, 1, 1},
                                               {0, 0, 0, 1, 1},
                                               {0, 0, 2, 0, 1},
                                               {0, 0, 2, 1, 0}}) {
    EXPECT_EQ(fb::device::exchange_gather_dcd(stream.get(), lc, ld, n_ao, d.data(), n_c, off.data(),
                                              n_d, off.data(), d.data()),
              wwr::wwrErrorInvalidValue)
        << lc << " " << ld << " " << n_ao << " " << n_c << " " << n_d;
  }
  for (const auto &[la, lb, n_ao, n_bra, off_b] : {std::array{kAboveD, 0, 2, 1, 0},
                                                   {0, -1, 2, 1, 0},
                                                   {0, 0, 0, 1, 0},
                                                   {0, 0, 2, 0, 0},
                                                   {0, 0, 2, 1, -1}}) {
    EXPECT_EQ(fb::device::exchange_scatter_block(stream.get(), la, lb, n_ao, n_bra, off.data(),
                                                 off_b, false, d.data(), d.data()),
              wwr::wwrErrorInvalidValue)
        << la << " " << lb << " " << n_ao << " " << n_bra << " " << off_b;
  }
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream.get()), wwr::wwrSuccess);
}

} // namespace
