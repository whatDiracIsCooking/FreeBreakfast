// GPU cases for the exchange build's hand-written device steps
// (src/integrals/exchange/helpers): max|D|, the Dcd gather, the block table,
// and the shell starts and mirror, each against a host loop. Each moves,
// compares or counts and does no arithmetic on a double, so each is checked
// exactly, on H2O/cc-pVDZ's recorded density and shells (s, p and d,
// interleaved by atom, so a shell's rank among its momentum differs from its
// index, and the shells of two momenta interleave in AO order).
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

std::vector<std::int32_t> download_ints(const Stream &stream, const std::int32_t *device,
                                        std::size_t n) {
  std::vector<std::int32_t> host(n);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(std::int32_t) * n,
                                wwr::wwrMemcpyDeviceToHost, stream.get()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream.get()), wwr::wwrSuccess);
  return host;
}

// One side of a kb launch, as a class's groups are: one group per shell, in
// basis order, of 1 + (shell's rank % 3) records, each record holding its
// first shell's AO offset.
struct Side {
  std::vector<std::int32_t> start; // groups + 1
  std::vector<std::int32_t> off;   // per record
};

Side side_of(const std::vector<std::int32_t> &shell_offsets) {
  Side out;
  for (std::size_t r = 0; r < shell_offsets.size(); ++r) {
    out.start.push_back(static_cast<std::int32_t>(out.off.size()));
    out.off.insert(out.off.end(), 1 + r % 3, shell_offsets[r]);
  }
  out.start.push_back(static_cast<std::int32_t>(out.off.size()));
  return out;
}

// Every (la, lb), symmetric and not: the table against a host enumeration of
// the same blocks, bra-major, a <= b by AO offset when symmetric. cc-pVDZ's
// shells of two momenta interleave by atom, so the symmetric cut falls inside
// the ket groups at a different place for each bra group.
TEST(ExchangeHelpers, BuildsTheBlockTable) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  const Golden g = read_golden("h2o_ccpvdz");
  ASSERT_FALSE(g.density.empty());
  const auto by_l = offsets_by_momentum(g.basis);
  constexpr auto kFields = static_cast<std::size_t>(fb::device::kExchangeBlockFields);
  for (const bool symmetric : {false, true}) {
    for (std::int32_t la = 0; la < kMomenta; ++la) {
      for (std::int32_t lb = 0; lb < kMomenta; ++lb) {
        const Side a = side_of(by_l.at(static_cast<std::size_t>(la)));
        const Side b = side_of(by_l.at(static_cast<std::size_t>(lb)));
        const auto n_a = static_cast<std::int32_t>(a.start.size() - 1);
        const auto n_b = static_cast<std::int32_t>(b.start.size() - 1);
        std::vector<std::array<std::int32_t, kFields>> want;
        for (std::int32_t ga = 0; ga < n_a; ++ga) {
          for (std::int32_t gb = 0; gb < n_b; ++gb) {
            const std::int32_t ko_a = a.off[static_cast<std::size_t>(a.start[ga])];
            const std::int32_t ko_b = b.off[static_cast<std::size_t>(b.start[gb])];
            if (!symmetric || ko_a <= ko_b) {
              want.push_back(
                  {a.start[ga], a.start[ga + 1], b.start[gb], b.start[gb + 1], ko_a, ko_b});
            }
          }
        }
        const std::string name =
            std::format("symmetric {}, (la, lb) = ({}, {})", symmetric, la, lb);
        // Off the diagonal of momenta some of each, so the cut is exercised.
        if (symmetric && la != lb) {
          ASSERT_GT(want.size(), 0U) << name;
          ASSERT_LT(want.size(), static_cast<std::size_t>(n_a) * n_b) << name;
        }
        if (want.empty()) {
          continue;
        }
        const auto n_blk = static_cast<std::int32_t>(want.size());
        const auto a_start = upload(stream, a.start);
        const auto a_off = upload(stream, a.off);
        const auto b_start = upload(stream, b.start);
        const auto b_off = upload(stream, b.off);
        DeviceArray<std::int32_t> table(kFields * want.size());
        ASSERT_EQ(fb::device::exchange_blocks(stream.get(), n_a, a_start.data(), a_off.data(), n_b,
                                              b_start.data(), b_off.data(), symmetric, n_blk,
                                              table.data()),
                  wwr::wwrSuccess)
            << name;
        const auto got = download_ints(stream, table.data(), kFields * want.size());
        for (std::size_t blk = 0; blk < want.size(); ++blk) {
          for (std::size_t f = 0; f < kFields; ++f) {
            ASSERT_EQ(got[f * want.size() + blk], want[blk][f])
                << name << ", block " << blk << ", field " << f;
          }
        }
      }
    }
  }
}

// The shell starts of every momentum, against the basis's; then the mirror of
// a K with a distinct value in every element: the lower shell blocks are the
// upper ones' transposes, and everything else, the diagonal blocks whole among
// it, is untouched.
TEST(ExchangeHelpers, MirrorsTheLowerShellBlocks) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  const Golden g = read_golden("h2o_ccpvdz");
  ASSERT_FALSE(g.density.empty());
  const auto n_ao = static_cast<std::int32_t>(g.basis.nao_cart());
  const auto nn = static_cast<std::size_t>(n_ao) * n_ao;
  const auto by_l = offsets_by_momentum(g.basis);
  std::vector<std::int32_t> want_start(static_cast<std::size_t>(n_ao));
  for (std::uint32_t sh = 0; sh < g.basis.nshells(); ++sh) {
    const auto o = static_cast<std::int32_t>(g.basis.ao_offset_cart(sh));
    for (std::int32_t i = 0; i < cartesians(g.basis.shells()[sh].angular_momentum()); ++i) {
      want_start[static_cast<std::size_t>(o + i)] = o;
    }
  }
  DeviceArray<std::int32_t> start(static_cast<std::size_t>(n_ao));
  std::vector<DeviceArray<std::int32_t>> offs;
  for (std::int32_t l = 0; l < kMomenta; ++l) {
    const auto &off = by_l.at(static_cast<std::size_t>(l));
    offs.push_back(upload(stream, off));
    ASSERT_EQ(fb::device::exchange_shell_starts(stream.get(), l,
                                                static_cast<std::int32_t>(off.size()),
                                                offs.back().data(), start.data()),
              wwr::wwrSuccess);
  }
  EXPECT_EQ(download_ints(stream, start.data(), want_start.size()), want_start);

  std::vector<double> k(nn);
  for (std::size_t e = 0; e < nn; ++e) {
    k[e] = 1.0 + static_cast<double>(e);
  }
  DeviceArray<double> k_d = upload(stream, k);
  ASSERT_EQ(fb::device::exchange_mirror(stream.get(), n_ao, start.data(), k_d.data()),
            wwr::wwrSuccess);
  const auto got = download(stream, k_d.data(), nn);
  const auto &aos = g.basis.aos_cart();
  for (std::size_t m = 0; m < static_cast<std::size_t>(n_ao); ++m) {
    for (std::size_t q = 0; q < static_cast<std::size_t>(n_ao); ++q) {
      const bool lower = aos[q].shell_index < aos[m].shell_index;
      const double want = lower ? k[q * n_ao + m] : k[m * n_ao + q];
      ASSERT_EQ(bits(got[m * n_ao + q]), bits(want)) << "K(" << m << ", " << q << ")";
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
  for (const auto &[n_a, n_b, n_blk] : {std::array{0, 1, 1},
                                        {1, 0, 1},
                                        {1, 1, 0},
                                        {1, 1, std::numeric_limits<std::int32_t>::max() / 2}}) {
    EXPECT_EQ(fb::device::exchange_blocks(stream.get(), n_a, off.data(), off.data(), n_b,
                                          off.data(), off.data(), true, n_blk, off.data()),
              wwr::wwrErrorInvalidValue)
        << n_a << " " << n_b << " " << n_blk;
  }
  for (const auto &[l, n_s] : {std::array{kAboveD, 1}, {-1, 1}, {0, 0}}) {
    EXPECT_EQ(fb::device::exchange_shell_starts(stream.get(), l, n_s, off.data(), off.data()),
              wwr::wwrErrorInvalidValue)
        << l << " " << n_s;
  }
  EXPECT_EQ(fb::device::exchange_mirror(stream.get(), 0, off.data(), d.data()),
            wwr::wwrErrorInvalidValue);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream.get()), wwr::wwrSuccess);
}

} // namespace
