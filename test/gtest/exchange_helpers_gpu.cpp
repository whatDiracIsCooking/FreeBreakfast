// GPU cases for the exchange build's hand-written device steps
// (src/integrals/eri/integral_direct/exchange/helpers): max|D|, the Dcd
// gather, the block table and its density factors, the shell starts and mirror, and the density decay
// the factors are cut from (the bound maximum, the AO centres, the group
// geometry and the decay fit), each against a host loop, on H2O/cc-pVDZ's
// recorded density and shells (s, p and d, interleaved by atom, so a shell's
// rank among its momentum differs from its index, and the shells of two
// momenta interleave in AO order) and the H10 chain's RHF density. What moves,
// compares or counts is checked exactly; what takes a square root or an
// exponential, to a few ulps.
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

// Relative agreement to a few ulps, for what takes a square root or an
// exponential: the device may contract a product into an fma and its exp may
// differ from the host's in the last bit.
constexpr double kUlps = 16 * std::numeric_limits<double>::epsilon();

// A side's group geometry (kExchangeGroupFields segments): the groups on the z
// axis, `spacing` apart from `z0`, every fifth one reaching nothing (-1), the
// rest 0.5, 0.75 or 1.0 bohr.
std::vector<double> geometry_of(const std::size_t n_groups, const double z0, const double spacing) {
  std::vector<double> geom(static_cast<std::size_t>(fb::device::kExchangeGroupFields) * n_groups);
  for (std::size_t g = 0; g < n_groups; ++g) {
    geom[2 * n_groups + g] = z0 + spacing * static_cast<double>(g);
    geom[3 * n_groups + g] = g % 5 == 4 ? -1.0 : 0.5 + 0.25 * static_cast<double>(g % 3);
  }
  return geom;
}

// F_ab as exchange_blocks states it, on the host.
double host_factor(const std::vector<double> &a, const std::size_t ga, const std::vector<double> &b,
                   const std::size_t gb, const std::array<double, 3> &decay) {
  const std::size_t n_a = a.size() / 4;
  const std::size_t n_b = b.size() / 4;
  const double ra = a[3 * n_a + ga];
  const double rb = b[3 * n_b + gb];
  if (ra < 0.0 || rb < 0.0) {
    return decay[0];
  }
  const double dx = a[ga] - b[gb];
  const double dy = a[n_a + ga] - b[n_b + gb];
  const double dz = a[2 * n_a + ga] - b[2 * n_b + gb];
  const double d = std::sqrt(dx * dx + dy * dy + dz * dz) - ra - rb;
  if (!(d > 2.0 * fb::device::kExchangeSameCentre)) {
    return decay[0];
  }
  return std::min(decay[0],
                  decay[2] * std::exp(-decay[1] * d) * (1.0 + fb::device::kExchangeDecayMargin));
}

// Every (la, lb), symmetric and not: the table against a host enumeration of
// the same blocks, bra-major, a <= b by AO offset when symmetric. cc-pVDZ's
// shells of two momenta interleave by atom, so the symmetric cut falls inside
// the ket groups at a different place for each bra group. Each block's factor
// against the rule, on synthetic geometry that takes every branch: a group
// that reaches nothing, a pair too close to bound, and pairs far enough apart
// that the decay is under max|D|.
TEST(ExchangeHelpers, BuildsTheBlockTable) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  const Golden g = read_golden("h2o_ccpvdz");
  ASSERT_FALSE(g.density.empty());
  const auto by_l = offsets_by_momentum(g.basis);
  constexpr auto kFields = static_cast<std::size_t>(fb::device::kExchangeBlockFields);
  const std::array decay{0.7, 0.4, 0.9};
  const auto decay_d = upload(stream, std::vector<double>(decay.begin(), decay.end()));
  std::size_t n_decayed = 0;
  std::size_t n_whole = 0;
  for (const bool symmetric : {false, true}) {
    for (std::int32_t la = 0; la < kMomenta; ++la) {
      for (std::int32_t lb = 0; lb < kMomenta; ++lb) {
        const Side a = side_of(by_l.at(static_cast<std::size_t>(la)));
        const Side b = side_of(by_l.at(static_cast<std::size_t>(lb)));
        const auto n_a = static_cast<std::int32_t>(a.start.size() - 1);
        const auto n_b = static_cast<std::int32_t>(b.start.size() - 1);
        const std::vector<double> a_geom = geometry_of(static_cast<std::size_t>(n_a), 0.0, 3.0);
        const std::vector<double> b_geom = geometry_of(static_cast<std::size_t>(n_b), 1.0, 2.0);
        std::vector<std::array<std::int32_t, kFields>> want;
        std::vector<double> want_factor;
        for (std::int32_t ga = 0; ga < n_a; ++ga) {
          for (std::int32_t gb = 0; gb < n_b; ++gb) {
            const std::int32_t ko_a = a.off[static_cast<std::size_t>(a.start[ga])];
            const std::int32_t ko_b = b.off[static_cast<std::size_t>(b.start[gb])];
            if (!symmetric || ko_a <= ko_b) {
              want.push_back(
                  {a.start[ga], a.start[ga + 1], b.start[gb], b.start[gb + 1], ko_a, ko_b});
              want_factor.push_back(host_factor(a_geom, static_cast<std::size_t>(ga), b_geom,
                                                static_cast<std::size_t>(gb), decay));
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
        const auto a_geom_d = upload(stream, a_geom);
        const auto b_geom_d = upload(stream, b_geom);
        DeviceArray<std::int32_t> table(kFields * want.size());
        DeviceArray<double> factor(want.size());
        ASSERT_EQ(fb::device::exchange_blocks(stream.get(), n_a, a_start.data(), a_off.data(),
                                              a_geom_d.data(), n_b, b_start.data(), b_off.data(),
                                              b_geom_d.data(), decay_d.data(), symmetric, n_blk,
                                              table.data(), factor.data()),
                  wwr::wwrSuccess)
            << name;
        const auto got = download_ints(stream, table.data(), kFields * want.size());
        const auto got_factor = download(stream, factor.data(), want.size());
        for (std::size_t blk = 0; blk < want.size(); ++blk) {
          for (std::size_t f = 0; f < kFields; ++f) {
            ASSERT_EQ(got[f * want.size() + blk], want[blk][f])
                << name << ", block " << blk << ", field " << f;
          }
          if (want_factor[blk] == decay[0]) {
            // max|D| is a copy, never computed.
            ASSERT_EQ(bits(got_factor[blk]), bits(decay[0])) << name << ", block " << blk;
            ++n_whole;
          } else {
            ASSERT_NEAR(got_factor[blk], want_factor[blk], kUlps * want_factor[blk])
                << name << ", block " << blk;
            ++n_decayed;
          }
        }
      }
    }
  }
  // Both branches taken.
  EXPECT_GT(n_whole, 0U);
  EXPECT_GT(n_decayed, 0U);
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

// The largest bound, folded from two launches into one g_max as two classes
// would be: exactly the host's maximum, whichever launch holds it.
TEST(ExchangeHelpers, BoundMaxIsTheHostMaximum) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  std::vector<double> first(1000);
  std::vector<double> second(37);
  for (std::size_t r = 0; r < first.size(); ++r) {
    first[r] = 1.0 / static_cast<double>(1 + (r * 7) % 13);
  }
  for (std::size_t r = 0; r < second.size(); ++r) {
    second[r] = 0.25 * static_cast<double>(r);
  }
  const auto first_d = upload(stream, first);
  const auto second_d = upload(stream, second);
  DeviceArray<double> g_max(1);
  ASSERT_EQ(wwr::wwrMemsetAsync(g_max.data(), 0, sizeof(double), stream.get()), wwr::wwrSuccess);
  ASSERT_EQ(fb::device::exchange_bound_max(stream.get(), static_cast<std::int32_t>(first.size()),
                                           first_d.data(), g_max.data()),
            wwr::wwrSuccess);
  EXPECT_EQ(bits(download(stream, g_max.data(), 1)[0]), bits(host_max_abs(first)));
  ASSERT_EQ(fb::device::exchange_bound_max(stream.get(), static_cast<std::int32_t>(second.size()),
                                           second_d.data(), g_max.data()),
            wwr::wwrSuccess);
  EXPECT_EQ(bits(download(stream, g_max.data(), 1)[0]), bits(host_max_abs(second)));
}

// A synthetic class (la, lc) over H2O/cc-pVDZ's shells, as ordered_pair_classes
// lays one out: a group per shell of la in basis order, each with two records
// per shell of lc, P a little off A, bounds descending within the group. The
// last group's bounds are scaled down so that none passes a cutoff.
struct Records {
  std::vector<std::int32_t> start;       // groups + 1
  std::vector<std::int32_t> off_a;       // per record
  std::vector<double> bound;             // per record
  std::array<std::vector<double>, 3> p;  // per record, x y z
  std::array<std::vector<double>, 3> pa; // P - A
  std::array<std::vector<double>, 3> pb; // P - C
};

// A centre's coordinates, indexable.
std::array<double, 3> coords(const fb::Vec<double, 3> &v) {
  return {v.x, v.y, v.z};
}

Records records_of(const fb::BasisSet &basis, const std::int32_t la, const std::int32_t lc) {
  Records out;
  std::vector<std::uint32_t> as;
  std::vector<std::uint32_t> cs;
  for (std::uint32_t s = 0; s < basis.nshells(); ++s) {
    const std::int32_t l = basis.shells()[s].angular_momentum();
    if (l == la) {
      as.push_back(s);
    }
    if (l == lc) {
      cs.push_back(s);
    }
  }
  for (std::size_t ga = 0; ga < as.size(); ++ga) {
    out.start.push_back(static_cast<std::int32_t>(out.bound.size()));
    const auto a = coords(basis.shells()[as[ga]].center());
    std::size_t i = 0;
    for (const std::uint32_t c_idx : cs) {
      const auto c = coords(basis.shells()[c_idx].center());
      for (std::int32_t k = 0; k < 2; ++k, ++i) {
        const std::array shift{0.01 * static_cast<double>(k + 1), -0.02,
                               0.03 * static_cast<double>(i % 4)};
        out.off_a.push_back(static_cast<std::int32_t>(basis.ao_offset_cart(as[ga])));
        const double scale = ga + 1 == as.size() ? 1e-6 : 1.0;
        out.bound.push_back(scale / static_cast<double>(1 + i));
        for (std::size_t x = 0; x < 3; ++x) {
          const double px = a[x] + shift[x];
          out.p[x].push_back(px);
          out.pa[x].push_back(px - a[x]);
          out.pb[x].push_back(px - c[x]);
        }
      }
    }
  }
  out.start.push_back(static_cast<std::int32_t>(out.bound.size()));
  return out;
}

// Every (la, lc): each group's centre is its first record's P - (P - A),
// exactly, and its reach the largest |AC| over the records over the cutoff,
// -1 for the group none passes; and for every (l, l), every AO's centre.
TEST(ExchangeHelpers, GroupGeometryAndAoCentres) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  const Golden g = read_golden("h2o_ccpvdz");
  ASSERT_FALSE(g.density.empty());
  const auto n_ao = static_cast<std::int32_t>(g.basis.nao_cart());
  // max|D| 0.5 and g_max 1 (the first record's): a record passes at bound >= 0.1.
  constexpr double kTau = 0.05;
  const std::vector<double> decay{0.5, 0.0, 0.0};
  const auto decay_d = upload(stream, decay);
  DeviceArray<double> centre(3 * static_cast<std::size_t>(n_ao));
  std::vector<double> want_centre(3 * static_cast<std::size_t>(n_ao));
  for (std::int32_t la = 0; la < kMomenta; ++la) {
    for (std::int32_t lc = 0; lc < kMomenta; ++lc) {
      const Records rec = records_of(g.basis, la, lc);
      const auto n_groups = static_cast<std::int32_t>(rec.start.size() - 1);
      const auto n_rec = static_cast<std::int32_t>(rec.bound.size());
      const auto start = upload(stream, rec.start);
      const auto off_a = upload(stream, rec.off_a);
      const auto bound = upload(stream, rec.bound);
      const std::array p_d{upload(stream, rec.p[0]), upload(stream, rec.p[1]),
                           upload(stream, rec.p[2])};
      const std::array pa_d{upload(stream, rec.pa[0]), upload(stream, rec.pa[1]),
                            upload(stream, rec.pa[2])};
      const std::array pb_d{upload(stream, rec.pb[0]), upload(stream, rec.pb[1]),
                            upload(stream, rec.pb[2])};
      const std::array p{p_d[0].data(), p_d[1].data(), p_d[2].data()};
      const std::array pa{pa_d[0].data(), pa_d[1].data(), pa_d[2].data()};
      const std::array pb{pb_d[0].data(), pb_d[1].data(), pb_d[2].data()};
      DeviceArray<double> g_max(1);
      ASSERT_EQ(wwr::wwrMemsetAsync(g_max.data(), 0, sizeof(double), stream.get()),
                wwr::wwrSuccess);
      ASSERT_EQ(fb::device::exchange_bound_max(stream.get(), n_rec, bound.data(), g_max.data()),
                wwr::wwrSuccess);
      DeviceArray<double> geom(static_cast<std::size_t>(fb::device::kExchangeGroupFields) *
                               static_cast<std::size_t>(n_groups));
      ASSERT_EQ(fb::device::exchange_group_geometry(
                    stream.get(), n_groups, start.data(), bound.data(), p.data(), pa.data(),
                    pb.data(), kTau, g_max.data(), decay_d.data(), geom.data()),
                wwr::wwrSuccess);
      const auto got = download(stream, geom.data(), 4 * static_cast<std::size_t>(n_groups));
      const std::string name = std::format("(la, lc) = ({}, {})", la, lc);
      const double scale = rec.bound[0] * decay[0];
      for (std::int32_t ga = 0; ga < n_groups; ++ga) {
        const auto r0 = static_cast<std::size_t>(rec.start[ga]);
        double reach = -1.0;
        for (auto r = r0; r < static_cast<std::size_t>(rec.start[ga + 1]); ++r) {
          if (rec.bound[r] * scale >= kTau * (1.0 - fb::device::kExchangeDecayMargin)) {
            const double dx = rec.pb[0][r] - rec.pa[0][r];
            const double dy = rec.pb[1][r] - rec.pa[1][r];
            const double dz = rec.pb[2][r] - rec.pa[2][r];
            reach = std::max(reach, std::sqrt(dx * dx + dy * dy + dz * dz));
          }
        }
        const auto at = static_cast<std::size_t>(ga);
        const auto n = static_cast<std::size_t>(n_groups);
        for (std::size_t x = 0; x < 3; ++x) {
          ASSERT_EQ(bits(got[x * n + at]), bits(rec.p[x][r0] - rec.pa[x][r0]))
              << name << ", group " << ga << ", axis " << x;
        }
        if (ga + 1 == n_groups) {
          ASSERT_EQ(reach, -1.0) << name << ": the last group passes nothing";
        }
        if (reach < 0.0) {
          ASSERT_EQ(got[3 * n + at], -1.0) << name << ", group " << ga;
        } else {
          ASSERT_NEAR(got[3 * n + at], reach, kUlps * reach) << name << ", group " << ga;
        }
      }
      if (la == lc) {
        ASSERT_EQ(fb::device::exchange_ao_centres(stream.get(), n_groups, start.data(),
                                                  cartesians(la), off_a.data(), p.data(), pa.data(),
                                                  n_ao, centre.data()),
                  wwr::wwrSuccess);
        for (std::int32_t ga = 0; ga < n_groups; ++ga) {
          const auto r0 = static_cast<std::size_t>(rec.start[ga]);
          for (std::int32_t i = 0; i < cartesians(la); ++i) {
            const auto m = static_cast<std::size_t>(rec.off_a[r0] + i);
            for (std::size_t x = 0; x < 3; ++x) {
              want_centre[x * static_cast<std::size_t>(n_ao) + m] = rec.p[x][r0] - rec.pa[x][r0];
            }
          }
        }
      }
    }
  }
  const auto got_centre = download(stream, centre.data(), want_centre.size());
  for (std::size_t e = 0; e < want_centre.size(); ++e) {
    ASSERT_EQ(bits(got_centre[e]), bits(want_centre[e])) << "centre element " << e;
  }
}

// Λ and H_off of a recorded density, against the same fit on the host from the
// basis's own centres: Λ to the host's logarithms, H_off to its exponentials;
// and the bound they make holds, with no tolerance, at every off-centre
// element. On the H10 chain's RHF density, which decays, Λ is positive.
class ExchangeDecay : public ::testing::TestWithParam<const char *> {};

TEST_P(ExchangeDecay, FitsAndBoundsTheDensity) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.density.empty());
  const auto n_ao = static_cast<std::int32_t>(g.basis.nao_cart());
  const auto n = static_cast<std::size_t>(n_ao);
  std::vector<double> centre(3 * n);
  for (std::uint32_t s = 0; s < g.basis.nshells(); ++s) {
    const auto &shell = g.basis.shells()[s];
    for (std::int32_t i = 0; i < cartesians(shell.angular_momentum()); ++i) {
      for (std::size_t x = 0; x < 3; ++x) {
        centre[x * n + g.basis.ao_offset_cart(s) + static_cast<std::size_t>(i)] =
            coords(shell.center())[x];
      }
    }
  }
  const auto distance = [&](const std::size_t m, const std::size_t q) {
    const double dx = centre[m] - centre[q];
    const double dy = centre[n + m] - centre[n + q];
    const double dz = centre[2 * n + m] - centre[2 * n + q];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };
  const auto density_d = upload(stream, g.density);
  const auto centre_d = upload(stream, centre);
  DeviceArray<double> decay(fb::device::kExchangeDecayFields);
  ASSERT_EQ(fb::device::exchange_density_decay(stream.get(), n_ao, density_d.data(),
                                               centre_d.data(), decay.data()),
            wwr::wwrSuccess);
  const auto got = download(stream, decay.data(), fb::device::kExchangeDecayFields);

  // The fit, as the kernel states it.
  constexpr std::size_t kBins = 1024;
  std::vector<double> top(kBins, 0.0);
  for (std::size_t m = 0; m < n; ++m) {
    for (std::size_t q = 0; q < n; ++q) {
      const double r = distance(m, q);
      if (r >= fb::device::kExchangeSameCentre) {
        const std::size_t k = std::min(static_cast<std::size_t>(r), kBins - 1);
        top[k] = std::max(top[k], std::abs(g.density[m * n + q]));
      }
    }
  }
  std::vector<std::pair<double, double>> xy;
  for (std::size_t k = 0; k < kBins; ++k) {
    if (top[k] > 0.0) {
      xy.emplace_back(static_cast<double>(k) + 0.5, std::log(top[k]));
    }
  }
  double lambda = 0.0;
  if (xy.size() >= 2) {
    double mx = 0.0;
    double my = 0.0;
    for (const auto &[x, y] : xy) {
      mx += x;
      my += y;
    }
    mx /= static_cast<double>(xy.size());
    my /= static_cast<double>(xy.size());
    double sxx = 0.0;
    double sxy = 0.0;
    for (const auto &[x, y] : xy) {
      sxx += (x - mx) * (x - mx);
      sxy += (x - mx) * (y - my);
    }
    lambda = std::max(0.0, -sxy / sxx);
  }
  EXPECT_NEAR(got[1], lambda, 1e-12 * std::max(lambda, 1.0)) << GetParam();
  double h_off = 0.0;
  for (std::size_t m = 0; m < n; ++m) {
    for (std::size_t q = 0; q < n; ++q) {
      const double r = distance(m, q);
      if (r >= fb::device::kExchangeSameCentre) {
        h_off = std::max(h_off, std::abs(g.density[m * n + q]) * std::exp(got[1] * r));
      }
    }
  }
  EXPECT_NEAR(got[2], h_off, kUlps * h_off) << GetParam();

  // The bound, at every off-centre element.
  for (std::size_t m = 0; m < n; ++m) {
    for (std::size_t q = 0; q < n; ++q) {
      const double r = distance(m, q);
      if (r >= fb::device::kExchangeSameCentre) {
        ASSERT_LE(std::abs(g.density[m * n + q]),
                  got[2] * std::exp(-got[1] * r) * (1.0 + fb::device::kExchangeDecayMargin))
            << GetParam() << ", (" << m << ", " << q << ") at " << r << " bohr";
      }
    }
  }
  std::cout << "[ decay    ] " << GetParam() << ": max|D| " << host_max_abs(g.density)
            << ", lambda " << got[1] << " / bohr, H_off " << got[2] << "\n";
  if (std::string_view{GetParam()}.starts_with("h10")) {
    EXPECT_GT(got[1], 0.0) << "the chain's density decays";
  }
}

INSTANTIATE_TEST_SUITE_P(Densities, ExchangeDecay,
                         ::testing::Values("h2o_ccpvdz", "h10_chain_ccpvdz"));

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
    EXPECT_EQ(fb::device::exchange_blocks(stream.get(), n_a, off.data(), off.data(), d.data(), n_b,
                                          off.data(), off.data(), d.data(), d.data(), true, n_blk,
                                          off.data(), d.data()),
              wwr::wwrErrorInvalidValue)
        << n_a << " " << n_b << " " << n_blk;
  }
  const std::array<const double *, 3> v{d.data(), d.data(), d.data()};
  EXPECT_EQ(fb::device::exchange_bound_max(stream.get(), 0, d.data(), d.data()),
            wwr::wwrErrorInvalidValue);
  for (const auto &[n_groups, nx, n_ao] : {std::array{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}) {
    EXPECT_EQ(fb::device::exchange_ao_centres(stream.get(), n_groups, off.data(), nx, off.data(),
                                              v.data(), v.data(), n_ao, d.data()),
              wwr::wwrErrorInvalidValue)
        << n_groups << " " << nx << " " << n_ao;
  }
  EXPECT_EQ(fb::device::exchange_group_geometry(stream.get(), 0, off.data(), d.data(), v.data(),
                                                v.data(), v.data(), 0.0, d.data(), d.data(),
                                                d.data()),
            wwr::wwrErrorInvalidValue);
  EXPECT_EQ(fb::device::exchange_density_decay(stream.get(), 0, d.data(), d.data(), d.data()),
            wwr::wwrErrorInvalidValue);
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
