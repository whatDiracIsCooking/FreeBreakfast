// GPU cases for the rolled K kernel's E tables (exchange_e_table,
// src/integrals/eri/integral_direct/exchange/helpers): every record of every
// ordered shell pair of H2O/cc-pVDZ (s, p, d) and CO/cc-pVTZ (through f, every
// class to (f, f)), as a bra (unsigned) and a ket (odd |t| negated), against
// FelizJueves's host launch_kblock.e_table (test/gtest/data/e_table/,
// tools/emit_e_table_golden.py), and the launcher's refusals.
//
// The records are the golden's own zeta, P - A and P - B, uploaded as they
// are, so what is compared is the table's arithmetic and layout alone: the
// card's records (mmd, exchange/pairs) are held to FelizJueves's in
// exchange_pairs_gpu.cpp. The E table is an intermediate with no libcint
// counterpart; K's vk check covers it end to end.
//
// The launcher is reached through its bridge header, which brings in wwr's
// textual runtime surface, so this TU imports nothing that carries
// wwr.runtime_api (exchange_helpers_gpu.cpp's header says why) and manages its
// device memory and stream through the textual wwr* calls.

#include <gtest/gtest.h>

// After every standard header: under HIP a libc++ header must not follow the
// vendor runtime header this brings in (see primitives.h).
#include "exchange_helpers_bridge.h"
#include "gpu_requirement.h"

import std;
import free_breakfast.io.json;

namespace {

namespace fb = free_breakfast;

// A device array of n doubles, freed on destruction.
class DeviceDoubles {
public:
  explicit DeviceDoubles(const std::size_t n) {
    void *p = nullptr;
    EXPECT_EQ(wwr::wwrMalloc(&p, sizeof(double) * n), wwr::wwrSuccess);
    data_ = static_cast<double *>(p);
  }
  DeviceDoubles(const DeviceDoubles &) = delete;
  DeviceDoubles &operator=(const DeviceDoubles &) = delete;
  ~DeviceDoubles() { EXPECT_EQ(wwr::wwrFree(data_), wwr::wwrSuccess); }
  [[nodiscard]] double *data() const { return data_; }

private:
  double *data_ = nullptr;
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

void upload(const Stream &stream, const std::vector<double> &host, const DeviceDoubles &dev) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(dev.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, stream.get()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream.get()), wwr::wwrSuccess);
}

std::vector<double> download(const Stream &stream, const DeviceDoubles &dev, const std::size_t n) {
  std::vector<double> host(n);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), dev.data(), sizeof(double) * n,
                                wwr::wwrMemcpyDeviceToHost, stream.get()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream.get()), wwr::wwrSuccess);
  return host;
}

// One class's records as the launcher reads them, and the golden's tables.
struct ClassRecords {
  std::int32_t width = 0;
  std::vector<std::int32_t> odd;
  std::vector<double> zeta;
  std::array<std::vector<double>, 3> pa, pb;
  std::vector<double> e; // n_rec x width, record-major
};

using ClassKey = std::pair<std::int32_t, std::int32_t>;

// Every record's table against FelizJueves's: each entry within kETableEps
// eps of the record's largest |entry|; the ket's the bra's with the odd-|t|
// entries negated, bit for bit.
class ExchangeETable : public ::testing::TestWithParam<std::string> {};

// The bar, in eps of the record's largest |entry|. The kernel runs
// FelizJueves's recurrence operation for operation, but the device fuses a
// product into an add where the host rounds twice, and its exp is not the
// host's: H2O/cc-pVDZ's worst is 2.0 eps and CO/cc-pVTZ's (to (ff)) 2.2 on the
// RTX 3080. 16 is FelizJueves's per-record bar for its generated kernels
// against their host (record_bound's, exchange_pairs_gpu.cpp). Relative to the
// record, not the entry: the recurrence adds terms of either sign, so a small
// entry can be a cancellation, and kbr contracts every entry of a record
// against the same [p|q], so the record's scale is what an error is felt at.
constexpr double kETableEps = 16.0;

TEST_P(ExchangeETable, MatchesFelizJuevesETable) {
  FBF_REQUIRE_GPU();
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "e_table" / (GetParam() + ".json"));
  ASSERT_TRUE(doc) << GetParam();
  std::map<ClassKey, ClassRecords> classes;
  for (const auto &c : (*doc)["classes"]) {
    const ClassKey key{c["la"].get<std::int32_t>(), c["lb"].get<std::int32_t>()};
    ClassRecords &cls = classes[key];
    cls.width = c["width"].get<std::int32_t>();
    cls.odd = c["odd"].get<std::vector<std::int32_t>>();
    // The host's width is FelizJueves's, the one #348's buffers are sized by.
    ASSERT_EQ(fb::device::exchange_e_table_width(key.first, key.second), cls.width)
        << key.first << ", " << key.second;
    ASSERT_EQ(cls.odd.size(), static_cast<std::size_t>(cls.width));
  }
  std::int64_t n_rec = 0;
  for (const auto &pair : (*doc)["pairs"]) {
    ClassRecords &cls =
        classes.at({pair["la"].get<std::int32_t>(), pair["lb"].get<std::int32_t>()});
    const auto &tables = pair["e"];
    ASSERT_EQ(tables.size(), pair["records"].size());
    for (std::size_t r = 0; r < tables.size(); ++r) {
      const auto rec = pair["records"][r].get<std::vector<double>>();
      ASSERT_EQ(rec.size(), 7U);
      cls.zeta.push_back(rec[0]);
      for (std::size_t d = 0; d < 3; ++d) {
        cls.pa[d].push_back(rec[1 + d]);
        cls.pb[d].push_back(rec[4 + d]);
      }
      const auto e = tables[r].get<std::vector<double>>();
      ASSERT_EQ(e.size(), static_cast<std::size_t>(cls.width));
      cls.e.insert(cls.e.end(), e.begin(), e.end());
      ++n_rec;
    }
  }
  ASSERT_EQ(n_rec, (*doc)["n_records"].get<std::int64_t>());

  const Stream stream;
  double worst = 0.0;
  for (const auto &[key, cls] : classes) {
    const std::size_t n = cls.zeta.size();
    ASSERT_GT(n, 0U) << key.first << ", " << key.second;
    const std::size_t w = static_cast<std::size_t>(cls.width);
    const DeviceDoubles zeta(n);
    upload(stream, cls.zeta, zeta);
    std::vector<std::unique_ptr<DeviceDoubles>> axes;
    std::array<const double *, 3> pa{};
    std::array<const double *, 3> pb{};
    for (std::size_t d = 0; d < 3; ++d) {
      axes.push_back(std::make_unique<DeviceDoubles>(n));
      upload(stream, cls.pa[d], *axes.back());
      pa[d] = axes.back()->data();
      axes.push_back(std::make_unique<DeviceDoubles>(n));
      upload(stream, cls.pb[d], *axes.back());
      pb[d] = axes.back()->data();
    }
    const DeviceDoubles bra_e(n * w);
    const DeviceDoubles ket_e(n * w);
    ASSERT_EQ(fb::device::exchange_e_table(stream.get(), key.first, key.second,
                                           static_cast<std::int32_t>(n), zeta.data(), pa.data(),
                                           pb.data(), false, bra_e.data()),
              wwr::wwrSuccess);
    ASSERT_EQ(fb::device::exchange_e_table(stream.get(), key.first, key.second,
                                           static_cast<std::int32_t>(n), zeta.data(), pa.data(),
                                           pb.data(), true, ket_e.data()),
              wwr::wwrSuccess);
    const std::vector<double> bra = download(stream, bra_e, n * w);
    const std::vector<double> ket = download(stream, ket_e, n * w);
    for (std::size_t r = 0; r < n; ++r) {
      double scale = 0.0;
      for (std::size_t k = 0; k < w; ++k) {
        scale = std::max(scale, std::abs(cls.e[r * w + k]));
      }
      for (std::size_t k = 0; k < w; ++k) {
        const std::size_t i = r * w + k;
        // A record whose pair factor underflowed is all zeros, exactly.
        const double diff = std::abs(bra[i] - cls.e[i]);
        const double err = scale == 0.0 ? (diff == 0.0 ? 0.0 : kETableEps + 1.0)
                                        : diff / (std::numeric_limits<double>::epsilon() * scale);
        worst = std::max(worst, err);
        EXPECT_LE(err, kETableEps) << "class " << key.first << ", " << key.second << " record " << r
                                   << " entry " << k << ": " << bra[i] << " against " << cls.e[i];
        // The ket's is the bra's with the odd-|t| entries negated, exactly.
        EXPECT_EQ(std::bit_cast<std::uint64_t>(ket[i]),
                  std::bit_cast<std::uint64_t>(cls.odd[k] != 0 ? -bra[i] : bra[i]))
            << "class " << key.first << ", " << key.second << " record " << r << " entry " << k;
      }
    }
  }
  std::cout << "[ e_table  ] " << GetParam() << ": " << n_rec << " records, worst " << worst
            << " eps of the record's largest entry\n";
  RecordProperty("worst_eps", std::format("{:.2f}", worst));
}

// A momentum past f, a negative one, or no records is refused before any
// launch: nothing is written.
TEST(ExchangeETableRefusals, OutOfRangeIsInvalidValue) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  const DeviceDoubles one(1);
  const std::array<const double *, 3> v{one.data(), one.data(), one.data()};
  for (const auto [la, lb, n] :
       {std::array<std::int32_t, 3>{4, 0, 1}, {0, 4, 1}, {-1, 0, 1}, {0, -1, 1}, {1, 1, 0}}) {
    EXPECT_EQ(fb::device::exchange_e_table(stream.get(), la, lb, n, one.data(), v.data(), v.data(),
                                           false, one.data()),
              wwr::wwrErrorInvalidValue)
        << la << ", " << lb << ", " << n;
  }
  // The widths through f are FelizJueves's (the golden's classes check them
  // too, where a molecule reaches them); a negative momentum has none.
  EXPECT_EQ(fb::device::exchange_e_table_width(0, 0), 1);
  EXPECT_EQ(fb::device::exchange_e_table_width(1, 0), 6);
  EXPECT_EQ(fb::device::exchange_e_table_width(-1, 0), 0);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream.get()), wwr::wwrSuccess);
}

INSTANTIATE_TEST_SUITE_P(Golden, ExchangeETable, ::testing::Values("h2o_ccpvdz", "co_ccpvtz"));

} // namespace
