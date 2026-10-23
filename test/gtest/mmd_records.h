/// @file
/// @brief A PrimitivePairClass copied back to the host, for the tests that
/// read mmd's records: they exist on the device only. Include it after
/// <gtest/gtest.h>, `import std;` and `import free_breakfast.integrals.mmd;`.

#pragma once

namespace free_breakfast::test {

// One class's records, every field copied back from the device, in record
// order.
struct MmdRecords {
  std::int32_t la;
  std::int32_t lb;
  std::int32_t n_rec;
  std::vector<double> zeta;
  std::vector<double> p_x;
  std::vector<double> p_y;
  std::vector<double> p_z;
  std::vector<double> pa_x;
  std::vector<double> pa_y;
  std::vector<double> pa_z;
  std::vector<double> pb_x;
  std::vector<double> pb_y;
  std::vector<double> pb_z;
  std::vector<double> s;
  std::vector<double> ea;
  std::vector<double> eb;
  std::vector<double> bound;
  std::vector<double> bound_ab;
  std::vector<double> bound_ba;
  std::vector<std::int32_t> off_a;
  std::vector<std::int32_t> off_b;
  std::vector<std::int32_t> rank_a;
  std::vector<std::int32_t> rank_b;

  bool operator==(const MmdRecords &) const = default;
};

// cls's records, each field copied back on its stream, then the stream waited
// for.
inline MmdRecords copy_back(const PrimitivePairClass &cls) {
  const wwr::wwrStream_t stream = cls.device_stream()->stream();
  const auto n = static_cast<std::size_t>(cls.n_rec);
  const auto back = [&]<typename T>(const StreamBuffers::device<T> &field) {
    std::vector<T> out(n);
    EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), field.data(), sizeof(T) * n,
                                  wwr::wwrMemcpyDeviceToHost, stream),
              wwr::wwrSuccess);
    return out;
  };
  MmdRecords out{.la = cls.la,
                 .lb = cls.lb,
                 .n_rec = cls.n_rec,
                 .zeta = back(cls.zeta),
                 .p_x = back(cls.p_x),
                 .p_y = back(cls.p_y),
                 .p_z = back(cls.p_z),
                 .pa_x = back(cls.pa_x),
                 .pa_y = back(cls.pa_y),
                 .pa_z = back(cls.pa_z),
                 .pb_x = back(cls.pb_x),
                 .pb_y = back(cls.pb_y),
                 .pb_z = back(cls.pb_z),
                 .s = back(cls.s),
                 .ea = back(cls.ea),
                 .eb = back(cls.eb),
                 .bound = back(cls.bound),
                 .bound_ab = back(cls.bound_ab),
                 .bound_ba = back(cls.bound_ba),
                 .off_a = back(cls.off_a),
                 .off_b = back(cls.off_b),
                 .rank_a = back(cls.rank_a),
                 .rank_b = back(cls.rank_b)};
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  return out;
}

// Every class of classes, copied back.
inline std::vector<MmdRecords> copy_back(const std::vector<PrimitivePairClass> &classes) {
  std::vector<MmdRecords> out;
  for (const PrimitivePairClass &cls : classes) {
    out.push_back(copy_back(cls));
  }
  return out;
}

} // namespace free_breakfast::test
