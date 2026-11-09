// Every allocation spelling devtools/alloc-sites.py inventories, and the
// look-alikes it must not. test_alloc_sites.py pins the exact rows; it reads
// this file and is never compiled.
module free_breakfast.fixture;

import free_breakfast.core;

namespace free_breakfast::fixture {

// A function returning a buffer is not a site; its body's allocation is.
Buffers::device<double> make_buffer(DeviceResources &res, const size_t count) {
  return Buffers::device<double>(count, res.device());
}

struct Holder {
  Holder(DeviceResources &res, size_t n)
      : n_{n}, values_(n * 2, res.device()), staging_{n, res.device()} {}
  size_t n_;
  Buffers::device<double> values_;
  StreamBuffers::pinned<int32_t> staging_;
};

Status every_spelling(DeviceResources &res, const Buffers::device<double> &in, size_t n, size_t m) {
  Buffers::device<double> named(n * m, res.device());
  Buffers::pinned<int32_t> braced{n + 1};
  Holder held{res, n};
  auto copy = Buffers::host<float>(static_cast<size_t>(m));
  std::vector<Buffers::device<double>> slabs;
  for (size_t i = 0; i < m; ++i) {
    slabs.emplace_back(n * i, res.device());
  }
  std::optional<Buffers::device<uint64_t>> maybe;
  maybe.emplace(m, res.device());
  auto shared = std::make_shared<StreamBuffers::device<double>>(n, res.device_stream());
  std::byte *base = res.device_workspace(n * sizeof(double)).data();
  const auto [work, block] = res.device_regions<double, int32_t>(
      static_cast<size_t>(n) * static_cast<size_t>(m) * sizeof(double), static_cast<size_t>(m));
  auto counted = res.counted_pinned<double>("every_spelling", n * m);
  // Buffers::device<double> commented(n, res.device());
  /* Buffers::device<double> blocked(n, res.device()); */
  const char *prose = "Buffers::device<double> quoted(n, res.device());";
  const Buffers::device<double> &alias = in;
  std::vector<double> plain;
  plain.emplace_back(1.0);
  return {};
}

} // namespace free_breakfast::fixture
