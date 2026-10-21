/// @file
/// @brief What the three atomic_scf test files share: the PySCF reference
/// document (data/atomic_scf.json, written by
/// experimental/atomic_scf/emit_reference.py) and the shells read from it.
/// Include it after <gtest/gtest.h> and the imports of std,
/// free_breakfast.basis, free_breakfast.io.json and free_breakfast.io.basis.

#pragma once

namespace free_breakfast::test {

inline const Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

inline double max_abs(const std::vector<double> &m) {
  double out = 0.0;
  for (const double v : m) {
    out = std::max(out, std::abs(v));
  }
  return out;
}

// Read once per binary: every case that wants a basis or an energy shares it.
inline const json &reference() {
  static const json doc = [] {
    auto read = read_json(std::filesystem::path{FBF_EXPERIMENTAL_DATA_DIR} / "atomic_scf.json");
    if (!read) {
      ADD_FAILURE() << read.error().message;
      return json{};
    }
    return *read;
  }();
  return doc;
}

// One element's shells from the reference file's BSE basis, at the origin, the
// way a production run gets them: parse_basis, primitive_normalized, at.
inline std::vector<Shell> reference_shells(const std::string &basis, std::int32_t z) {
  auto stamp = parse_basis(reference()["bases"][basis].dump());
  EXPECT_TRUE(stamp) << basis;
  if (!stamp) {
    return {};
  }
  const BasisStamp normalized = stamp->primitive_normalized();
  const std::vector<ShellStamp> *element = normalized.shells_for(z);
  EXPECT_NE(element, nullptr) << basis << " Z=" << z;
  std::vector<Shell> shells;
  if (element != nullptr) {
    for (const ShellStamp &s : *element) {
      shells.push_back(s.at(kOrigin));
    }
  }
  return shells;
}

} // namespace free_breakfast::test
