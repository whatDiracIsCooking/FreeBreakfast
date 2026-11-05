/// @file
/// @brief examples/nevpt2.toml's water (C2v, cc-pVDZ / cc-pVDZ-JKFIT,
/// CASSCF(4,4)) as a df_*_nevpt2 test start, and its symmetry-adapted
/// coordinates, for the SC and PC Hessians' symmetric-water cases (#952).
///
/// Include it AFTER `import std;`, `import free_breakfast.core;` and
/// `import free_breakfast.io.json;`, like scratch_dir.h: it includes nothing
/// itself.

#pragma once

namespace free_breakfast::test {

/// @brief The water's atoms in bohr: O on the C2 axis z, the molecule in
/// the yz plane, H1 at +y and H2 at -y.
inline std::array<std::array<double, 3>, 3> symmetric_water_atoms() {
  constexpr std::array<std::array<double, 3>, 3> kAngstrom{
      {{0.0, 0.0, 0.1173}, {0.0, 0.7572, -0.4692}, {0.0, -0.7572, -0.4692}}};
  std::array<std::array<double, 3>, 3> bohr{};
  for (std::size_t a = 0; a < 3; ++a) {
    for (std::size_t x = 0; x < 3; ++x) {
      bohr[a][x] = kAngstrom[a][x] / kAngstromPerBohr<double>;
    }
  }
  return bohr;
}

/// @brief The test start: the active space the CASSCF step picks there
/// (orbitals 3 to 6).
inline json symmetric_water() {
  json atoms = json::array();
  const auto bohr = symmetric_water_atoms();
  for (std::size_t a = 0; a < 3; ++a) {
    atoms.push_back(
        {{"symbol", a == 0 ? "O" : "H"}, {"coord", {bohr[a][0], bohr[a][1], bohr[a][2]}}});
  }
  return {{"name", "h2o_c2v_ccpvdz"},
          {"atoms", atoms},
          {"charge", 0},
          {"basis", {{"name", "cc-pVDZ"}}},
          {"aux_basis", {{"name", "cc-pVDZ-JKFIT"}}},
          {"ncas", 4},
          {"nelecas", 4},
          {"active", {3, 4, 5, 6}}};
}

/// @brief Nine orthonormal symmetry-adapted coordinates, each a Cartesian
/// displacement (atom-major x, y, z), and their labels: per irrep the rigid
/// translation and rotations (about the unweighted centroid) first, then
/// the vibrations, Gram-Schmidt in that order. 0 Tz, 1-2 A1 vibrations,
/// 3 Ty, 4 Rx, 5 the B2 vibration, 6 Tx, 7 Ry, 8 Rz.
struct WaterCoordinates {
  std::vector<std::array<double, 9>> rows;
  std::vector<std::string> labels;
};

inline WaterCoordinates symmetric_water_coordinates() {
  using Row = std::array<double, 9>;
  const auto at = symmetric_water_atoms();
  std::array<double, 3> centroid{};
  for (const auto &a : at) {
    for (std::size_t x = 0; x < 3; ++x) {
      centroid[x] += a[x] / 3.0;
    }
  }
  const auto unit = [](std::size_t atom, std::size_t axis, double w = 1.0) {
    Row r{};
    r[(3 * atom) + axis] = w;
    return r;
  };
  const auto sum = [](const Row &a, const Row &b) {
    Row r{};
    for (std::size_t i = 0; i < 9; ++i) {
      r[i] = a[i] + b[i];
    }
    return r;
  };
  const auto translation = [](std::size_t axis) {
    Row r{};
    for (std::size_t a = 0; a < 3; ++a) {
      r[(3 * a) + axis] = 1.0;
    }
    return r;
  };
  const auto rotation = [&](std::size_t axis) {
    Row r{};
    for (std::size_t a = 0; a < 3; ++a) {
      std::array<double, 3> d{};
      for (std::size_t x = 0; x < 3; ++x) {
        d[x] = at[a][x] - centroid[x];
      }
      std::array<double, 3> w{};
      w[axis] = 1.0;
      r[(3 * a) + 0] = (w[1] * d[2]) - (w[2] * d[1]);
      r[(3 * a) + 1] = (w[2] * d[0]) - (w[0] * d[2]);
      r[(3 * a) + 2] = (w[0] * d[1]) - (w[1] * d[0]);
    }
    return r;
  };
  WaterCoordinates out;
  const auto block = [&](std::string_view irrep, std::size_t external,
                         const std::vector<Row> &spanning) {
    std::size_t kept = 0;
    for (Row r : spanning) {
      for (const Row &b : out.rows) {
        double d = 0.0;
        for (std::size_t i = 0; i < 9; ++i) {
          d += r[i] * b[i];
        }
        for (std::size_t i = 0; i < 9; ++i) {
          r[i] -= d * b[i];
        }
      }
      double n = 0.0;
      for (const double x : r) {
        n += x * x;
      }
      if (std::sqrt(n) < 1e-8) {
        continue;
      }
      for (double &x : r) {
        x /= std::sqrt(n);
      }
      out.rows.push_back(r);
      out.labels.push_back(std::format("{} {}", irrep, kept < external ? "external" : "vibration"));
      ++kept;
    }
  };
  block("A1", 1,
        {translation(2), unit(0, 2), sum(unit(1, 2), unit(2, 2)), sum(unit(1, 1), unit(2, 1, -1))});
  block("B2", 2,
        {translation(1), rotation(0), unit(0, 1), sum(unit(1, 1), unit(2, 1)),
         sum(unit(1, 2), unit(2, 2, -1))});
  block("B1", 2, {translation(0), rotation(1), unit(0, 0), sum(unit(1, 0), unit(2, 0))});
  block("A2", 1, {rotation(2), sum(unit(1, 0), unit(2, 0, -1))});
  return out;
}

/// @brief The rows @p keep of symmetric_water_coordinates, row-major.
inline std::vector<double> symmetric_water_rows(std::span<const std::size_t> keep) {
  const WaterCoordinates c = symmetric_water_coordinates();
  std::vector<double> q;
  for (const std::size_t i : keep) {
    q.insert(q.end(), c.rows[i].begin(), c.rows[i].end());
  }
  return q;
}

} // namespace free_breakfast::test
