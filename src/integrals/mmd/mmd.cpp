/// @file
/// @brief primitive_pair_classes: the shell-pair sweep, the la <= lb swap and
/// the per-record arithmetic.

module free_breakfast.integrals.mmd;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

namespace free_breakfast {

// A shell-pair class, (la, lb) with la <= lb.
using ClassKey = std::pair<int32_t, int32_t>;

// Shell pair (i, j) as (bra, ket) shell indices, swapped when i has the higher
// angular momentum so the pair lands in its la <= lb class.
std::pair<uint32_t, uint32_t> bra_ket_order(const std::vector<Shell> &shells, const uint32_t i,
                                            const uint32_t j) {
  if (shells[i].angular_momentum() > shells[j].angular_momentum()) {
    return {j, i};
  }
  return {i, j};
}

// Size every per-record field of cls for its n_rec records.
void resize_records(PrimitivePairClass &cls) {
  const auto n = static_cast<size_t>(cls.n_rec);
  cls.zeta.resize(n);
  for (std::vector<double> *axis : {&cls.p_x, &cls.p_y, &cls.p_z, &cls.pa_x, &cls.pa_y, &cls.pa_z,
                                    &cls.pb_x, &cls.pb_y, &cls.pb_z}) {
    axis->resize(n);
  }
  cls.s.resize(n);
  cls.eb.resize(n);
  cls.off_a.resize(n);
  cls.off_b.resize(n);
}

// Write the records of shell pair (a, b), already in la <= lb order, into cls
// from record index rec; return the index one past the last record written.
int32_t write_shell_pair_records(PrimitivePairClass &cls, const int32_t rec, const Shell &a,
                                 const int32_t off_a, const Shell &b, const int32_t off_b) {
  const Vec<double, 3> &ca = a.center();
  const Vec<double, 3> &cb = b.center();
  auto r = static_cast<size_t>(rec);
  for (const Primitive &pa : a.primitives()) {
    for (const Primitive &pb : b.primitives()) {
      const double zeta = pa.exponent + pb.exponent;
      const double px = ((pa.exponent * ca.x) + (pb.exponent * cb.x)) / zeta;
      const double py = ((pa.exponent * ca.y) + (pb.exponent * cb.y)) / zeta;
      const double pz = ((pa.exponent * ca.z) + (pb.exponent * cb.z)) / zeta;
      cls.zeta[r] = zeta;
      cls.p_x[r] = px;
      cls.p_y[r] = py;
      cls.p_z[r] = pz;
      cls.pa_x[r] = px - ca.x;
      cls.pa_y[r] = py - ca.y;
      cls.pa_z[r] = pz - ca.z;
      cls.pb_x[r] = px - cb.x;
      cls.pb_y[r] = py - cb.y;
      cls.pb_z[r] = pz - cb.z;
      cls.s[r] = pa.coefficient * pb.coefficient * std::pow(kPi<double> / zeta, 1.5);
      cls.eb[r] = pb.exponent;
      cls.off_a[r] = off_a;
      cls.off_b[r] = off_b;
      ++r;
    }
  }
  return static_cast<int32_t>(r);
}

std::vector<PrimitivePairClass> primitive_pair_classes(const BasisSet &basis) {
  // BasisSet guarantees nao_cart() fits an int32_t, so the offset casts below are exact.
  const std::vector<Shell> &shells = basis.shells();
  const uint32_t nsh = basis.nshells();

  // Pass 1: every reached class's record count, in int64_t so a class past
  // INT32_MAX is caught before it is sized. Keyed by (la, lb), so the result
  // comes out in ascending class order.
  std::map<ClassKey, int64_t> counts;
  for (uint32_t i = 0; i < nsh; ++i) {
    for (uint32_t j = i; j < nsh; ++j) {
      const auto [bra, ket] = bra_ket_order(shells, i, j);
      counts[{shells[bra].angular_momentum(), shells[ket].angular_momentum()}] +=
          static_cast<int64_t>(shells[bra].primitives().size()) *
          static_cast<int64_t>(shells[ket].primitives().size());
    }
  }

  // Size each class once, at its final record count.
  std::vector<PrimitivePairClass> out;
  out.reserve(counts.size());
  std::map<ClassKey, size_t> slot;
  for (const auto &[key, n] : counts) {
    if (n > std::numeric_limits<int32_t>::max()) {
      check_or_abort(
          Status{err_config("primitive_pair_classes: class ({}, {}) has {} records, "
                            "past {}",
                            key.first, key.second, n, std::numeric_limits<int32_t>::max())});
    }
    slot.emplace(key, out.size());
    PrimitivePairClass &cls = out.emplace_back();
    cls.la = key.first;
    cls.lb = key.second;
    cls.n_rec = static_cast<int32_t>(n);
    resize_records(cls);
  }

  // Pass 2: the same sweep, each shell pair's records written at its class's
  // cursor, so records keep their shell-pair, then bra-primitive-major, order.
  std::vector<int32_t> cursor(out.size(), 0);
  for (uint32_t i = 0; i < nsh; ++i) {
    for (uint32_t j = i; j < nsh; ++j) {
      const auto [bra, ket] = bra_ket_order(shells, i, j);
      const size_t c = slot.at({shells[bra].angular_momentum(), shells[ket].angular_momentum()});
      cursor[c] = write_shell_pair_records(
          out[c], cursor[c], shells[bra], static_cast<int32_t>(basis.ao_offset_cart(bra)),
          shells[ket], static_cast<int32_t>(basis.ao_offset_cart(ket)));
    }
  }
  return out;
}

} // namespace free_breakfast
