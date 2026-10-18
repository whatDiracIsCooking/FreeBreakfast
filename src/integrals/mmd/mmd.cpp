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

// Append the records of shell pair (a, b), already in la <= lb order, to cls.
void append_shell_pair_records(PrimitivePairClass &cls, const Shell &a, const int32_t off_a,
                               const Shell &b, const int32_t off_b) {
  const Vec<double, 3> &ca = a.center();
  const Vec<double, 3> &cb = b.center();
  const int64_t n_new =
      static_cast<int64_t>(a.primitives().size()) * static_cast<int64_t>(b.primitives().size());
  if (n_new > std::numeric_limits<int32_t>::max() - cls.n_rec) {
    check_or_abort(Status{err_config("primitive_pair_classes: class ({}, {}) exceeds {} records",
                                     cls.la, cls.lb, std::numeric_limits<int32_t>::max())});
  }
  cls.n_rec += static_cast<int32_t>(n_new);
  for (const Primitive &pa : a.primitives()) {
    for (const Primitive &pb : b.primitives()) {
      const double zeta = pa.exponent + pb.exponent;
      const double px = ((pa.exponent * ca.x) + (pb.exponent * cb.x)) / zeta;
      const double py = ((pa.exponent * ca.y) + (pb.exponent * cb.y)) / zeta;
      const double pz = ((pa.exponent * ca.z) + (pb.exponent * cb.z)) / zeta;
      cls.zeta.push_back(zeta);
      cls.p.insert(cls.p.end(), {px, py, pz});
      cls.xpa.insert(cls.xpa.end(), {px - ca.x, py - ca.y, pz - ca.z});
      cls.xpb.insert(cls.xpb.end(), {px - cb.x, py - cb.y, pz - cb.z});
      cls.s.push_back(pa.coefficient * pb.coefficient * std::pow(kPi<double> / zeta, 1.5));
      cls.eb.push_back(pb.exponent);
      cls.off_a.push_back(off_a);
      cls.off_b.push_back(off_b);
    }
  }
}

std::vector<PrimitivePairClass> primitive_pair_classes(const BasisSet &basis) {
  // BasisSet guarantees nao_cart() fits an int32_t, so the offset casts below are exact.
  // Keyed by (la, lb), so the result comes out in ascending class order.
  std::map<std::pair<int32_t, int32_t>, PrimitivePairClass> classes;
  const std::vector<Shell> &shells = basis.shells();
  for (uint32_t i = 0; i < basis.nshells(); ++i) {
    for (uint32_t j = i; j < basis.nshells(); ++j) {
      uint32_t bra = i;
      uint32_t ket = j;
      if (shells[i].angular_momentum() > shells[j].angular_momentum()) {
        std::swap(bra, ket);
      }
      const int32_t la = shells[bra].angular_momentum();
      const int32_t lb = shells[ket].angular_momentum();
      PrimitivePairClass &cls = classes[{la, lb}];
      cls.la = la;
      cls.lb = lb;
      append_shell_pair_records(cls, shells[bra], static_cast<int32_t>(basis.ao_offset_cart(bra)),
                                shells[ket], static_cast<int32_t>(basis.ao_offset_cart(ket)));
    }
  }
  std::vector<PrimitivePairClass> out;
  out.reserve(classes.size());
  for (auto &[key, cls] : classes) {
    out.push_back(std::move(cls));
  }
  return out;
}

} // namespace free_breakfast
