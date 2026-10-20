/// @file
/// @brief primitive_pair_classes and write_primitive_pairs: the shell-pair
/// sweep, the la <= lb swap and the per-record arithmetic.

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

// One class's caller-owned records, addressed by field and record index.
struct RecordWriter {
  int32_t n_rec;
  PrimitivePairRecords records;

  double &operator()(const PairReal f, const size_t r) const {
    return records.reals[field_offset(f, n_rec) + r];
  }
  int32_t &operator()(const PairOffset f, const size_t r) const {
    return records.offsets[field_offset(f, n_rec) + r];
  }
};

// Write the records of shell pair (a, b), already in la <= lb order, into w
// from record index rec; return the index one past the last record written.
int32_t write_shell_pair_records(const RecordWriter &w, const int32_t rec, const Shell &a,
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
      w(PairReal::Zeta, r) = zeta;
      w(PairReal::PX, r) = px;
      w(PairReal::PY, r) = py;
      w(PairReal::PZ, r) = pz;
      w(PairReal::PaX, r) = px - ca.x;
      w(PairReal::PaY, r) = py - ca.y;
      w(PairReal::PaZ, r) = pz - ca.z;
      w(PairReal::PbX, r) = px - cb.x;
      w(PairReal::PbY, r) = py - cb.y;
      w(PairReal::PbZ, r) = pz - cb.z;
      w(PairReal::S, r) = pa.coefficient * pb.coefficient * std::pow(kPi<double> / zeta, 1.5);
      w(PairReal::Eb, r) = pb.exponent;
      w(PairOffset::A, r) = off_a;
      w(PairOffset::B, r) = off_b;
      ++r;
    }
  }
  return static_cast<int32_t>(r);
}

std::vector<PrimitivePairClass> primitive_pair_classes(const BasisSet &basis) {
  const std::vector<Shell> &shells = basis.shells();
  const uint32_t nsh = basis.nshells();

  // Every reached class's record count, in int64_t so a class past INT32_MAX
  // is caught before anything is sized for it. Keyed by (la, lb), so the
  // result comes out in ascending class order.
  std::map<ClassKey, int64_t> counts;
  for (uint32_t i = 0; i < nsh; ++i) {
    for (uint32_t j = i; j < nsh; ++j) {
      const auto [bra, ket] = bra_ket_order(shells, i, j);
      counts[{shells[bra].angular_momentum(), shells[ket].angular_momentum()}] +=
          static_cast<int64_t>(shells[bra].primitives().size()) *
          static_cast<int64_t>(shells[ket].primitives().size());
    }
  }

  std::vector<PrimitivePairClass> out;
  out.reserve(counts.size());
  for (const auto &[key, n] : counts) {
    if (n > std::numeric_limits<int32_t>::max()) {
      check_or_abort(
          Status{err_config("primitive_pair_classes: class ({}, {}) has {} records, "
                            "past {}",
                            key.first, key.second, n, std::numeric_limits<int32_t>::max())});
    }
    out.push_back({.la = key.first, .lb = key.second, .n_rec = static_cast<int32_t>(n)});
  }
  return out;
}

void write_primitive_pairs(const BasisSet &basis, const std::span<const PrimitivePairClass> classes,
                           const std::span<const PrimitivePairRecords> records) {
  // The sweep below writes by index into caller-sized spans, so a stale class
  // list or a short span would write out of bounds: refuse both up front.
  const std::vector<PrimitivePairClass> want = primitive_pair_classes(basis);
  const bool same_classes = std::ranges::equal(
      classes, want, [](const PrimitivePairClass &a, const PrimitivePairClass &b) {
        return a.la == b.la && a.lb == b.lb && a.n_rec == b.n_rec;
      });
  if (!same_classes || records.size() != classes.size()) {
    check_or_abort(Status{
        err_config("write_primitive_pairs: {} classes and {} record sets given, for a basis of {} "
                   "classes, or the classes are not this basis's",
                   classes.size(), records.size(), want.size())});
  }
  std::map<ClassKey, size_t> slot;
  std::vector<RecordWriter> writers;
  writers.reserve(classes.size());
  for (size_t c = 0; c < classes.size(); ++c) {
    const auto n = static_cast<size_t>(classes[c].n_rec);
    if (records[c].reals.size() != kPairReals * n ||
        records[c].offsets.size() != kPairOffsets * n) {
      check_or_abort(Status{err_config(
          "write_primitive_pairs: class ({}, {}) of {} records given {} reals and {} offsets",
          classes[c].la, classes[c].lb, n, records[c].reals.size(), records[c].offsets.size())});
    }
    slot.emplace(ClassKey{classes[c].la, classes[c].lb}, c);
    writers.push_back({.n_rec = classes[c].n_rec, .records = records[c]});
  }

  // BasisSet guarantees nao_cart() fits an int32_t, so the offset casts below
  // are exact. Each shell pair's records go at its class's cursor, so records
  // keep their shell-pair, then bra-primitive-major, order.
  const std::vector<Shell> &shells = basis.shells();
  const uint32_t nsh = basis.nshells();
  std::vector<int32_t> cursor(classes.size(), 0);
  for (uint32_t i = 0; i < nsh; ++i) {
    for (uint32_t j = i; j < nsh; ++j) {
      const auto [bra, ket] = bra_ket_order(shells, i, j);
      const size_t c = slot.at({shells[bra].angular_momentum(), shells[ket].angular_momentum()});
      cursor[c] = write_shell_pair_records(
          writers[c], cursor[c], shells[bra], static_cast<int32_t>(basis.ao_offset_cart(bra)),
          shells[ket], static_cast<int32_t>(basis.ao_offset_cart(ket)));
    }
  }
}

} // namespace free_breakfast
