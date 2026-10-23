// GPU cases for integrals.eri.integral_direct.exchange.pairs: the ordered pair
// classes K's bras and kets are drawn from, on H2O in STO-3G (O's p shell
// precedes the H s shells, so (p, s) is reached only by swapping mmd's (s|p))
// and cc-pVDZ (s, p, d: all nine ordered classes), and the H10 chain. Each
// class's record count against a host count of ordered primitive pairs; its
// groups (group_start) contiguous, one first shell each with its AO offset, in
// basis order and non-increasing in the dressed record bound; each record's
// rank_b; the (la, lb) and (lb, la) records the same multiset with P - A and
// P - B swapped; every class bit for bit against a host gather of mmd's
// records; the K build's sizes, Dcd rows and a <= b block tables against the
// basis; the view naming the device fields; and each record's dressed bound
// against FelizJueves's
// ao.k.record_bounds (test/gtest/data/record_bound/,
// tools/emit_record_bound_golden.py), on H2O/cc-pVDZ and the chain, with its
// centres those of its two shells. The classes are device only: every check
// reads a copy taken back explicitly.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.integral_direct.exchange.pairs;

#include "mmd_records.h"

namespace {

namespace fb = free_breakfast;

using ClassKey = std::pair<std::int32_t, std::int32_t>;

// H2O in the basis the stem names, as coulomb_gpu.cpp reads it.
fb::BasisSet h2o(const std::string &stem) {
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
  return fb::BasisSet(std::move(shells));
}

// A point as three doubles, x, y, z.
std::array<double, 3> coords(const fb::Vec<double, 3> &v) {
  return {v.x, v.y, v.z};
}

// A device field of n elements, copied back to the host on the class's
// stream, which every write of it was queued on.
template <typename T>
std::vector<T> download(const fb::OrderedPairClass &cls, const fb::StreamBuffers::device<T> &field,
                        const std::size_t n) {
  std::vector<T> out(n);
  const wwr::wwrStream_t stream = cls.device_stream()->stream();
  EXPECT_EQ(field.num_elements(), n);
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), field.data(), sizeof(T) * n, wwr::wwrMemcpyDeviceToHost,
                                stream),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  return out;
}

// An ordered class's fields on the host, named as there.
struct Host {
  std::vector<double> zeta, p_x, p_y, p_z, pa_x, pa_y, pa_z, pb_x, pb_y, pb_z, s, record_bound;
  std::vector<std::int32_t> off_a, off_b, rank_b, group_start;
};

Host fetch(const fb::OrderedPairClass &cls) {
  const auto n = static_cast<std::size_t>(cls.n_rec);
  const auto f = [&cls, n](const auto &field) { return download(cls, field, n); };
  return {.zeta = f(cls.zeta),
          .p_x = f(cls.p_x),
          .p_y = f(cls.p_y),
          .p_z = f(cls.p_z),
          .pa_x = f(cls.pa_x),
          .pa_y = f(cls.pa_y),
          .pa_z = f(cls.pa_z),
          .pb_x = f(cls.pb_x),
          .pb_y = f(cls.pb_y),
          .pb_z = f(cls.pb_z),
          .s = f(cls.s),
          .record_bound = f(cls.record_bound),
          .off_a = f(cls.off_a),
          .off_b = f(cls.off_b),
          .rank_b = f(cls.rank_b),
          .group_start =
              download(cls, cls.group_start, static_cast<std::size_t>(cls.n_groups) + 1)};
}

// The basis's ordered classes, with the mmd classes they were drawn from (the
// ordered classes do not hold them, but the test keeps them alive anyway).
struct Ordered {
  std::vector<fb::PrimitivePairClass> mmd;
  std::vector<fb::OrderedPairClass> ordered;
};

Ordered ordered_of(fb::DeviceResources &res, const fb::BasisSet &basis) {
  auto mmd = fb::primitive_pair_classes(res, basis);
  EXPECT_TRUE(mmd);
  if (!mmd) {
    return {};
  }
  auto ordered = fb::ordered_pair_classes(res, basis, *mmd);
  EXPECT_TRUE(ordered);
  if (!ordered) {
    return {};
  }
  return {std::move(*mmd), std::move(*ordered)};
}

// The host's count of ordered primitive pairs per ordered class: every
// ordered shell pair (A, B), a shell against itself once.
std::map<ClassKey, std::int64_t> host_counts(const fb::BasisSet &basis) {
  std::map<ClassKey, std::int64_t> n;
  for (const fb::Shell &a : basis.shells()) {
    for (const fb::Shell &b : basis.shells()) {
      n[{a.angular_momentum(), b.angular_momentum()}] +=
          static_cast<std::int64_t>(a.primitives().size()) *
          static_cast<std::int64_t>(b.primitives().size());
    }
  }
  return n;
}

// The shell starting at each AO offset, by its momentum and host rank: its
// index among the basis's shells of the same momentum, in basis order.
struct ShellAt {
  std::int32_t l;
  std::int32_t rank;
};

std::map<std::int32_t, ShellAt> shells_by_offset(const fb::BasisSet &basis) {
  std::map<std::int32_t, std::int32_t> next;
  std::map<std::int32_t, ShellAt> out;
  for (std::uint32_t i = 0; i < basis.nshells(); ++i) {
    const std::int32_t l = basis.shells()[i].angular_momentum();
    out[static_cast<std::int32_t>(basis.ao_offset_cart(i))] = {l, next[l]++};
  }
  return out;
}

// A record as the multiset test compares it: zeta, P, P - A, P - B, s, and the
// two AO offsets, P - A and P - B (and the offsets) swapped when asked.
using Record = std::array<double, 13>;

std::vector<Record> records(const fb::OrderedPairClass &cls, const bool swap) {
  const Host h = fetch(cls);
  std::vector<Record> out;
  for (std::size_t k = 0; k < static_cast<std::size_t>(cls.n_rec); ++k) {
    const std::array<double, 3> pa{h.pa_x[k], h.pa_y[k], h.pa_z[k]};
    const std::array<double, 3> pb{h.pb_x[k], h.pb_y[k], h.pb_z[k]};
    const std::array<double, 3> &one = swap ? pb : pa;
    const std::array<double, 3> &two = swap ? pa : pb;
    const auto off_one = static_cast<double>(swap ? h.off_b[k] : h.off_a[k]);
    const auto off_two = static_cast<double>(swap ? h.off_a[k] : h.off_b[k]);
    out.push_back({h.zeta[k], h.p_x[k], h.p_y[k], h.p_z[k], one[0], one[1], one[2], two[0], two[1],
                   two[2], h.s[k], off_one, off_two});
  }
  std::ranges::sort(out);
  return out;
}

class ExchangePairs : public ::testing::TestWithParam<std::string> {};

TEST_P(ExchangePairs, RecordCountsAreTheOrderedPrimitivePairs) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = h2o(GetParam());
  const Ordered o = ordered_of(res, basis);
  const std::map<ClassKey, std::int64_t> want = host_counts(basis);
  ASSERT_EQ(o.ordered.size(), want.size());
  auto w = want.begin();
  for (const fb::OrderedPairClass &cls : o.ordered) {
    EXPECT_EQ(ClassKey(cls.la, cls.lb), w->first);
    EXPECT_EQ(cls.n_rec, w->second) << "(" << cls.la << ", " << cls.lb << ")";
    for (const fb::StreamBuffers::device<double> *field :
         {&cls.zeta, &cls.pb_z, &cls.record_bound}) {
      EXPECT_EQ(field->num_elements(), static_cast<std::size_t>(cls.n_rec));
    }
    EXPECT_EQ(cls.rank_b.num_elements(), static_cast<std::size_t>(cls.n_rec));
    ++w;
  }
}

TEST_P(ExchangePairs, GroupsAreContiguousOneFirstShellAndDescendingInBound) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = h2o(GetParam());
  const Ordered o = ordered_of(res, basis);
  const std::map<std::int32_t, ShellAt> by_offset = shells_by_offset(basis);
  const std::vector<fb::Shell> &shells = basis.shells();
  for (const fb::OrderedPairClass &cls : o.ordered) {
    const std::string name = std::format("({}, {})", cls.la, cls.lb);
    // One group per shell of momentum la, in basis order.
    std::vector<std::uint32_t> firsts;
    for (std::uint32_t sh = 0; sh < basis.nshells(); ++sh) {
      if (shells[sh].angular_momentum() == cls.la) {
        firsts.push_back(sh);
      }
    }
    ASSERT_FALSE(firsts.empty()) << name;
    ASSERT_EQ(cls.n_groups, static_cast<std::int32_t>(firsts.size())) << name;
    ASSERT_EQ(cls.group_start.num_elements(), firsts.size() + 1) << name;
    const Host h = fetch(cls);
    std::int32_t next = 0;
    for (std::size_t gi = 0; gi < firsts.size(); ++gi) {
      const fb::Shell &a = shells[firsts[gi]];
      // Contiguous and in basis order; the groups cover the class exactly.
      const std::int32_t start = h.group_start[gi];
      const std::int32_t count = h.group_start[gi + 1] - start;
      EXPECT_EQ(start, next) << name << " group " << gi;
      EXPECT_GT(count, 0) << name;
      next = start + count;
      // Every primitive pair of a against every lb shell.
      std::int64_t want = 0;
      for (const fb::Shell &b : shells) {
        if (b.angular_momentum() == cls.lb) {
          want += static_cast<std::int64_t>(a.primitives().size()) *
                  static_cast<std::int64_t>(b.primitives().size());
        }
      }
      EXPECT_EQ(count, want) << name << " shell " << firsts[gi];
      const auto a_off = static_cast<std::int32_t>(basis.ao_offset_cart(firsts[gi]));
      for (std::int32_t k = start; k < start + count; ++k) {
        const auto r = static_cast<std::size_t>(k);
        EXPECT_EQ(h.off_a[r], a_off) << name << " record " << k;
        // kb's sorted breaks need it; a far pair's bound may be exactly 0.
        if (k > start) {
          EXPECT_LE(h.record_bound[r], h.record_bound[r - 1]) << name << " record " << k;
        }
        EXPECT_GE(h.record_bound[r], 0.0) << name << " record " << k;
        // The second shell has momentum lb and its host rank.
        const auto b = by_offset.find(h.off_b[r]);
        ASSERT_NE(b, by_offset.end()) << name << " record " << k;
        EXPECT_EQ(b->second.l, cls.lb) << name << " record " << k;
        EXPECT_EQ(h.rank_b[r], b->second.rank) << name << " record " << k;
      }
    }
    EXPECT_EQ(next, cls.n_rec) << name;
    EXPECT_EQ(h.group_start[firsts.size()], cls.n_rec) << name;
  }
}

// What the K build reads of a class and the ordered classes make once per
// basis: the basis's AO count and shells per momentum; a bra's Dcd rows,
// rank_b times each momentum's shell count; and its a <= b block table against
// every ket class of the set, in set order, each against a host enumeration
// of the same blocks from the groups, bra-major. cc-pVDZ's shells of two
// momenta interleave by atom, so the a <= b cut falls inside the ket groups
// at a different place for each bra group.
TEST_P(ExchangePairs, BraTablesAreTheBasissABlocks) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = h2o(GetParam());
  const Ordered o = ordered_of(res, basis);
  ASSERT_FALSE(o.ordered.empty());
  std::vector<std::int32_t> n_shells(static_cast<std::size_t>(fb::kMaxAngularMomentum + 1));
  for (const fb::Shell &sh : basis.shells()) {
    const auto l = static_cast<std::size_t>(sh.angular_momentum());
    if (l >= n_shells.size()) {
      n_shells.resize(l + 1);
    }
    ++n_shells[l];
  }
  std::vector<Host> hosts;
  for (const fb::OrderedPairClass &cls : o.ordered) {
    hosts.push_back(fetch(cls));
  }
  // bra0, bra1, ket0, ket1, ko_a, ko_b: exchange_helpers_bridge.h's
  // kExchangeBlockFields, which this test cannot include beside its imports
  // (exchange_helpers_gpu.cpp's header says why).
  constexpr std::size_t kFields = 6;
  std::size_t n_cut = 0;
  for (std::size_t a = 0; a < o.ordered.size(); ++a) {
    const fb::OrderedPairClass &bra = o.ordered[a];
    const Host &hb = hosts[a];
    const std::string name = std::format("({}, {})", bra.la, bra.lb);
    EXPECT_EQ(bra.n_ao, static_cast<std::int32_t>(basis.nao_cart())) << name;
    EXPECT_EQ(bra.n_shells, n_shells) << name;
    ASSERT_EQ(bra.n_blk.size(), o.ordered.size()) << name;

    const auto n_rec = static_cast<std::size_t>(bra.n_rec);
    const auto dc = download(bra, bra.dc, n_shells.size() * n_rec);
    for (std::size_t ld = 0; ld < n_shells.size(); ++ld) {
      if (n_shells[ld] == 0) {
        continue;
      }
      for (std::size_t r = 0; r < n_rec; ++r) {
        ASSERT_EQ(dc[ld * n_rec + r], hb.rank_b[r] * n_shells[ld]) << name << " ld " << ld;
      }
    }

    std::vector<std::array<std::int32_t, kFields>> want;
    for (std::size_t b = 0; b < o.ordered.size(); ++b) {
      const fb::OrderedPairClass &ket = o.ordered[b];
      const Host &hk = hosts[b];
      std::size_t count = 0;
      for (std::int32_t ga = 0; ga < bra.n_groups; ++ga) {
        for (std::int32_t gb = 0; gb < ket.n_groups; ++gb) {
          const std::int32_t ko_a = hb.off_a[static_cast<std::size_t>(hb.group_start[ga])];
          const std::int32_t ko_b = hk.off_a[static_cast<std::size_t>(hk.group_start[gb])];
          if (ko_a <= ko_b) {
            want.push_back({hb.group_start[ga], hb.group_start[ga + 1], hk.group_start[gb],
                            hk.group_start[gb + 1], ko_a, ko_b});
            ++count;
          }
        }
      }
      EXPECT_EQ(static_cast<std::size_t>(bra.n_blk[b]), count)
          << name << " against (" << ket.la << ", " << ket.lb << ")";
      n_cut += static_cast<std::size_t>(bra.n_groups) * ket.n_groups - count;
    }
    if (want.empty()) {
      continue;
    }
    // The tables back to back, each kFields segments of its own block count.
    const auto table = download(bra, bra.blocks, kFields * want.size());
    std::size_t at = 0;
    std::size_t blk0 = 0;
    for (const std::int32_t count : bra.n_blk) {
      const auto n = static_cast<std::size_t>(count);
      for (std::size_t blk = 0; blk < n; ++blk) {
        for (std::size_t f = 0; f < kFields; ++f) {
          ASSERT_EQ(table[at + f * n + blk], want[blk0 + blk][f])
              << name << ", block " << blk0 + blk << ", field " << f;
        }
      }
      at += kFields * n;
      blk0 += n;
    }
  }
  // The a > b blocks are left out.
  EXPECT_GT(n_cut, 0U);
}

TEST_P(ExchangePairs, SwappedClassesHoldTheSameRecordsWithPaAndPbSwapped) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const Ordered o = ordered_of(res, h2o(GetParam()));
  std::map<ClassKey, const fb::OrderedPairClass *> by_key;
  for (const fb::OrderedPairClass &cls : o.ordered) {
    by_key[{cls.la, cls.lb}] = &cls;
  }
  for (const auto &[key, cls] : by_key) {
    const auto mirror = by_key.find({key.second, key.first});
    ASSERT_NE(mirror, by_key.end()) << "(" << key.first << ", " << key.second << ")";
    EXPECT_EQ(records(*cls, false), records(*mirror->second, true))
        << "(" << key.first << ", " << key.second << ")";
  }
}

// Every class bit for bit against a host gather of mmd's records, copied
// back from the device: each mmd record
// taken in record order to its first shell's group, as stored and, unless
// diagonal, swapped; each group stably sorted by the dressed bound,
// descending. The order, the bounds and every field the device's gather
// wrote.
TEST_P(ExchangePairs, MatchesAHostGatherOfMmd) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  Ordered o = ordered_of(res, h2o(GetParam()));
  ASSERT_FALSE(o.ordered.empty());
  // mmd's records exist on the device only: copy back every field read below.
  const std::vector<fb::test::MmdRecords> mmd = fb::test::copy_back(o.mmd);
  const auto pin = [](const auto &field, const std::size_t r) { return field[r]; };
  for (const fb::OrderedPairClass &cls : o.ordered) {
    const std::string name = std::format("({}, {})", cls.la, cls.lb);
    const auto m = std::ranges::find_if(mmd, [&cls](const fb::test::MmdRecords &c) {
      return c.la == std::min(cls.la, cls.lb) && c.lb == std::max(cls.la, cls.lb);
    });
    ASSERT_NE(m, mmd.end()) << name;
    // Each group's (record, swapped), by the first shell's rank.
    std::vector<std::vector<std::pair<std::size_t, bool>>> groups(
        static_cast<std::size_t>(cls.n_groups));
    for (std::size_t r = 0; r < static_cast<std::size_t>(m->n_rec); ++r) {
      if (cls.la == m->la) {
        groups.at(static_cast<std::size_t>(pin(m->rank_a, r))).emplace_back(r, false);
      }
      if (cls.la == m->lb && (m->la != m->lb || pin(m->off_a, r) != pin(m->off_b, r))) {
        groups.at(static_cast<std::size_t>(pin(m->rank_b, r))).emplace_back(r, true);
      }
    }
    std::vector<std::pair<std::size_t, bool>> want;
    for (auto &g : groups) {
      std::ranges::stable_sort(g, std::ranges::greater{}, [&](const auto &rec) {
        return pin(rec.second ? m->bound_ba : m->bound_ab, rec.first);
      });
      want.insert(want.end(), g.begin(), g.end());
    }
    ASSERT_EQ(want.size(), static_cast<std::size_t>(cls.n_rec)) << name;
    const Host h = fetch(cls);
    for (std::size_t k = 0; k < want.size(); ++k) {
      const auto [r, sw] = want[k];
      const auto &[pa_x, pb_x] = sw ? std::tie(m->pb_x, m->pa_x) : std::tie(m->pa_x, m->pb_x);
      const auto &[pa_y, pb_y] = sw ? std::tie(m->pb_y, m->pa_y) : std::tie(m->pa_y, m->pb_y);
      const auto &[pa_z, pb_z] = sw ? std::tie(m->pb_z, m->pa_z) : std::tie(m->pa_z, m->pb_z);
      const auto &[off_a, off_b] = sw ? std::tie(m->off_b, m->off_a) : std::tie(m->off_a, m->off_b);
      const std::array<double, 12> want_d{
          pin(m->zeta, r), pin(m->p_x, r), pin(m->p_y, r), pin(m->p_z, r),
          pin(pa_x, r),    pin(pa_y, r),   pin(pa_z, r),   pin(pb_x, r),
          pin(pb_y, r),    pin(pb_z, r),   pin(m->s, r),   pin(sw ? m->bound_ba : m->bound_ab, r)};
      const std::array<double, 12> got_d{h.zeta[k], h.p_x[k],  h.p_y[k],  h.p_z[k],
                                         h.pa_x[k], h.pa_y[k], h.pa_z[k], h.pb_x[k],
                                         h.pb_y[k], h.pb_z[k], h.s[k],    h.record_bound[k]};
      ASSERT_EQ(got_d, want_d) << name << " record " << k;
      const std::array<std::int32_t, 3> want_i{pin(off_a, r), pin(off_b, r),
                                               pin(sw ? m->rank_a : m->rank_b, r)};
      const std::array<std::int32_t, 3> got_i{h.off_a[k], h.off_b[k], h.rank_b[k]};
      ASSERT_EQ(got_i, want_i) << name << " record " << k;
    }
  }
}

TEST_P(ExchangePairs, ViewNamesTheDeviceFields) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const Ordered o = ordered_of(res, h2o(GetParam()));
  ASSERT_FALSE(o.ordered.empty());
  for (const fb::OrderedPairClass &cls : o.ordered) {
    const fb::OrderedPairClassView v = cls.view();
    EXPECT_EQ(v.la, cls.la);
    EXPECT_EQ(v.lb, cls.lb);
    EXPECT_EQ(v.n_rec, cls.n_rec);
    EXPECT_EQ(v.zeta, cls.zeta.data());
    EXPECT_EQ(v.p_x, cls.p_x.data());
    EXPECT_EQ(v.pa_y, cls.pa_y.data());
    EXPECT_EQ(v.pb_z, cls.pb_z.data());
    EXPECT_EQ(v.s, cls.s.data());
    EXPECT_EQ(v.record_bound, cls.record_bound.data());
    EXPECT_EQ(v.off_a, cls.off_a.data());
    EXPECT_EQ(v.off_b, cls.off_b.data());
    EXPECT_EQ(v.rank_b, cls.rank_b.data());
    EXPECT_EQ(v.n_groups, cls.n_groups);
    EXPECT_EQ(v.group_start, cls.group_start.data());
    EXPECT_EQ(cls.device_stream()->dev_idx(), res.dev_idx());
  }
}

// Every record's dressed bound against FelizJueves's ao.k.record_bounds of the
// same ordered shell pair, matched by its two shells and its zeta (the two
// sides order their records differently): to a few ulps, a host zero (a pair
// factor that underflowed) exactly zero. And every record's centres, P - (P -
// A) and P - (P - B), those of its two shells, so the sort that put the bound
// in order carried P, P - A, P - B and both offsets with it.
class ExchangeRecordBound : public ::testing::TestWithParam<std::string> {};

// FelizJueves holds the generated kernel to its host within 16 eps per record
// (test_kernel_record_bound.py); the same bar here. The kernel forms each I_p in
// closed form and sums plainly, the host by the [p|q] recurrence and fsum.
constexpr double kBoundUlps = 16.0;

TEST_P(ExchangeRecordBound, MatchesFelizJuevesRecordBounds) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = h2o(GetParam());
  const auto doc = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "record_bound" /
                                 (GetParam() + ".json"));
  ASSERT_TRUE(doc) << GetParam();
  ASSERT_EQ((*doc)["n_shells"].get<std::uint32_t>(), basis.nshells());
  // Per ordered shell pair, its records as (zeta, bound), sorted.
  using Recs = std::vector<std::pair<double, double>>;
  std::map<std::pair<std::int32_t, std::int32_t>, Recs> want;
  for (const auto &pair : (*doc)["pairs"]) {
    Recs &recs = want[{pair["a"].get<std::int32_t>(), pair["b"].get<std::int32_t>()}];
    for (const auto &rec : pair["records"]) {
      recs.emplace_back(rec[0].get<double>(), rec[1].get<double>());
    }
  }
  std::map<std::int32_t, std::int32_t> shell_at;
  for (std::uint32_t i = 0; i < basis.nshells(); ++i) {
    shell_at[static_cast<std::int32_t>(basis.ao_offset_cart(i))] = static_cast<std::int32_t>(i);
  }
  const Ordered o = ordered_of(res, basis);
  std::map<std::pair<std::int32_t, std::int32_t>, Recs> got;
  std::int64_t n_rec = 0;
  std::int64_t n_zero = 0;
  for (const fb::OrderedPairClass &cls : o.ordered) {
    const Host h = fetch(cls);
    for (std::size_t k = 0; k < static_cast<std::size_t>(cls.n_rec); ++k) {
      const std::int32_t a = shell_at.at(h.off_a[k]);
      const std::int32_t b = shell_at.at(h.off_b[k]);
      const auto ca = coords(basis.shells()[static_cast<std::size_t>(a)].center());
      const auto cb = coords(basis.shells()[static_cast<std::size_t>(b)].center());
      const std::array p{h.p_x[k], h.p_y[k], h.p_z[k]};
      const std::array pa{h.pa_x[k], h.pa_y[k], h.pa_z[k]};
      const std::array pb{h.pb_x[k], h.pb_y[k], h.pb_z[k]};
      for (std::size_t x = 0; x < 3; ++x) {
        ASSERT_NEAR(p[x] - pa[x], ca[x], 1e-12)
            << "record " << k << " of shells " << a << ", " << b;
        ASSERT_NEAR(p[x] - pb[x], cb[x], 1e-12)
            << "record " << k << " of shells " << a << ", " << b;
      }
      got[{a, b}].emplace_back(h.zeta[k], h.record_bound[k]);
      ++n_rec;
    }
  }
  ASSERT_EQ(n_rec, (*doc)["n_records"].get<std::int64_t>());
  ASSERT_EQ(got.size(), want.size());
  double worst = 0.0;
  for (auto &[key, recs] : want) {
    const auto it = got.find(key);
    ASSERT_NE(it, got.end()) << "shells " << key.first << ", " << key.second;
    Recs &mine = it->second;
    std::ranges::sort(recs);
    std::ranges::sort(mine);
    ASSERT_EQ(mine.size(), recs.size()) << "shells " << key.first << ", " << key.second;
    for (std::size_t r = 0; r < recs.size(); ++r) {
      const auto &[zeta, bound] = recs[r];
      ASSERT_EQ(mine[r].first, zeta) << "shells " << key.first << ", " << key.second;
      if (bound == 0.0) {
        EXPECT_EQ(mine[r].second, 0.0) << "shells " << key.first << ", " << key.second;
        ++n_zero;
        continue;
      }
      const double ulps =
          std::abs(mine[r].second - bound) / (std::numeric_limits<double>::epsilon() * bound);
      worst = std::max(worst, ulps);
      EXPECT_LE(ulps, kBoundUlps) << "shells " << key.first << ", " << key.second << ", zeta "
                                  << zeta;
    }
  }
  std::cout << "[ bound    ] " << GetParam() << ": " << n_rec << " records, worst " << worst
            << " eps, " << n_zero << " exact zeros\n";
  RecordProperty("worst_eps", std::format("{:.2f}", worst));
  RecordProperty("exact_zeros", std::format("{}", n_zero));
}

INSTANTIATE_TEST_SUITE_P(Golden, ExchangeRecordBound,
                         ::testing::Values("h2o_ccpvdz", "h10_chain_ccpvdz"));

// The ordered classes of one basis built from another basis's mmd classes are
// refused: a class the basis does not reach, or of a record count not the
// basis's, is an InvalidConfig error, not a garbage gather.
TEST(ExchangePairsMismatch, ClassesOfAnotherBasisAreRefused) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet big = h2o("h2o_ccpvdz");
  const fb::BasisSet small = h2o("h2o_sto3g");
  auto mmd = fb::primitive_pair_classes(res, big);
  ASSERT_TRUE(mmd);
  const auto ordered = fb::ordered_pair_classes(res, small, *mmd);
  ASSERT_FALSE(ordered);
  EXPECT_EQ(ordered.error().kind, fb::ErrorKind::InvalidConfig);
}

INSTANTIATE_TEST_SUITE_P(H2O, ExchangePairs, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));
// Five H2 4 bohr apart in cc-pVDZ, with a converged RHF density.
INSTANTIATE_TEST_SUITE_P(Chain, ExchangePairs, ::testing::Values("h10_chain_ccpvdz"));

} // namespace
