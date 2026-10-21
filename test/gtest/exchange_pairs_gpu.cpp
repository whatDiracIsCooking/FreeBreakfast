// GPU cases for integrals.exchange.pairs: the ordered pair classes K's bras
// and kets are drawn from, on H2O in STO-3G (O's p shell precedes the H s
// shells, so (p, s) is reached only by swapping mmd's (s|p)) and cc-pVDZ (s,
// p, d: all nine ordered classes). Each class's record count against a host
// count of ordered primitive pairs; its groups contiguous, one first shell
// each, in basis order and non-increasing in bound, and their bounds on the
// device side as group_start; each record's rank_b; the
// (la, lb) and (lb, la) records the same multiset with P - A and P - B
// swapped; and the device side matching the pinned side.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.exchange.pairs;

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

// Record k of a field's pinned side.
template <typename T>
T at(const fb::MirroredBuffer<T> &field, const std::size_t k) {
  return field.pinned().data()[k];
}

// A field's pinned side.
template <typename T>
std::vector<T> pinned(const fb::MirroredBuffer<T> &field) {
  const T *p = field.pinned().data();
  return {p, p + field.size()};
}

// A field's device side, copied back to the host on the field's stream.
template <typename T>
std::vector<T> download(const fb::MirroredBuffer<T> &field) {
  std::vector<T> out(field.size());
  const wwr::wwrStream_t stream = field.handle()->stream();
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), field.device().data(), sizeof(T) * out.size(),
                                wwr::wwrMemcpyDeviceToHost, stream),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  return out;
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
  std::vector<Record> out;
  for (std::size_t k = 0; k < static_cast<std::size_t>(cls.n_rec); ++k) {
    const std::array<double, 3> pa{at(cls.pa_x, k), at(cls.pa_y, k), at(cls.pa_z, k)};
    const std::array<double, 3> pb{at(cls.pb_x, k), at(cls.pb_y, k), at(cls.pb_z, k)};
    const std::array<double, 3> &one = swap ? pb : pa;
    const std::array<double, 3> &two = swap ? pa : pb;
    const auto off_one = static_cast<double>(swap ? at(cls.off_b, k) : at(cls.off_a, k));
    const auto off_two = static_cast<double>(swap ? at(cls.off_a, k) : at(cls.off_b, k));
    out.push_back({at(cls.zeta, k), at(cls.p_x, k), at(cls.p_y, k), at(cls.p_z, k), one[0], one[1],
                   one[2], two[0], two[1], two[2], at(cls.s, k), off_one, off_two});
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
    for (const fb::MirroredBuffer<double> *field : {&cls.zeta, &cls.pb_z, &cls.bound}) {
      EXPECT_EQ(field->size(), static_cast<std::size_t>(cls.n_rec));
    }
    EXPECT_EQ(cls.rank_b.size(), static_cast<std::size_t>(cls.n_rec));
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
    ASSERT_FALSE(cls.groups.empty()) << name;
    ASSERT_EQ(cls.group_start.size(), cls.groups.size() + 1) << name;
    std::int32_t next = 0;
    std::int32_t prev_shell = -1;
    for (std::size_t gi = 0; gi < cls.groups.size(); ++gi) {
      const fb::OrderedPairGroup &g = cls.groups[gi];
      // Contiguous and in basis order; the groups cover the class exactly.
      EXPECT_EQ(g.start, next) << name;
      EXPECT_EQ(at(cls.group_start, gi), g.start) << name << " group " << gi;
      EXPECT_GT(g.count, 0) << name;
      EXPECT_GT(g.shell, prev_shell) << name;
      next = g.start + g.count;
      prev_shell = g.shell;
      ASSERT_GE(g.shell, 0) << name;
      ASSERT_LT(static_cast<std::uint32_t>(g.shell), basis.nshells()) << name;
      const fb::Shell &a = shells[static_cast<std::size_t>(g.shell)];
      EXPECT_EQ(a.angular_momentum(), cls.la) << name;
      // Every primitive pair of a against every lb shell.
      std::int64_t want = 0;
      for (const fb::Shell &b : shells) {
        if (b.angular_momentum() == cls.lb) {
          want += static_cast<std::int64_t>(a.primitives().size()) *
                  static_cast<std::int64_t>(b.primitives().size());
        }
      }
      EXPECT_EQ(g.count, want) << name << " shell " << g.shell;
      const auto a_off = static_cast<std::int32_t>(basis.ao_offset_cart(g.shell));
      for (std::int32_t k = g.start; k < g.start + g.count; ++k) {
        const auto r = static_cast<std::size_t>(k);
        EXPECT_EQ(at(cls.off_a, r), a_off) << name << " record " << k;
        if (k > g.start) {
          EXPECT_LE(at(cls.bound, r), at(cls.bound, r - 1)) << name << " record " << k;
        }
        EXPECT_GT(at(cls.bound, r), 0.0) << name << " record " << k;
        // The second shell has momentum lb and its host rank.
        const auto b = by_offset.find(at(cls.off_b, r));
        ASSERT_NE(b, by_offset.end()) << name << " record " << k;
        EXPECT_EQ(b->second.l, cls.lb) << name << " record " << k;
        EXPECT_EQ(at(cls.rank_b, r), b->second.rank) << name << " record " << k;
      }
    }
    EXPECT_EQ(next, cls.n_rec) << name;
    EXPECT_EQ(at(cls.group_start, cls.groups.size()), cls.n_rec) << name;
  }
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

TEST_P(ExchangePairs, DeviceSideMatchesThePinnedSide) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const Ordered o = ordered_of(res, h2o(GetParam()));
  ASSERT_FALSE(o.ordered.empty());
  for (const fb::OrderedPairClass &cls : o.ordered) {
    const fb::OrderedPairClassView v = cls.view();
    EXPECT_EQ(v.n_rec, cls.n_rec);
    EXPECT_EQ(v.rank_b, cls.rank_b.device().data());
    EXPECT_EQ(v.n_groups, static_cast<std::int32_t>(cls.groups.size()));
    EXPECT_EQ(v.group_start, cls.group_start.device().data());
    for (const fb::MirroredBuffer<double> *field :
         {&cls.zeta, &cls.p_x, &cls.p_y, &cls.p_z, &cls.pa_x, &cls.pa_y, &cls.pa_z, &cls.pb_x,
          &cls.pb_y, &cls.pb_z, &cls.s, &cls.bound}) {
      EXPECT_EQ(download(*field), pinned(*field)) << "(" << cls.la << ", " << cls.lb << ")";
    }
    for (const fb::MirroredBuffer<std::int32_t> *field :
         {&cls.off_a, &cls.off_b, &cls.rank_b, &cls.group_start}) {
      EXPECT_EQ(download(*field), pinned(*field)) << "(" << cls.la << ", " << cls.lb << ")";
    }
  }
}

// The ordered classes of one basis built from another basis's mmd classes are
// refused: an AO offset that starts no shell is an InvalidConfig error, not a garbage
// gather.
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
