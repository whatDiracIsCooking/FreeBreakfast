/// @file
/// @brief The Cartesian Coulomb matrix J on the GPU, by McMurchie-Davidson,
/// over the dq, e_hermite and jab kernels FelizJueves generates, screened by
/// the record bound mmd forms (PrimitivePairClass::bound).
///
/// The records are J's own, CoulombPairClass: coulomb_pair_classes' copy of
/// mmd's, made on the device, with the primitive pairs J cannot need pruned
/// and only the fields J reads, plus the class's per-build scratch. Every class
/// is both a ket class (its density pre-contracted, its records sorted by the
/// screen's bound, then contracted into every bra class's J_p) and a bra class
/// (its J_p scattered into J).

export module free_breakfast.integrals.coulomb;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.linalg.matrix_ao;

export namespace free_breakfast {

/// @brief One class of J's primitive-pair records: the records of one of
/// mmd's classes (la, lb) that J can need (@ref coulomb_pair_classes), with
/// the fields J reads, and the scratch a J build writes, on the device only,
/// on the class's own work stream.
///
/// Each record field is mmd's (@ref PrimitivePairClass has their meaning) for
/// the kept records, in mmd's order: descending zeta, so J's bras keep their
/// load-balancing order. @ref bound is mmd's undressed bound, e_hermite's Ip
/// and Iq. There is no pinned side, and no ea, eb, dressed bound or rank:
/// nothing J runs reads them.
///
/// **The scratch is the class's, not the build's.** Every @ref coulomb call
/// rewrites it: the weights, D_q and Dmax in record order, the class as a ket
/// (the `ket_` fields: zeta, P, Iq, Dmax and D_q sorted descending by
/// Iq * Dmax, the order e_hermite's screen breaks on, and the sort's own
/// scratch), and J_p. It is allocated with the records, once per basis, so an
/// SCF loop allocates nothing per build; and, being the class's, it allows one
/// @ref coulomb at a time per set of classes: two builds over the same classes
/// must be queued on one stream, which orders them.
///
/// Movable, not copyable.
struct CoulombPairClass {
  /// @brief The bra shell's angular momentum, <= @ref lb.
  int32_t la;
  /// @brief The ket shell's angular momentum.
  int32_t lb;
  /// @brief The number of records, positive: the length of every record field.
  int32_t n_rec;
  /// @brief The number of Hermite functions of order up to la + lb: one
  /// record's row of @ref jp, and the number of rows of @ref dq.
  int32_t n_hermite;
  /// @brief The class's own work stream: every field was allocated and written
  /// on it, and @ref coulomb launches the class on it. Declared before the
  /// fields, so it outlives their stream-ordered frees.
  std::shared_ptr<DeviceStream> handle;
  /// @brief zeta = alpha + beta.
  StreamBuffers::device<double> zeta;
  /// @brief P: x.
  StreamBuffers::device<double> p_x;
  /// @brief P: y.
  StreamBuffers::device<double> p_y;
  /// @brief P: z.
  StreamBuffers::device<double> p_z;
  /// @brief P - A: x.
  StreamBuffers::device<double> pa_x;
  /// @brief P - A: y.
  StreamBuffers::device<double> pa_y;
  /// @brief P - A: z.
  StreamBuffers::device<double> pa_z;
  /// @brief P - B: x.
  StreamBuffers::device<double> pb_x;
  /// @brief P - B: y.
  StreamBuffers::device<double> pb_y;
  /// @brief P - B: z.
  StreamBuffers::device<double> pb_z;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  StreamBuffers::device<double> s;
  /// @brief The record's Schwarz bound I = pair_bound(zeta, la + lb), mmd's.
  StreamBuffers::device<double> bound;
  /// @brief The bra shell's first Cartesian AO.
  StreamBuffers::device<int32_t> off_a;
  /// @brief The ket shell's first Cartesian AO.
  StreamBuffers::device<int32_t> off_b;

  /// @brief Scratch: each record's density weight, 2 for two shells, 1 for a
  /// shell against itself.
  StreamBuffers::device<double> w;
  /// @brief Scratch: D_q, q-major, n_hermite x n_rec, in record order.
  StreamBuffers::device<double> dq;
  /// @brief Scratch: Dmax = max_q |D_q|, in record order.
  StreamBuffers::device<double> dmax;
  /// @brief Scratch: J_p, record-major, n_rec x n_hermite, summed over every
  /// ket class.
  StreamBuffers::device<double> jp;
  /// @brief Scratch: the ket sort's keys, -(Iq * Dmax).
  StreamBuffers::device<double> ket_key;
  /// @brief Scratch: the ket sort's second key buffer, used past one tile.
  StreamBuffers::device<double> ket_key_alt;
  /// @brief Scratch: the ket sort's permutation.
  StreamBuffers::device<uint32_t> ket_perm;
  /// @brief Scratch: the ket sort's second permutation buffer.
  StreamBuffers::device<uint32_t> ket_perm_alt;
  /// @brief Scratch: zeta, in the ket order.
  StreamBuffers::device<double> ket_zeta;
  /// @brief Scratch: P: x, in the ket order.
  StreamBuffers::device<double> ket_p_x;
  /// @brief Scratch: P: y, in the ket order.
  StreamBuffers::device<double> ket_p_y;
  /// @brief Scratch: P: z, in the ket order.
  StreamBuffers::device<double> ket_p_z;
  /// @brief Scratch: Iq (@ref bound), in the ket order.
  StreamBuffers::device<double> ket_bound;
  /// @brief Scratch: Dmax, in the ket order.
  StreamBuffers::device<double> ket_dmax;
  /// @brief Scratch: D_q, q-major, each row in the ket order.
  StreamBuffers::device<double> ket_dq;

  /// @brief @ref handle, as launch_each takes a class's stream.
  const std::shared_ptr<DeviceStream> &device_stream() const noexcept { return handle; }

  /// @brief The records on the device by pointer, in record order, for a
  /// launcher; the fields J's class does not carry are null. Valid while this
  /// is alive.
  [[nodiscard]] PrimitivePairClassView view() const {
    return {
        .la = la,
        .lb = lb,
        .n_rec = n_rec,
        .zeta = zeta.data(),
        .p_x = p_x.data(),
        .p_y = p_y.data(),
        .p_z = p_z.data(),
        .pa_x = pa_x.data(),
        .pa_y = pa_y.data(),
        .pa_z = pa_z.data(),
        .pb_x = pb_x.data(),
        .pb_y = pb_y.data(),
        .pb_z = pb_z.data(),
        .s = s.data(),
        .ea = nullptr,
        .eb = nullptr,
        .bound = bound.data(),
        .bound_ab = nullptr,
        .bound_ba = nullptr,
        .off_a = off_a.data(),
        .off_b = off_b.data(),
        .rank_a = nullptr,
        .rank_b = nullptr,
    };
  }

  /// @brief The class as e_hermite's ket: la, lb and n_rec, and zeta, P and
  /// the bound (Iq) in the ket order the last build sorted them into. The
  /// other fields are null; e_hermite does not read them of a ket.
  [[nodiscard]] PrimitivePairClassView ket_view() const {
    return {
        .la = la,
        .lb = lb,
        .n_rec = n_rec,
        .zeta = ket_zeta.data(),
        .p_x = ket_p_x.data(),
        .p_y = ket_p_y.data(),
        .p_z = ket_p_z.data(),
        .pa_x = nullptr,
        .pa_y = nullptr,
        .pa_z = nullptr,
        .pb_x = nullptr,
        .pb_y = nullptr,
        .pb_z = nullptr,
        .s = nullptr,
        .ea = nullptr,
        .eb = nullptr,
        .bound = ket_bound.data(),
        .bound_ab = nullptr,
        .bound_ba = nullptr,
        .off_a = nullptr,
        .off_b = nullptr,
        .rank_a = nullptr,
        .rank_b = nullptr,
    };
  }
};

/// @brief The Cartesian Coulomb matrix J_mn = sum_ls (mn|ls) D_ls of the AO
/// density @p density into @p j, screened at @p tau.
///
/// Screened by e_hermite's Schwarz bound: a primitive quartet's [p|q] D_q is
/// skipped when `Dmax * Ip * Iq < tau`, Ip and Iq being the bra's and the
/// ket's record bounds (mmd's bound field) and Dmax the ket's max_q |D_q|. That bounds
/// each term of the bra's J_p, not of J: what jab then scatters is weighted by
/// the bra's E^ab_p. @p tau = 0 skips nothing. The default, 1e-10, is the J
/// threshold of EXESS's J engine (Palethorpe and Barca), whose [p|q] kernel and
/// screen these are.
///
/// Ket-major, in two phases. Zeroes @p j and every class's J_p; then, per
/// class on that class's own stream, pre-contracts the density into the
/// class's Hermite basis (dq), sorts the class as a ket descending by
/// Iq * Dmax (the screen breaks at a lane's first ket under @p tau, so that is
/// the order it needs), and contracts every bra class against it (e_hermite,
/// added atomically into the bra's J_p, which every ket class's stream adds
/// to). Once every class's stream is joined, each class's J_p is complete and
/// it scatters it into @p j on its own stream (jab, added at (m, n) and
/// (n, m)). A class's sorted ket copies are read only on its own stream; the
/// bra side reads the class's records in record order, its Ip the class's own
/// bound, so jab's rows stay the identity. @p density must be symmetric; it is
/// not checked. The classes must come from @ref coulomb_pair_classes over the
/// basis @p density and @p j were made for (at tau = 0 for every record): a
/// record's AO offsets are not checked against j.n().
///
/// Queued on res.stream(), each phase's classes forked from it and joined back
/// (@c launch_each), with no host synchronization, and every class's stream
/// forked from it once more at the end, succeeded or not, so nothing queued
/// there later (a class's own frees) runs ahead of another class's stream
/// still reading it. Writes the classes' scratch (@ref CoulombPairClass), so
/// one call at a time per set of classes; it allocates only a max(n_rec) row
/// map, from res.device()'s pool in res.stream()'s order. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return Success; InvalidConfig if @p tau is negative or NaN, if @p res is
///         not on @p j's, @p density's or @p classes' device, if @p density and
///         @p j differ in order, if a class is above d (or not la <= lb), or if
///         j.n() is too large for the kernels' 32-bit index (j.n()^2 above
///         INT32_MAX); or the GpuRuntime error of the first memset, fork,
///         launch or join that failed, which leaves @p j unspecified.
[[nodiscard]] Status coulomb(DeviceResources &res, std::span<CoulombPairClass> classes,
                             const CartesianMatrixAO &density, CartesianMatrixAO &j,
                             double tau = 1e-10);

/// @brief The records of @p classes (primitive_pair_classes of one basis)
/// that J can need at @p tau: J's own classes, each a device copy of one of
/// @p classes with its negligible primitive pairs pruned, for @ref coulomb.
/// Once per basis; mmd's own records (which S, T and V read) are untouched.
///
/// J's term of a bra record (a, b) and a ket record (c, d) is
/// `sum_p s E^ab_p sum_q [p|q] D_q`, D_q = w sum_cd D_cd s E^cd_q (w = 2 for
/// two shells, 1 for a shell against itself). With |[p|q]| <= I_p I_q (Schwarz)
/// and **|D_cd| <= Dc = 2, the density this assumes and @ref coulomb does not
/// check**. It is an assumption, not a theorem: a closed-shell density has
/// |D_cd| <= sqrt(D_cc D_dd) and D_cc <= 2 (S^-1)_cc, which is 2 for a
/// normalized AO that overlaps nothing and grows with the overlap, without
/// bound toward linear dependence; the H10 chain's RHF density has max|D|
/// 0.33. Under it, every such term of an element
/// of J is at most `B_ab * Dc * K_cd`, where B is a record's dressed bound
/// (mmd's bound_ab: max_{i in a} sum_{k in b} sum_p |s E^{ik}_p| I_p, which
/// bounds every single (i, k) as a bra) and
/// K = w ncart(la) B (which bounds the sum over every (c, d) as a ket). A
/// record is pruned when it is negligible both ways: `B_r Dc max K < tau` and
/// `K_r Dc max B < tau`, the maxima over every record of @p classes. So each
/// term a pruned record would have given, as bra or as ket, is under @p tau.
///
/// What it saves: mmd's bound, J's Ip, is pair-factor free, so J's screen
/// never drops a far bra, and each of its warps runs every ket a near bra
/// would. On long chains that is most of J's time (README.md, "Pruning", has
/// the measurement).
///
/// The default, 1e-13, is three orders under @ref coulomb's screen: each
/// pruned term is under it, but an element sums many, and on the H10 chain
/// pruning at 1e-10 moves J by 6.6e-12 max|J|, past the 1e-12 libcint bar; at
/// 1e-13, by 7.9e-15. The record count barely depends on it (the pair factor
/// falls as exp(-ab/z |AB|^2)). @p tau = 0 keeps every record.
///
/// All on the device: the two maxima as reductions, then per class the keep
/// flag, a stable scan and a scatter of each field J reads, so each kept
/// record lands in its class's order, mmd's descending zeta; an emptied class
/// is left out, the others in @p classes' order. Queued on res.stream(), which
/// first waits on each class's stream (an event, not a host synchronization:
/// mmd's records may still be being ordered and bounded there); each class's
/// stream then waits on res.stream() in turn, so a free of @p classes queued
/// there is ordered after every read of them. One host synchronization, of
/// res.stream(), reads the kept counts back to size J's classes; each is on a
/// new @c DeviceStream of res.device(), its records and its build scratch
/// (@ref CoulombPairClass) allocated there and its scatter queued there and
/// joined back into res.stream(). The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @return J's classes (none for no @p classes); InvalidConfig if @p tau is
///         negative or NaN, or a class is above d (it has no bound kernel, and
///         @ref coulomb refuses it); or the GpuRuntime error of the first
///         event, launch, copy or synchronization that failed.
[[nodiscard]] Result<std::vector<CoulombPairClass>>
coulomb_pair_classes(DeviceResources &res, std::span<const PrimitivePairClass> classes,
                     double tau = 1e-13);

} // namespace free_breakfast
