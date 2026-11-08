/// @file
/// @brief `free-breakfast-qd-nevpt2-gradient`: central-difference nuclear
///        gradients of this tree's SC- and PC-QD-NEVPT2 energies, written as
///        `test/gtest/data/qd_nevpt2_gradient/<case>.json`.
///
/// No other program has a QD-NEVPT2 energy, and an oracle must differentiate
/// exactly the functional the analytic gradient does (the same B, the core and
/// virtuals canonical over the state-averaged Fock, the same PC cut and gap
/// rule), so the energy differenced is the tree's own: at every geometry the
/// chain the `free-breakfast` binary runs for `[casscf]` + density-fitted
/// `[eri]` + `[nevpt2] quasi_degenerate = true`, rerun as one nested
/// `Pipeline`. The energy at the reference is gated against independent code
/// already (`qd_nevpt2/`); this tool only establishes the derivative of it.
///
/// Per case (`kCases`; geometry, basis, active space and weights from
/// `qd_nevpt2/qd_nevpt2.json`):
///
/// - **The reference.** RHF (SAD guess), then the DF-SA-CASSCF over the RHF's
///   frontier orbitals at the case's weights, to ||g|| < kConvTolGrad and
///   |Delta E| < kConvTol, and SC- and PC-QD-NEVPT2 over it. Each root's
///   analytic DF-SA-CASSCF gradient by `mcscf::df_gradient`.
/// - **Every displaced point** re-converges the DF-SA-CASSCF from the
///   reference's orbitals, Loewdin-orthonormalized in the displaced overlap (as
///   `[optimize]` carries them), at the same weights, auxiliary basis and
///   bars, and runs the same QD-NEVPT2. Each displaced CASSCF root is matched
///   to the reference's by `mcscf::state_overlaps`, and each eigenstate K of
///   H_eff by the overlap of u_K in that matched root basis, never by rank:
///   pipeline::hessian's `tracked_qd_state`, the tree's one R_P match.
///   E_K is invariant to the signs of the roots and of u_K.
/// - **The continuity guard** (nevpt2_gradient_reference.py's, #555): the
///   CASSCF converged to the bars; each root's energy within
///   kPredictionBound h^2 of E_r + g_r . dR; the active space's smallest
///   singular value of overlap with the reference's (through the AO cross
///   overlap) within kSubspaceTol of 1; each matched root's CI overlap within
///   kCiTol of 1 in magnitude; each matched u_K's overlap within 1e-3 of 1,
///   with no second state within 1e-3 of it (`tracked_qd_state`'s bar);
///   and for PC, every coupling's metrics cut exactly the reference's modes.
///   A miss refuses the case.
/// - **The differences.** Along three seeded unit directions u (translation
///   and rotation projected out), and, for `Spec::full`, along every Cartesian
///   component: D(h) = (E(+h) - E(-h)) / 2h at the three steps, Richardson
///   (4 D(h/2) - D(h)) / 3 over the two finest, and its error bar, the
///   distance from the same step over the two coarser plus the energies'
///   noise over twice the finest step (`Spec::noise`). Checked before writing:
///   Richardson against the finest D (kRichardsonTol), the error bar
///   (kErrorBarTol), the step ladder's ratio in kLadder wherever its
///   denominator is above kLadderFloor, a full gradient's translational sum
///   within the sum of its components' error bars, and each root's CASSCF difference against its analytic
///   gradient within the component's error bar, which checks the differencing
///   machinery itself.
///
/// The cross-check (`h2o_skew_ccpvdz`): nevpt2_gradient.json's single-state
/// case through the same chain without quasi_degenerate (the per-root SC- and
/// PC-NEVPT2), at its steps, every Cartesian component; its SC and PC
/// gradients must match that file's PySCF/block2-built ones within the two
/// error bars combined, that file's carrying its own noise term
/// (kCrossCheckNoise). Its CASSCF is checked against the analytic gradient,
/// as every case's is.
///
/// **The coupling mode** (`--coupling`, `kCouplingCases`, #743) writes
/// `test/gtest/data/qd_nevpt2_nac/coupling/<case>.json` instead: through the
/// same points, matching and guard, for every pair P < Q of H_eff's
/// eigenstates, SC and PC, central differences of
///
/// - **h_PQ** = d/dR R_P(R0)^T H_eff(R) R_Q(R0), H_eff(R) carried into the
///   reference's root basis (each displaced root matched by
///   `mcscf::state_overlaps` and signed so that overlap is positive), as its
///   reference part sum_r R_rP R_rQ E_r(R) (`h_reference`), its PT2 part
///   R_P^T (H_eff - diag E) R_Q (`h_pt2`) and their sum (`h`);
/// - **the zeroth-order overlap** sum_MN R_MP(R0) R_NQ(R) S_MN(R0, R)
///   (`zeroth`, d^NEV + d^CAS, Park 2020 Eqs. 47 + 48), R(R) matched to
///   R(R0) in that basis and signed so that its overlap is positive;
/// - and, R = 1 (QD off), the roots' own S_IJ(R0, R) (`casscf_zeroth`),
///   checked against `mcscf::df_nac`'s analytic total d_IJ
///   (`use_etfs = false`) within its error bar.
///
/// An overlap and an off-diagonal of H_eff feel a residual ||g|| to first
/// order, so in this mode every point, the reference's too, is polished
/// (`ChainSampler`): rerun from its own orbitals until ||g|| < kPolishTarget.
/// Each family's noise is measured: the reference geometry is re-converged
/// from the orbitals of the three coarsest +h points, and the noise is
/// three times the largest distance of any of the family's values from the
/// reference's own (a point error e moves a Richardson value by up to
/// 1.5 e / h), at least the case's energy noise for the h families and
/// kOverlapNoiseFloor, or the case's `overlap_noise`, for the overlaps. From the
/// Richardson values come d^NEV = h_PQ / (E_Q - E_P) and d^CAS = 2
/// sum_{M<N} T^PQ_MN d_MN (T^PQ = 1/2 (R_P R_Q^T - R_Q R_P^T)), whose sum
/// must match `zeroth`, the visibility of every term, and, where #744's
/// brute-force file has the case, the cross-check of `zeroth` against it.
///
/// The chain at every point, its CASSCF-root guard, the polish and
/// Richardson are the module library beside this file
/// (`free_breakfast.tools.qd_nevpt2_gradient`), which also gives a GPU test
/// the numerical dH_eff/dR in process (#783); the H_eff-eigenstate matching,
/// the energy prediction, the checks and the files are this tool's.
///
/// `free-breakfast-qd-nevpt2-gradient <repo>/test/gtest/data [--coupling]
/// [case ...]`: the cases named, or all of the mode's. Nothing run-dependent
/// is written and every
/// point runs in one order, but the tree's GPU sums reassociate run to run,
/// so a rerun reproduces the energies to ~1e-12 hartree and the gradients to
/// ~1e-9 hartree/bohr, not byte for byte. A refused case writes nothing and
/// fails the run.

#include "diagnostics.h"

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.resources;
import free_breakfast.atom;    // element_from_symbol
import free_breakfast.io.base; // write_file
import free_breakfast.io.json;
import free_breakfast.mcscf.gradient;
import free_breakfast.mcscf.nac;
import free_breakfast.pipeline.nevpt2;
// tracked_qd_state: the tree's one R_P match (#1107).
import free_breakfast.pipeline.hessian;
// The chain at every point, its CASSCF guard and Richardson (#783).
import free_breakfast.tools.qd_nevpt2_gradient;

namespace {

namespace fb = free_breakfast;

// The fitting set every case was made over, compiled in.
constexpr std::string_view kAuxBasis = "cc-pVDZ-JKFIT";

// The chain's guard bounds and the coupling mode's polish and error-bar
// bound are the library's (qd_nevpt2_gradient.cppm).
using fb::nevpt2::kCouplingErrorBarTol;
using fb::nevpt2::kPolishTarget;
using fb::nevpt2::kPredictionBound;

// The checks before writing (hartree/bohr). The gradient tests' gate is
// ~1e-6, not the tree's usual 1e-7, because the oracle is itself a
// difference; every bar here sits at least a factor five inside it.
constexpr double kRichardsonTol = 2e-7;
constexpr double kErrorBarTol = 1e-7;
// nevpt2_gradient.json's own energy noise, as its README states it (about
// 1e-11 hartree in SC and PC, under 1e-12 in the CASSCF): the cross-check
// gives that file's error bars the same noise term.
constexpr double kCrossCheckNoise = 1e-11;
constexpr std::array<double, 2> kLadder{3.2, 4.8};
// The rung difference the ladder check needs, at the least: and five times
// the noise over the finest step (`differences`).
constexpr double kLadderFloor = 1e-7;
// The reference point's energies against the files the case came from: a
// different CASSCF solution misses by millihartree.
constexpr double kReferenceAgreement = 1e-6;

// The directions' generator: one seed, one std::mt19937_64 per case (fully
// specified by the standard, so every library draws the same numbers).
constexpr std::uint64_t kSeed = 696;
constexpr std::size_t kDirections = 3;

// The least noise an overlap family carries in the coupling mode: two full
// runs' Richardson values of skew H2CO's `casscf_zeroth` differ by up to
// 5.3e-9 at 2h = 5e-4, a point error of 2.6e-12 the restarts (1.4e-13) miss.
constexpr double kOverlapNoiseFloor = 3e-12;

// One QD case of qd_nevpt2.json.
struct Spec {
  std::string_view name;
  // The central-difference steps (bohr), halving.
  std::array<double, 3> steps;
  // PC-QD-NEVPT2 differenced beside SC.
  bool pc;
  // The full 3N gradient beside the directions.
  bool full;
  // The energies' noise (hartree), which each error bar carries as noise / 2h
  // at the finest step: the tree's GPU sums reassociate run to run (atomics in
  // integrals/active_space and recipes/einsum), and each point's CASSCF stops
  // at its own ||g||, which a root's energy (not variational in a state
  // average) and E_corr feel to first order, mostly the same way every run.
  // Set above both the largest spread of any point's energy over reruns and
  // the per-point error the CASSCF difference shows against its analytic
  // gradient (qd_nevpt2_gradient/README.md has both, per case).
  double noise;
  // The coupling mode's least noise of SC's and PC's `zeroth`, above the
  // measured one where the data shows a point error the restarts do not
  // (`kCouplingCases`).
  std::array<double, 2> overlap_noise{kOverlapNoiseFloor, kOverlapNoiseFloor};
};

constexpr std::array kCases{
    Spec{"h2co_skew_ccpvdz_sa3", {1e-3, 5e-4, 2.5e-4}, true, true, 1e-11},
    Spec{"h8_ring_sto3g_sa3", {1e-4, 5e-5, 2.5e-5}, true, false, 1e-12},
    Spec{"h6_631g_sa2", {1e-3, 5e-4, 2.5e-4}, true, false, 1e-12},
    Spec{"h8_sto3g_sa3", {5e-4, 2.5e-4, 1.25e-4}, false, false, 1e-11},
    Spec{"h2co_ccpvdz_sa2", {1e-3, 5e-4, 2.5e-4}, true, false, 1e-11},
    Spec{"h2co_ccpvdz_sa2_weighted", {1e-3, 5e-4, 2.5e-4}, true, false, 1e-11},
    Spec{"h2co_skew_ccpvdz_sa3_weighted", {1e-3, 5e-4, 2.5e-4}, true, false, 1e-11},
    Spec{"h8_ring12_ccpvdz_cas66_sa2", {4e-4, 2e-4, 1e-4}, true, true, 8e-12},
};

// The coupling mode's cases (#743), `noise` the h families' least energy
// noise (`kCases`' for the same case). Not h2co_ccpvdz_sa2: its A1 and A2
// roots couple only along an A2 displacement, which H2CO has none of.
constexpr std::array kCouplingCases{
    // SC's and PC's overlaps carry a point error of up to ~1e-10 at a few
    // points, the same on every run (README.md): along Cartesian components
    // that 1e-3 resolves, and smaller steps magnify.
    Spec{"h2co_skew_ccpvdz_sa3", {1e-3, 5e-4, 2.5e-4}, true, true, 1e-11, {2e-10, 2e-10}},
    // Its SC overlaps curve on a length of ~1e-3 bohr along the second
    // direction (its two closest core orbitals are 2.6e-3 hartree apart), so
    // a quarter of the others' steps. PC's overlap carries a point error of
    // ~2e-10 at a few points the restarts do not show (README.md).
    Spec{"h2co_skew_ccpvdz_sa3_weighted",
         {2.5e-4, 1.25e-4, 6.25e-5},
         true,
         false,
         1e-11,
         {kOverlapNoiseFloor, 3e-10}},
    // #744's steps: the SC pair (1, 2), 4.7 mEh apart, needs them. Its SC
    // overlaps carry H_eff's own rounding through 1 / (E_2 - E_1): up to
    // 2e-14 Eh in Srsi's (1, 2) coupling, from ket 2's small norms, which
    // each backend repeats as its own. The largest point difference of any
    // two of ten runs, five per card, is 4.3e-12 (SC) and 2.0e-12 (PC), and
    // the noise is three times that (README.md, "Across backends", #1111).
    Spec{"h8_ring_sto3g_sa3", {6e-5, 3e-5, 1.5e-5}, true, false, 1e-12, {1.3e-11, 6e-12}},
};

// The measured noise's margin: a point error e moves a Richardson value by
// up to 1.5 e / h, the noise term is noise / 2h.
constexpr double kNoiseMargin = 3.0;
// A term is visible where some component's |Richardson| clears this many
// times its consumer's bar, twice its error bar.
constexpr double kVisibility = 10.0;

// The cross-check's noise: H2O's spread over reruns is 1.4e-12.
constexpr double kCrossCheckOwnNoise = 2e-12;

// nevpt2_gradient.json's case the single-state cross-check runs.
constexpr std::string_view kCrossCheck = "h2o_skew_ccpvdz";

// A molecule and its CASSCF, read from a reference file.
struct System {
  std::string name;
  std::string basis;
  std::vector<std::string> symbols;
  std::vector<fb::Atom> atoms; // bohr
  std::int32_t ncas = 0;
  std::int32_t nelecas = 0;
  std::vector<std::int32_t> active; // empty: the RHF's frontier orbitals
  std::vector<double> weights;
};

// The scheme energies at one geometry, in the reference's order: the
// chain's point (each CASSCF root matched), and each SC and PC state (QD:
// H_eff's eigenstates, matched here; one root: its SC- and PC-NEVPT2 total),
// with what the guard measured there.
struct Point {
  fb::nevpt2::ChainPoint at;
  std::vector<double> sc;
  std::vector<double> pc;
  // 1 - overlap: the SC and PC u_K.
  double sc_vectors = 0.0;
  double pc_vectors = 0.0;
  // max_r |E_r - (E_r0 + g_r . dR)| / |dR|^2, set by the caller.
  double prediction = 0.0;
  // SC's and PC's eigenvectors in the reference's root basis, state K at
  // [K n, (K + 1) n), matched to the reference's and signed so that that
  // overlap is positive.
  std::vector<double> sc_vectors_matched;
  std::vector<double> pc_vectors_matched;
};

// nlohmann's .at() throws on a missing key: the one place it is called, the
// exception becomes a Result.
fb::Result<System> read_system(const fb::json &doc, std::string_view name, bool qd) {
  try {
    for (const fb::json &entry : doc.at("cases")) {
      if (entry.at("name").get<std::string>() != name) {
        continue;
      }
      System sys{.name = std::string(name), .basis = entry.at("basis").get<std::string>()};
      for (const fb::json &atom : entry.at("atoms")) {
        // qd_nevpt2.json: [symbol, [x, y, z]]; nevpt2_gradient.json:
        // {"symbol", "coord"}.
        const std::string symbol =
            qd ? atom.at(0).get<std::string>() : atom.at("symbol").get<std::string>();
        const auto coord = (qd ? atom.at(1) : atom.at("coord")).get<std::vector<double>>();
        const fb::Element element = fb::element_from_symbol(symbol);
        if (element == fb::Element::Sentinel || coord.size() != 3) {
          return fb::err_config("{}: atom {} at {} coordinates", name, symbol, coord.size());
        }
        sys.symbols.push_back(symbol);
        sys.atoms.push_back(fb::Atom{{coord[0], coord[1], coord[2]}, element});
      }
      sys.ncas = entry.at("ncas").get<std::int32_t>();
      sys.nelecas = entry.at("nelecas").get<std::int32_t>();
      if (!qd && !entry.at("active").is_null()) {
        sys.active = entry.at("active").get<std::vector<std::int32_t>>();
      }
      sys.weights = entry.at("weights").get<std::vector<double>>();
      return sys;
    }
  } catch (const std::exception &e) {
    return fb::err_io("{}: {}", name, e.what());
  }
  return fb::err_config("no case {} in the reference file", name);
}

// The eigenstates of a displaced H_eff in the reference's order: @p vectors
// (state k at [k n, (k + 1) n), in the reference's root basis, as
// ChainPoint's, so every root sign 1) each matched to the reference state K
// by pipeline::hessian's tracked_qd_state (the tree's one R_P match, its
// 1e-3 bar and its tie), which must make a permutation. Returns
// the matched energies; @p miss takes the worst 1 - |overlap|, and
// @p matched u_k, state K at [K n, (K + 1) n), signed so that its overlap
// with u_K is positive.
fb::Result<std::vector<double>> matched_states(std::span<const double> energies,
                                               std::span<const double> vectors,
                                               std::span<const double> reference, double &miss,
                                               std::vector<double> &matched) {
  const std::size_t n = energies.size();
  const std::vector<double> signs(n, 1.0);
  std::vector<double> out(n);
  std::vector<bool> taken(n, false);
  matched.assign(n * n, 0.0);
  for (std::size_t big_k = 0; big_k < n; ++big_k) {
    const auto [best, overlap] =
        FBF_TRY(fb::tracked_qd_state(reference.subspan(big_k * n, n), vectors, signs));
    if (taken[best]) {
      return fb::err_numerical("H_eff's eigenstate {} matches displaced state {}, already taken",
                               big_k, best);
    }
    taken[best] = true;
    out[big_k] = energies[best];
    const double sign = overlap < 0.0 ? -1.0 : 1.0;
    for (std::size_t i = 0; i < n; ++i) {
      matched[(big_k * n) + i] = sign * vectors[(best * n) + i];
    }
    miss = std::max(miss, 1.0 - std::abs(overlap));
  }
  return out;
}

// Three seeded unit directions in the 3N space, translation and rotation (about
// the centroid) projected out, orthonormalized, each first projected onto the
// totally symmetric displacements of the geometry's symmetry: the operations
// of D2h about the coordinate axes (x -> +-x, ...) that map the atoms onto
// themselves. The energy is invariant under them, so its gradient is totally
// symmetric and loses nothing to the projection; and a symmetry-breaking step
// mixes canonical virtuals of different irreps, which on h2co_ccpvdz_sa2 are
// 8.8e-5 hartree apart, and SC, not invariant to that rotation, curves too
// sharply there for any step the noise allows. C1 geometries have the
// identity alone, and their directions are the plain draws.
std::vector<std::vector<double>> directions(const std::vector<fb::Atom> &atoms) {
  const std::size_t dim = 3 * atoms.size();
  // Each symmetry operation as its signs and the atom it sends each atom to.
  std::vector<std::pair<std::array<double, 3>, std::vector<std::size_t>>> group;
  for (std::size_t bits = 0; bits < 8; ++bits) {
    const std::array<double, 3> signs{(bits & 1U) != 0 ? -1.0 : 1.0, (bits & 2U) != 0 ? -1.0 : 1.0,
                                      (bits & 4U) != 0 ? -1.0 : 1.0};
    std::vector<std::size_t> image;
    for (const fb::Atom &atom : atoms) {
      const std::array<double, 3> r{signs[0] * atom.position.x, signs[1] * atom.position.y,
                                    signs[2] * atom.position.z};
      for (std::size_t b = 0; b < atoms.size(); ++b) {
        if (atoms[b].element == atom.element && std::abs(atoms[b].position.x - r[0]) < 1e-8 &&
            std::abs(atoms[b].position.y - r[1]) < 1e-8 &&
            std::abs(atoms[b].position.z - r[2]) < 1e-8) {
          image.push_back(b);
          break;
        }
      }
    }
    if (image.size() == atoms.size()) {
      group.emplace_back(signs, std::move(image));
    }
  }
  const auto symmetrized = [&](const std::vector<double> &v) {
    std::vector<double> out(dim, 0.0);
    for (const auto &[signs, image] : group) {
      for (std::size_t a = 0; a < atoms.size(); ++a) {
        for (std::size_t k = 0; k < 3; ++k) {
          out[(3 * image[a]) + k] += signs[k] * v[(3 * a) + k] / static_cast<double>(group.size());
        }
      }
    }
    return out;
  };
  std::array<double, 3> centroid{};
  for (const fb::Atom &atom : atoms) {
    centroid[0] += atom.position.x / static_cast<double>(atoms.size());
    centroid[1] += atom.position.y / static_cast<double>(atoms.size());
    centroid[2] += atom.position.z / static_cast<double>(atoms.size());
  }
  std::vector<std::vector<double>> basis;
  for (std::size_t k = 0; k < 3; ++k) {
    std::vector<double> t(dim, 0.0);
    std::vector<double> r(dim, 0.0);
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      t[(3 * a) + k] = 1.0;
      const std::array<double, 3> d{atoms[a].position.x - centroid[0],
                                    atoms[a].position.y - centroid[1],
                                    atoms[a].position.z - centroid[2]};
      // e_k x d.
      const std::size_t k1 = (k + 1) % 3;
      const std::size_t k2 = (k + 2) % 3;
      r[(3 * a) + k2] = d[k1];
      r[(3 * a) + k1] = -d[k2];
    }
    basis.push_back(std::move(t));
    basis.push_back(std::move(r));
  }
  // Gram-Schmidt, twice over: the rigid motions, then each drawn vector
  // against everything before it. A rigid motion that vanishes (a linear
  // molecule's) is dropped.
  const auto orthogonalized = [&](std::vector<double> v) {
    for (std::size_t pass = 0; pass < 2; ++pass) {
      for (const std::vector<double> &b : basis) {
        const double dot = std::inner_product(v.begin(), v.end(), b.begin(), 0.0);
        for (std::size_t i = 0; i < dim; ++i) {
          v[i] -= dot * b[i];
        }
      }
    }
    const double norm = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    for (double &x : v) {
      x /= norm;
    }
    return std::pair{v, norm};
  };
  std::vector<std::vector<double>> rigid;
  std::swap(rigid, basis);
  for (std::vector<double> &v : rigid) {
    auto [unit, norm] = orthogonalized(std::move(v));
    if (norm > 1e-8) {
      basis.push_back(std::move(unit));
    }
  }
  const std::size_t n_rigid = basis.size();
  std::mt19937_64 gen{kSeed};
  // A uniform double in [-1, 1) from the top 53 bits: std::uniform_real_
  // distribution's algorithm is the library's, not the standard's.
  const auto uniform = [&gen] { return (static_cast<double>(gen() >> 11U) * 0x1.0p-52) - 1.0; };
  while (basis.size() < n_rigid + kDirections) {
    std::vector<double> v(dim);
    std::ranges::generate(v, uniform);
    basis.push_back(orthogonalized(symmetrized(v)).first);
  }
  return {basis.begin() + static_cast<std::ptrdiff_t>(n_rigid), basis.end()};
}

// @p g (n_atom x 3 row-major) as rows of three.
fb::json rows(std::span<const double> g) {
  fb::json out = fb::json::array();
  for (std::size_t a = 0; a + 2 < g.size(); a += 3) {
    out.push_back(fb::json::array({g[a], g[a + 1], g[a + 2]}));
  }
  return out;
}

// What a full (every Cartesian component) family's translational sum is held
// to: the sum of its components' error bars (#696's energies), twice its
// largest error bar (#743's h_PQ), or nothing (an overlap, which the basis
// functions moving with the atoms make translation-variant).
enum class Translation : std::uint8_t { SumOfBars, TwiceLargestBar, None };

// One family of differenced quantities ("casscf", "sc" or "pc"; the coupling
// mode's per scheme and term): per index (root, state or pair) and per
// component, central differences per step, Richardson, its error bar, and
// the checks. @p values[i][c][s] = {F(-h), F(+h)}. @p gradients, when not
// empty, is per index an analytic 3N derivative, and @p analytic its
// projection per component. @p noise is the quantity's per-point noise.
struct Family {
  std::string name;
  double noise = 0.0;
  Translation translation = Translation::SumOfBars;
  std::vector<std::vector<double>> gradients;
  std::vector<std::vector<std::vector<std::array<double, 2>>>> values;
  std::vector<std::vector<double>> analytic;
};

// @p full lays components out as n_atom x 3 rows.
// @p richardson_tol bounds Richardson's distance from the finest difference
// (the finest step's truncation), @p error_bar_tol the error bar.
fb::Result<fb::json> differences(const Family &family, std::span<const double> steps, bool full,
                                 double richardson_tol, double error_bar_tol,
                                 std::string_view where, fb::json &checks) {
  const double noise = family.noise;
  // Per full index and axis: the translational sum and its components' bars.
  std::vector<std::pair<double, double>> translation_sums;
  fb::json out = fb::json::array();
  double worst_richardson = 0.0;
  double worst_bar = 0.0;
  double worst_translation = 0.0;
  double worst_translation_ratio = 0.0;
  double worst_analytic = 0.0;
  double worst_analytic_ratio = 0.0;
  std::vector<double> ratios;
  const std::size_t nsteps = steps.size();
  // A rung difference the noise can push out of the window is no evidence
  // either way: noise / h on each rung's D moves the ratio by about
  // 4 noise / (h |D(2h) - D(h)|), the window's half width 0.8 at 5 noise / h.
  const double ladder_floor = std::max(kLadderFloor, 5.0 * noise / steps[nsteps - 1]);
  for (std::size_t i = 0; i < family.values.size(); ++i) {
    const auto &per_component = family.values[i];
    const std::size_t ncomp = per_component.size();
    std::vector<std::vector<double>> central(nsteps, std::vector<double>(ncomp));
    for (std::size_t c = 0; c < ncomp; ++c) {
      for (std::size_t s = 0; s < nsteps; ++s) {
        central[s][c] = (per_component[c][s][1] - per_component[c][s][0]) / (2.0 * steps[s]);
      }
    }
    std::vector<double> richardson(ncomp);
    std::vector<double> bar(ncomp);
    for (std::size_t c = 0; c < ncomp; ++c) {
      const double fine = central[nsteps - 1][c];
      const double middle = central[nsteps - 2][c];
      const double coarse = central[nsteps - 3][c];
      const fb::nevpt2::Richardson r =
          fb::nevpt2::richardson(coarse, middle, fine, noise, steps[nsteps - 1]);
      richardson[c] = r.value;
      bar[c] = r.error_bar;
      worst_richardson = std::max(worst_richardson, std::abs(richardson[c] - fine));
      worst_bar = std::max(worst_bar, bar[c]);
      if (std::abs(middle - fine) > ladder_floor) {
        ratios.push_back((coarse - middle) / (middle - fine));
        if (ratios.back() <= kLadder[0] || ratios.back() >= kLadder[1]) {
          std::println("{} {} [{}] component {}: D = {:.12e}, {:.12e}, {:.12e}", where, family.name,
                       i, c, coarse, middle, fine);
        }
      }
      if (!family.analytic.empty()) {
        const double miss = std::abs(richardson[c] - family.analytic[i][c]);
        worst_analytic = std::max(worst_analytic, miss);
        worst_analytic_ratio = std::max(worst_analytic_ratio, miss / bar[c]);
      }
    }
    if (full && family.translation != Translation::None) {
      for (std::size_t k = 0; k < 3; ++k) {
        double sum = 0.0;
        double bars = 0.0;
        for (std::size_t a = k; a < ncomp; a += 3) {
          sum += richardson[a];
          bars += bar[a];
        }
        translation_sums.emplace_back(sum, bars);
      }
    }
    fb::json entry;
    fb::json central_json = fb::json::array();
    for (const std::vector<double> &d : central) {
      central_json.push_back(full ? rows(d) : fb::json(d));
    }
    entry["central"] = central_json;
    entry["richardson"] = full ? rows(richardson) : fb::json(richardson);
    entry["error_bar"] = full ? rows(bar) : fb::json(bar);
    if (!family.analytic.empty()) {
      entry["analytic"] = full ? rows(family.analytic[i]) : fb::json(family.analytic[i]);
    }
    out.push_back(entry);
  }
  // The translational sums against their bars, once the family's largest
  // error bar is known.
  for (const auto &[sum, bars] : translation_sums) {
    const double against =
        family.translation == Translation::TwiceLargestBar ? 2.0 * worst_bar : bars;
    worst_translation = std::max(worst_translation, std::abs(sum));
    worst_translation_ratio = std::max(worst_translation_ratio, std::abs(sum) / against);
  }
  const bool translation = full && family.translation != Translation::None;
  const auto [low, high] = std::ranges::minmax(ratios.empty() ? std::vector<double>{0.0} : ratios);
  std::println("{} {}: Richardson - finest {:.1e}, error bar {:.1e}, ladder {:.3f}..{:.3f} over "
               "{}{}{}",
               where, family.name, worst_richardson, worst_bar, low, high, ratios.size(),
               translation ? std::format(", translation {:.1e} ({:.2f} of its bar)",
                                         worst_translation, worst_translation_ratio)
                           : "",
               family.analytic.empty() ? ""
                                       : std::format(", analytic {:.1e} ({:.2f} of the error bar)",
                                                     worst_analytic, worst_analytic_ratio));
  if (worst_richardson >= richardson_tol) {
    return fb::err_numerical("{} {}: Richardson off the finest difference by {:.2e}", where,
                             family.name, worst_richardson);
  }
  if (worst_bar >= error_bar_tol) {
    return fb::err_numerical("{} {}: error bar {:.2e}", where, family.name, worst_bar);
  }
  for (const double ratio : ratios) {
    if (ratio <= kLadder[0] || ratio >= kLadder[1]) {
      return fb::err_numerical("{} {}: ladder ratio {:.3f}", where, family.name, ratio);
    }
  }
  if (translation && worst_translation_ratio > 1.0) {
    return fb::err_numerical("{} {}: translational sum {:.2e}, {:.2f} times its bar", where,
                             family.name, worst_translation, worst_translation_ratio);
  }
  if (!family.analytic.empty() && worst_analytic_ratio > 1.0) {
    return fb::err_numerical("{} {}: Richardson off the analytic gradient by {:.2e}, {:.2f} times "
                             "its error bar",
                             where, family.name, worst_analytic, worst_analytic_ratio);
  }
  fb::json check{
      {"richardson_minus_finest", worst_richardson},
      {"error_bar", worst_bar},
      {"ladder_ratios",
       ratios.empty() ? fb::json(nullptr) : fb::json(std::vector<double>{low, high})},
      {"ladder_components", ratios.size()},
  };
  if (translation) {
    check["translation"] = worst_translation;
    check[family.translation == Translation::TwiceLargestBar ? "translation_over_twice_error_bar"
                                                             : "translation_over_error_bars"] =
        worst_translation_ratio;
  }
  if (!family.analytic.empty()) {
    check["analytic"] = worst_analytic;
    check["analytic_over_error_bar"] = worst_analytic_ratio;
  }
  checks[family.name] = check;
  return out;
}

// A Richardson or error-bar entry as a flat list: a direction list as it is,
// n_atom x 3 rows atom-major.
std::vector<double> flat(const fb::json &values) {
  std::vector<double> out;
  for (const fb::json &v : values) {
    if (v.is_array()) {
      for (const fb::json &x : v) {
        out.push_back(x.get<double>());
      }
    } else {
      out.push_back(v.get<double>());
    }
  }
  return out;
}

// The coupling mode's derived terms in one block (`directional` or `full`),
// per scheme and pair, from its Richardson values and error bars:
// d^NEV = h_PQ / (E_Q - E_P), its bar h's over |E_Q - E_P|, and d^CAS =
// sum_{M<N} (R_MP R_NQ - R_MQ R_NP) d_MN over the roots' `casscf_zeroth`,
// its bar the same sum of |coefficient| bar_MN; and the check that
// d^NEV + d^CAS is `zeroth` within the three bars summed.
fb::Status coupling_parts(fb::json &block,
                          std::span<const std::pair<std::size_t, std::size_t>> pairs,
                          std::span<const std::string> schemes, const Point &at0,
                          const fb::Nevpt2Output &nev0, std::size_t n, std::string_view where) {
  try {
    const fb::json &roots = block.at("casscf_zeroth");
    for (const std::string &scheme : schemes) {
      const std::vector<double> &e = scheme == "sc" ? at0.sc : at0.pc;
      const std::vector<double> &r = scheme == "sc" ? nev0.qd_vectors : nev0.pc_qd_vectors;
      fb::json d_nev = fb::json::array();
      fb::json d_cas = fb::json::array();
      double worst = 0.0;
      double worst_ratio = 0.0;
      for (std::size_t pq = 0; pq < pairs.size(); ++pq) {
        const auto [big_p, big_q] = pairs[pq];
        const fb::json &h = block.at(std::format("{}_h", scheme)).at(pq);
        const fb::json &zeroth = block.at(std::format("{}_zeroth", scheme)).at(pq);
        const std::vector<double> h_value = flat(h.at("richardson"));
        const std::vector<double> h_bar = flat(h.at("error_bar"));
        const std::vector<double> z_value = flat(zeroth.at("richardson"));
        const std::vector<double> z_bar = flat(zeroth.at("error_bar"));
        const std::size_t ncomp = h_value.size();
        const double gap = e[big_q] - e[big_p];
        std::vector<double> nev_value(ncomp);
        std::vector<double> nev_bar(ncomp);
        std::vector<double> cas_value(ncomp, 0.0);
        std::vector<double> cas_bar(ncomp, 0.0);
        for (std::size_t c = 0; c < ncomp; ++c) {
          nev_value[c] = h_value[c] / gap;
          nev_bar[c] = h_bar[c] / std::abs(gap);
        }
        for (std::size_t mn = 0; mn < pairs.size(); ++mn) {
          const auto [m, nn] = pairs[mn];
          const double coefficient = (r[(big_p * n) + m] * r[(big_q * n) + nn]) -
                                     (r[(big_q * n) + m] * r[(big_p * n) + nn]);
          const std::vector<double> d_value = flat(roots.at(mn).at("richardson"));
          const std::vector<double> d_bar = flat(roots.at(mn).at("error_bar"));
          for (std::size_t c = 0; c < ncomp; ++c) {
            cas_value[c] += coefficient * d_value[c];
            cas_bar[c] += std::abs(coefficient) * d_bar[c];
          }
        }
        for (std::size_t c = 0; c < ncomp; ++c) {
          const double miss = std::abs(z_value[c] - nev_value[c] - cas_value[c]);
          worst = std::max(worst, miss);
          worst_ratio = std::max(worst_ratio, miss / (z_bar[c] + nev_bar[c] + cas_bar[c]));
        }
        d_nev.push_back({{"richardson", nev_value}, {"error_bar", nev_bar}});
        d_cas.push_back({{"richardson", cas_value}, {"error_bar", cas_bar}});
      }
      block[std::format("{}_d_nev", scheme)] = d_nev;
      block[std::format("{}_d_cas", scheme)] = d_cas;
      block.at("checks")[std::format("{}_zeroth_vs_parts", scheme)] = {
          {"max_difference", worst}, {"max_over_error_bars", worst_ratio}};
      std::println("{} {}: zeroth - (d_nev + d_cas) {:.1e}, {:.2f} of the bars", where, scheme,
                   worst, worst_ratio);
      if (worst_ratio > 1.0) {
        return fb::err_numerical("{} {}: zeroth off d_nev + d_cas by {:.2e}, {:.2f} times the "
                                 "bars",
                                 where, scheme, worst, worst_ratio);
      }
    }
  } catch (const std::exception &e) {
    return fb::err_io("{}: {}", where, e.what());
  }
  return {};
}

// Per scheme and pair, each separately coded term's largest |Richardson| over
// every component of every block, against its consumer's bar (twice the
// error bar): `max`, the largest such bar `bar`, and `best`, the largest
// per-component |Richardson| / (2 error bar); `visible` where `best` clears
// kVisibility. The roots' own coupling (`casscf_zeroth`) per pair beside.
fb::json visibility(const fb::json &doc, std::span<const std::pair<std::size_t, std::size_t>> pairs,
                    std::span<const std::string> schemes) {
  const auto measure = [&](const std::string &family, std::size_t pq) {
    double largest = 0.0;
    double bar = 0.0;
    double best = 0.0;
    for (const std::string_view block : {"directional", "full"}) {
      if (!doc.contains(block)) {
        continue;
      }
      const fb::json &entry = doc.at(block).at(family).at(pq);
      const std::vector<double> value = flat(entry.at("richardson"));
      const std::vector<double> error = flat(entry.at("error_bar"));
      for (std::size_t c = 0; c < value.size(); ++c) {
        largest = std::max(largest, std::abs(value[c]));
        bar = std::max(bar, 2.0 * error[c]);
        best = std::max(best, std::abs(value[c]) / (2.0 * error[c]));
      }
    }
    return fb::json{
        {"max", largest}, {"bar", bar}, {"best", best}, {"visible", best >= kVisibility}};
  };
  fb::json out = fb::json::array();
  for (std::size_t pq = 0; pq < pairs.size(); ++pq) {
    fb::json entry{{"pair", {pairs[pq].first, pairs[pq].second}},
                   {"casscf_zeroth", measure("casscf_zeroth", pq)}};
    for (const std::string &scheme : schemes) {
      for (const std::string_view term :
           {"h_reference", "h_pt2", "h", "d_nev", "d_cas", "zeroth"}) {
        entry[scheme][term] = measure(std::format("{}_{}", scheme, term), pq);
      }
    }
    out.push_back(entry);
  }
  return out;
}

// `zeroth` along the directions against #744's brute-force file's (PySCF,
// full-space overlaps), per scheme, pair and direction, within the two
// error bars combined. The phases differ by the states' signs, s_P s_Q per
// pair, which is read off the data: the sign of sum_c ours_c theirs_c, and
// over three states their product must be +1.
fb::Result<fb::json> brute_force_check(const fb::json &doc, const fb::json &file,
                                       std::span<const std::string> schemes) {
  fb::json result;
  try {
    const auto ours_dirs = doc.at("directions").get<std::vector<std::vector<double>>>();
    const auto their_dirs = file.at("directions").get<std::vector<std::vector<double>>>();
    double direction_miss = 0.0;
    for (std::size_t c = 0; c < ours_dirs.size(); ++c) {
      for (std::size_t x = 0; x < ours_dirs[c].size(); ++x) {
        direction_miss = std::max(direction_miss, std::abs(ours_dirs[c][x] - their_dirs[c][x]));
      }
    }
    if (direction_miss > 1e-12 || doc.at("pairs") != file.at("pairs")) {
      return fb::err_config("brute-force cross-check: other directions ({:.1e}) or pairs",
                            direction_miss);
    }
    result["direction_miss"] = direction_miss;
    for (const std::string &scheme : schemes) {
      const fb::json &ours = doc.at("directional").at(std::format("{}_zeroth", scheme));
      const fb::json &theirs = file.at("directional").at(scheme).at("zeroth");
      std::vector<double> signs;
      double worst = 0.0;
      double worst_ratio = 0.0;
      for (std::size_t pq = 0; pq < ours.size(); ++pq) {
        const std::vector<double> value = flat(ours.at(pq).at("richardson"));
        const std::vector<double> bar = flat(ours.at(pq).at("error_bar"));
        double dot = 0.0;
        for (std::size_t c = 0; c < value.size(); ++c) {
          dot += value[c] * theirs.at(c).at("richardson").at(pq).get<double>();
        }
        const double sign = dot < 0.0 ? -1.0 : 1.0;
        signs.push_back(sign);
        for (std::size_t c = 0; c < value.size(); ++c) {
          const double miss =
              std::abs(value[c] - (sign * theirs.at(c).at("richardson").at(pq).get<double>()));
          const double bars = bar[c] + theirs.at(c).at("error_bar").at(pq).get<double>();
          worst = std::max(worst, miss);
          worst_ratio = std::max(worst_ratio, miss / bars);
        }
      }
      // Pairs (0, 1), (0, 2), (1, 2): s0 s1 . s0 s2 . s1 s2 = 1.
      const bool consistent = signs.size() != 3 || signs[0] * signs[1] * signs[2] > 0.0;
      std::println("brute-force cross-check {}: |ours - #744's| {:.1e}, at most {:.2f} of the two "
                   "error bars, pair signs {}",
                   scheme, worst, worst_ratio, signs);
      if (!consistent || worst_ratio > 1.0) {
        return fb::err_numerical("brute-force cross-check {}: off by {:.2e}, {:.2f} times the "
                                 "two error bars (pair signs {})",
                                 scheme, worst, worst_ratio, signs);
      }
      result[scheme] = {
          {"max_difference", worst}, {"max_over_error_bars", worst_ratio}, {"pair_signs", signs}};
    }
  } catch (const std::exception &e) {
    return fb::err_io("brute-force cross-check: {}", e.what());
  }
  return result;
}

// One case: the reference, every displaced point, the differences and their
// checks. One root (the single-state cross-check) differences its SC- and
// PC-NEVPT2 over every Cartesian component; two or more, H_eff's eigenstates
// along the directions, and along every component when @p full. With
// @p coupling, the coupling mode's families instead (main.cpp's header), and
// the cross-check against @p brute_force, #744's file for the case (null
// when it has none).
fb::Result<fb::json> run_case(fb::NodeResources &node, const System &sys,
                              std::span<const double> steps, double noise, bool pc, bool full,
                              const fb::json &qd_doc, bool coupling, const fb::json &brute_force,
                              std::array<double, 2> overlap_noise = {}) {
  const std::size_t n = sys.weights.size();
  const bool qd = n > 1;
  const std::size_t natom = sys.atoms.size();
  const std::size_t dim = 3 * natom;
  if (coupling && !qd) {
    return fb::err_config("{}: the coupling mode needs two roots or more", sys.name);
  }

  // The reference, and the chain at every point. In the coupling mode it is
  // polished, the reference too: an overlap and an off-diagonal of H_eff feel
  // a residual ||g|| to first order, where an eigenvalue of H_eff feels it
  // only through the roots' energies.
  const auto started = std::chrono::steady_clock::now();
  fb::nevpt2::ChainSampler sampler =
      FBF_TRY(fb::nevpt2::ChainSampler::create(node,
                                               fb::nevpt2::NumericalChain{
                                                   .atoms = sys.atoms,
                                                   .basis = sys.basis,
                                                   .aux_basis = std::string(kAuxBasis),
                                                   .ncas = sys.ncas,
                                                   .nelecas = sys.nelecas,
                                                   .active = sys.active,
                                                   .weights = sys.weights,
                                                   .pc = pc,
                                               },
                                               coupling));
  fb::DeviceResources &res = node.device(0);
  const fb::CasscfOutput &cas0 = sampler.casscf();
  const fb::Nevpt2Output &nev0 = sampler.nevpt2();
  const fb::mcscf::Solution &solution0 = sampler.solution();
  const fb::DensityFitted &fit = sampler.fit();
  // Each root's analytic DF-SA-CASSCF gradient.
  std::vector<std::vector<double>> casscf_gradient;
  for (std::size_t r = 0; r < n; ++r) {
    const fb::mcscf::GradientTerms g = FBF_TRY(
        fb::mcscf::df_gradient(res, sampler.molecule(), sampler.basis(), sampler.classes(), fit.aux,
                               sampler.core_hamiltonian(), fit.b, fit.l.data(), solution0,
                               cas0.ncore, cas0.ncas, static_cast<std::int32_t>(r)));
    casscf_gradient.push_back(g.total);
  }
  // The pairs P < Q, and in the coupling mode each pair of roots' analytic
  // total d_IJ = <I|dJ/dR> (use_etfs = false), over the reference's own CI
  // phases, which every displaced root is signed to.
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = p + 1; q < n; ++q) {
      pairs.emplace_back(p, q);
    }
  }
  std::vector<fb::mcscf::Nac> nacs;
  if (coupling) {
    for (const auto &[bra, ket] : pairs) {
      nacs.push_back(FBF_TRY(fb::mcscf::df_nac(
          res, sampler.molecule(), sampler.basis(), sampler.classes(), fit.aux,
          sampler.core_hamiltonian(), fit.b, fit.l.data(), solution0, cas0.ncore, cas0.ncas,
          static_cast<std::int32_t>(bra), static_cast<std::int32_t>(ket))));
    }
  }
  const auto ncore = static_cast<std::size_t>(cas0.ncore);
  const auto ncas = static_cast<std::size_t>(cas0.ncas);
  // The PC cut at every coupling (QD) or of the root (one state), for the
  // file: the sampler checks every point's against it.
  std::vector<std::array<std::int64_t, 8>> dropped0;
  if (qd) {
    for (const fb::nevpt2::PcClassEnergies &c : nev0.pc_qd_couplings) {
      dropped0.push_back(c.dropped);
    }
  } else if (nev0.pc_roots[0]) {
    dropped0.push_back(nev0.pc_roots[0]->dropped);
  }

  const fb::nevpt2::ChainPoint &ref0 = sampler.reference();
  Point at0;
  at0.at = ref0;
  at0.sc = ref0.sc;
  at0.pc = ref0.pc;

  // The reference against the file the case came from.
  fb::json agreement;
  try {
    for (const fb::json &entry : qd_doc.at("cases")) {
      if (entry.at("name").get<std::string>() != sys.name) {
        continue;
      }
      double casscf_miss = 0.0;
      double sc_miss = 0.0;
      double pc_miss = 0.0;
      if (qd) {
        const auto e_cas = entry.at("casscf_energies").get<std::vector<double>>();
        const auto e_sc = entry.at("eigenvalues").get<std::vector<double>>();
        for (std::size_t r = 0; r < n; ++r) {
          casscf_miss = std::max(casscf_miss, std::abs(e_cas[r] - at0.at.casscf[r]));
          sc_miss = std::max(sc_miss, std::abs(e_sc[r] - at0.sc[r]));
        }
        for (const fb::json &p : qd_doc.at("pc").at("cases")) {
          if (pc && p.at("name").get<std::string>() == sys.name) {
            const auto e_pc = p.at("eigenvalues").get<std::vector<double>>();
            for (std::size_t r = 0; r < n; ++r) {
              pc_miss = std::max(pc_miss, std::abs(e_pc[r] - at0.pc[r]));
            }
          }
        }
      } else {
        const fb::json &e = entry.at("energies");
        casscf_miss = std::abs(e.at("casscf").get<double>() - at0.at.casscf[0]);
        sc_miss = std::abs(e.at("sc").get<double>() - at0.sc[0]);
        pc_miss = std::abs(e.at("pc").get<double>() - at0.pc[0]);
      }
      agreement = {{"casscf", casscf_miss}, {"sc", sc_miss}};
      if (pc) {
        agreement["pc"] = pc_miss;
      }
    }
  } catch (const std::exception &e) {
    return fb::err_io("{}: {}", sys.name, e.what());
  }
  if (agreement.is_null()) {
    return fb::err_config("{}: not in its reference file", sys.name);
  }
  for (const std::string_view key : {"casscf", "sc", "pc"}) {
    if (agreement.contains(key) && agreement[key].get<double>() >= kReferenceAgreement) {
      return fb::err_numerical("{}: the reference {} energies off the file's by {:.2e}", sys.name,
                               key, agreement[key].get<double>());
    }
  }
  std::println("{}: reference |g| {:.1e}, agreement with its file casscf {:.1e} sc {:.1e}{}",
               sys.name, cas0.gradient_norm, agreement["casscf"].get<double>(),
               agreement["sc"].get<double>(),
               pc ? std::format(" pc {:.1e}", agreement["pc"].get<double>()) : "");

  Point worst; // the worst guard values over every point
  worst.at.core_gap = at0.at.core_gap;
  worst.at.virtual_gap = at0.at.virtual_gap;
  std::size_t points = 0;
  // A point from the reference's orbitals, or from @p from's: the sampler's
  // CASSCF guard, then H_eff's eigenstates matched by u_K and guarded.
  const auto point = [&](std::span<const double> displacement,
                         const fb::nevpt2::ChainPoint *from) -> fb::Result<Point> {
    Point p;
    p.at = from == nullptr ? FBF_TRY(sampler.at(displacement))
                           : FBF_TRY(sampler.at(displacement, *from));
    // tracked_qd_state refuses a u_K off 1 by 1e-3 or more; the case and
    // the scheme are named here.
    const auto named =
        [&](std::string_view scheme,
            fb::Result<std::vector<double>> matched) -> fb::Result<std::vector<double>> {
      if (!matched) {
        fb::Error error = std::move(matched).error();
        error.message = std::format("{}: {}: {}", sys.name, scheme, error.message);
        return std::unexpected(std::move(error));
      }
      return matched;
    };
    if (qd) {
      p.sc = FBF_TRY(named("sc", matched_states(p.at.sc, p.at.sc_vectors, nev0.qd_vectors,
                                                p.sc_vectors, p.sc_vectors_matched)));
      if (pc) {
        p.pc = FBF_TRY(named("pc", matched_states(p.at.pc, p.at.pc_vectors, nev0.pc_qd_vectors,
                                                  p.pc_vectors, p.pc_vectors_matched)));
      }
    } else {
      p.sc = p.at.sc;
      p.pc = p.at.pc;
    }
    ++points;
    return p;
  };

  // What every point gives each family, per index (root, state or pair).
  // The gradient mode's families are energies: each root's CASSCF, and SC's
  // and PC's per H_eff eigenstate. The coupling mode's are per pair P < Q:
  // the roots' overlap, then per scheme h's reference and PT2 parts, h, and
  // the zeroth-order overlap, all in the reference's root basis with R held
  // at the reference's eigenvectors @p r0 (state K at [K n, (K + 1) n)) but
  // in the overlap, whose ket takes the point's own.
  const auto coupling_terms = [&](const Point &p, std::span<const double> r0,
                                  std::span<const double> heff, std::span<const double> r,
                                  std::vector<std::vector<double>> &out) {
    std::vector<double> h_reference;
    std::vector<double> h_pt2;
    std::vector<double> h;
    std::vector<double> zeroth;
    for (const auto &[big_p, big_q] : pairs) {
      double ref_part = 0.0;
      double pt2_part = 0.0;
      double whole = 0.0;
      double overlap = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double rp = r0[(big_p * n) + i];
        ref_part += rp * r0[(big_q * n) + i] * p.at.casscf[i];
        for (std::size_t j = 0; j < n; ++j) {
          const double rq = r0[(big_q * n) + j];
          const double off = heff[(i * n) + j] - (i == j ? p.at.casscf[i] : 0.0);
          pt2_part += rp * off * rq;
          whole += rp * heff[(i * n) + j] * rq;
          overlap += rp * r[(big_q * n) + j] * p.at.s[(i * n) + j];
        }
      }
      h_reference.push_back(ref_part);
      h_pt2.push_back(pt2_part);
      h.push_back(whole);
      zeroth.push_back(overlap);
    }
    out.push_back(std::move(h_reference));
    out.push_back(std::move(h_pt2));
    out.push_back(std::move(h));
    out.push_back(std::move(zeroth));
  };
  const auto observe = [&](const Point &p) {
    std::vector<std::vector<double>> out;
    if (!coupling) {
      out.push_back(p.at.casscf);
      out.push_back(p.sc);
      if (pc) {
        out.push_back(p.pc);
      }
      return out;
    }
    std::vector<double> roots;
    for (const auto &[i, j] : pairs) {
      roots.push_back(p.at.s[(i * n) + j]);
    }
    out.push_back(std::move(roots));
    coupling_terms(p, nev0.qd_vectors, p.at.sc_heff, p.sc_vectors_matched, out);
    if (pc) {
      coupling_terms(p, nev0.pc_qd_vectors, p.at.pc_heff, p.pc_vectors_matched, out);
    }
    return out;
  };
  std::vector<std::string> schemes{"sc"};
  if (pc) {
    schemes.emplace_back("pc");
  }
  std::vector<Family> families;
  if (coupling) {
    families.push_back({.name = "casscf_zeroth",
                        .translation = Translation::None,
                        .gradients = {},
                        .values = {},
                        .analytic = {}});
    for (const fb::mcscf::Nac &d : nacs) {
      families.back().gradients.push_back(d.total);
    }
    for (const std::string &scheme : schemes) {
      for (const std::string_view term : {"h_reference", "h_pt2", "h"}) {
        families.push_back({.name = std::format("{}_{}", scheme, term),
                            .translation = Translation::TwiceLargestBar});
      }
      families.push_back(
          {.name = std::format("{}_zeroth", scheme), .translation = Translation::None});
    }
  } else {
    families.push_back({.name = "casscf", .noise = noise, .gradients = casscf_gradient});
    for (const std::string &scheme : schemes) {
      families.push_back({.name = scheme, .noise = noise});
    }
  }

  // The coupling mode's noise: the reference geometry re-converged from the
  // orbitals of the three coarsest +h points along the directions, each
  // family's largest distance from the reference's own values.
  const std::vector<std::vector<double>> dirs = directions(sys.atoms);
  fb::json noise_json;
  if (coupling) {
    Point at_ref;
    at_ref.at = ref0;
    at_ref.sc_vectors_matched = nev0.qd_vectors;
    at_ref.pc_vectors_matched = nev0.pc_qd_vectors;
    const std::vector<std::vector<double>> baseline = observe(at_ref);
    std::vector<double> measured(families.size(), 0.0);
    const std::vector<double> origin(dim, 0.0);
    for (const std::vector<double> &u : dirs) {
      std::vector<double> dr(dim);
      for (std::size_t x = 0; x < dim; ++x) {
        dr[x] = steps[0] * u[x];
      }
      const Point away = FBF_TRY(point(dr, nullptr));
      const Point again = FBF_TRY(point(origin, &away.at));
      const std::vector<std::vector<double>> values = observe(again);
      for (std::size_t f = 0; f < families.size(); ++f) {
        for (std::size_t i = 0; i < values[f].size(); ++i) {
          measured[f] = std::max(measured[f], std::abs(values[f][i] - baseline[f][i]));
        }
      }
    }
    for (std::size_t f = 0; f < families.size(); ++f) {
      const bool energy = families[f].translation == Translation::TwiceLargestBar;
      const double floor = energy                                ? noise
                           : families[f].name.starts_with("sc_") ? overlap_noise[0]
                           : families[f].name.starts_with("pc_") ? overlap_noise[1]
                                                                 : kOverlapNoiseFloor;
      families[f].noise = std::max(floor, kNoiseMargin * measured[f]);
      noise_json[families[f].name] = families[f].noise;
      noise_json["measured"][families[f].name] = measured[f];
      std::println("{}: {} noise measured {:.1e}, carried {:.1e}", sys.name, families[f].name,
                   measured[f], families[f].noise);
    }
  }

  // The components: three directions, or every Cartesian one.
  const auto run_components = [&](const std::vector<std::vector<double>> &components,
                                  bool cartesian) -> fb::Result<fb::json> {
    const std::size_t ncomp = components.size();
    std::vector<Family> run = families;
    fb::json displaced = fb::json::array();
    for (std::size_t c = 0; c < ncomp; ++c) {
      fb::json entry;
      if (cartesian) {
        entry["atom"] = c / 3;
        entry["axis"] = c % 3;
      } else {
        entry["direction"] = c;
      }
      fb::json per_step = fb::json::array();
      for (std::size_t si = 0; si < steps.size(); ++si) {
        fb::json pair = fb::json::array();
        for (const double sign : {-1.0, 1.0}) {
          std::vector<double> dr(dim);
          for (std::size_t x = 0; x < dim; ++x) {
            dr[x] = sign * steps[si] * components[c][x];
          }
          Point p = FBF_TRY(point(dr, nullptr));
          if (coupling && p.at.gradient_norm >= kPolishTarget) {
            std::println("{}: component {} step {} side {}: polished to ||g|| {:.1e} only",
                         sys.name, c, si, sign, p.at.gradient_norm);
          }
          const double h2 = steps[si] * steps[si];
          for (std::size_t r = 0; r < n; ++r) {
            const double predicted =
                at0.at.casscf[r] + std::inner_product(casscf_gradient[r].begin(),
                                                      casscf_gradient[r].end(), dr.begin(), 0.0);
            p.prediction = std::max(p.prediction, std::abs(p.at.casscf[r] - predicted) / h2);
          }
          if (p.prediction >= kPredictionBound) {
            return fb::err_numerical("{}: a displaced CASSCF {:.3f} h^2 off the first-order "
                                     "prediction",
                                     sys.name, p.prediction);
          }
          worst.prediction = std::max(worst.prediction, p.prediction);
          worst.at.subspace = std::max(worst.at.subspace, p.at.subspace);
          worst.at.ci = std::max(worst.at.ci, p.at.ci);
          worst.sc_vectors = std::max(worst.sc_vectors, p.sc_vectors);
          worst.pc_vectors = std::max(worst.pc_vectors, p.pc_vectors);
          worst.at.gradient_norm = std::max(worst.at.gradient_norm, p.at.gradient_norm);
          worst.at.core_gap = std::min(worst.at.core_gap, p.at.core_gap);
          worst.at.virtual_gap = std::min(worst.at.virtual_gap, p.at.virtual_gap);
          const std::size_t side = sign < 0.0 ? 0 : 1;
          const std::vector<std::vector<double>> values = observe(p);
          fb::json energies{{"gradient_norm", p.at.gradient_norm}};
          for (std::size_t f = 0; f < run.size(); ++f) {
            if (run[f].values.empty()) {
              run[f].values.assign(values[f].size(),
                                   std::vector<std::vector<std::array<double, 2>>>(
                                       ncomp, std::vector<std::array<double, 2>>(steps.size())));
            }
            for (std::size_t i = 0; i < values[f].size(); ++i) {
              run[f].values[i][c][si][side] = values[f][i];
            }
            energies[run[f].name] = values[f];
          }
          pair.push_back(energies);
        }
        per_step.push_back(pair);
      }
      entry["points"] = per_step;
      displaced.push_back(entry);
      std::println(
          "{}: {} {} done ({} points, {:.0f} s)", sys.name, cartesian ? "component" : "direction",
          c, points,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    }
    // The analytic derivatives along each component.
    for (Family &f : run) {
      for (const std::vector<double> &g : f.gradients) {
        std::vector<double> along(ncomp);
        for (std::size_t c = 0; c < ncomp; ++c) {
          along[c] = std::inner_product(g.begin(), g.end(), components[c].begin(), 0.0);
        }
        f.analytic.push_back(std::move(along));
      }
    }
    fb::json out;
    fb::json checks;
    for (const Family &f : run) {
      out[f.name] = FBF_TRY(differences(
          f, steps, cartesian, coupling ? std::numeric_limits<double>::infinity() : kRichardsonTol,
          coupling ? kCouplingErrorBarTol : kErrorBarTol,
          std::format("{} {}", sys.name, cartesian ? "full" : "directional"), checks));
    }
    out["checks"] = checks;
    out["displaced"] = displaced;
    return out;
  };

  fb::json doc;
  doc["name"] = sys.name;
  doc["basis"] = sys.basis;
  doc["aux_basis"] = kAuxBasis;
  fb::json atoms_json = fb::json::array();
  for (std::size_t a = 0; a < natom; ++a) {
    atoms_json.push_back(
        {{"symbol", sys.symbols[a]},
         {"coord", {sys.atoms[a].position.x, sys.atoms[a].position.y, sys.atoms[a].position.z}}});
  }
  doc["atoms"] = atoms_json;
  doc["ncore"] = ncore;
  doc["ncas"] = ncas;
  doc["nelecas"] = sys.nelecas;
  doc["active"] = sys.active.empty() ? fb::json(nullptr) : fb::json(sys.active);
  doc["weights"] = sys.weights;
  doc["schemes"] = pc ? std::vector<std::string>{"sc", "pc"} : std::vector<std::string>{"sc"};
  doc["steps"] = std::vector<double>(steps.begin(), steps.end());
  doc["energy_noise"] = noise;
  fb::json reference{
      {"casscf_energies", at0.at.casscf},
      {"sc_energies", at0.sc},
      {"gradient_norm", at0.at.gradient_norm},
      {"casscf_gradient", fb::json::array()},
      {"inactive_gaps", {{"core", at0.at.core_gap}, {"virtual", at0.at.virtual_gap}}},
      {"agreement", agreement},
  };
  for (const std::vector<double> &g : casscf_gradient) {
    reference["casscf_gradient"].push_back(rows(g));
  }
  if (pc) {
    reference["pc_energies"] = at0.pc;
    std::vector<std::vector<std::int64_t>> cut;
    for (const auto &d : dropped0) {
      cut.emplace_back(d.begin(), d.end());
    }
    reference["pc_dropped"] = cut;
  }
  if (qd) {
    std::vector<double> sc_gaps;
    std::vector<double> pc_gaps;
    for (std::size_t k = 1; k < n; ++k) {
      sc_gaps.push_back(at0.sc[k] - at0.sc[k - 1]);
      if (pc) {
        pc_gaps.push_back(at0.pc[k] - at0.pc[k - 1]);
      }
    }
    reference["sc_heff_gaps"] = sc_gaps;
    reference["sc_heff_vectors"] = nev0.qd_vectors;
    if (pc) {
      reference["pc_heff_gaps"] = pc_gaps;
      reference["pc_heff_vectors"] = nev0.pc_qd_vectors;
    }
  }
  if (coupling) {
    std::vector<double> root_gaps;
    for (std::size_t k = 1; k < n; ++k) {
      root_gaps.push_back(at0.at.casscf[k] - at0.at.casscf[k - 1]);
    }
    reference["casscf_gaps"] = root_gaps;
    reference["sc_heff"] = nev0.qd_heff;
    if (pc) {
      reference["pc_heff"] = nev0.pc_qd_heff;
    }
    reference["nac"] = fb::json::array();
    for (const fb::mcscf::Nac &d : nacs) {
      reference["nac"].push_back({{"total", rows(d.total)},
                                  {"numerator", rows(d.numerator)},
                                  {"csf", rows(d.csf)},
                                  {"iterations", d.iterations},
                                  {"residual", d.residual}});
    }
    doc["pairs"] = pairs;
    doc["noise"] = noise_json;
  }
  doc["reference"] = reference;
  if (qd) {
    doc["seed"] = kSeed;
    doc["directions"] = dirs;
    doc["directional"] = FBF_TRY(run_components(dirs, false));
  }
  if (full) {
    std::vector<std::vector<double>> cartesian(dim, std::vector<double>(dim, 0.0));
    for (std::size_t x = 0; x < dim; ++x) {
      cartesian[x][x] = 1.0;
    }
    doc["full"] = FBF_TRY(run_components(cartesian, true));
  }
  if (coupling) {
    for (const std::string_view block : {"directional", "full"}) {
      if (doc.contains(block)) {
        FBF_TRY(coupling_parts(doc[block], pairs, schemes, at0, nev0, n,
                               std::format("{} {}", sys.name, block)));
      }
    }
    doc["visibility"] = visibility(doc, pairs, schemes);
    if (!brute_force.is_null()) {
      doc["brute_force_cross_check"] = FBF_TRY(brute_force_check(doc, brute_force, schemes));
    }
  }
  fb::json guard{
      {"prediction", worst.prediction},
      {"subspace", worst.at.subspace},
      {"ci", worst.at.ci},
      {"sc_vectors", worst.sc_vectors},
      {"gradient_norm", worst.at.gradient_norm},
      {"inactive_gap", {{"core", worst.at.core_gap}, {"virtual", worst.at.virtual_gap}}},
      {"points", points},
  };
  if (pc) {
    guard["pc_vectors"] = worst.pc_vectors;
  }
  if (coupling) {
    guard["polish_passes"] = sampler.polish_passes();
  }
  doc["guard"] = guard;
  std::println("{}: {} points, worst prediction {:.3f}, subspace {:.1e}, CI {:.1e}, u_K {:.1e} / "
               "{:.1e}, |g| {:.1e} ({:.0f} s)",
               sys.name, points, worst.prediction, worst.at.subspace, worst.at.ci, worst.sc_vectors,
               worst.pc_vectors, worst.at.gradient_norm,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
  return doc;
}

// The single-state cross-check against nevpt2_gradient.json's PySCF/block2
// gradients: per quantity and component, |ours - theirs| within the two
// error bars combined.
fb::Status cross_check(fb::json &doc, const fb::json &file_case) {
  fb::json result;
  try {
    for (const std::string_view q : {"sc", "pc"}) {
      const fb::json &ours = doc.at("full").at(q).at(0);
      const fb::json &theirs = file_case.at("gradients").at(q);
      double worst = 0.0;
      double worst_ratio = 0.0;
      const double finest = file_case.at("steps").back().get<double>();
      for (std::size_t a = 0; a < ours.at("richardson").size(); ++a) {
        for (std::size_t k = 0; k < 3; ++k) {
          const double diff = std::abs(ours.at("richardson").at(a).at(k).get<double>() -
                                       theirs.at("richardson").at(a).at(k).get<double>());
          const double bars = ours.at("error_bar").at(a).at(k).get<double>() +
                              theirs.at("error_bar").at(a).at(k).get<double>() +
                              (kCrossCheckNoise / (2.0 * finest));
          worst = std::max(worst, diff);
          worst_ratio = std::max(worst_ratio, diff / bars);
        }
      }
      std::println("cross-check {}: |ours - PySCF's| {:.1e}, at most {:.2f} of the two error "
                   "bars",
                   q, worst, worst_ratio);
      if (worst_ratio > 1.0) {
        return fb::err_numerical("cross-check {}: off the reference file's gradient by {:.2e}, "
                                 "{:.2f} times the two error bars",
                                 q, worst, worst_ratio);
      }
      result[q] = {{"max_difference", worst}, {"max_over_error_bars", worst_ratio}};
    }
  } catch (const std::exception &e) {
    return fb::err_io("cross-check: {}", e.what());
  }
  doc["cross_check"] = result;
  return {};
}

fb::Status run(std::span<char *const> args) {
  if (args.size() < 2) {
    return fb::err_config("usage: {} <repo>/test/gtest/data [--coupling] [case ...]",
                          args.empty() ? "free-breakfast-qd-nevpt2-gradient" : args[0]);
  }
  const std::filesystem::path data{args[1]};
  const bool coupling = args.size() > 2 && std::string_view{args[2]} == "--coupling";
  std::vector<std::string_view> only;
  for (const char *arg : args.subspan(coupling ? 3 : 2)) {
    only.emplace_back(arg);
  }
  const auto wanted = [&](std::string_view name) {
    return only.empty() || std::ranges::contains(only, name);
  };
  for (const std::string_view name : only) {
    if (coupling ? !std::ranges::contains(kCouplingCases, name, &Spec::name)
                 : name != kCrossCheck && !std::ranges::contains(kCases, name, &Spec::name)) {
      return fb::err_config("no case {}", name);
    }
  }
  const fb::json qd_doc = FBF_TRY(fb::read_json(data / "qd_nevpt2" / "qd_nevpt2.json"));
  const fb::json single_doc =
      FBF_TRY(fb::read_json(data / "nevpt2_gradient" / "nevpt2_gradient.json"));
  const std::filesystem::path out_dir =
      coupling ? data / "qd_nevpt2_nac" / "coupling" : data / "qd_nevpt2_gradient";
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    return fb::err_io("cannot create {}: {}", out_dir.string(), ec.message());
  }
  // The CASSCF logs every macro iteration at info; this tool prints its own
  // progress.
  fb::log::set_level(fb::log::LogLevel::Warn);
  fb::NodeResources node;
  if (node.device_count() == 0) {
    return fb::err_gpu("no GPU device is visible");
  }

  if (coupling) {
    for (const Spec &spec : kCouplingCases) {
      if (!wanted(spec.name)) {
        continue;
      }
      const System sys = FBF_TRY(read_system(qd_doc, spec.name, true));
      const std::filesystem::path brute =
          data / "qd_nevpt2_nac" / "brute_force" / std::format("{}.json", spec.name);
      const fb::json brute_force =
          std::filesystem::exists(brute) ? FBF_TRY(fb::read_json(brute)) : fb::json(nullptr);
      const fb::json doc = FBF_TRY(run_case(node, sys, spec.steps, spec.noise, spec.pc, spec.full,
                                            qd_doc, true, brute_force, spec.overlap_noise));
      const std::filesystem::path out = out_dir / std::format("{}.json", spec.name);
      FBF_TRY(fb::write_file(out, fb::json_to_string(doc, 1) + '\n'));
      std::println("wrote {}", out.string());
    }
    return {};
  }
  if (wanted(kCrossCheck)) {
    const System sys = FBF_TRY(read_system(single_doc, kCrossCheck, false));
    fb::json file_case;
    try {
      for (const fb::json &entry : single_doc.at("cases")) {
        if (entry.at("name").get<std::string>() == kCrossCheck) {
          file_case = entry;
        }
      }
    } catch (const std::exception &e) {
      return fb::err_io("{}: {}", kCrossCheck, e.what());
    }
    std::vector<double> steps;
    try {
      steps = file_case.at("steps").get<std::vector<double>>();
    } catch (const std::exception &e) {
      return fb::err_io("{}: {}", kCrossCheck, e.what());
    }
    fb::json doc = FBF_TRY(
        run_case(node, sys, steps, kCrossCheckOwnNoise, true, true, single_doc, false, nullptr));
    FBF_TRY(cross_check(doc, file_case));
    FBF_TRY(fb::write_file(out_dir / std::format("single_state_{}.json", kCrossCheck),
                           fb::json_to_string(doc, 1) + '\n'));
  }
  for (const Spec &spec : kCases) {
    if (!wanted(spec.name)) {
      continue;
    }
    const System sys = FBF_TRY(read_system(qd_doc, spec.name, true));
    const fb::json doc = FBF_TRY(
        run_case(node, sys, spec.steps, spec.noise, spec.pc, spec.full, qd_doc, false, nullptr));
    FBF_TRY(fb::write_file(out_dir / std::format("{}.json", spec.name),
                           fb::json_to_string(doc, 1) + '\n'));
    std::println("wrote {}", (out_dir / std::format("{}.json", spec.name)).string());
  }
  return {};
}

} // namespace

// int-type-ok: argc's type is fixed by the standard's signature for main.
int main(int argc, char **argv) {
  const fb::Status status = run(std::span<char *const>(argv, static_cast<std::size_t>(argc)));
  if (!status) {
    std::println(std::cerr, "error: {}", status.error().message);
    return fb::to_exit_code(status.error().kind);
  }
  return 0;
}
