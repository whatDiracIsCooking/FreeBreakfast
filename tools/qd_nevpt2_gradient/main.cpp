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
///   H_eff by the overlap of u_K in that matched root basis, never by rank.
///   E_K is invariant to the signs of the roots and of u_K.
/// - **The continuity guard** (nevpt2_gradient_reference.py's, #555): the
///   CASSCF converged to the bars; each root's energy within
///   kPredictionBound h^2 of E_r + g_r . dR; the active space's smallest
///   singular value of overlap with the reference's (through the AO cross
///   overlap) within kSubspaceTol of 1; each matched root's CI overlap within
///   kCiTol of 1 in magnitude; each matched u_K's overlap within kCiTol of 1;
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
/// `free-breakfast-qd-nevpt2-gradient <repo>/test/gtest/data [case ...]`: the
/// cases named, or all of them. Nothing run-dependent is written and every
/// point runs in one order, but the tree's GPU sums reassociate run to run,
/// so a rerun reproduces the energies to ~1e-12 hartree and the gradients to
/// ~1e-9 hartree/bohr, not byte for byte. A refused case writes nothing and
/// fails the run.

#include "diagnostics.h"

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.atom;    // element_from_symbol
import free_breakfast.io.base; // write_file
import free_breakfast.io.json;
import free_breakfast.integrals.one_electron.overlap_mixed;
import free_breakfast.mcscf.gradient;
import free_breakfast.mcscf.overlap;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.orthogonalize;
import free_breakfast.pipeline.pipeline;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.overlap;
import free_breakfast.pipeline.nevpt2;

namespace {

namespace fb = free_breakfast;

// The CASSCF's bars at every point. NEVPT2 is not variational in the
// orbitals, so a residual gradient puts an error linear in ||g|| into E_corr;
// over a step of 1e-3 bohr that is the size of the 1e-6 gate unless ||g|| is
// held near the formaldehyde fixtures' polish (< 1e-10).
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-13;
constexpr std::int32_t kMaxCycle = 200;

// The fitting set every case was made over, compiled in.
constexpr std::string_view kAuxBasis = "cc-pVDZ-JKFIT";

// The continuity guard (nevpt2_gradient_reference.py's bounds): hartree/bohr^2
// for the prediction, 1 - overlap for the rest. Another stationary point or
// another state misses each by orders of magnitude.
constexpr double kPredictionBound = 2.0;
constexpr double kSubspaceTol = 1e-3;
constexpr double kCiTol = 1e-3;

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

// The chain's root at one geometry: the molecule from atoms in memory
// (pipeline/optimize's TrialGeometryStep).
struct TrialGeometryStep {
  static constexpr std::string_view kName = "molecule";

  std::vector<fb::Atom> atoms;

  [[nodiscard]] fb::Result<fb::MoleculeOutput> run(fb::View<> /*in*/,
                                                   fb::DeviceResources &res) const {
    return fb::MoleculeOutput{fb::Molecule(res, atoms, 0, 1)};
  }
};

// Every displaced point's ScfOutput: the reference CASSCF's orbitals,
// Loewdin-orthonormalized in this geometry's overlap, C (C^T S C)^-1/2
// (pipeline/optimize's CarriedOrbitalsStep), so the CasscfStep starts from the
// active space the reference converged. Only the CasscfStep reads it, by index.
struct CarriedOrbitalsStep {
  static constexpr std::string_view kName = "scf";

  std::vector<double> c; // AO by MO, column-major
  std::vector<double> mo_energies;
  std::int32_t n_alpha = 0;
  std::int32_t n_beta = 0;

  [[nodiscard]] fb::Result<fb::ScfOutput> run(fb::View<fb::BasisSetOutput, fb::OverlapOutput> in,
                                              fb::DeviceResources &res) const {
    const fb::BasisSet &basis = in.get<fb::BasisSetOutput>().basis;
    const fb::SphericalMatrixAO &s = in.get<fb::OverlapOutput>().s;
    if (c.size() != s.num_elements() || mo_energies.size() != s.n()) {
      return fb::err_config("carried {} orbital coefficients and {} energies into a basis of {}",
                            c.size(), mo_energies.size(), s.n());
    }
    const fb::ScopedDeviceIndex on{res.dev_idx()};
    const auto n = static_cast<std::int32_t>(s.n());
    const std::size_t n2 = s.num_elements();
    fb::SphericalMatrixAO previous(res, basis);
    fb::Buffers::device<double> m(n2, res.device());
    fb::Buffers::device<double> m_inv_sqrt(n2, res.device());
    FBF_TRY(fb::gpu_status(wwr::wwrMemcpyAsync(previous.data(), c.data(), n2 * sizeof(double),
                                               wwr::wwrMemcpyHostToDevice, res.stream())));
    {
      const auto [work] = res.device_regions<double>(n2);
      FBF_TRY(
          fb::gpu_status(fb::gemmwich(res.blas(), n, previous.data(), s.data(), work, m.data())));
    }
    FBF_TRY(fb::orthogonalize<double>(res, n, m.data(), m_inv_sqrt.data()));
    fb::ScfOutput out{
        .energy = std::numeric_limits<double>::quiet_NaN(),
        .iterations = 0,
        .n_alpha = n_alpha,
        .n_beta = n_beta,
        .mo_energies = mo_energies,
        .c = fb::SphericalMatrixAO(res, basis),
        .d = fb::SphericalMatrixAO(res, basis),
    };
    FBF_TRY(fb::gpu_status(fb::gemm(res.blas(), n, &fb::kOne<double>, previous.data(),
                                    m_inv_sqrt.data(), &fb::kZero<double>, out.c.data())));
    FBF_TRY(res.sync());
    return out;
  }
};

// The energies at one geometry, in the reference's order: each CASSCF root,
// and each SC and PC state (QD: H_eff's eigenstates; one root: its SC- and
// PC-NEVPT2 total), with what the guard measured there.
struct Point {
  std::vector<double> casscf;
  std::vector<double> sc;
  std::vector<double> pc;
  double gradient_norm = 0.0;
  // 1 - overlap: the active space, the roots' CI, the SC and PC u_K.
  double subspace = 0.0;
  double ci = 0.0;
  double sc_vectors = 0.0;
  double pc_vectors = 0.0;
  // max_r |E_r - (E_r0 + g_r . dR)| / |dR|^2, set by the caller.
  double prediction = 0.0;
  // The smallest gap between two canonical core, or virtual, energies.
  double core_gap = std::numeric_limits<double>::infinity();
  double virtual_gap = std::numeric_limits<double>::infinity();
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

// The mcscf::Solution a CasscfOutput is (pipeline/casscf_gradient's): its
// orbitals and CI vectors back on the device, the weighted gamma from the
// per-root ones.
fb::Result<fb::mcscf::Solution> solution_of(fb::DeviceResources &res, const fb::BasisSet &basis,
                                            const fb::CasscfOutput &cas) {
  const auto n2 = static_cast<std::size_t>(cas.ncas) * static_cast<std::size_t>(cas.ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas.weights.size(); ++r) {
    for (std::size_t i = 0; i < n2; ++i) {
      gamma[i] += cas.weights[r] * cas.rdm1[(r * n2) + i];
    }
  }
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots =
          {
              .energies = cas.energies,
              .spin_squares = cas.spin_squares,
              .vectors = fb::Buffers::device<double>(cas.vectors.size(), res.device()),
              .davidson = {},
              .fp32_iterations = 0,
          },
      .energy = cas.energy,
      .gradient_norm = cas.gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas.rdm2,
      .root_gammas = cas.rdm1,
      .iterations = cas.iterations,
  };
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(fb::gpu_status(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                             cas.vectors.size() * sizeof(double),
                                             wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(fb::gpu_status(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                             cas.c.num_elements() * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
  FBF_TRY(res.sync());
  return solution;
}

// @p n device doubles on the host.
fb::Result<std::vector<double>> download(fb::DeviceResources &res, const double *device,
                                         std::size_t n) {
  std::vector<double> host(n);
  FBF_TRY(fb::gpu_status(wwr::wwrMemcpyAsync(host.data(), device, n * sizeof(double),
                                             wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return host;
}

// 1 - the smallest singular value of C_a^T S_x C'_a, the two active spaces'
// overlap through the AO cross overlap of the two geometries.
fb::Result<double> subspace_miss(fb::DeviceResources &res, const fb::BasisSet &bra_basis,
                                 const fb::SphericalMatrixAO &bra, const fb::BasisSet &ket_basis,
                                 const fb::SphericalMatrixAO &ket, std::size_t ncore,
                                 std::size_t ncas) {
  const std::size_t nb = bra.n();
  const std::size_t nk = ket.n();
  fb::Buffers::device<double> s_dev(nb * nk, res.device());
  FBF_TRY(fb::overlap_mixed_spherical(res, bra_basis, ket_basis, s_dev));
  const std::vector<double> s = FBF_TRY(download(res, s_dev.data(), nb * nk));
  const std::vector<double> cb = FBF_TRY(download(res, bra.data(), bra.num_elements()));
  const std::vector<double> ck = FBF_TRY(download(res, ket.data(), ket.num_elements()));
  // sc = S C'_a (nb x ncas), then M = C_a^T sc.
  std::vector<double> sc(nb * ncas, 0.0);
  for (std::size_t j = 0; j < ncas; ++j) {
    for (std::size_t n = 0; n < nk; ++n) {
      const double x = ck[n + (nk * (ncore + j))];
      for (std::size_t m = 0; m < nb; ++m) {
        sc[m + (nb * j)] += s[m + (nb * n)] * x;
      }
    }
  }
  std::vector<double> mat(ncas * ncas, 0.0);
  for (std::size_t j = 0; j < ncas; ++j) {
    for (std::size_t i = 0; i < ncas; ++i) {
      for (std::size_t m = 0; m < nb; ++m) {
        mat[i + (ncas * j)] += cb[m + (nb * (ncore + i))] * sc[m + (nb * j)];
      }
    }
  }
  // M^T M's least eigenvalue is the least singular value squared.
  std::vector<double> mtm(ncas * ncas, 0.0);
  for (std::size_t j = 0; j < ncas; ++j) {
    for (std::size_t i = 0; i < ncas; ++i) {
      for (std::size_t k = 0; k < ncas; ++k) {
        mtm[i + (ncas * j)] += mat[k + (ncas * i)] * mat[k + (ncas * j)];
      }
    }
  }
  fb::Buffers::device<double> a(mtm.size(), res.device());
  fb::Buffers::device<double> w(ncas, res.device());
  FBF_TRY(fb::gpu_status(wwr::wwrMemcpyAsync(a.data(), mtm.data(), mtm.size() * sizeof(double),
                                             wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(fb::diagonalize(res, static_cast<std::int32_t>(ncas), a.data(), w.data()));
  const std::vector<double> values = FBF_TRY(download(res, w.data(), ncas));
  return 1.0 - std::sqrt(std::max(values.front(), 0.0));
}

// The smallest gap between neighbouring values of @p sorted (infinity for
// fewer than two).
double smallest_gap(std::span<const double> sorted) {
  double gap = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i < sorted.size(); ++i) {
    gap = std::min(gap, sorted[i] - sorted[i - 1]);
  }
  return gap;
}

// The eigenstates of a displaced H_eff in the reference's order: @p vectors
// (state k at [k n, (k + 1) n), over the displaced roots) carried into the
// reference's root basis by @p pi and @p signs, each reference state K matched
// to the displaced k of largest |<u_K|u_k>|, which must be a permutation.
// Returns the matched energies; @p miss takes the worst 1 - |overlap|.
fb::Result<std::vector<double>> matched_states(std::span<const double> energies,
                                               std::span<const double> vectors,
                                               std::span<const double> reference,
                                               std::span<const std::size_t> pi,
                                               std::span<const double> signs, double &miss) {
  const std::size_t n = energies.size();
  std::vector<double> out(n);
  std::vector<bool> taken(n, false);
  for (std::size_t big_k = 0; big_k < n; ++big_k) {
    std::size_t best = n;
    double best_overlap = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
      double overlap = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        overlap += reference[(big_k * n) + i] * signs[i] * vectors[(k * n) + pi[i]];
      }
      if (std::abs(overlap) > best_overlap) {
        best_overlap = std::abs(overlap);
        best = k;
      }
    }
    if (best == n || taken[best]) {
      return fb::err_numerical("H_eff's eigenstate {} matches no displaced state alone", big_k);
    }
    taken[best] = true;
    out[big_k] = energies[best];
    miss = std::max(miss, 1.0 - best_overlap);
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

// One family of differenced energies ("casscf", "sc" or "pc"): per index
// (root or state) and per component, central differences per step, Richardson,
// its error bar, and the checks. @p values[i][c][s] = {E(-h), E(+h)}.
// @p analytic, when not empty, is per index the analytic derivative per
// component. @p full lays components out as n_atom x 3 rows.
struct Family {
  std::string name;
  std::vector<std::vector<std::vector<std::array<double, 2>>>> values;
  std::vector<std::vector<double>> analytic;
};

fb::Result<fb::json> differences(const Family &family, std::span<const double> steps, double noise,
                                 bool full, std::string_view where, fb::json &checks) {
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
      richardson[c] = ((4.0 * fine) - middle) / 3.0;
      bar[c] = std::abs(richardson[c] - (((4.0 * middle) - coarse) / 3.0)) +
               (noise / (2.0 * steps[nsteps - 1]));
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
    if (full) {
      for (std::size_t k = 0; k < 3; ++k) {
        double sum = 0.0;
        double bars = 0.0;
        for (std::size_t a = k; a < ncomp; a += 3) {
          sum += richardson[a];
          bars += bar[a];
        }
        worst_translation = std::max(worst_translation, std::abs(sum));
        worst_translation_ratio = std::max(worst_translation_ratio, std::abs(sum) / bars);
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
  const auto [low, high] = std::ranges::minmax(ratios.empty() ? std::vector<double>{0.0} : ratios);
  std::println("{} {}: Richardson - finest {:.1e}, error bar {:.1e}, ladder {:.3f}..{:.3f} over "
               "{}{}{}",
               where, family.name, worst_richardson, worst_bar, low, high, ratios.size(),
               full ? std::format(", translation {:.1e} ({:.2f} of the error bars)",
                                  worst_translation, worst_translation_ratio)
                    : "",
               family.analytic.empty() ? ""
                                       : std::format(", analytic {:.1e} ({:.2f} of the error bar)",
                                                     worst_analytic, worst_analytic_ratio));
  if (worst_richardson >= kRichardsonTol) {
    return fb::err_numerical("{} {}: Richardson off the finest difference by {:.2e}", where,
                             family.name, worst_richardson);
  }
  if (worst_bar >= kErrorBarTol) {
    return fb::err_numerical("{} {}: error bar {:.2e}", where, family.name, worst_bar);
  }
  for (const double ratio : ratios) {
    if (ratio <= kLadder[0] || ratio >= kLadder[1]) {
      return fb::err_numerical("{} {}: ladder ratio {:.3f}", where, family.name, ratio);
    }
  }
  if (full && worst_translation_ratio > 1.0) {
    return fb::err_numerical("{} {}: translational sum {:.2e}, {:.2f} times its components' "
                             "error bars",
                             where, family.name, worst_translation, worst_translation_ratio);
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
  if (full) {
    check["translation"] = worst_translation;
    check["translation_over_error_bars"] = worst_translation_ratio;
  }
  if (!family.analytic.empty()) {
    check["analytic"] = worst_analytic;
    check["analytic_over_error_bar"] = worst_analytic_ratio;
  }
  checks[family.name] = check;
  return out;
}

// One case: the reference, every displaced point, the differences and their
// checks. One root (the single-state cross-check) differences its SC- and
// PC-NEVPT2 over every Cartesian component; two or more, H_eff's eigenstates
// along the directions, and along every component when @p full.
fb::Result<fb::json> run_case(fb::NodeResources &node, const System &sys,
                              std::span<const double> steps, double noise, bool pc, bool full,
                              const fb::json &qd_doc) {
  fb::DeviceResources &res = node.device(0);
  const std::size_t n = sys.weights.size();
  const bool qd = n > 1;
  const std::size_t natom = sys.atoms.size();
  const std::size_t dim = 3 * natom;

  fb::BasisSetStep basis_set;
  basis_set.basis = sys.basis;
  fb::EriStep eri;
  eri.mode = fb::EriMode::DensityFitting;
  eri.aux_basis = std::string(kAuxBasis);
  fb::CasscfStep casscf;
  casscf.ncas = sys.ncas;
  casscf.nelecas = sys.nelecas;
  casscf.active = sys.active;
  casscf.nroots = static_cast<std::int32_t>(n);
  casscf.weights = sys.weights;
  casscf.conv_tol = kConvTol;
  casscf.conv_tol_grad = kConvTolGrad;
  casscf.max_cycle = kMaxCycle;
  fb::Nevpt2Step nevpt2;
  nevpt2.quasi_degenerate = qd;

  const auto started = std::chrono::steady_clock::now();
  auto ref = FBF_TRY((fb::Pipeline<TrialGeometryStep, fb::BasisSetStep, fb::SphericalTransformStep,
                                   fb::PrimitivePairClassesStep, fb::EriStep, fb::OverlapStep,
                                   fb::OrthogonalizerStep, fb::CoreHamiltonianStep, fb::GuessStep,
                                   fb::ScfStep, fb::CasscfStep, fb::Nevpt2Step>::create({
      TrialGeometryStep{sys.atoms},
      basis_set,
      fb::SphericalTransformStep{},
      fb::PrimitivePairClassesStep{},
      eri,
      fb::OverlapStep{},
      fb::OrthogonalizerStep{},
      fb::CoreHamiltonianStep{},
      fb::GuessStep{},
      fb::ScfStep{},
      casscf,
      nevpt2,
  })));
  FBF_TRY(ref.run(node));
  const fb::CasscfOutput &cas0 = *ref.get<fb::CasscfOutput>();
  const fb::Nevpt2Output &nev0 = *ref.get<fb::Nevpt2Output>();
  const fb::ScfOutput &rhf = *ref.get<fb::ScfOutput>();
  const fb::BasisSet &basis0 = ref.get<fb::BasisSetOutput>()->basis;
  const auto *const fit = std::get_if<fb::DensityFitted>(&ref.get<fb::EriOutput>()->eri);
  if (fit == nullptr) {
    return fb::err_config("{}: the ERIs are not density-fitted", sys.name);
  }
  if (cas0.gradient_norm >= kConvTolGrad) {
    return fb::err_numerical("{}: the reference CASSCF stopped at ||g|| {:.2e}", sys.name,
                             cas0.gradient_norm);
  }
  if (qd && (nev0.qd_energies.size() != n || (pc && nev0.pc_qd_energies.size() != n))) {
    return fb::err_numerical("{}: no QD-NEVPT2 at the reference: {}", sys.name, nev0.pc_qd_refusal);
  }
  if (!qd && !nev0.pc_roots[0]) {
    return fb::err_numerical("{}: PC refused at the reference: {}", sys.name, nev0.pc_refusals[0]);
  }
  const fb::mcscf::Solution solution0 = FBF_TRY(solution_of(res, basis0, cas0));
  // Each root's analytic DF-SA-CASSCF gradient.
  std::vector<std::vector<double>> casscf_gradient;
  for (std::size_t r = 0; r < n; ++r) {
    const fb::mcscf::GradientTerms g = FBF_TRY(
        fb::mcscf::df_gradient(res, ref.get<fb::MoleculeOutput>()->molecule, basis0,
                               ref.get<fb::PrimitivePairClassesOutput>()->classes, fit->aux,
                               ref.get<fb::CoreHamiltonianOutput>()->h, fit->b, fit->l.data(),
                               solution0, cas0.ncore, cas0.ncas, static_cast<std::int32_t>(r)));
    casscf_gradient.push_back(g.total);
  }
  const auto ncore = static_cast<std::size_t>(cas0.ncore);
  const auto ncas = static_cast<std::size_t>(cas0.ncas);
  const std::size_t nocc = ncore + ncas;
  // The PC cut at every coupling (QD) or of the root (one state).
  const auto dropped_of = [&](const fb::Nevpt2Output &nev) {
    std::vector<std::array<std::int64_t, 8>> dropped;
    if (qd) {
      for (const fb::nevpt2::PcClassEnergies &c : nev.pc_qd_couplings) {
        dropped.push_back(c.dropped);
      }
    } else if (nev.pc_roots[0]) {
      dropped.push_back(nev.pc_roots[0]->dropped);
    }
    return dropped;
  };
  const std::vector<std::array<std::int64_t, 8>> dropped0 = dropped_of(nev0);

  Point at0;
  at0.casscf = cas0.energies;
  if (qd) {
    at0.sc = nev0.qd_energies;
    at0.pc = nev0.pc_qd_energies;
  } else {
    at0.sc = {cas0.energies[0] + nev0.roots[0].total};
    at0.pc = {cas0.energies[0] + nev0.pc_roots[0]->total};
  }
  at0.gradient_norm = cas0.gradient_norm;
  at0.core_gap = smallest_gap(std::span{cas0.mo_energies}.subspan(0, ncore));
  at0.virtual_gap = smallest_gap(std::span{cas0.mo_energies}.subspan(nocc));

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
          casscf_miss = std::max(casscf_miss, std::abs(e_cas[r] - at0.casscf[r]));
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
        casscf_miss = std::abs(e.at("casscf").get<double>() - at0.casscf[0]);
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

  // What every displaced point starts from.
  const CarriedOrbitalsStep carried{
      .c = FBF_TRY(download(res, cas0.c.data(), cas0.c.num_elements())),
      .mo_energies = cas0.mo_energies,
      .n_alpha = rhf.n_alpha,
      .n_beta = rhf.n_beta,
  };
  fb::CasscfStep carried_casscf = casscf;
  carried_casscf.active.clear();

  Point worst; // the worst guard values over every point
  worst.core_gap = at0.core_gap;
  worst.virtual_gap = at0.virtual_gap;
  std::size_t points = 0;
  const auto point = [&](std::span<const double> displacement) -> fb::Result<Point> {
    std::vector<fb::Atom> atoms = sys.atoms;
    for (std::size_t a = 0; a < natom; ++a) {
      atoms[a].position = {atoms[a].position.x + displacement[3 * a],
                           atoms[a].position.y + displacement[(3 * a) + 1],
                           atoms[a].position.z + displacement[(3 * a) + 2]};
    }
    auto chain =
        FBF_TRY((fb::Pipeline<TrialGeometryStep, fb::BasisSetStep, fb::SphericalTransformStep,
                              fb::PrimitivePairClassesStep, fb::EriStep, fb::OverlapStep,
                              fb::CoreHamiltonianStep, CarriedOrbitalsStep, fb::CasscfStep,
                              fb::Nevpt2Step>::create({
            TrialGeometryStep{std::move(atoms)},
            basis_set,
            fb::SphericalTransformStep{},
            fb::PrimitivePairClassesStep{},
            eri,
            fb::OverlapStep{},
            fb::CoreHamiltonianStep{},
            carried,
            carried_casscf,
            nevpt2,
        })));
    FBF_TRY(chain.run(node));
    const fb::CasscfOutput &cas = *chain.get<fb::CasscfOutput>();
    const fb::Nevpt2Output &nev = *chain.get<fb::Nevpt2Output>();
    const fb::BasisSet &basis = chain.get<fb::BasisSetOutput>()->basis;
    Point p;
    p.gradient_norm = cas.gradient_norm;
    if (cas.gradient_norm >= kConvTolGrad) {
      return fb::err_numerical("{}: a displaced CASSCF stopped at ||g|| {:.2e}", sys.name,
                               cas.gradient_norm);
    }
    // The roots, matched by their overlaps with the reference's.
    const fb::mcscf::Solution solution = FBF_TRY(solution_of(res, basis, cas));
    const std::vector<double> s = FBF_TRY(fb::mcscf::state_overlaps(
        res, basis0, solution0, basis, solution, cas0.ncore, cas0.ncas, cas0.nelecas, 1));
    std::vector<std::size_t> pi(n);
    std::vector<double> signs(n);
    std::vector<bool> taken(n, false);
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t best = 0;
      for (std::size_t j = 1; j < n; ++j) {
        if (std::abs(s[(i * n) + j]) > std::abs(s[(i * n) + best])) {
          best = j;
        }
      }
      if (taken[best]) {
        return fb::err_numerical("{}: reference root {} matches no displaced root alone", sys.name,
                                 i);
      }
      taken[best] = true;
      pi[i] = best;
      signs[i] = s[(i * n) + best] < 0.0 ? -1.0 : 1.0;
      p.ci = std::max(p.ci, 1.0 - std::abs(s[(i * n) + best]));
      p.casscf.push_back(cas.energies[best]);
    }
    p.subspace = FBF_TRY(subspace_miss(res, basis0, cas0.c, basis, cas.c, ncore, ncas));
    if (p.ci >= kCiTol || p.subspace >= kSubspaceTol) {
      return fb::err_numerical("{}: a displaced CASSCF's CI overlap off 1 by {:.2e}, its active "
                               "space by {:.2e}",
                               sys.name, p.ci, p.subspace);
    }
    if (qd) {
      if (nev.qd_energies.size() != n) {
        return fb::err_numerical("{}: no SC-QD-NEVPT2 at a displaced point", sys.name);
      }
      p.sc = FBF_TRY(matched_states(nev.qd_energies, nev.qd_vectors, nev0.qd_vectors, pi, signs,
                                    p.sc_vectors));
      if (pc) {
        if (nev.pc_qd_energies.size() != n) {
          return fb::err_numerical("{}: PC-QD-NEVPT2 refused at a displaced point: {}", sys.name,
                                   nev.pc_qd_refusal);
        }
        p.pc = FBF_TRY(matched_states(nev.pc_qd_energies, nev.pc_qd_vectors, nev0.pc_qd_vectors, pi,
                                      signs, p.pc_vectors));
      }
    } else {
      if (!nev.pc_roots[0]) {
        return fb::err_numerical("{}: PC refused at a displaced point: {}", sys.name,
                                 nev.pc_refusals[0]);
      }
      p.sc = {cas.energies[0] + nev.roots[0].total};
      p.pc = {cas.energies[0] + nev.pc_roots[0]->total};
    }
    if (std::max(p.sc_vectors, p.pc_vectors) >= kCiTol) {
      return fb::err_numerical("{}: a displaced H_eff eigenvector's overlap off 1 by {:.2e}",
                               sys.name, std::max(p.sc_vectors, p.pc_vectors));
    }
    // PC's QD couplings are taken over the displaced roots' order: compare the
    // cut as a multiset, which a permutation of the roots leaves alone.
    if (pc) {
      auto dropped = dropped_of(nev);
      auto expected = dropped0;
      std::ranges::sort(dropped);
      std::ranges::sort(expected);
      if (dropped != expected) {
        return fb::err_numerical("{}: PC cuts other modes at a displaced point: a rank change",
                                 sys.name);
      }
    }
    p.core_gap = smallest_gap(std::span{cas.mo_energies}.subspan(0, ncore));
    p.virtual_gap = smallest_gap(std::span{cas.mo_energies}.subspan(nocc));
    ++points;
    return p;
  };

  // The components: three directions, or every Cartesian one.
  const auto run_components = [&](const std::vector<std::vector<double>> &components,
                                  bool cartesian) -> fb::Result<fb::json> {
    const std::size_t ncomp = components.size();
    std::vector<Family> families{{"casscf", {}, {}}, {"sc", {}, {}}};
    if (pc) {
      families.push_back({"pc", {}, {}});
    }
    for (Family &f : families) {
      f.values.assign(n, std::vector<std::vector<std::array<double, 2>>>(
                             ncomp, std::vector<std::array<double, 2>>(steps.size())));
    }
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
          Point p = FBF_TRY(point(dr));
          const double h2 = steps[si] * steps[si];
          for (std::size_t r = 0; r < n; ++r) {
            const double predicted =
                at0.casscf[r] + std::inner_product(casscf_gradient[r].begin(),
                                                   casscf_gradient[r].end(), dr.begin(), 0.0);
            p.prediction = std::max(p.prediction, std::abs(p.casscf[r] - predicted) / h2);
          }
          if (p.prediction >= kPredictionBound) {
            return fb::err_numerical("{}: a displaced CASSCF {:.3f} h^2 off the first-order "
                                     "prediction",
                                     sys.name, p.prediction);
          }
          worst.prediction = std::max(worst.prediction, p.prediction);
          worst.subspace = std::max(worst.subspace, p.subspace);
          worst.ci = std::max(worst.ci, p.ci);
          worst.sc_vectors = std::max(worst.sc_vectors, p.sc_vectors);
          worst.pc_vectors = std::max(worst.pc_vectors, p.pc_vectors);
          worst.gradient_norm = std::max(worst.gradient_norm, p.gradient_norm);
          worst.core_gap = std::min(worst.core_gap, p.core_gap);
          worst.virtual_gap = std::min(worst.virtual_gap, p.virtual_gap);
          const std::size_t side = sign < 0.0 ? 0 : 1;
          for (std::size_t r = 0; r < n; ++r) {
            families[0].values[r][c][si][side] = p.casscf[r];
            families[1].values[r][c][si][side] = p.sc[r];
            if (pc) {
              families[2].values[r][c][si][side] = p.pc[r];
            }
          }
          fb::json energies{{"casscf", p.casscf}, {"sc", p.sc}, {"gradient_norm", p.gradient_norm}};
          if (pc) {
            energies["pc"] = p.pc;
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
    // Each root's analytic CASSCF derivative along each component.
    for (std::size_t r = 0; r < n; ++r) {
      std::vector<double> along(ncomp);
      for (std::size_t c = 0; c < ncomp; ++c) {
        along[c] = std::inner_product(casscf_gradient[r].begin(), casscf_gradient[r].end(),
                                      components[c].begin(), 0.0);
      }
      families[0].analytic.push_back(std::move(along));
    }
    fb::json out;
    fb::json checks;
    for (const Family &f : families) {
      out[f.name] = FBF_TRY(
          differences(f, steps, noise, cartesian,
                      std::format("{} {}", sys.name, cartesian ? "full" : "directional"), checks));
    }
    out["checks"] = checks;
    out["displaced"] = displaced;
    return out;
  };

  const std::vector<std::vector<double>> dirs = directions(sys.atoms);
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
      {"casscf_energies", at0.casscf},
      {"sc_energies", at0.sc},
      {"gradient_norm", at0.gradient_norm},
      {"casscf_gradient", fb::json::array()},
      {"inactive_gaps", {{"core", at0.core_gap}, {"virtual", at0.virtual_gap}}},
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
  fb::json guard{
      {"prediction", worst.prediction},
      {"subspace", worst.subspace},
      {"ci", worst.ci},
      {"sc_vectors", worst.sc_vectors},
      {"gradient_norm", worst.gradient_norm},
      {"inactive_gap", {{"core", worst.core_gap}, {"virtual", worst.virtual_gap}}},
      {"points", points},
  };
  if (pc) {
    guard["pc_vectors"] = worst.pc_vectors;
  }
  doc["guard"] = guard;
  std::println("{}: {} points, worst prediction {:.3f}, subspace {:.1e}, CI {:.1e}, u_K {:.1e} / "
               "{:.1e}, |g| {:.1e} ({:.0f} s)",
               sys.name, points, worst.prediction, worst.subspace, worst.ci, worst.sc_vectors,
               worst.pc_vectors, worst.gradient_norm,
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
    return fb::err_config("usage: {} <repo>/test/gtest/data [case ...]",
                          args.empty() ? "free-breakfast-qd-nevpt2-gradient" : args[0]);
  }
  const std::filesystem::path data{args[1]};
  std::vector<std::string_view> only;
  for (const char *arg : args.subspan(2)) {
    only.emplace_back(arg);
  }
  const auto wanted = [&](std::string_view name) {
    return only.empty() || std::ranges::contains(only, name);
  };
  for (const std::string_view name : only) {
    if (name != kCrossCheck && !std::ranges::contains(kCases, name, &Spec::name)) {
      return fb::err_config("no case {}", name);
    }
  }
  const fb::json qd_doc = FBF_TRY(fb::read_json(data / "qd_nevpt2" / "qd_nevpt2.json"));
  const fb::json single_doc =
      FBF_TRY(fb::read_json(data / "nevpt2_gradient" / "nevpt2_gradient.json"));
  const std::filesystem::path out_dir = data / "qd_nevpt2_gradient";
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
    fb::json doc = FBF_TRY(run_case(node, sys, steps, kCrossCheckOwnNoise, true, true, single_doc));
    FBF_TRY(cross_check(doc, file_case));
    FBF_TRY(fb::write_file(out_dir / std::format("single_state_{}.json", kCrossCheck),
                           fb::json_to_string(doc, 1) + '\n'));
  }
  for (const Spec &spec : kCases) {
    if (!wanted(spec.name)) {
      continue;
    }
    const System sys = FBF_TRY(read_system(qd_doc, spec.name, true));
    const fb::json doc =
        FBF_TRY(run_case(node, sys, spec.steps, spec.noise, spec.pc, spec.full, qd_doc));
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
