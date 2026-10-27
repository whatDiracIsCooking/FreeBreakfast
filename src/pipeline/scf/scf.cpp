/// @file
/// @brief ScfStep::run: the step's checks and its integral-direct Fock
/// builder, run by the RHF loop of `:loop`.

module free_breakfast.pipeline.scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.recipes.diis;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import :loop;

#include "diagnostics.h"

namespace free_breakfast {

// The integral-direct Fock build, F = H + G(D), G(D) = J(D) - K(D)/2, J and K
// built from a Cartesian density, and its policy: which builds are full. A full
// build makes G of D itself; an incremental one adds G(D - d_built) to the F of
// d_built, which G's linearity makes the same matrix. J's and K's screens scale
// with the density they read, and the difference shrinks as the SCF converges,
// so they skip more of it. The J and K classes are made from the shared
// PrimitivePairClassesOutput and never published.
struct DirectFockBuilder {
  std::vector<CoulombPairClass> j_classes;
  std::vector<OrderedPairClass> k_classes;
  SphericalMatrixAO d_built; // the density f is the Fock matrix of
  MatrixAOPair dd;           // what J and K read: D on a full build, D - d_built otherwise
  MatrixAOPair g;            // J - K/2 of dd, built Cartesian and carried onto the spherical layout
  CartesianMatrixAO k;       // K, before it is folded into g
  double tau_j;
  int32_t fock_rebuild_interval;
  double commutator_tolerance;
  bool built = false;             // any build yet
  bool full = false;              // the last build was
  int32_t incremental_builds = 0; // since the last full build
  // Set once the commutator is under its tolerance, and every build from then
  // on is full. By then D - d_built is small enough that J's and K's screening
  // error is a good part of its G, and an incremental F gains a fresh error
  // every build: the energy wanders by about that much each iteration and
  // never meets energy_tolerance (H2O/STO-3G: 4.5e-10 Eh a build). A full F's
  // error is a fixed function of D, so the loop keeps a fixed point; and
  // convergence needs a full build anyway.
  bool full_from_now = false;

  // The orbitals are not read: J and K are both built from D.
  [[nodiscard]] Result<bool> build(DeviceResources &res, const SphericalMatrixAO &d,
                                   const SphericalMatrixAO * /*c*/, const SphericalMatrixAO &h,
                                   SphericalMatrixAO &f) {
    // Only a full build may converge: an incremental F carries the screening
    // error of every build since the last full one.
    full = !built || full_from_now || incremental_builds + 1 >= fock_rebuild_interval;
    built = true;
    incremental_builds = full ? 0 : incremental_builds + 1;
    const wwr::wwrblasHandle_t blas = res.blas().get();
    const auto n = static_cast<int32_t>(d.n());
    const size_t bytes = d.num_elements() * sizeof(double);
    if (full) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(dd.spherical().data(), d.data(), bytes,
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
    } else {
      constexpr double kMinusOne = -1.0;
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n,
                                                    n, &kOne<double>, d.data(), n, &kMinusOne,
                                                    d_built.data(), n, dd.spherical().data(), n)));
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_built.data(), d.data(), bytes,
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    FBF_TRY(dd.spherical_to_cartesian(res));
    FBF_TRY(coulomb(res, j_classes, dd.cartesian(), g.cartesian(), tau_j));
    FBF_TRY(exchange(res, k_classes, dd.cartesian(), k));
    constexpr double kMinusHalf = -0.5;
    FBF_TRY(
        gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(k.num_elements()),
                                              &kMinusHalf, k.data(), 1, g.cartesian().data(), 1)));
    FBF_TRY(g.cartesian_to_spherical(res));
    if (full) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(f.data(), h.data(), bytes,
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
    }
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(f.num_elements()),
                                                  &kOne<double>, g.spherical().data(), 1, f.data(),
                                                  1)));
    return full;
  }

  // Switching off an incremental build, the history is as stale as its
  // residuals are small: it fits the drifted F's, not the full ones to come.
  // Dropped once, after that step's extrapolation, and not earlier: a reset
  // further from convergence restarts DIIS where it is still needed
  // (gly5/cc-pVDZ: 3 full builds to finish against 4 kept, 5 and 6 reset at
  // 10x and 100x commutator_tolerance).
  [[nodiscard]] bool reset_diis(double norm) {
    const bool switching = !full_from_now && norm < commutator_tolerance;
    full_from_now = full_from_now || switching;
    return switching && !full;
  }
};

[[nodiscard]] Result<ScfOutput> ScfStep::run(
    View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput,
         OverlapOutput, OrthogonalizerOutput, CoreHamiltonianOutput, GuessOutput> in,
    DeviceResources &res) const {
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const std::vector<double> &guess = in.get<GuessOutput>().d;

  if (max_iterations < 1) {
    return err_config("[scf] max_iterations is {}, below 1", max_iterations);
  }
  if (!(energy_tolerance >= 0.0) || !(commutator_tolerance >= 0.0)) {
    return err_config("[scf] energy_tolerance ({}) and commutator_tolerance ({}) must be "
                      "non-negative",
                      energy_tolerance, commutator_tolerance);
  }
  // Written so that NaN fails too: a NaN tau would never screen, silently.
  if (!(tau_j >= 0.0)) {
    return err_config("[scf] tau_j is {}, but must be 0 or above", tau_j);
  }
  if (fock_rebuild_interval < 1) {
    return err_config("[scf] fock_rebuild_interval is {}, below 1", fock_rebuild_interval);
  }
  if (diis_history < 1 || std::cmp_greater(diis_history, kDiisMaxHistoryLimit)) {
    return err_config("[scf] diis_history is {}, outside 1..{}", diis_history,
                      kDiisMaxHistoryLimit);
  }
  const int32_t electrons = molecule.electron_count();
  if (molecule.multiplicity() != 1 || electrons % 2 != 0) {
    return err_unsupported("scf: only closed-shell RHF is implemented, and the molecule has {} "
                           "electrons at multiplicity {}",
                           electrons, molecule.multiplicity());
  }
  const uint32_t n = basis.nao_sphe();
  const int32_t nocc = electrons / 2;
  if (electrons < 0 || std::cmp_greater(nocc, n)) {
    return err_config("scf: {} electrons do not fit {} spatial orbitals", electrons, n);
  }
  if (guess.size() != std::size_t{n} * n) {
    return err_config("scf: the guess has {} elements, not {} squared", guess.size(), n);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const std::shared_ptr<const Buffers::device<double>> &u = in.get<SphericalTransformOutput>().u;
  const std::vector<PrimitivePairClass> &pairs = in.get<PrimitivePairClassesOutput>().classes;
  std::vector<CoulombPairClass> j_classes = FBF_TRY(coulomb_pair_classes(res, pairs));
  std::vector<OrderedPairClass> k_classes = FBF_TRY(ordered_pair_classes(res, basis, pairs));
  DirectFockBuilder builder{
      .j_classes = std::move(j_classes),
      .k_classes = std::move(k_classes),
      .d_built = SphericalMatrixAO(res, basis),
      .dd = MatrixAOPair(res, basis, u),
      .g = MatrixAOPair(res, basis, u),
      .k = CartesianMatrixAO(res, basis),
      .tau_j = tau_j,
      .fock_rebuild_interval = fock_rebuild_interval,
      .commutator_tolerance = commutator_tolerance,
  };
  return rhf_loop(res, builder, molecule, basis, in.get<OverlapOutput>().s,
                  in.get<OrthogonalizerOutput>().x, in.get<CoreHamiltonianOutput>().h, guess,
                  max_iterations, energy_tolerance, commutator_tolerance, diis_history, molden);
}

} // namespace free_breakfast
