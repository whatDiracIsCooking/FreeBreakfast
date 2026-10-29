/// @file
/// @brief CisStep: the refusals, then Δε, the sigma, the guess and the
/// preconditioner over the ScfOutput, davidson_solve, and the roots copied
/// off the device.

module free_breakfast.pipeline.cis;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.davidson;
import free_breakfast.cis.precondition;
import free_breakfast.cis.sigma;

#include "diagnostics.h"

namespace free_breakfast {

[[nodiscard]] Result<CisOutput> CisStep::run(
    View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, EriOutput, ScfOutput> in,
    DeviceResources &res) const {
  const int32_t n_states = options.n_states;
  if (n_states < 1) {
    return err_config("cis: n_states is {}, not at least 1", n_states);
  }
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const ScfOutput &scf = in.get<ScfOutput>();
  const int32_t n_electrons = molecule.electron_count();
  if (molecule.multiplicity() != 1 || n_electrons % 2 != 0) {
    return err_config("cis: needs a closed-shell singlet reference, and the molecule has {} "
                      "electrons at multiplicity {}",
                      n_electrons, molecule.multiplicity());
  }
  const int32_t n_occ = n_electrons / 2;
  if (scf.n_alpha != n_occ || scf.n_beta != n_occ) {
    return err_config("cis: needs the closed-shell RHF of the molecule's {} electrons, and the "
                      "ScfOutput has {} alpha and {} beta",
                      n_electrons, scf.n_alpha, scf.n_beta);
  }
  const auto nmo = static_cast<int64_t>(scf.mo_energies.size());
  const int64_t n_virt = nmo - n_occ;
  const int64_t n_ov_wide = n_occ * n_virt;
  if (n_virt < 1 || !std::in_range<int32_t>(n_ov_wide)) {
    return err_config("cis: {} occupied of {} orbitals leave n_ov = {}, not in [1, int32_t]", n_occ,
                      nmo, n_ov_wide);
  }
  const auto n_ov = static_cast<int32_t>(n_ov_wide);
  if (n_states > n_ov) {
    return err_config("cis: n_states is {}, past the {} single excitations of {} occupied and {} "
                      "virtual orbitals",
                      n_states, n_ov, n_occ, n_virt);
  }
  const auto *const direct = std::get_if<IntegralDirect>(&in.get<EriOutput>().eri);
  if (direct == nullptr) {
    return err_unsupported("cis: runs over integral-direct ERIs only so far, and the EriOutput is "
                           "density-fitted: set [eri] mode = 'integral_direct'");
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const auto n = static_cast<size_t>(n_ov);
  const auto roots = static_cast<size_t>(n_states);

  // Δε over the ScfOutput's orbital energies, which the sigma, the guess and
  // the preconditioner all borrow.
  Buffers::device<double> energies(scf.mo_energies.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(energies.data(), scf.mo_energies.data(),
                                         scf.mo_energies.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  Buffers::device<double> delta(n, res.device());
  FBF_TRY(cis::diagonal(res, n_occ, static_cast<int32_t>(n_virt), energies.data(), delta.data()));
  cis::Sigma sigma = FBF_TRY(cis::Sigma::create(
      res, in.get<BasisSetOutput>().basis, in.get<SphericalTransformOutput>().u, direct->j_classes,
      direct->k_classes, scf.c, n_occ, delta.data(), options.spin));

  // davidson_solve needs max_subspace >= 2 n_states, and past n_ov the
  // subspace has nothing left to hold, so the lower bound wins when the two
  // cross.
  const int32_t subspace = std::max(2 * n_states, std::min(max_subspace, n_ov));
  const int32_t guess_count = std::min(n_ov, 2 * n_states);
  Buffers::device<double> start(n * static_cast<size_t>(guess_count), res.device());
  FBF_TRY(cis::guess(res, n_ov, delta.data(), guess_count, start.data()));
  const cis::Preconditioner precondition{res, n_ov, delta.data(), n_states};
  Buffers::device<double> values(roots, res.device());
  Buffers::device<double> vectors(n * roots, res.device());
  const DavidsonInfo info = FBF_TRY(
      davidson_solve<double>(res, sigma, precondition, n_ov, n_states, subspace, start.data(),
                             guess_count, values.data(), vectors.data(), options.davidson));

  // The Ritz vectors are column-major n_ov x n_states, so each root's column
  // is already its amplitudes in (i, a) row-major order.
  std::vector<double> omega(roots);
  std::vector<double> x(n * roots);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(omega.data(), values.data(), roots * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(x.data(), vectors.data(), x.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());

  CisOutput out{
      .iterations = info.iterations,
      .n_occ = n_occ,
      .n_virt = static_cast<int32_t>(n_virt),
      .reference_energy = scf.energy,
      .states = {},
  };
  out.states.reserve(roots);
  for (size_t r = 0; r < roots; ++r) {
    const auto first = x.begin() + static_cast<std::ptrdiff_t>(r * n);
    out.states.push_back(CisState{
        .excitation_energy = omega[r],
        .oscillator_strength = 0.0,
        .transition_dipole = {0.0, 0.0, 0.0},
        .amplitudes = std::vector<double>(first, first + static_cast<std::ptrdiff_t>(n)),
    });
  }
  return out;
}

} // namespace free_breakfast
