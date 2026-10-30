/// @file
/// @brief CisStep: the refusals, then Δε, the guess and the preconditioner
/// (diagonal, clustered or RIS-J) over the ScfOutput, the sigma the EriOutput selects (integral-direct or
/// fitted), davidson_solve, a singlet's transition dipoles over the ov block
/// of the dipole matrices, and the roots copied off the device.

module free_breakfast.pipeline.cis;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kSqrt2
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr; // Buffers, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.recipes.davidson;
import free_breakfast.integrals.one_electron.dipole;
import free_breakfast.cis.precondition;
import free_breakfast.cis.ris_precondition;
import free_breakfast.cis.sigma;

#include "diagnostics.h"

namespace free_breakfast {

// The singlets' transition dipoles μⁿ_c = √2 Σ_ia X_ia μ^ov_c,ia, into d, 3 x
// n_states column-major (root n's x, y, z at 3 n): the spherical dipole
// matrices from the classes, their ov blocks μ^ov_c = C_virtᵀ μ_c C_occ (each
// n_virt x n_occ, so element a + n_virt i, the amplitudes' order) side by side
// as the n_ov x 3 matrix M, and d = √2 Mᵀ X over the n_ov x n_states Ritz
// vectors x. Queued on res.stream(), whose pool also orders the scratch's
// frees.
[[nodiscard]] Status cis_transition_dipoles(DeviceResources &res, const BasisSet &basis,
                                            const SphericalTransformOutput &transform,
                                            std::span<const PrimitivePairClass> classes,
                                            const SphericalMatrixAO &c, const int32_t n_occ,
                                            const int32_t n_states, const double *const x,
                                            double *const d) {
  std::array<MatrixAOPair, 3> mu{MatrixAOPair(res, basis, transform.u),
                                 MatrixAOPair(res, basis, transform.u),
                                 MatrixAOPair(res, basis, transform.u)};
  FBF_TRY(dipole(res, classes, mu[0].cartesian(), mu[1].cartesian(), mu[2].cartesian()));
  const auto nao = static_cast<int32_t>(c.n());
  const int32_t n_virt = nao - n_occ;
  const auto n_ov = static_cast<size_t>(n_occ) * static_cast<size_t>(n_virt);
  Buffers::device<double> half(static_cast<size_t>(nao) * static_cast<size_t>(n_occ), res.device());
  Buffers::device<double> ov(3 * n_ov, res.device());
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const double *const c_virt = c.data() + (static_cast<size_t>(n_occ) * static_cast<size_t>(nao));
  for (size_t k = 0; k < mu.size(); ++k) {
    FBF_TRY(mu[k].cartesian_to_spherical(res));
    // μ_c C_occ, nao x n_occ, then C_virtᵀ of it.
    FBF_TRY(gpu_status(gemm(blas, nao, n_occ, nao, &kOne<double>, mu[k].spherical().data(),
                            c.data(), &kZero<double>, half.data())));
    FBF_TRY(
        gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_virt, n_occ, nao, &kOne<double>, c_virt,
                                           half.data(), &kZero<double>, ov.data() + (k * n_ov))));
  }
  return gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, 3, n_states, static_cast<int32_t>(n_ov),
                                            &kSqrt2<double>, ov.data(), x, &kZero<double>, d));
}

[[nodiscard]] Result<CisOutput>
CisStep::run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput,
                  PrimitivePairClassesOutput, EriOutput, ScfOutput> in,
             DeviceResources &res) const {
  const int32_t n_states = options.n_states;
  if (n_states < 1) {
    return err_config("cis: n_states is {}, not at least 1", n_states);
  }
  if (!(options.cluster_gap >= 0.0)) {
    return err_config("cis: cluster_gap is {}, not at least 0", options.cluster_gap);
  }
  if (options.ris && options.cluster_gap > 0.0) {
    return err_config("cis: ris and a cluster_gap of {} each pick the preconditioner; set one",
                      options.cluster_gap);
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

  // davidson_solve needs max_subspace >= 2 n_states, and past n_ov the
  // subspace has nothing left to hold, so the lower bound wins when the two
  // cross.
  const int32_t subspace = std::max(2 * n_states, std::min(max_subspace, n_ov));
  const int32_t guess_count = std::min(n_ov, 2 * n_states);
  Buffers::device<double> start(n * static_cast<size_t>(guess_count), res.device());
  FBF_TRY(cis::guess(res, n_ov, delta.data(), guess_count, start.data()));
  cis::Preconditioner precondition{res, n_ov, delta.data(), n_states};
  // RIS-J (or RIS-JK, with its exchange) in the diagonal correction's place
  // when options.ris asks, over its own fit of the RIS basis, whichever [eri]
  // the sigma has.
  const auto &classes = in.get<PrimitivePairClassesOutput>().classes;
  std::optional<cis::RisPreconditioner> ris;
  if (options.ris) {
    ris = FBF_TRY(cis::RisPreconditioner::create(res, in.get<BasisSetOutput>().basis, classes,
                                                 molecule.atoms(), scf.c, n_occ, delta.data(),
                                                 n_states, *options.ris));
  }
  Buffers::device<double> values(roots, res.device());
  Buffers::device<double> vectors(n * roots, res.device());
  // Otherwise the clusters over the guess's entries, where the subspace
  // starts; at the default gap of zero, none, and the diagonal correction
  // alone.
  const auto solve = [&](auto &sigma) -> Result<DavidsonInfo> {
    const auto davidson = [&](const auto &correct) {
      return davidson_solve<double>(res, sigma, correct, n_ov, n_states, subspace, start.data(),
                                    guess_count, values.data(), vectors.data(), options.davidson);
    };
    if (ris) {
      return davidson(*ris);
    }
    FBF_TRY(precondition.cluster(res, sigma, guess_count, options.cluster_gap));
    return davidson(precondition);
  };
  // The sigma the EriOutput's alternative selects. The fitted one's block is
  // the widest davidson_solve applies: the guess's columns first, then at
  // most n_states corrections, which the guess's min(n_ov, 2 n_states) is
  // never below.
  const EriOutput &eri = in.get<EriOutput>();
  DavidsonInfo info;
  if (const auto *const fit = std::get_if<DensityFitted>(&eri.eri)) {
    cis::DfSigma sigma = FBF_TRY(cis::DfSigma::create(res, fit->b, fit->naux, scf.c, n_occ,
                                                      delta.data(), options.spin, guess_count));
    info = FBF_TRY(solve(sigma));
  } else {
    const auto &direct = std::get<IntegralDirect>(eri.eri);
    cis::Sigma sigma = FBF_TRY(cis::Sigma::create(
        res, in.get<BasisSetOutput>().basis, in.get<SphericalTransformOutput>().u, direct.j_classes,
        direct.k_classes, scf.c, n_occ, delta.data(), options.spin));
    info = FBF_TRY(solve(sigma));
  }

  // A singlet's transition dipoles; a triplet's are zero by spin
  // orthogonality, as are a singlet's past the dipole kernels' d, where the
  // step still returns the energies rather than failing.
  std::vector<double> mu(3 * roots, 0.0);
  const bool past_dipole = std::ranges::any_of(
      classes, [](const PrimitivePairClass &cls) { return cls.lb > kDipoleMaxL; });
  if (options.spin == cis::Spin::Singlet && past_dipole) {
    FBF_LOG_WARN("cis: the basis has a shell above l = {}, where the dipole integrals stop, so "
                 "the transition dipoles and oscillator strengths are left zero",
                 kDipoleMaxL);
  }
  Buffers::device<double> dipoles(mu.size(), res.device());
  if (options.spin == cis::Spin::Singlet && !past_dipole) {
    FBF_TRY(cis_transition_dipoles(res, in.get<BasisSetOutput>().basis,
                                   in.get<SphericalTransformOutput>(), classes, scf.c, n_occ,
                                   n_states, vectors.data(), dipoles.data()));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(mu.data(), dipoles.data(), mu.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
  }

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
    const std::array<double, 3> dipole_r{mu[3 * r], mu[(3 * r) + 1], mu[(3 * r) + 2]};
    const double norm2 =
        (dipole_r[0] * dipole_r[0]) + (dipole_r[1] * dipole_r[1]) + (dipole_r[2] * dipole_r[2]);
    out.states.push_back(CisState{
        .excitation_energy = omega[r],
        .oscillator_strength = 2.0 / 3.0 * omega[r] * norm2,
        .transition_dipole = dipole_r,
        .amplitudes = std::vector<double>(first, first + static_cast<std::ptrdiff_t>(n)),
    });
  }
  return out;
}

} // namespace free_breakfast
