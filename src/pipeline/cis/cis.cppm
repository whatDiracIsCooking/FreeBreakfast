/// @file
/// @brief The `Cis` step: the lowest singlet or triplet excitations of CIS
/// (the Tamm-Dancoff approximation) over the @ref ScfOutput's closed-shell
/// RHF, by `davidson_solve` over `cis::Sigma` or, density-fitted,
/// `cis::DfSigma`, from `cis::guess`'s start with `cis::Preconditioner`.
///
/// Device-shaped. The sigma, the guess and the preconditioner are the `cis`
/// modules'; `run` only wires them over the pipeline's products and copies the
/// roots off the device. See README.md.

export module free_breakfast.pipeline.cis;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.cis.sigma; // cis::Options and cis::Spin, in the step
export import free_breakfast.pipeline.scf;

export namespace free_breakfast {

/// @brief One CIS excited state.
struct CisState {
  /// @brief ω, the excitation energy above the reference (hartree).
  double excitation_energy;
  /// @brief The length-gauge oscillator strength (2/3) ω |μ|². Zero for a
  /// triplet (spin-forbidden), and for a singlet in a basis with a shell above
  /// `kDipoleMaxL` (d), where the dipole integrals stop.
  double oscillator_strength;
  /// @brief The transition dipole μ = <0|r|n> (x, y, z, atomic units, origin
  /// 0), √2 Σ_ia X_ia <i|r|a> over the spin-adapted singlet. Its sign is the
  /// amplitudes'. Zero exactly where @ref oscillator_strength is.
  std::array<double, 3> transition_dipole;
  /// @brief The unit-norm amplitudes X_ia, n_occ · n_virt of them, flat
  /// row-major (i, a): element i · n_virt + a, i over the occupied orbitals
  /// and a over the virtual ones, both in the ScfOutput's order. The overall
  /// sign is arbitrary, and within a degenerate level so is the rotation.
  std::vector<double> amplitudes;
};

/// @brief The lowest CIS excitations of one spin over a closed-shell RHF.
/// There is no converged flag: a solve that does not converge is an error
/// (MaxIterationsExceeded), not a product.
struct CisOutput {
  /// @brief The Davidson's Rayleigh-Ritz steps, the converging one included.
  int32_t iterations;
  /// @brief The doubly occupied orbitals, the first n_occ of the ScfOutput's.
  int32_t n_occ;
  /// @brief The virtual orbitals, the rest of the ScfOutput's.
  int32_t n_virt;
  /// @brief The ground-state (RHF) total energy the excitations sit above
  /// (hartree): the ScfOutput's.
  double reference_energy;
  /// @brief The states, ascending in excitation energy.
  std::vector<CisState> states;
};

/// @brief CIS over the @ref ScfOutput's canonical orbitals, on @p res's
/// device.
struct CisStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "cis";

  /// @brief The roots (`n_states`, at least 1), their spin, and the
  /// `davidson_solve` options, at cis::Options' own defaults.
  cis::Options options{};
  /// @brief The Davidson subspace's columns, clamped to
  /// [2 · n_states, n_ov] when the step runs: never past the space itself,
  /// and never below what `davidson_solve` needs.
  int32_t max_subspace = 40;

  /// @brief The @ref cis::Options::n_states lowest excitations of
  /// @ref cis::Options::spin. n_occ is half the molecule's electrons; C, the
  /// orbital energies and the reference energy are the ScfOutput's; σ = A b is
  /// `cis::Sigma` over the EriOutput's J and K classes when it is
  /// IntegralDirect, and `cis::DfSigma` over its B when it is DensityFitted,
  /// at a block of the guess's column count; with @ref options' method
  /// cis::Method::Ris, whatever the EriOutput, `cis::DfSigma` over TDA-ris's
  /// two RIS fits of its own (the J fit's B_vo over RisShells::Sp at
  /// theta_j, the K fit's B_oo and B_vv over RisShells::S at theta_k). The
  /// start is
  /// `cis::guess`'s min(n_ov, 2 · n_states) unit vectors, the preconditioner
  /// `cis::Preconditioner`, clustered over the guess's entries at
  /// @ref options' cluster_gap (none at its default 0), or, when
  /// @ref options' ris is set, `cis::RisPreconditioner` over its own fit of
  /// the RIS basis, whichever EriOutput the sigma has, and `davidson_solve`
  /// runs at @ref options' davidson. A singlet's transition dipoles are then √2 (μ^ov)ᵀ X over
  /// the ov block of the spherical dipole matrices, μ^ov_c = C_virtᵀ μ_c C_occ,
  /// formed once from `dipole` over the PrimitivePairClassesOutput's classes,
  /// and f = (2/3) ω |μ|². A triplet's are zero; so are a singlet's in a basis
  /// with a shell above `kDipoleMaxL` (d), with a warning logged, the energies
  /// still returned. Synchronizes res.stream(): every Davidson iteration reads
  /// back to the host, and so do the roots and their dipoles.
  ///
  /// @return The states; InvalidConfig, before anything is allocated, for
  ///         n_states below 1, a negative or NaN cluster_gap or a positive
  ///         one beside ris, with method Ris a theta_j or theta_k that is
  ///         not finite and positive, a molecule that is not a closed-shell singlet
  ///         (odd electrons or multiplicity other than 1) or whose electrons
  ///         the ScfOutput does not hold as n_alpha = n_beta, no virtual
  ///         orbital, an n_ov past int32_t, or more states than n_ov; or the
  ///         error of the first call that failed: `cis::diagonal`'s,
  ///         `cis::guess`'s, `cis::Sigma::create`'s or
  ///         `cis::DfSigma::create`'s (Unsupported for MO blocks past the
  ///         device's free memory), TDA-ris's fits' (Unsupported for a B
  ///         past it; else ris_basis's, metric's, three_center's,
  ///         density_fit's or b_mo's), `cis::Preconditioner::cluster`'s,
  ///         `cis::RisPreconditioner::create`'s or its correction's, or
  ///         `davidson_solve`'s
  ///         (MaxIterationsExceeded, Divergence), and GpuRuntime for a failed
  ///         copy, dipole launch or gemm.
  [[nodiscard]] Result<CisOutput> run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput,
                                           PrimitivePairClassesOutput, EriOutput, ScfOutput> in,
                                      DeviceResources &res) const;
};

} // namespace free_breakfast
