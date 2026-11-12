/// @file
/// @brief harmonic_analysis: the harmonic frequencies and normal modes of a
/// Cartesian Hessian, the rigid-body motions projected out.

module free_breakfast.geometry;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.atom;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.diagonalize;
import :rigid_body;
import :masses;

#include "diagnostics.h"

namespace free_breakfast::geometry {

// CODATA 2018: the dalton in electron masses, and the hartree in cm^-1. An
// eigenvalue of the Hessian mass-weighted in amu, over kElectronMassesPerDalton,
// is omega^2 in atomic units, and omega in hartree times
// kWavenumbersPerHartree is the wavenumber.
constexpr double kElectronMassesPerDalton = 1822.888486209;
constexpr double kWavenumbersPerHartree = 219474.6313632;

Result<Vibrations> harmonic_analysis(DeviceResources &res, std::span<const Atom> atoms,
                                     std::span<const double> hessian) {
  const size_t points = atoms.size();
  const size_t n = 3 * points;
  if (points == 0 || hessian.size() != n * n) {
    return err_config("geometry::harmonic_analysis: needs at least one atom and a 3N x 3N "
                      "Hessian, got {} atoms and {} elements",
                      points, hessian.size());
  }
  if (!std::ranges::all_of(hessian, [](double value) { return std::isfinite(value); })) {
    return err_config("geometry::harmonic_analysis: the Hessian is not finite");
  }
  // The coordinates, flat, and sqrt(m) per atom: the mass weighting.
  std::vector<double> x(n);
  std::vector<double> sqrt_mass(points);
  for (size_t i = 0; i < points; ++i) {
    const auto z = static_cast<size_t>(std::to_underlying(atoms[i].element));
    if (z < 1 || z > kAverageMasses.size()) {
      return err_config("geometry::harmonic_analysis: atom {} has Z = {}, not 1 to {}", i, z,
                        kAverageMasses.size());
    }
    sqrt_mass[i] = std::sqrt(kAverageMasses[z - 1]);
    x[3 * i] = atoms[i].position.x;
    x[3 * i + 1] = atoms[i].position.y;
    x[3 * i + 2] = atoms[i].position.z;
  }
  if (!std::ranges::all_of(x, [](double value) { return std::isfinite(value); })) {
    return err_config("geometry::harmonic_analysis: an atom's position is not finite");
  }

  // The rigid-body motions in mass-weighted coordinates, orthonormal: 6, 5
  // for a linear molecule, 3 for one atom. m internal modes remain.
  const std::vector<std::vector<double>> rigid = rigid_body_basis(x, sqrt_mass);
  const size_t k = rigid.size();
  const size_t m = n - k;
  Vibrations vibrations;
  if (m == 0) {
    return vibrations;
  }

  // Column-major n x n on the host: P = I - Q Q^T, the projector onto the
  // complement of the rigid motions, and H_mw = M^-1/2 H M^-1/2.
  std::vector<double> projector(n * n, 0.0);
  std::vector<double> weighted(n * n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      double value = i == j ? 1.0 : 0.0;
      for (const std::vector<double> &u : rigid) {
        value -= u[i] * u[j];
      }
      projector[i + j * n] = value;
      weighted[i + j * n] = hessian[i * n + j] / (sqrt_mass[i / 3] * sqrt_mass[j / 3]);
    }
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const auto n32 = static_cast<int32_t>(n);
  const auto m32 = static_cast<int32_t>(m);
  // Held across diagonalize, which takes the workspace for syevd's work, so
  // these are the call's own: the projector (then its eigenvectors), H_mw,
  // the projected Hessian (then its eigenvectors) and the two spectra.
  Buffers::device<double> basis(n * n, res.device());
  Buffers::device<double> h_mw(n * n, res.device());
  Buffers::device<double> h_internal(m * m, res.device());
  Buffers::device<double> projector_spectrum(n, res.device());
  Buffers::device<double> spectrum(m, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(basis.data(), projector.data(), n * n * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h_mw.data(), weighted.data(), n * n * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));

  // 1. P's eigenvalues are k zeros and m ones, ascending, so its last m
  //    eigenvectors B are an orthonormal basis of the internal motions:
  //    PySCF's bvec.
  FBF_TRY(diagonalize<double>(res, n32, basis.data(), projector_spectrum.data()));
  const double *const internal = basis.data() + (k * n);

  // 2. B^T H_mw B (m x m), and its eigendecomposition V diag(lambda) V^T.
  const wwr::wwrblasHandle_t blas = res.blas().get();
  {
    const auto [work] = FBF_TRY(res.device_regions<double>("geometry::harmonic_analysis", n * m));
    FBF_TRY(gpu_status(gemmwich(blas, m32, n32, internal, h_mw.data(), work, h_internal.data())));
  }
  FBF_TRY(diagonalize<double>(res, m32, h_internal.data(), spectrum.data()));

  // 3. The mass-weighted modes B V (n x m), each column one mode.
  const auto [modes] = FBF_TRY(res.device_regions<double>("geometry::harmonic_analysis", n * m));
  {
    const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t,
                                            CreatePolicy<wwr::wwrblasStatus_t>>
        host_scalars{blas, wwr::WWRBLAS_POINTER_MODE_HOST};
    FBF_TRY(gpu_status(gemm(blas, n32, m32, m32, &kOne<double>, internal, h_internal.data(),
                            &kZero<double>, modes)));
  }
  std::vector<double> eigenvalues(m);
  vibrations.modes.resize(n * m);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(eigenvalues.data(), spectrum.data(), m * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(vibrations.modes.data(), modes, n * m * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());

  // lambda (hartree / bohr^2 / amu) to cm^-1, an imaginary frequency as
  // minus its magnitude; each mode back to Cartesian displacements.
  vibrations.frequencies.resize(m);
  for (size_t j = 0; j < m; ++j) {
    const double omega =
        std::sqrt(std::abs(eigenvalues[j]) / kElectronMassesPerDalton) * kWavenumbersPerHartree;
    vibrations.frequencies[j] = eigenvalues[j] < 0.0 ? -omega : omega;
    for (size_t i = 0; i < n; ++i) {
      vibrations.modes[j * n + i] /= sqrt_mass[i / 3];
    }
  }
  return vibrations;
}

MemoryFootprint harmonic_analysis_footprint(std::span<const Atom> atoms,
                                            std::span<const size_t> syevd_lwork) {
  const size_t n = 3 * atoms.size();
  if (n == 0) {
    return {};
  }
  std::vector<double> x(n);
  for (size_t i = 0; i < atoms.size(); ++i) {
    x[3 * i] = atoms[i].position.x;
    x[3 * i + 1] = atoms[i].position.y;
    x[3 * i + 2] = atoms[i].position.z;
  }
  // The rigid motions' count does not depend on the masses: unweighted.
  const size_t m = n - rigid_body_basis(x, {}).size();
  if (m == 0) {
    return {};
  }
  constexpr size_t kAlignment = DeviceResources::kWorkspaceAlignment;
  MemoryFootprint out = diagonalize_footprint<double>(syevd_lwork[n]);
  out.device.workspace = std::max({out.device.workspace,
                                   diagonalize_footprint<double>(syevd_lwork[m]).device.workspace,
                                   math::align_up(n * m * sizeof(double), kAlignment)});
  out.device.scratch = sizeof(double) * ((2 * n * n) + (m * m) + n + m);
  return out;
}

} // namespace free_breakfast::geometry
