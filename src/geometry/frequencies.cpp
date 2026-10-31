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
import free_breakfast.atom;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.diagonalize;
import :rigid_body;

#include "diagnostics.h"

namespace free_breakfast::geometry {

// The isotope-averaged mass of each element (amu), Z = 1 to 118: PySCF's
// pyscf.data.elements.MASSES (the IUPAC standard atomic weights, and the
// longest-lived isotope's mass where an element has none), copied so the
// frequencies match pyscf.hessian.thermo.harmonic_analysis's.
constexpr std::array<double, 118> kAverageMasses{
    1.008,        4.002602,  6.94,        9.0121831, 10.81,      12.011,    14.007,       15.999,
    18.998403163, 20.1797,   22.98976928, 24.305,    26.9815385, 28.085,    30.973761998, 32.06,
    35.45,        39.948,    39.0983,     40.078,    44.955908,  47.867,    50.9415,      51.9961,
    54.938044,    55.845,    58.933194,   58.6934,   63.546,     65.38,     69.723,       72.63,
    74.921595,    78.971,    79.904,      83.798,    85.4678,    87.62,     88.90584,     91.224,
    92.90637,     95.95,     97.90721,    101.07,    102.9055,   106.42,    107.8682,     112.414,
    114.818,      118.71,    121.76,      127.6,     126.90447,  131.293,   132.90545196, 137.327,
    138.90547,    140.116,   140.90766,   144.242,   144.91276,  150.36,    151.964,      157.25,
    158.92535,    162.5,     164.93033,   167.259,   168.93422,  173.054,   174.9668,     178.49,
    180.94788,    183.84,    186.207,     190.23,    192.217,    195.084,   196.966569,   200.592,
    204.38,       207.2,     208.9804,    208.98243, 209.98715,  222.01758, 223.01974,    226.02541,
    227.02775,    232.0377,  231.03588,   238.02891, 237.04817,  244.06421, 243.06138,    247.07035,
    247.07031,    251.07959, 252.083,     257.09511, 258.09843,  259.101,   262.11,       267.122,
    268.126,      271.134,   270.133,     269.1338,  278.156,    281.165,   281.166,      285.177,
    286.182,      289.19,    289.194,     293.204,   293.208,    294.214,
};

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
    const auto [work] = res.device_regions<double>(n * m);
    FBF_TRY(gpu_status(gemmwich(blas, m32, n32, internal, h_mw.data(), work, h_internal.data())));
  }
  FBF_TRY(diagonalize<double>(res, m32, h_internal.data(), spectrum.data()));

  // 3. The mass-weighted modes B V (n x m), each column one mode.
  const auto [modes] = res.device_regions<double>(n * m);
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

} // namespace free_breakfast::geometry
