/// @file
/// @brief rhf_dipole_derivative: the closed-shell RHF dipole derivative from
/// the CPHF U^x, its explicit term over the dipole integrals' derivative and
/// its response terms as gemms over the MO blocks of mu; and
/// infrared_intensities, its projection onto the normal modes.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kTwo, kAvogadro, kEpsilon0, kElementaryCharge
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, geam, wwrMemcpyAsync, wwrMemGetInfo
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.dipole;
import free_breakfast.integrals.one_electron.dipole_gradient;
import free_breakfast.integrals.one_electron.gradient;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast::hessian {

// The speed of light (m / s, exact) and the dalton (kg, CODATA 2018), which
// core/constants does not carry: only the intensity's unit needs them.
constexpr double kSpeedOfLight = 299792458.0;
constexpr double kDalton = 1.66053906660e-27;
// km / mol per (e / amu^1/2)^2: N_A / (12 eps_0 c^2) e^2 / u, 974.88.
constexpr double kKmPerMolPerSquaredCharge =
    kAvogadro<double> / (12.0 * kEpsilon0<double> * kSpeedOfLight * kSpeedOfLight) *
    kElementaryCharge<double> * kElementaryCharge<double> / kDalton / kKilo<double>;

Result<std::vector<double>> rhf_dipole_derivative(DeviceResources &res, const Molecule &molecule,
                                                  const BasisSet &basis,
                                                  const std::span<const PrimitivePairClass> classes,
                                                  const Buffers::device<double> &c,
                                                  const int32_t nocc,
                                                  const Buffers::device<double> &u) {
  const size_t nao = basis.nao_sphe();
  const size_t n_atom = basis.natoms();
  const size_t n_x = 3 * n_atom;
  const size_t nmo = nao == 0 ? 0 : c.num_elements() / nao;
  const auto occ = static_cast<size_t>(std::max(nocc, 0));
  if (molecule.natoms() != n_atom || nao == 0 || c.num_elements() != nao * nmo || nocc < 1 ||
      occ >= nmo || u.num_elements() != (nmo - occ) * occ * n_x) {
    return err_config("hessian::rhf_dipole_derivative: needs a molecule of the basis's {} atoms, "
                      "C nao nmo long, 1 <= nocc < nmo and U (nmo - nocc) nocc 3 n_atom long ({} "
                      "atoms, nao {}, C {} doubles, nocc {}, U {} doubles)",
                      n_atom, molecule.natoms(), nao, c.num_elements(), nocc, u.num_elements());
  }
  const size_t nvir = nmo - occ;
  const size_t nn = nao * nao;
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (n_x * nn > kInt32Max || 3 * nmo * occ > kInt32Max) {
    return err_unsupported("hessian::rhf_dipole_derivative: 3 n_atom nao^2 = {} exceeds int32_t",
                           n_x * nn);
  }
  // Held at the peak: S^x, the three P^c and the MO blocks of mu, beside
  // overlap_derivative's own scratch (counted as one more S^x).
  const size_t held = (3 * nn) + (3 * nmo * occ) + (3 * nvir * occ) + (nao * occ) + (n_x * 3);
  const size_t bytes = sizeof(double) * (held + (2 * n_x * nn));
  const ScopedDeviceIndex on{res.dev_idx()};
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (bytes > free_bytes) {
    return err_unsupported("hessian::rhf_dipole_derivative: the overlap derivative and dipole "
                           "blocks of {} coordinates over {} orbital functions need {} bytes of "
                           "device memory, and device {} has {} free",
                           n_x, nao, bytes, res.dev_idx(), free_bytes);
  }

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto n_ao = static_cast<int32_t>(nao);
  const auto n_mo = static_cast<int32_t>(nmo);
  const auto n_vir = static_cast<int32_t>(nvir);
  const auto n_coord = static_cast<int32_t>(n_x);
  const double *const c_occ = c.data();

  // The explicit term's contraction, G[c, A, k] = Sum_mn D_mn d mu^c_mn / dR_{A,k},
  // D = 2 C_o C_o^T.
  std::vector<double> explicit_term;
  {
    SphericalMatrixAO d(res, basis);
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, n_ao, n_ao, nocc, &kTwo<double>, c_occ, c_occ, &kZero<double>, d.data())));
    explicit_term = FBF_TRY(dipole_gradient(res, basis, classes, d));
  }

  // mu^c's MO blocks C^T mu^c C_o (nmo x nocc each, side by side), and
  // P^c = C_o mu^c_oo C_o^T (nao x nao each, side by side).
  Buffers::device<double> mo(3 * nmo * occ, res.device());
  Buffers::device<double> vo(3 * nvir * occ, res.device());
  Buffers::device<double> p(3 * nn, res.device());
  Buffers::device<double> half(nao * occ, res.device());
  Buffers::device<double> out(n_x * 3, res.device());
  {
    const auto transform = FBF_TRY(upload_spherical_transform_matrix(res, basis));
    std::array<MatrixAOPair, 3> mu{MatrixAOPair(res, basis, transform),
                                   MatrixAOPair(res, basis, transform),
                                   MatrixAOPair(res, basis, transform)};
    FBF_TRY(dipole(res, classes, mu[0].cartesian(), mu[1].cartesian(), mu[2].cartesian()));
    for (size_t comp = 0; comp < 3; ++comp) {
      double *const mo_c = mo.data() + (comp * nmo * occ);
      FBF_TRY(mu[comp].cartesian_to_spherical(res));
      FBF_TRY(gpu_status(gemm(blas, n_ao, nocc, n_ao, &kOne<double>, mu[comp].spherical().data(),
                              c_occ, &kZero<double>, half.data())));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_mo, nocc, n_ao, &kOne<double>, c.data(),
                                                 half.data(), &kZero<double>, mo_c)));
      // C_o mu_oo (mu_oo the first nocc rows of the slab), then times C_o^T.
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_ao, nocc, nocc, &kOne<double>, c_occ, n_ao,
          mo_c, n_mo, &kZero<double>, half.data(), n_ao)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, n_ao, n_ao, nocc, &kOne<double>, half.data(), c_occ, &kZero<double>,
          p.data() + (comp * nn))));
    }
  }
  // The vo blocks of the three slabs are one nvir x (3 nocc) matrix of leading
  // dimension nmo: packed, they are three columns of U's (n_vir n_occ) rows.
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_vir, 3 * nocc, &kOne<double>, mo.data() + occ,
      n_mo, &kZero<double>, mo.data() + occ, n_mo, vo.data(), n_vir)));
  // out = -4 U^T mu_vo + 2 S^x . P, 3 n_atom x 3 column-major: S^x's slabs
  // are the columns of one nao^2 x 3 n_atom matrix, and S^x and P^c are
  // symmetric, so the dot of slab x with P^c is Sum_ij S^x_ij mu^c_ij.
  const auto n_pairs = static_cast<int32_t>(nvir * occ);
  constexpr double kNegFour = -4.0;
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_coord, 3, n_pairs, &kNegFour, u.data(),
                                             vo.data(), &kZero<double>, out.data())));
  {
    const Buffers::device<double> sx = FBF_TRY(overlap_derivative(res, basis, classes));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_coord, 3, static_cast<int32_t>(nn),
                                               &kTwo<double>, sx.data(), p.data(), &kOne<double>,
                                               out.data())));
    std::vector<double> response(n_x * 3);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(response.data(), out.data(),
                                           sizeof(double) * response.size(),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());

    // dmu_c/dx = -G[c, A, k] + response (x, c) + Z_A delta_kc, row x.
    const std::vector<Atom> &atoms = molecule.atoms();
    std::vector<double> derivative(n_x * 3);
    for (size_t x = 0; x < n_x; ++x) {
      const size_t atom = x / 3;
      const size_t k = x % 3;
      for (size_t comp = 0; comp < 3; ++comp) {
        double v = response[x + (n_x * comp)] - explicit_term[(3 * ((n_atom * comp) + atom)) + k];
        if (comp == k) {
          v += static_cast<double>(std::to_underlying(atoms[atom].element));
        }
        derivative[(3 * x) + comp] = v;
      }
    }
    return derivative;
  }
}

Result<std::vector<double>> infrared_intensities(const std::span<const double> modes,
                                                 const std::span<const double> dipole_derivative) {
  const size_t n_x = dipole_derivative.size() / 3;
  if (n_x == 0 || dipole_derivative.size() % 9 != 0 || modes.size() % n_x != 0) {
    return err_config("hessian::infrared_intensities: needs a 3 n_atom x 3 dipole derivative and "
                      "modes of 3 n_atom each ({} and {} doubles)",
                      dipole_derivative.size(), modes.size());
  }
  std::vector<double> intensities(modes.size() / n_x);
  for (size_t q = 0; q < intensities.size(); ++q) {
    std::array<double, 3> dq{};
    for (size_t x = 0; x < n_x; ++x) {
      for (size_t comp = 0; comp < 3; ++comp) {
        dq[comp] += modes[(n_x * q) + x] * dipole_derivative[(3 * x) + comp];
      }
    }
    intensities[q] =
        kKmPerMolPerSquaredCharge * ((dq[0] * dq[0]) + (dq[1] * dq[1]) + (dq[2] * dq[2]));
  }
  return intensities;
}

} // namespace free_breakfast::hessian
