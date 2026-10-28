/// @file
/// @brief EriStep: J's and K's pair classes; or the aux basis placed, then
/// metric, three_center and density_fit, in that order, on res.stream().

module free_breakfast.pipeline.eri;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemGetInfo, gpu_status
import free_breakfast.core.resources;
import free_breakfast.io.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.density_fitting;

#include "diagnostics.h"

namespace free_breakfast {

// The aux basis an empty EriStep::aux_basis stands for: the orbital set's
// acCD-6-6 fitting set. Keyed off BasisSet::name(), which only a compiled-in
// set has, so a basis read from a path gets none whatever its document claims
// (the SAD stamps' rule).
[[nodiscard]] Result<std::string> default_aux_basis(const BasisSet &basis) {
  if (basis.name().empty()) {
    return err_config("[eri] mode = 'density_fitting' has no default aux_basis for a basis read "
                      "from a file: set aux_basis");
  }
  std::string aux = basis.name() + "-acCD-6-6";
  if (!is_builtin_aux_basis(aux)) {
    return err_config("[eri] mode = 'density_fitting' has no default aux_basis for {}, which has "
                      "no carried {}: set aux_basis",
                      basis.name(), aux);
  }
  return aux;
}

// B over the aux basis named @p spec.
[[nodiscard]] Result<DensityFitted>
density_fitted(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput> in,
               DeviceResources &res, std::string_view spec) {
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  // Placed as the orbital basis is: the fitting sets' coefficients are for
  // normalized primitives too.
  const BasisStamp stamp = FBF_TRY(resolve_aux_basis(spec));
  const BasisSet aux =
      FBF_TRY(place(stamp.primitive_normalized(), in.get<MoleculeOutput>().molecule));
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();

  // B is the largest thing DF holds, and past the device's memory its
  // allocation would abort (CreatePolicy), not return; three_center then
  // counts its own scratch against what B leaves.
  const size_t b_bytes = sizeof(double) * naux * nao * nao;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (b_bytes > free_bytes) {
    return err_unsupported("eri: B over {} orbital and {} fitting functions needs {} bytes of "
                           "device memory, and device {} has {} free",
                           nao, naux, b_bytes, res.dev_idx(), free_bytes);
  }

  MatrixAOPair m(res, aux, FBF_TRY(upload_spherical_transform_matrix(res, aux)));
  FBF_TRY(metric(res, aux, m));
  Buffers::device<double> b(naux * nao * nao, res.device());
  FBF_TRY(three_center(res, basis, in.get<PrimitivePairClassesOutput>().classes, aux, b.data()));
  // L overwrites the metric, which is freed at return, stream-ordered after
  // the trsm that reads it.
  FBF_TRY(density_fit(res, static_cast<int32_t>(nao), static_cast<int32_t>(naux),
                      m.spherical().data(), b.data()));
  return DensityFitted{.b = std::move(b), .naux = static_cast<int32_t>(naux)};
}

Result<EriOutput> EriStep::run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput> in,
                               DeviceResources &res) const {
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const ScopedDeviceIndex on{res.dev_idx()};
  if (mode == EriMode::IntegralDirect) {
    if (!aux_basis.empty()) {
      return err_config("[eri] aux_basis needs mode = 'density_fitting'");
    }
    const std::vector<PrimitivePairClass> &pairs = in.get<PrimitivePairClassesOutput>().classes;
    return EriOutput{IntegralDirect{
        .j_classes = FBF_TRY(coulomb_pair_classes(res, pairs)),
        .k_classes = FBF_TRY(ordered_pair_classes(res, basis, pairs)),
    }};
  }
  const std::string spec = aux_basis.empty() ? FBF_TRY(default_aux_basis(basis)) : aux_basis;
  return EriOutput{FBF_TRY(density_fitted(in, res, spec))};
}

} // namespace free_breakfast
