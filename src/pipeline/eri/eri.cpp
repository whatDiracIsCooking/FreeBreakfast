/// @file
/// @brief EriStep: J's and K's pair classes; or the aux basis placed, then
/// metric, three_center and density_fit, in that order, on res.stream(),
/// the aux basis and L kept with B.

module free_breakfast.pipeline.eri;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemcpyAsync, gpu_status
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
  BasisSet aux = FBF_TRY(place(stamp.primitive_normalized(), in.get<MoleculeOutput>().molecule));
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();

  // B is the largest thing DF holds, and past the device's memory its
  // allocation would abort (CreatePolicy), not return; three_center then
  // counts its own scratch against what B and L leave.
  const size_t b_bytes = sizeof(double) * naux * (nao * nao + naux);
  if (const size_t budget = FBF_TRY(res.memory_budget()); b_bytes > budget) {
    return err_unsupported("eri: B and L over {} orbital and {} fitting functions need {} bytes "
                           "of device memory, and device {}'s budget is {}",
                           nao, naux, b_bytes, res.dev_idx(), budget);
  }

  // The spherical metric copied into L's own buffer, which density_fit
  // factors in place and the product keeps. The pair and the aux U are freed
  // at the block's end, stream-ordered after the copy, so B and the
  // three-center scratch never share the device with them.
  //
  // So that B can take the block's place, nothing the block allocates
  // outlives it, and its frees complete before B is allocated
  // (pipeline/footprint's README, "Fragmentation headroom"): metric's staging
  // is the device workspace's, held from construction, never a block above
  // the pair; and a chunk whose free is still pending on the stream is not
  // remapped into a new allocation, so B would take new chunks above them.
  Buffers::device<double> l(naux * naux, res.device());
  {
    MatrixAOPair m(res, aux, FBF_TRY(upload_spherical_transform_matrix(res, aux)));
    FBF_TRY(metric(res, aux, m));
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(l.data(), m.spherical().data(), sizeof(double) * naux * naux,
                                       wwr::wwrMemcpyDeviceToDevice, res.stream())));
  }
  FBF_TRY(res.sync());
  Buffers::device<double> b(naux * nao * nao, res.device());
  FBF_TRY(three_center(res, basis, in.get<PrimitivePairClassesOutput>().classes, aux, b.data()));
  FBF_TRY(
      density_fit(res, static_cast<int32_t>(nao), static_cast<int32_t>(naux), l.data(), b.data()));
  return DensityFitted{.b = std::move(b),
                       .naux = static_cast<int32_t>(naux),
                       .aux = std::move(aux),
                       .l = std::move(l)};
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

MemoryFootprint EriStep::footprint(const FootprintDimensions &dims) const {
  const BasisSet &basis = *dims.basis;
  if (mode == EriMode::IntegralDirect) {
    return then(coulomb_pair_classes_footprint(dims.mmd_records),
                ordered_pair_classes_footprint(basis));
  }
  const BasisSet &aux = *dims.aux;
  const auto held = [](size_t bytes) { return MemoryFootprint{.device = {.persistent = bytes}}; };
  // density_fitted's order: L; the aux U and the metric's pair, the metric
  // into them, freed; B; the three-center tensor; the Cholesky fit.
  const MemoryFootprint metric_fp =
      then(then(upload_spherical_transform_matrix_footprint(aux), matrix_ao_pair_footprint(aux)),
           metric_footprint(aux));
  const MemoryFootprint l = held(sizeof(double) * dims.naux * dims.naux);
  const MemoryFootprint b = held(sizeof(double) * dims.naux * dims.nao * dims.nao);
  return then(
      then(then(then(l, drop(metric_fp)), b), three_center_footprint(basis, aux)),
      density_fit_footprint(static_cast<int32_t>(dims.nao), static_cast<int32_t>(dims.naux), 0));
}

} // namespace free_breakfast
