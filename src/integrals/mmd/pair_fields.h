/// @file
/// @brief PairFields: the fields of a primitive-pair record, named once, over
/// whatever holds each field.
///
/// This header is the single definition. The launchers reach it through
/// device_views.h (`#include`); C++ modules reach the same entity by
/// `import free_breakfast.integrals.mmd;`, which includes this header in its
/// global module fragment and re-exports the name (the `atom` pattern).
///
/// One struct serves every holder of a class's records, so the field names are
/// written here and nowhere else:
///
/// | Instance | Holds |
/// |---|---|
/// | `PairFields<std::span<double>, std::span<int32_t>>` | the storage mmd writes into (`PrimitivePairRecords`) |
/// | `PairFields<MirroredBuffer<double>, MirroredBuffer<int32_t>>` | the device owner's buffers (one_electron.upload) |
/// | `PairFields<const double *, const int32_t *>` | the kernels' pointers (`PrimitivePairClassView`) |
///
/// The module's @c apply_pair_fields and @c map_pair_fields go from one to
/// another without naming a field.

#pragma once

namespace free_breakfast {

/// @brief One holder per field of a primitive-pair record: @p Real for each
/// real field, @p Offset for each AO offset.
///
/// A vector is three fields, one per axis, named as the generated kernels name
/// their parameters. Every field holds one value per record of its class.
///
/// Adding a field means adding it here and to @c apply_pair_fields, which
/// static_asserts the count.
template <typename Real, typename Offset>
struct PairFields {
  /// @brief zeta = alpha + beta.
  Real zeta;
  /// @brief P, the product centre: x.
  Real p_x;
  /// @brief P: y.
  Real p_y;
  /// @brief P: z.
  Real p_z;
  /// @brief P - A: x.
  Real pa_x;
  /// @brief P - A: y.
  Real pa_y;
  /// @brief P - A: z.
  Real pa_z;
  /// @brief P - B: x.
  Real pb_x;
  /// @brief P - B: y.
  Real pb_y;
  /// @brief P - B: z.
  Real pb_z;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  Real s;
  /// @brief beta, the ket primitive's exponent.
  Real eb;
  /// @brief The bra shell's first Cartesian AO (@c BasisSet::ao_offset_cart).
  Offset off_a;
  /// @brief The ket shell's first Cartesian AO.
  Offset off_b;
};

} // namespace free_breakfast
