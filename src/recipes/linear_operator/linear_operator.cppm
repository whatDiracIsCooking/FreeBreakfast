/// @file
/// @brief LinearOperator: a matrix known only by what it does to a block of
/// device vectors, Y = A X.
///
/// The concept an iterative eigensolver applies its operator through, so the
/// solver knows nothing of what the operator is: a dense matrix, or a sigma
/// build (CIS, CASCI) that never forms one.

export module free_breakfast.recipes.linear_operator;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief An operator on length-n vectors applied a block at a time:
/// `op.apply(res, k, x, y)` writes Y = A X for the n x k column-major device
/// blocks @p x and @p y (leading dimension n), and returns a Status.
///
/// n is the operator's own: the concept does not pass it. The work is queued
/// on res.stream() with res's handles, and the operator may take
/// res.device_workspace(), so a caller holds no region of it across apply.
/// apply is non-const and the operator is taken by lvalue reference: a model
/// may keep state, such as an apply count or a cached intermediate.
template <class Op, class T>
concept LinearOperator = requires(Op &op, DeviceResources &res, int32_t k, const T *x, T *y) {
  { op.apply(res, k, x, y) } -> std::convertible_to<Status>;
};

} // namespace free_breakfast
