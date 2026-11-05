// Compile-time checks of the concepts src/recipes exports: LinearOperator,
// DavidsonPreconditioner and UnitaryCostFunction, each against a model that
// satisfies it and models that miss it in one way each. Every check is a
// static_assert, so this TU fails the build rather than a run, and declares
// no case: its models are declared and never defined, because a concept only
// reads their signatures.
//
// blas/'s BlasOp and SandwichOp are not here: they live in the private
// partition blas:shared and are not exported, so no importer can name them.
// What they decide is checked through the exported functions they constrain,
// in gemm_gpu.cpp (kGemmTakes, kRectGemmTakes), gemmwich_gpu.cpp and
// gmmwich_gpu.cpp (kSquareTakes, kGeneralTakes, kComplexTakes): every op each
// takes, and OP_C refused on a real T and OP_T on a complex one.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

// ---------------------------------------------------------------------------
// LinearOperator<Op, T>: op.apply(res, k, x, y) -> Status
// ---------------------------------------------------------------------------

template <typename T>
struct Applies {
  fb::Status apply(fb::DeviceResources &res, std::int32_t k, const T *x, T *y);
};

// The same shape, returning something no Status converts from.
template <typename T>
struct AppliesReturningInteger {
  std::int32_t apply(fb::DeviceResources &res, std::int32_t k, const T *x, T *y);
};

// One argument short: no block width.
template <typename T>
struct AppliesWithoutK {
  fb::Status apply(fb::DeviceResources &res, const T *x, T *y);
};

// The work under another name.
template <typename T>
struct Multiplies {
  fb::Status multiply(fb::DeviceResources &res, std::int32_t k, const T *x, T *y);
};

static_assert(fb::LinearOperator<Applies<double>, double>);
static_assert(fb::LinearOperator<Applies<float>, float>);
static_assert(fb::LinearOperator<Applies<wwr::wwrDoubleComplex>, wwr::wwrDoubleComplex>);
static_assert(!fb::LinearOperator<Applies<double>, float>);        // the wrong scalar
static_assert(!fb::LinearOperator<Applies<float>, double>);        // and the other way
static_assert(!fb::LinearOperator<const Applies<double>, double>); // apply is non-const
static_assert(!fb::LinearOperator<AppliesReturningInteger<double>, double>);
static_assert(!fb::LinearOperator<AppliesWithoutK<double>, double>);
static_assert(!fb::LinearOperator<Multiplies<double>, double>); // no apply at all
static_assert(!fb::LinearOperator<double, double>);

// ---------------------------------------------------------------------------
// DavidsonPreconditioner<F, T>: f(res, theta, ritz_vectors, residual,
// correction) -> Status, callable on a const F
// ---------------------------------------------------------------------------

template <typename T>
struct Preconditions {
  fb::Status operator()(fb::DeviceResources &res, std::span<const T> theta, const T *ritz_vectors,
                        const T *residual, T *correction) const;
};

// Callable only on a non-const F: davidson_solve holds its preconditioner
// const.
template <typename T>
struct PreconditionsMutably {
  fb::Status operator()(fb::DeviceResources &res, std::span<const T> theta, const T *ritz_vectors,
                        const T *residual, T *correction);
};

// Without the Ritz vectors Olsen's correction needs.
template <typename T>
struct PreconditionsWithoutRitzVectors {
  fb::Status operator()(fb::DeviceResources &res, std::span<const T> theta, const T *residual,
                        T *correction) const;
};

template <typename T>
struct PreconditionsReturningVoid {
  void operator()(fb::DeviceResources &res, std::span<const T> theta, const T *ritz_vectors,
                  const T *residual, T *correction) const;
};

static_assert(fb::DavidsonPreconditioner<Preconditions<double>, double>);
static_assert(fb::DavidsonPreconditioner<Preconditions<float>, float>);
static_assert(!fb::DavidsonPreconditioner<Preconditions<double>, float>); // the wrong scalar
static_assert(!fb::DavidsonPreconditioner<PreconditionsMutably<double>, double>);
static_assert(!fb::DavidsonPreconditioner<PreconditionsWithoutRitzVectors<double>, double>);
static_assert(!fb::DavidsonPreconditioner<PreconditionsReturningVoid<double>, double>);
static_assert(!fb::DavidsonPreconditioner<Applies<double>, double>); // not callable

// ---------------------------------------------------------------------------
// UnitaryCostFunction<F, T>: a usual_fp T, F::kOrder, and const
// euclidean_gradient(res, n, w, psi) and value(res, n, w, psi, out) -> Status
// ---------------------------------------------------------------------------

template <typename T>
struct Cost {
  static constexpr std::int32_t kOrder = 2;
  fb::Status euclidean_gradient(fb::DeviceResources &res, std::int32_t n, const T *w, T *psi) const;
  fb::Status value(fb::DeviceResources &res, std::int32_t n, const T *w, const T *psi,
                   T *out) const;
};

template <typename T>
struct CostWithoutOrder {
  fb::Status euclidean_gradient(fb::DeviceResources &res, std::int32_t n, const T *w, T *psi) const;
  fb::Status value(fb::DeviceResources &res, std::int32_t n, const T *w, const T *psi,
                   T *out) const;
};

template <typename T>
struct CostWithoutValue {
  static constexpr std::int32_t kOrder = 2;
  fb::Status euclidean_gradient(fb::DeviceResources &res, std::int32_t n, const T *w, T *psi) const;
};

template <typename T>
struct CostWithoutGradient {
  static constexpr std::int32_t kOrder = 2;
  fb::Status value(fb::DeviceResources &res, std::int32_t n, const T *w, const T *psi,
                   T *out) const;
};

// Both calls non-const: cg_unitary holds its cost function const.
template <typename T>
struct CostMutably {
  static constexpr std::int32_t kOrder = 2;
  fb::Status euclidean_gradient(fb::DeviceResources &res, std::int32_t n, const T *w, T *psi);
  fb::Status value(fb::DeviceResources &res, std::int32_t n, const T *w, const T *psi, T *out);
};

template <typename T>
struct CostReturningDouble {
  static constexpr std::int32_t kOrder = 2;
  fb::Status euclidean_gradient(fb::DeviceResources &res, std::int32_t n, const T *w, T *psi) const;
  double value(fb::DeviceResources &res, std::int32_t n, const T *w, const T *psi, T *out) const;
};

// Every arm, in every usual_fp element type.
static_assert(fb::UnitaryCostFunction<Cost<float>, float>);
static_assert(fb::UnitaryCostFunction<Cost<double>, double>);
static_assert(fb::UnitaryCostFunction<Cost<wwr::wwrFloatComplex>, wwr::wwrFloatComplex>);
static_assert(fb::UnitaryCostFunction<Cost<wwr::wwrDoubleComplex>, wwr::wwrDoubleComplex>);
// A model whose signatures fit, over a T that is no usual_fp.
static_assert(!fb::UnitaryCostFunction<Cost<std::int32_t>, std::int32_t>);
static_assert(!fb::UnitaryCostFunction<Cost<double>, float>); // the wrong scalar
static_assert(!fb::UnitaryCostFunction<CostWithoutOrder<double>, double>);
static_assert(!fb::UnitaryCostFunction<CostWithoutValue<double>, double>);
static_assert(!fb::UnitaryCostFunction<CostWithoutGradient<double>, double>);
static_assert(!fb::UnitaryCostFunction<CostMutably<double>, double>);
static_assert(!fb::UnitaryCostFunction<CostReturningDouble<double>, double>);

} // namespace
