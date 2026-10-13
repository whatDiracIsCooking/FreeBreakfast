/// @file
/// @brief Map a scalar type and an order N onto the corresponding GPU builtin
/// vector type, and construct one generically (header for global-module-fragment
/// inclusion and for direct `#include` from a device translation unit).
///
/// @c Vec<T, N> resolves to the builtin vector type with scalar @c T and @c N
/// components, e.g. @c Vec<double,4> is @c double4 and @c Vec<float,3> is
/// @c float3. @c make_vec constructs one from its components in either host or
/// device code. The builtins themselves are declared by the backend's toolkit,
/// under the same global names on CUDA and HIP; this header only names the
/// mapping and a factory over it.

#pragma once

// The builtin vector types (float1..4, double1..4, int1..4, and the rest) and
// the __host__/__device__ macros, for whichever backend this build selected.
// Reached through WarpWraps' src/vector_types.h -- the one neutral include point
// for the vendor vector types -- rather than naming <cuda_runtime_api.h> or
// <hip/hip_vector_types.h> here. Angle brackets resolve it on wwr's src/ include
// root (linked via wwr::vector_types::host); this header is named vec.h, not
// vector_types.h, precisely so that spelling cannot find this file instead.
//
// Only the global ::float4 / ::double4 / ... names are used below, not wwr's
// wwr::float4 aliases: wwr deliberately omits the 64-bit 4-vectors (double4,
// long4, ...) because CUDA 13 deprecates them, and this mapping keeps them.
#include <vector_types.h>

namespace free_breakfast {

namespace detail {

/// @brief Selects the @p N-th (1-based) type from four candidates.
///
/// The primary template is intentionally left undefined, so an @p N outside
/// 1..4 is a compile error rather than a silent pick.
template <int N, typename T1, typename T2, typename T3, typename T4> struct nth_type;
template <typename T1, typename T2, typename T3, typename T4> struct nth_type<1, T1, T2, T3, T4> { using type = T1; };
template <typename T1, typename T2, typename T3, typename T4> struct nth_type<2, T1, T2, T3, T4> { using type = T2; };
template <typename T1, typename T2, typename T3, typename T4> struct nth_type<3, T1, T2, T3, T4> { using type = T3; };
template <typename T1, typename T2, typename T3, typename T4> struct nth_type<4, T1, T2, T3, T4> { using type = T4; };

} // namespace detail

/// @brief Trait mapping a scalar type @c T and an order @c N (1..4) to the
/// matching builtin vector type, exposed as the nested @c type.
///
/// The primary template is intentionally left undefined: an unsupported
/// combination (an order outside 1..4, or a scalar with no builtin vector type)
/// is a compile error rather than a silent fallback. Each scalar row below
/// names all four builtins because their identifiers cannot be synthesised from
/// the scalar type; @c detail::nth_type then keeps the N axis DRY.

template <typename T, int N> struct vector_type;

// CUDA 13 deprecates the 32-byte builtins (double4, long4, longlong4, ulong4,
// ulonglong4) in favour of explicit _16a / _32a alignment variants, which HIP
// does not have -- so the plain names are the only portable spelling. Naming
// them in the mapping below is not itself a use of the deprecated alignment, so
// silence the warning here; a caller that reaches the type through Vec inherits
// no warning. (Under nvcc this guard no-ops -- its front end is not clang -- so
// a .cu build still prints the deprecation notes; harmless without -Werror.
// Under HIP nothing is deprecated and the guard is inert.)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

// Scalar-to-builtin rows follow the toolkits' own naming: charN carries a
// `signed char`, ucharN an `unsigned char`, and `long` / `long long` map to the
// distinct longN / longlongN families even where they share a width.
template <int N> struct vector_type<signed char, N>        { using type = typename detail::nth_type<N, char1,      char2,      char3,      char4>::type; };
template <int N> struct vector_type<unsigned char, N>      { using type = typename detail::nth_type<N, uchar1,     uchar2,     uchar3,     uchar4>::type; };
template <int N> struct vector_type<short, N>              { using type = typename detail::nth_type<N, short1,     short2,     short3,     short4>::type; };
template <int N> struct vector_type<unsigned short, N>     { using type = typename detail::nth_type<N, ushort1,    ushort2,    ushort3,    ushort4>::type; };
template <int N> struct vector_type<int, N>                { using type = typename detail::nth_type<N, int1,       int2,       int3,       int4>::type; };
template <int N> struct vector_type<unsigned int, N>       { using type = typename detail::nth_type<N, uint1,      uint2,      uint3,      uint4>::type; };
template <int N> struct vector_type<long, N>               { using type = typename detail::nth_type<N, long1,      long2,      long3,      long4>::type; };
template <int N> struct vector_type<unsigned long, N>      { using type = typename detail::nth_type<N, ulong1,     ulong2,     ulong3,     ulong4>::type; };
template <int N> struct vector_type<long long, N>          { using type = typename detail::nth_type<N, longlong1,  longlong2,  longlong3,  longlong4>::type; };
template <int N> struct vector_type<unsigned long long, N> { using type = typename detail::nth_type<N, ulonglong1, ulonglong2, ulonglong3, ulonglong4>::type; };
template <int N> struct vector_type<float, N>              { using type = typename detail::nth_type<N, float1,     float2,     float3,     float4>::type; };
template <int N> struct vector_type<double, N>             { using type = typename detail::nth_type<N, double1,    double2,    double3,    double4>::type; };

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/// @brief The builtin vector type with scalar @c T and @c N components.
///
/// Example: @c Vec<double,4> is @c double4, @c Vec<float,3> is @c float3.
template <typename T, int N> using Vec = typename vector_type<T, N>::type;

/// @brief Construct a @c Vec<T, sizeof...(Args)> by brace initialisation,
/// usable from host or device code.
///
/// The component count fixes the order, so @c make_vec<double>(1,2,3,4) yields a
/// @c double4 and @c make_vec<float>(x,y,z) a @c float3. Every argument is
/// converted to @c T, so mixed-type components do not narrow. The brace is
/// portable: aggregate initialisation of a CUDA builtin, a constructor call on
/// HIP's HIP_vector_type. This generalises @c make_double4 / @c make_float3
/// without naming each factory; those remain available for callers that prefer
/// them.
template <typename T, typename... Args>
__host__ __device__ constexpr Vec<T, static_cast<int>(sizeof...(Args))> make_vec(Args... args) {
  return Vec<T, static_cast<int>(sizeof...(Args))>{static_cast<T>(args)...};
}

} // namespace free_breakfast
