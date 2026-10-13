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

/// @brief Trait mapping a scalar type @c T and an order @c N (1..4) to the
/// matching builtin vector type, exposed as the nested @c type.
///
/// The primary template is intentionally left undefined: an unsupported
/// combination (an order outside 1..4, or a scalar with no builtin vector type)
/// is a compile error rather than a silent fallback. Each scalar row below
/// names all four builtins because their identifiers cannot be synthesised from
/// the scalar type.

template <typename T, int N>
struct vector_type;

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
//
// Spelled out in full rather than factored through a helper metafunction: this
// header is #included by device code, where no module partition can keep a
// helper private, so any helper declared here would be public API.
// clang-format off
template <> struct vector_type<signed char,        1> { using type = char1; };
template <> struct vector_type<signed char,        2> { using type = char2; };
template <> struct vector_type<signed char,        3> { using type = char3; };
template <> struct vector_type<signed char,        4> { using type = char4; };

template <> struct vector_type<unsigned char,      1> { using type = uchar1; };
template <> struct vector_type<unsigned char,      2> { using type = uchar2; };
template <> struct vector_type<unsigned char,      3> { using type = uchar3; };
template <> struct vector_type<unsigned char,      4> { using type = uchar4; };

template <> struct vector_type<short,              1> { using type = short1; };
template <> struct vector_type<short,              2> { using type = short2; };
template <> struct vector_type<short,              3> { using type = short3; };
template <> struct vector_type<short,              4> { using type = short4; };

template <> struct vector_type<unsigned short,     1> { using type = ushort1; };
template <> struct vector_type<unsigned short,     2> { using type = ushort2; };
template <> struct vector_type<unsigned short,     3> { using type = ushort3; };
template <> struct vector_type<unsigned short,     4> { using type = ushort4; };

template <> struct vector_type<int,                1> { using type = int1; };
template <> struct vector_type<int,                2> { using type = int2; };
template <> struct vector_type<int,                3> { using type = int3; };
template <> struct vector_type<int,                4> { using type = int4; };

template <> struct vector_type<unsigned int,       1> { using type = uint1; };
template <> struct vector_type<unsigned int,       2> { using type = uint2; };
template <> struct vector_type<unsigned int,       3> { using type = uint3; };
template <> struct vector_type<unsigned int,       4> { using type = uint4; };

template <> struct vector_type<long,               1> { using type = long1; };
template <> struct vector_type<long,               2> { using type = long2; };
template <> struct vector_type<long,               3> { using type = long3; };
template <> struct vector_type<long,               4> { using type = long4; };

template <> struct vector_type<unsigned long,      1> { using type = ulong1; };
template <> struct vector_type<unsigned long,      2> { using type = ulong2; };
template <> struct vector_type<unsigned long,      3> { using type = ulong3; };
template <> struct vector_type<unsigned long,      4> { using type = ulong4; };

template <> struct vector_type<long long,          1> { using type = longlong1; };
template <> struct vector_type<long long,          2> { using type = longlong2; };
template <> struct vector_type<long long,          3> { using type = longlong3; };
template <> struct vector_type<long long,          4> { using type = longlong4; };

template <> struct vector_type<unsigned long long, 1> { using type = ulonglong1; };
template <> struct vector_type<unsigned long long, 2> { using type = ulonglong2; };
template <> struct vector_type<unsigned long long, 3> { using type = ulonglong3; };
template <> struct vector_type<unsigned long long, 4> { using type = ulonglong4; };

template <> struct vector_type<float,              1> { using type = float1; };
template <> struct vector_type<float,              2> { using type = float2; };
template <> struct vector_type<float,              3> { using type = float3; };
template <> struct vector_type<float,              4> { using type = float4; };

template <> struct vector_type<double,             1> { using type = double1; };
template <> struct vector_type<double,             2> { using type = double2; };
template <> struct vector_type<double,             3> { using type = double3; };
template <> struct vector_type<double,             4> { using type = double4; };
// clang-format on

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/// @brief The builtin vector type with scalar @c T and @c N components.
///
/// Example: @c Vec<double,4> is @c double4, @c Vec<float,3> is @c float3.
template <typename T, int N>
using Vec = typename vector_type<T, N>::type;

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
