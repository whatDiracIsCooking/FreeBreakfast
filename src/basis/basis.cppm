/// @file
/// @brief Gaussian basis-set description: shells and a basis set over a
/// molecule.
///
/// This module carries only the *description* of a Gaussian basis -- the data
/// the integral engines will consume -- and evaluates no integrals itself.
/// `import free_breakfast.basis;` gives @ref free_breakfast::Shell and
/// @ref free_breakfast::BasisSet along with the center-independent primitives
/// they build on.
///
/// The aggregate is split into six partitions: @c :shell defines @ref
/// free_breakfast::Shell (and re-exports the center-independent names @ref
/// free_breakfast::kMaxAngularMomentum, @ref free_breakfast::cartesian_function_count,
/// @ref free_breakfast::spherical_function_count and @ref
/// free_breakfast::Primitive from @c basis.h); @c :ao defines the
/// atomic-orbital view -- @ref free_breakfast::cartesian_powers (the canonical
/// Cartesian ordering) and @ref free_breakfast::AO; @c :harmonics defines the
/// closed-form real solid harmonic table (@ref free_breakfast::solid_harmonic_block)
/// that each shell's Cartesian-to-spherical transform is built from;
/// @c :basis_set defines
/// @ref free_breakfast::BasisSet over those shells, laying out its @ref
/// free_breakfast::AO list in that order; @c :shell_stamp defines the
/// geometry-free @ref free_breakfast::ShellStamp; and @c :basis_stamp defines
/// the per-element @ref free_breakfast::BasisStamp with @ref
/// free_breakfast::place, which stamps one onto a molecule to make a @ref
/// free_breakfast::BasisSet. A shell center is a @c Vec<double, 3> from
/// @c free_breakfast.core.types.
///
/// Everything is in atomic units: shell centers are in bohr (a @c
/// Vec<double, 3>), and primitive exponents are in inverse
/// bohr squared. A shell is a fixed contraction of primitive Gaussians that all
/// share one center and one angular momentum L; each primitive is an
/// (exponent, coefficient) pair (@ref free_breakfast::Primitive).
///
/// **Function convention: Cartesian at the kernel, spherical at the shell.**
/// Every @ref free_breakfast::Shell means the 2L+1 real solid harmonics of
/// its angular momentum L. The GPU integral engines evaluate the Cartesian
/// Gaussians x^a y^b z^c with a+b+c = L those harmonics are embedded in, of
/// which there are (L+1)(L+2)/2 -- so 1 for L=0 (s), 3 for L=1 (p), 6 for L=2
/// (d); the two sets coincide for L <= 1. The working target is up to d, i.e.
/// L in {0, 1, 2}. The Cartesian ordering within a shell is the canonical
/// lexicographic-on-(a,b,c)-with-a-highest order used by most integral
/// libraries; for p it is (x, y, z), and for d it is (xx, xy, xz, yy, yz, zz).
/// The harmonic combination is formed afterward via @c :harmonics' transform,
/// applied at the @ref free_breakfast::BasisSet level by @ref
/// free_breakfast::spherical_transform_matrix.

export module free_breakfast.basis;

// Re-exported so one import is enough to build a basis: a shell center is a
// Vec<double, 3> and place() stamps onto a Molecule, both of which come from
// here.
export import free_breakfast.molecule;

export import :shell;
export import :ao;
export import :harmonics;
export import :basis_set;
export import :shell_stamp;
export import :basis_stamp;
