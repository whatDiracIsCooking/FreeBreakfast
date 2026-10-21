/// @file
/// @brief The spherically averaged atomic SCF a production SAD guess is made
/// of: one atom's restricted Hartree-Fock with fractional occupations, on the
/// host, over its own one-center integrals (@ref free_breakfast::atomic_scf),
/// and the block-diagonal molecular guess assembled from it (@ref
/// free_breakfast::atomic_scf_density).
///
/// Experimental: built only under FREE_BREAKFAST_BUILD_EXPERIMENTAL. The
/// README has the model and how it relates to free_breakfast.sad.
///
/// **The model.** free_breakfast.sad's occupations, with orbitals in place of
/// basis functions. The atom's Z electrons fill subshells in @ref kAufbauOrder;
/// the k-th subshell of momentum l is seated on the k-th lowest orbital of the
/// l block, its electrons spread evenly over the 2l+1 m. That density is
/// spherical, so its Fock matrix is block-diagonal by (l, m) and the same for
/// every m: the SCF diagonalizes one small radial block per l -- its shells of
/// that momentum -- which is PySCF's @c AtomSphAverageRHF exactly. The energy
/// is the restricted closed-shell expression over the spin-summed density,
/// E = 1/2 tr D (H + F) with F = H + J - K/2, so a fractionally occupied
/// orbital keeps part of its own self-interaction (hydrogen is not -0.5).
///
/// **One center, so no Boys function.** Every integral here has all four
/// centers at the nucleus. In McMurchie-Davidson terms every distance is zero,
/// the Hermite expansion of a product reduces to its same-center recurrence,
/// and the Boys function is only ever evaluated at 0, where F_n(0) = 1/(2n+1).
/// The integrals are exact, closed form, with no quadrature or table.
///
/// The exported surface is three partitions: @c :integrals, @c :scf and
/// @c :density. Everything else is an internal partition, imported only by
/// this module's implementation units.
export module free_breakfast.experimental.atomic_scf;

export import :integrals;
export import :scf;
export import :density;
