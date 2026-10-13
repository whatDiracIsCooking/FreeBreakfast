/// @file
/// @brief Molecular geometry types and the nuclear repulsion energy.
///
/// Everything here is in atomic units: positions are in bohr, energies in
/// hartree, and a nucleus is identified by its element (its atomic number Z).
/// That choice is what keeps the nuclear repulsion a bare Sum Z_A Z_B / R_AB
/// with no physical constants in front -- in Hartree atomic units the Coulomb
/// prefactor 1/(4*pi*epsilon_0) is exactly 1.

export module free_breakfast.molecule;

import std;
import free_breakfast.core.types;
export import free_breakfast.atom;
export import wwr.vector_types;

export namespace free_breakfast {

/// @brief Euclidean distance between two points @p a and @p b, in bohr.
///
/// A point or displacement in 3D space is a @c wwr::double3 -- the backend's own
/// @c double3, so a host-built value and a kernel-built one share a layout. Its
/// components are public, so callers brace-initialise it directly
/// (`wwr::double3{x, y, z}`).
[[nodiscard]] double distance(const wwr::double3 &a, const wwr::double3 &b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

/// @brief The atomic number Z of @p atom (its nuclear charge).
[[nodiscard]] int32_t nuclear_charge(const Atom &atom) {
  return static_cast<int32_t>(atom.element);
}

/// @brief A collection of atoms with an overall charge and spin multiplicity.
///
/// The geometry is fixed at construction; @c charge is the net molecular charge
/// in units of the elementary charge (positive for a cation) and @c
/// multiplicity is 2S+1, defaulting to a singlet. Each nucleus is an
/// @ref free_breakfast::Atom -- a @c double3 position in bohr plus an
/// Element; Z is read with @ref nuclear_charge above.
class Molecule {
public:
  Molecule() = default;

  /// @param atoms         The nuclei and their positions (bohr).
  /// @param charge        Net molecular charge (default neutral).
  /// @param multiplicity  Spin multiplicity 2S+1 (default singlet, 1).
  explicit Molecule(std::vector<Atom> atoms, int32_t charge = 0, int32_t multiplicity = 1)
      : atoms_(std::move(atoms)), charge_(charge), multiplicity_(multiplicity) {}

  /// @brief The atoms, in the order supplied.
  [[nodiscard]] const std::vector<Atom> &atoms() const { return atoms_; }

  /// @brief Number of nuclei in the molecule.
  [[nodiscard]] uint32_t natoms() const { return static_cast<uint32_t>(atoms_.size()); }

  /// @brief Net molecular charge, in units of the elementary charge.
  [[nodiscard]] int32_t charge() const { return charge_; }

  /// @brief Spin multiplicity 2S+1.
  [[nodiscard]] int32_t multiplicity() const { return multiplicity_; }

  /// @brief Total nuclear charge, Sum Z_A over all atoms.
  [[nodiscard]] int32_t total_nuclear_charge() const {
    int32_t sum = 0;
    for (const Atom &atom : atoms_) {
      sum += nuclear_charge(atom);
    }
    return sum;
  }

  /// @brief Number of electrons: (Sum Z_A) - charge.
  ///
  /// Removing an electron leaves a cation of charge +1, so the electron count
  /// falls as the charge rises -- hence the subtraction.
  [[nodiscard]] int32_t electron_count() const { return total_nuclear_charge() - charge_; }

private:
  std::vector<Atom> atoms_{};
  int32_t charge_{0};
  int32_t multiplicity_{1};
};

/// @brief Nuclear repulsion energy E_nn = Sum_{A<B} Z_A Z_B / R_AB, in hartree.
///
/// Each unordered pair of nuclei is counted once. With fewer than two atoms
/// there are no pairs and the energy is zero.
[[nodiscard]] double nuclear_repulsion_energy(const Molecule &molecule) {
  const std::vector<Atom> &atoms = molecule.atoms();
  double energy = 0.0;
  for (uint32_t a = 0; a < atoms.size(); ++a) {
    for (uint32_t b = a + 1; b < atoms.size(); ++b) {
      const double r = distance(atoms[a].position, atoms[b].position);
      energy += static_cast<double>(nuclear_charge(atoms[a])) *
                static_cast<double>(nuclear_charge(atoms[b])) / r;
    }
  }
  return energy;
}

/// @brief Gradient of the nuclear repulsion energy, dE_nn/dR, in hartree/bohr.
///
/// The classical derivative of @ref nuclear_repulsion_energy with respect to
/// every nuclear coordinate:
///
///     dE_nn/dR_A = -Sum_{B != A} Z_A Z_B (R_A - R_B) / |R_A - R_B|^3.
///
/// Returned flat and row-major as @c (natoms, 3): @c grad[3*a + axis] is the
/// derivative with respect to atom @c a's position along @c axis (0=x, 1=y,
/// 2=z). Each unordered pair contributes once, with equal and opposite pulls
/// on its two atoms (@c +f on @c A, @c -f on @c B), so the per-axis sum over
/// atoms is zero by construction (translational invariance) -- no cancellation
/// of separately accumulated large terms. With fewer than two atoms there are
/// no pairs and the gradient is all zeros.
[[nodiscard]] std::vector<double> nuclear_repulsion_gradient(const Molecule &molecule) {
  const std::vector<Atom> &atoms = molecule.atoms();
  std::vector<double> grad(static_cast<size_t>(atoms.size()) * 3, 0.0);
  for (uint32_t a = 0; a < atoms.size(); ++a) {
    for (uint32_t b = a + 1; b < atoms.size(); ++b) {
      const wwr::double3 ra = atoms[a].position;
      const wwr::double3 rb = atoms[b].position;
      const double r = distance(ra, rb);
      const double zz = static_cast<double>(nuclear_charge(atoms[a])) *
                        static_cast<double>(nuclear_charge(atoms[b]));
      // gA = -Z_A Z_B (R_A - R_B) / r^3 is dE_nn/dR_A from this pair; by
      // symmetry dE_nn/dR_B = -gA. Accumulating both as an equal-and-opposite
      // pair keeps Sum_A dE/dR_A = 0 to machine eps.
      const double neg_inv_r3 = -zz / (r * r * r);
      const double gx = neg_inv_r3 * (ra.x - rb.x);
      const double gy = neg_inv_r3 * (ra.y - rb.y);
      const double gz = neg_inv_r3 * (ra.z - rb.z);
      grad[3 * a + 0] += gx;
      grad[3 * a + 1] += gy;
      grad[3 * a + 2] += gz;
      grad[3 * b + 0] -= gx;
      grad[3 * b + 1] -= gy;
      grad[3 * b + 2] -= gz;
    }
  }
  return grad;
}

} // namespace free_breakfast
