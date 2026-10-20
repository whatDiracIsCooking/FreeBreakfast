/// @file
/// @brief Molecular geometry types and the nuclear repulsion energy.
///
/// Everything here is in atomic units: positions are in bohr, energies in
/// hartree, and a nucleus is identified by its element (its atomic number Z).
/// That choice is what keeps the nuclear repulsion a bare Sum Z_A Z_B / R_AB
/// with no physical constants in front -- in Hartree atomic units the Coulomb
/// prefactor 1/(4*pi*epsilon_0) is exactly 1.
///
/// A Molecule also owns its nuclei on the device, uploaded once at
/// construction: its geometry is fixed for its lifetime, so one upload serves
/// every build that reads it. The non-owning view, NucleiView, is
/// nuclei_view.h's, included in the global module fragment and re-exported so
/// importers and the launchers' #include share one entity (atom.cppm has the
/// why of the in-namespace using-declaration).

module;

#include "nuclei_view.h"

export module free_breakfast.molecule;

import std;
export import free_breakfast.core.types;
export import free_breakfast.atom;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.mirrored_buffer;
import free_breakfast.core.resources;

export namespace free_breakfast {

using free_breakfast::NucleiView;

/// @brief Euclidean distance between two points @p a and @p b, in bohr.
///
/// A point or displacement in 3D space is a @c Vec<double, 3> -- the backend's
/// own @c double3, so a host-built value and a kernel-built one share a layout.
/// Its components are public, so callers brace-initialise it directly
/// (`Vec<double, 3>{x, y, z}`).
[[nodiscard]] double distance(const Vec<double, 3> &a, const Vec<double, 3> &b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

/// @brief The atomic number Z of @p atom (its nuclear charge).
[[nodiscard]] int32_t nuclear_charge(const Atom &atom) {
  return static_cast<int32_t>(atom.element);
}

/// @brief A collection of atoms with an overall charge and spin multiplicity,
/// and their nuclei on the device.
///
/// The geometry is fixed at construction; @c charge is the net molecular charge
/// in units of the elementary charge (positive for a cation) and @c
/// multiplicity is 2S+1, defaulting to a singlet. Each nucleus is an
/// @ref free_breakfast::Atom -- a @c double3 position in bohr plus an
/// Element; Z is read with @ref nuclear_charge above.
///
/// The nuclei -- Z and the positions, in atom order, the point charges V
/// attracts to -- are copied to the device once, at construction, one
/// @c MirroredBuffer per field on a new @c DeviceStream of res.device(). The
/// pinned side they were written into is kept for the Molecule's lifetime.
/// Movable, not copyable.
class Molecule {
public:
  /// @brief Hold @p atoms and copy their nuclei to @p res's device.
  ///
  /// The device res.dev_idx() names is made current for the call and the
  /// previous one restored. The copies' stream is synchronized once, after
  /// them: every class's stream reads the nuclei, and none is ordered after
  /// that stream. A molecule with no atoms, or more than an int32_t counts,
  /// aborts; so does a failed allocation (CreatePolicy) or copy. The copies
  /// only move data just created, so their failure means a broken context, not
  /// a bad input.
  ///
  /// @param res           The device the nuclei go to.
  /// @param atoms         The nuclei and their positions (bohr).
  /// @param charge        Net molecular charge (default neutral).
  /// @param multiplicity  Spin multiplicity 2S+1 (default singlet, 1).
  Molecule(DeviceResources &res, std::vector<Atom> atoms, int32_t charge = 0,
           int32_t multiplicity = 1)
      : atoms_(std::move(atoms)), charge_(charge), multiplicity_(multiplicity),
        nuclei_(upload(res, atoms_)) {}

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

  /// @brief The nuclei's own work stream: they were allocated and copied on
  /// it, and its dev_idx() is the device they are on.
  [[nodiscard]] const std::shared_ptr<DeviceStream> &device_stream() const noexcept {
    return nuclei_.z.handle();
  }

  /// @brief The nuclei on the device by pointer, for a launcher. Valid while
  /// this is alive.
  [[nodiscard]] NucleiView view() const {
    return {.n_nuc = static_cast<int32_t>(atoms_.size()),
            .z = nuclei_.z.device().data(),
            .c = nuclei_.c.device().data()};
  }

private:
  struct Nuclei {
    MirroredBuffer<double> z;
    MirroredBuffer<double> c;
  };

  // Write the nuclei into the pinned side and copy them to the device, on a
  // stream of their own.
  [[nodiscard]] static Nuclei upload(DeviceResources &res, const std::vector<Atom> &atoms) {
    if (atoms.empty() || atoms.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      check_or_abort(err_config("Molecule: a molecule of {} atoms", atoms.size()));
    }
    const ScopedDeviceIndex on{res.dev_idx()};
    const auto ds = std::make_shared<DeviceStream>(res.device());
    Nuclei out{
        .z = MirroredBuffer<double>(atoms.size(), ds),
        .c = MirroredBuffer<double>(3 * atoms.size(), ds),
    };
    double *z = out.z.pinned().data();
    double *c = out.c.pinned().data();
    for (size_t a = 0; a < atoms.size(); ++a) {
      z[a] = static_cast<double>(nuclear_charge(atoms[a]));
      c[(3 * a) + 0] = atoms[a].position.x;
      c[(3 * a) + 1] = atoms[a].position.y;
      c[(3 * a) + 2] = atoms[a].position.z;
    }
    check_or_abort(gpu_status(out.z.to_device()));
    // Both copies are on one stream, so this one sync covers z's as well.
    check_or_abort(gpu_status(out.c.to_device_sync()));
    return out;
  }

  std::vector<Atom> atoms_;
  int32_t charge_;
  int32_t multiplicity_;
  // After atoms_: it is uploaded from them.
  Nuclei nuclei_;
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
      const Vec<double, 3> ra = atoms[a].position;
      const Vec<double, 3> rb = atoms[b].position;
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
