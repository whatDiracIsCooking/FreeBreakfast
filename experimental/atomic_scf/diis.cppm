/// @file
/// @brief Pulay DIIS on the host, over flat vectors.

// An internal partition: nothing here is exported. src/linalg/diis is the
// device-resident one; this one holds a few hundred doubles per iterate.
module free_breakfast.experimental.atomic_scf:diis;

import std;
import free_breakfast.core.types;

namespace free_breakfast {

/// @brief Pulay DIIS: keeps the last few (Fock, error) pairs and extrapolates
/// the Fock vector whose error combination is smallest.
class AtomicDiis {
public:
  /// @param depth Pairs kept; the oldest is dropped past it.
  explicit AtomicDiis(size_t depth) : depth_(depth) {}

  /// @brief Record @p fock and its @p error, then return the extrapolated Fock
  /// vector over everything held.
  ///
  /// When the DIIS matrix is singular -- the residuals have aligned -- the
  /// oldest pairs are dropped until it is not; with one pair left that pair's
  /// Fock vector is returned as is.
  [[nodiscard]] std::vector<double> extrapolate(std::vector<double> fock,
                                                std::vector<double> error);

private:
  size_t depth_;
  std::deque<std::vector<double>> fock_;
  std::deque<std::vector<double>> error_;
};

} // namespace free_breakfast
