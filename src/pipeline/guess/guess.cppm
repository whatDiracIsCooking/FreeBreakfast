/// @file
/// @brief The `Guess` step: the initial-guess density, SAD for now.
///
/// A thin wrapper over sad's task form, its atomic SCFs spread over the step's
/// subflow, with the compiled-in stamps of io.sad standing in for the SCFs of a
/// compiled-in basis set. HostSubflow-shaped: the pipeline's first host step meant to run
/// beside the device chain. See README.md.

export module free_breakfast.pipeline.guess;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.sad;
import free_breakfast.io.sad;
import free_breakfast.pipeline.step;
export import free_breakfast.basis;
export import free_breakfast.third_party.taskflow;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief Which initial guess @ref GuessStep builds.
enum class GuessType : uint8_t {
  Sad, ///< Superposition of atomic densities (@ref sad_density)
};

/// @brief The initial-guess density, spin-summed, on the host.
///
/// @c nao_sphe square, flat column-major (element (i, j) at i + j*n), in the
/// spherical layout. Host data because the guess is made on the host: the
/// device step that first reads it uploads it.
struct Guess {
  std::vector<double> d;
};

/// @brief Builds the initial-guess density of @ref type over the
/// @ref BasisSet placed on the @ref Molecule.
struct GuessStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "guess";

  /// @brief The kind of guess.
  GuessType type = GuessType::Sad;

  /// @brief The guess, its atomic SCFs run in parallel on @p subflow.
  ///
  /// A basis resolved by name (@ref BasisSet::name) takes each element's
  /// density from the stamps compiled into the binary (@ref
  /// builtin_sad_density) and runs an SCF only for an element they lack; a
  /// basis read from a path runs every one. The errors are sad_density's: a basis not placed over these atoms, or one too
  /// small to seat an atom's occupied subshells, is ErrorKind::InvalidConfig;
  /// an atom too large for the atomic SCF is ErrorKind::Unsupported; a
  /// non-converged atomic SCF is ErrorKind::MaxIterationsExceeded.
  [[nodiscard]] Result<Guess> run(View<Molecule, BasisSet> in, tf::Subflow &subflow) const {
    Result<std::vector<double>> d;
    switch (type) {
    case GuessType::Sad:
      sad_density(subflow, in.get<BasisSet>(), in.get<Molecule>().atoms(), d, builtin_sad_density);
      break;
    }
    // Join here, explicitly: the task writes d, a local of this frame, and
    // Taskflow's implicit join would run it only after run has returned.
    subflow.join();
    return Guess{FBF_TRY(std::move(d))};
  }
};

} // namespace free_breakfast
