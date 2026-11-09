/// @file
/// @brief Footprints as the pipeline sees them: the plan dimensions a step's
///        footprint is computed from, one step's footprint, and the plan fold
///        that finds the peak.
///
/// A step's footprint is the member
/// `MemoryFootprint footprint(const FootprintDimensions &) const`, which
/// unpacks the dimensions into its routines' arguments and composes their
/// footprints with `then` / `drop`. See README.md for the peak model.

export module free_breakfast.pipeline.footprint;

import std;
import free_breakfast.core.types;
export import free_breakfast.core.footprint;
export import free_breakfast.basis;

export namespace free_breakfast {

/// @brief The plan's dimensions, before the first step runs: what every
///        step's footprint is computed from.
///
/// pipeline.shape's @ref plan_shape fills it from a planned Calculation. The
/// bases are carried rather than counted, because the library footprints mmd's,
/// K's and the three-center tensor's take the host basis, their routines' own
/// argument, and a table of per-momentum counts would only restate it. Zero
/// (or null, or empty) means the plan has none: no basis step leaves
/// @ref basis null, no density fitting leaves @ref aux null and @ref naux at 0,
/// no planned active space leaves @ref ncas and @ref nelecas at 0.
struct FootprintDimensions {
  /// @brief The orbital basis, placed and primitive-normalized as the
  ///        BasisSet step places it; owned by whoever filled the dimensions.
  const BasisSet *basis = nullptr;
  /// @brief The density-fitting aux basis, placed as the Eri step places it;
  ///        null without density fitting.
  const BasisSet *aux = nullptr;
  /// @brief MINAO, placed and primitive-normalized as AVAS places it; null
  ///        when no planned active space is picked by AVAS.
  const BasisSet *minao = nullptr;
  /// @brief The molecule's atoms, in order: what AVAS's labels are matched
  ///        against (avas_minao_aos).
  std::vector<Atom> atoms;
  /// @brief Spherical AOs: @ref basis's.
  size_t nao = 0;
  /// @brief Auxiliary basis functions: @ref aux's; 0 without density fitting.
  size_t naux = 0;
  /// @brief Doubly occupied orbitals: n_beta.
  size_t nocc = 0;
  /// @brief Alpha and beta electrons.
  size_t n_alpha = 0;
  size_t n_beta = 0;
  size_t natoms = 0;
  /// @brief mmd's primitive-pair records of class (la, lb), for la <= lb; 0
  ///        below the diagonal. Square, indexed `[la][lb]` over the momenta
  ///        @ref basis reaches: what J's classes are sized from.
  std::vector<std::vector<size_t>> mmd_records;
  /// @brief syevd's work by order, in doubles, as the solver sizes it
  ///        (diagonalize's own query): syevd_lwork[k] at order k, for every
  ///        order through the plan's largest eigensolve, nao or a planned
  ///        step's subspace (a Davidson's 16 nroots columns, the augmented
  ///        Hessian's ah_max_subspace + 1); empty for no basis. The step
  ///        footprints read it here, so none holds a solver handle.
  std::vector<size_t> syevd_lwork;
  /// @brief Worst-case active orbitals and electrons; 0 when no active space
  ///        is planned.
  size_t ncas = 0;
  size_t nelecas = 0;
};

} // namespace free_breakfast

// Unexported, which in a module is the privacy; in the interface so the
// exported template below can reach it from an importer's TU. Taking
// &S::footprint is what rejects an overloaded or template member, and its type
// what rejects a static, non-const or wrong-signature one. #1122 deletes it,
// when the footprint becomes a requirement of PipelineStep.
namespace free_breakfast {

template <class S>
constexpr bool kHasFootprintMember = requires {
  requires std::same_as<decltype(&S::footprint),
                        MemoryFootprint (S::*)(const FootprintDimensions &) const> ||
               std::same_as<decltype(&S::footprint),
                            MemoryFootprint (S::*)(const FootprintDimensions &) const noexcept>;
};

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief Step @p step's footprint at @p dims, or none for a step without a
///        `MemoryFootprint footprint(const FootprintDimensions &) const`
///        member (non-const, static, overloaded or of another signature).
///
/// The optional is transitional: once every step carries a footprint (#1122),
/// the fold's caller reads the member directly.
template <class S>
[[nodiscard]] constexpr std::optional<MemoryFootprint>
step_footprint(const S &step, const FootprintDimensions &dims) {
  if constexpr (kHasFootprintMember<S>) {
    return step.footprint(dims);
  } else {
    return std::nullopt;
  }
}

/// @brief One device's plan, folded: @ref footprint is `then` over its steps'
///        footprints in plan order, and each kind's peak step is the position
///        of the first step at which that kind's peak is reached.
///
/// At step k the device holds every earlier step's persistent bytes, its own
/// persistent and scratch, and the plan's maximum workspace, so
/// `footprint.device.peak()` is the largest of those sums over k, and
/// @ref device_peak_step the first k that reaches it. Pinned alike. An empty
/// plan's peaks are 0, at step 0.
struct PlanFootprint {
  MemoryFootprint footprint;
  size_t device_peak_step = 0;
  size_t pinned_peak_step = 0;
};

/// @brief Fold @p steps, one device's step footprints in plan order: see
///        @ref PlanFootprint.
[[nodiscard]] constexpr PlanFootprint plan_footprint(std::span<const MemoryFootprint> steps) {
  PlanFootprint plan;
  // Each kind's held-at-step-k sum without the workspace, which is the plan's
  // maximum at every step and so moves no peak.
  size_t device_persistent = 0;
  size_t pinned_persistent = 0;
  size_t device_best = 0;
  size_t pinned_best = 0;
  for (size_t k = 0; k < steps.size(); ++k) {
    const MemoryFootprint &step = steps[k];
    plan.footprint = then(plan.footprint, step);
    device_persistent += step.device.persistent;
    pinned_persistent += step.pinned.persistent;
    if (const size_t held = device_persistent + step.device.scratch; held > device_best) {
      device_best = held;
      plan.device_peak_step = k;
    }
    if (const size_t held = pinned_persistent + step.pinned.scratch; held > pinned_best) {
      pinned_best = held;
      plan.pinned_peak_step = k;
    }
  }
  return plan;
}

} // namespace free_breakfast
