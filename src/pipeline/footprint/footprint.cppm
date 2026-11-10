/// @file
/// @brief Footprints as the pipeline sees them: the plan dimensions a step's
///        footprint is computed from, one step's footprint, and the plan fold
///        that finds the peak.
///
/// A step's footprint is the member
/// `MemoryFootprint footprint(const FootprintDimensions &) const`, which
/// unpacks the dimensions into its routines' arguments and composes their
/// footprints with `then` / `drop`. See README.md for the peak model, and for
/// check_plan, the fold held to the `[memory]` caps.

export module free_breakfast.pipeline.footprint;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
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
  /// @brief MINAO, placed and primitive-normalized as AVAS and the
  ///        Localization step place it; null when neither a planned active
  ///        space is picked by AVAS nor a planned localization builds IAOs
  ///        (IBO, or the LIVVOs).
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
  /// @brief skew_eig's work at order nao, in complex doubles, as the solver
  ///        sizes it (skew_eig_lwork): the CASSCF orbital step's
  ///        exponential. 0 when no CASSCF is planned.
  size_t skew_eig_lwork = 0;
  /// @brief getrf's work at the order of the plan's LU solve, in doubles, as
  ///        the solver sizes it (getrf_lwork): a CIS step's RIS
  ///        preconditioner's capacitance matrix, at its RIS basis's function
  ///        count. 0 when no RIS preconditioner is planned.
  size_t getrf_lwork = 0;
  /// @brief Worst-case active orbitals and electrons; 0 when no active space
  ///        is planned.
  size_t ncas = 0;
  size_t nelecas = 0;
  /// @brief The planned CASSCF step's roots and whether two of its weights
  ///        differ (response::rotating_pairs over them, equal when none are
  ///        given): what its gradient and couplings are sized over. 0 and
  ///        false when no CASSCF step is planned.
  size_t casscf_nroots = 0;
  bool casscf_unequal_weights = false;
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

// The transitional form, until #1122: a step that reruns a nested plan
// ([optimize], the semi-numerical [hessian]) has a footprint for the
// configurations whose every library call has one, and returns none for the
// rest, which pre-flight then refuses under a cap.
template <class S>
constexpr bool kHasOptionalFootprintMember = requires {
  requires std::same_as<decltype(&S::footprint),
                        std::optional<MemoryFootprint> (S::*)(const FootprintDimensions &) const> ||
               std::same_as<decltype(&S::footprint),
                            std::optional<MemoryFootprint> (S::*)(const FootprintDimensions &)
                                const noexcept>;
};

// The same for the optional `size_t adaptive_unit(const FootprintDimensions &)
// const`: the bytes one more unit of an adaptive step's adaptive part takes.
template <class S>
constexpr bool kHasAdaptiveUnitMember = requires {
  requires std::same_as<decltype(&S::adaptive_unit),
                        size_t (S::*)(const FootprintDimensions &) const> ||
               std::same_as<decltype(&S::adaptive_unit),
                            size_t (S::*)(const FootprintDimensions &) const noexcept>;
};

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief Step @p step's footprint at @p dims, or none for a step without a
///        `MemoryFootprint footprint(const FootprintDimensions &) const`
///        member (non-const, static, overloaded or of another signature).
///        A member returning `std::optional<MemoryFootprint>` instead is
///        passed through: a nested step's, none where its chain is not
///        footprinted yet.
///
/// The optional is transitional: once every step carries a footprint (#1122),
/// the fold's caller reads the member directly.
template <class S>
[[nodiscard]] constexpr std::optional<MemoryFootprint>
step_footprint(const S &step, const FootprintDimensions &dims) {
  if constexpr (kHasFootprintMember<S> || kHasOptionalFootprintMember<S>) {
    return step.footprint(dims);
  } else {
    return std::nullopt;
  }
}

/// @brief For an adaptive step, whose footprint's scratch is the least it
///        runs with: the bytes one more unit of its adaptive part takes at
///        @p dims (a tile's row, a batch's member). None for a step without
///        a `size_t adaptive_unit(const FootprintDimensions &) const` member,
///        which is every step that is not adaptive.
///
/// Pre-flight warns when the cap leaves an adaptive step less than this above
/// its least: the plan fits, but that step runs at its narrowest.
template <class S>
[[nodiscard]] constexpr std::optional<size_t> step_adaptive_unit(const S &step,
                                                                 const FootprintDimensions &dims) {
  if constexpr (kHasAdaptiveUnitMember<S>) {
    return step.adaptive_unit(dims);
  } else {
    return std::nullopt;
  }
}

/// @brief One planned step as pre-flight sees it: its name, its footprint
///        (none for a step without one yet), and its adaptive unit (none for
///        a step that is not adaptive). Pipeline::footprints lists them in
///        plan order.
struct StepFootprint {
  std::string_view name;
  std::optional<MemoryFootprint> footprint;
  std::optional<size_t> adaptive_unit;
};

/// @brief The `[memory]` caps as one device's fold is held to them.
///
/// @ref device_room is what the fold's device peak may reach: the pool's cap
/// (max_device_mem_per_gpu less the reserve outside the pool) less the
/// fragmentation headroom and, on a pool that rounds a reused block, its
/// rounding; pipeline.shape's preflight computes it. The caps themselves are
/// carried for the messages. 0 is no cap, for either kind.
struct PlanCaps {
  /// @brief `[memory] max_device_mem_per_gpu`; 0 for none.
  size_t max_device_mem_per_gpu = 0;
  /// @brief The most the fold's device peak may reach under it; 0 for no cap.
  size_t device_room = 0;
  /// @brief `[memory] max_pinned_mem_per_gpu`, which the fold's pinned peak
  ///        may reach exactly: pinned allocations are not pooled, so nothing
  ///        is left above it; 0 for none.
  size_t max_pinned_mem_per_gpu = 0;
};

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

/// @brief Fold @p steps, one device's plan in plan order, and hold it to
///        @p caps; at least one cap is set (an uncapped plan is not checked).
///
/// Logs, at info, each kind's peak and the step reaching it, and the
/// workspace gap: the workspace is held throughout at the plan's maximum,
/// and the peak step uses only its own part of it. For each adaptive step
/// (an @ref StepFootprint::adaptive_unit) under a device cap, the bytes the
/// cap leaves it above its least scratch at its point in the fold, at warn
/// level when that is under one more unit.
///
/// @return The folded plan; ErrorKind::InvalidConfig naming the first step
///         with no footprint, which no cap can be checked against, or the
///         step that reaches a kind's peak past its cap, with the peak and
///         the cap.
[[nodiscard]] Result<PlanFootprint> check_plan(std::span<const StepFootprint> steps,
                                               const PlanCaps &caps);

/// @brief The machine check under a pinned cap: @p devices DeviceResources,
///        each allowed @p max_pinned_mem_per_gpu of pinned memory, must fit
///        the node's @p mem_available (`/proc/meminfo`'s MemAvailable).
///
/// Pinned memory is non-pageable, so the node's RAM bounds it, and this
/// stands in for a node-wide host cap. ErrorKind::InvalidConfig naming the
/// product and MemAvailable when it is above.
[[nodiscard]] Status check_pinned_node(size_t max_pinned_mem_per_gpu, size_t devices,
                                       size_t mem_available);

} // namespace free_breakfast
