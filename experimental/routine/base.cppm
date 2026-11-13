/// @file
/// @brief Routine<Self, Alloc>: the CRTP base whose static operator() runs
/// Self::check and then Self::invoke over one designated-initializer In.

export module free_breakfast.experimental.routine:base;

import :key;

export namespace free_breakfast::routine {

/// @brief The call shape of a routine: `Self::check(in)`, then
/// `Self::invoke`, over one `const Self::In &`, from a static operator() so
/// a routine is called through an inline constexpr object of its type.
///
/// The parameter is spelled `typename S::In` with `S` defaulted to `Self`:
/// `Self::In` directly does not compile, because `Self` is incomplete while
/// this base is instantiated (at Self's base-clause). The default makes the
/// lookup wait for the call. `In` is a non-deduced context, so `S` is always
/// `Self` and a braced list converts to `In` by aggregate initialization.
template <class Self, Alloc A>
struct Routine;

/// @brief A routine that allocates: it takes the caller's key and passes it
/// to `Self::invoke(key, in)`.
template <class Self>
struct Routine<Self, Alloc::Yes> {
  /// @brief Check, then invoke with @p key.
  template <class S = Self>
  static decltype(auto) operator()(const AllocKey &key, const typename S::In &in) {
    S::check(in);
    return S::invoke(key, in);
  }
};

/// @brief A routine that does not: it takes no key, so it has none to pass
/// to an Alloc::Yes callee or to a KeyedBuffer.
template <class Self>
struct Routine<Self, Alloc::No> {
  /// @brief Check, then invoke.
  template <class S = Self>
  static decltype(auto) operator()(const typename S::In &in) {
    S::check(in);
    return S::invoke(in);
  }
};

} // namespace free_breakfast::routine
