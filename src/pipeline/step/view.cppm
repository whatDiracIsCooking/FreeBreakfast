/// @file
/// @brief `:view`: a step's read-only inputs, and what may be a product.

export module free_breakfast.pipeline.step:view;

import std;
import free_breakfast.core.types;

// Unexported, which in a module is the privacy. The exported templates below
// are instantiated in an importer's TU, so these sit in an interface partition
// to be reachable from there; :pipeline_step uses them too.
namespace free_breakfast {

// What a product may be: an object a slot can own and move. Not copyable on
// purpose, since products own device buffers.
template <class T>
concept Product =
    std::is_object_v<T> && std::same_as<T, std::remove_cv_t<T>> && std::move_constructible<T>;

template <class T, class... Ts>
constexpr size_t kProductCount = (size_t{0} + ... + size_t{std::same_as<T, Ts>});

// Distinct products: get<T> picks a tuple element by type, which is ambiguous
// for a repeated one.
template <class... Ts>
concept ProductSet = (Product<Ts> && ...) && ((kProductCount<Ts, Ts...> == 1) && ...);

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief A step's read-only inputs: a `const` reference to each of `Ts`.
///
/// A step's `run` takes one as its first parameter, and that parameter is the
/// step's whole declaration of what it reads. Every type must be a
/// non-`const` movable object type, and none may repeat: `View<OverlapOutput,
/// OverlapOutput>` does not compile.
///
/// Holds pointers into the producers' output slots and owns nothing, so it
/// lives for one `run` call. A product therefore never borrows from another
/// product: it owns its data, or shares it as `std::shared_ptr<const ...>`.
template <class... Ts>
  requires ProductSet<Ts...>
class View {
public:
  /// @brief Binds one input of each declared type. The inputs must outlive
  ///        the view.
  explicit View(const Ts &...inputs) : inputs_{&inputs...} {}

  /// @brief The bound input of type `T`. Constrained to the declared `Ts`,
  ///        so an undeclared read fails to compile rather than at run time.
  template <class T>
    requires(std::same_as<T, Ts> || ...)
  const T &get() const {
    return *std::get<const T *>(inputs_);
  }

private:
  std::tuple<const Ts *...> inputs_;
};

} // namespace free_breakfast
