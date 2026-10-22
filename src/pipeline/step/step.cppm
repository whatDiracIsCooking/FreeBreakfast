/// @file
/// @brief The pipeline step contract: the view a step reads its inputs
///        through, and the concept a step type satisfies.
///
/// A step's `run` signature is its whole declaration: the `View` it takes is
/// what it reads, the `Result<T>` it returns is the one product it writes, and
/// the resources it takes are where it runs. It never sees a shared store, so
/// an undeclared read is a compile error. See README.md.

export module free_breakfast.pipeline.step;

export import :view;
export import :run_shape;
export import :pipeline_step;
