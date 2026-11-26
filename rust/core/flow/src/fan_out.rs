//! An all-GPU step's inner work: one scoped thread per GPU, joined before
//! it returns. The one place the v0.0.4 layer spawns
//! (`docs/programming_model.md`, "CPU workers").

use fb_core_error_handling::Result;

/// Run `per_gpu` once per item of `gpus` (one per GPU of the step, gpu0
/// first: its guards from [`Held::all_gpus`](crate::Held::all_gpus), or
/// what the step pairs with them), each on a scoped thread of its own when
/// there is more than one, and join: whoever fans out, joins. Each thread
/// makes its own device current, as every `GpuResources` and `Stream`
/// call does for itself. The results in GPU order, or the first error in
/// GPU order; a thread's panic is resumed on the caller's.
///
/// ```
/// let squares = fb_core_flow::fan_out(vec![1, 2, 3], |x| Ok(x * x)).unwrap();
/// assert_eq!(squares, [1, 4, 9]);
/// ```
///
/// # Errors
/// The first GPU's failure, in GPU order.
pub fn fan_out<C: Send, R: Send>(
    gpus: Vec<C>,
    per_gpu: impl Fn(C) -> Result<R> + Sync,
) -> Result<Vec<R>> {
    if gpus.len() <= 1 {
        return gpus.into_iter().map(per_gpu).collect();
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "an all-GPU step's fan-out: one scoped thread per GPU, joined before it returns (docs/programming_model.md, \"CPU workers\")"
    )]
    std::thread::scope(|scope| {
        let per_gpu = &per_gpu;
        let threads: Vec<_> = gpus
            .into_iter()
            .map(|gpu| scope.spawn(move || per_gpu(gpu)))
            .collect();
        let joined: Vec<Result<R>> = threads
            .into_iter()
            .map(|t| {
                t.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect();
        joined.into_iter().collect()
    })
}
