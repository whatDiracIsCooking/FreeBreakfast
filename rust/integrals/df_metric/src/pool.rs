//! The host integrals' pool: a rayon pool of a fixed size, whose threads are
//! joined when it drops.

use std::thread::JoinHandle;

/// The pool the host integrals run on: `threads` rayon threads, made once
/// and named `host-integrals-<i>`. Every call of a host integral crate made
/// inside [`HostPool::install`] runs its tasks on them, never on rayon's
/// global pool, which is never built. Dropping the pool joins its threads,
/// so none outlives it.
///
/// It lives here because only the four host integral crates may name rayon
/// (`docs/programming_model.md`, "CPU workers"), and this one is already a
/// dependency of another of them; `fb-steps` makes one in setup, of
/// `[resources] host_threads` threads.
///
/// ```
/// use fb_basis::BasisSet;
/// use fb_integrals_df_metric::{HostPool, metric_into};
///
/// let pool = HostPool::new(2).unwrap();
/// assert_eq!(pool.threads(), 2);
/// let aux = BasisSet::new(Vec::new());
/// let mut m: [f64; 0] = [];
/// pool.install(|| metric_into(&aux, &mut m));
/// ```
pub struct HostPool {
    /// `None` only while dropping, so the threads can be told to end before
    /// they are joined.
    pool: Option<rayon::ThreadPool>,
    workers: Vec<JoinHandle<()>>,
}

impl HostPool {
    /// A pool of `threads` threads, each started before this returns.
    ///
    /// # Errors
    /// The operating system's refusal to start a thread.
    #[allow(
        clippy::disallowed_methods,
        reason = "the one place the host integrals' threads are started, once per run in setup (docs/programming_model.md, \"CPU workers\")"
    )]
    pub fn new(threads: usize) -> std::io::Result<HostPool> {
        let mut workers = Vec::with_capacity(threads);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .spawn_handler(|thread| {
                let worker = std::thread::Builder::new()
                    .name(format!("host-integrals-{}", thread.index()))
                    .spawn(|| thread.run())?;
                workers.push(worker);
                Ok(())
            })
            .build()
            .map_err(std::io::Error::other)?;
        Ok(HostPool {
            pool: Some(pool),
            workers,
        })
    }

    /// The pool's threads.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    /// Run `f` on the pool, blocking until it returns: every rayon task `f`
    /// starts runs on the pool's threads.
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(pool) => pool.install(f),
            // INVARIANT: `pool` is `None` only inside `drop`.
            None => unreachable!("the pool is taken only in drop"),
        }
    }
}

impl Drop for HostPool {
    fn drop(&mut self) {
        // Dropping the rayon pool tells its threads to end once idle, and
        // none is busy, since `install` blocks; then each is joined.
        drop(self.pool.take());
        for worker in self.workers.drain(..) {
            // A worker ends by returning from `run`; a panic in a task was
            // already re-raised by the `install` that ran it.
            let _ = worker.join();
        }
    }
}
