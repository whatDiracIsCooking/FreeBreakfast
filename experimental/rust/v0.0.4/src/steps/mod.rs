//! The steps of the DF-RHF chain, each in its crudest form. They and the
//! bus are one crate, so the bus names every `Output` and a step's writes
//! are private to its module.
//!
//! ```text
//! overlap ──> orthogonalizer ─┐
//! core_hamiltonian ───────────┤
//! metric ──> eri ─────────────┼─> scf
//! guess ──────────────────────┘
//! ```
//!
//! | Step | Runs on | Locks | `Output` | Scratch |
//! |---|---|---|---|---|
//! | [`overlap`] | host, then gpu0 | none | S on gpu0 | the pinned staging |
//! | [`core_hamiltonian`] | host, then gpu0 | none | H = T + V on gpu0 | the pinned staging |
//! | [`metric`] | host, then every GPU | none | (P\|Q) on every GPU | the pinned staging |
//! | [`guess`] | host, then gpu0 | none | the SAD density on gpu0 | the pinned staging |
//! | [`orthogonalizer`] | gpu0 | gpu0 | X = S^(-1/2) | syevd's work, the eigenvectors |
//! | [`eri`] | every GPU | the `EriManager`, every GPU | the `EriManager` | the orbital and aux records |
//! | [`scf`] | every GPU | the `EriManager`, every GPU | C, the orbital energies, D, the energy | on gpu0: F, the DIIS ring, syevd's work |

pub mod core_hamiltonian;
pub mod eri;
pub mod guess;
pub mod metric;
pub mod orthogonalizer;
pub mod overlap;
pub mod scf;

use fb_core_error_handling::{Error, Result};

/// The privileged GPU.
pub(crate) const GPU0: usize = 0;

/// A packed lower triangle (`(p, q)`, `q <= p`, at `p (p + 1) / 2 + q`)
/// written out as the full symmetric `n × n` matrix.
pub(crate) fn unpack_lower(packed: &[f64], n: usize, full: &mut [f64]) {
    assert!(packed.len() == n * (n + 1) / 2 && full.len() >= n * n);
    for p in 0..n {
        let row = &packed[p * (p + 1) / 2..p * (p + 1) / 2 + p + 1];
        for (q, &v) in row.iter().enumerate() {
            full[p + n * q] = v;
            full[q + n * p] = v;
        }
    }
}

/// Run `gpu_run` once per GPU of a step, each on a thread of its own when
/// there is more than one, and join: whoever fans out, joins. The first
/// error, by GPU order.
pub(crate) fn fan_out<C: Send, R: Send>(
    gpus: Vec<C>,
    gpu_run: impl Fn(C) -> Result<R> + Sync,
) -> Result<Vec<R>> {
    if gpus.len() == 1 {
        return gpus.into_iter().map(gpu_run).collect();
    }
    std::thread::scope(|scope| {
        let gpu_run = &gpu_run;
        let handles: Vec<_> = gpus
            .into_iter()
            .map(|gpu| scope.spawn(move || gpu_run(gpu)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(Error::gpu("a GPU's inner task panicked")))
            })
            .collect()
    })
}
