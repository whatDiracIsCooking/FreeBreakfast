//! The steps, each in its crudest form. They and the bus are one crate, so
//! the bus names every `Output` and a step's writes are private to its
//! module.

pub mod core_hamiltonian;
pub mod guess;
pub mod metric;
pub mod orthogonalizer;
pub mod overlap;

use contracts::requires;
use fb_core_budget::{DeviceBuf, HostPart, Layout, Span};
use fb_core_error_handling::{Error, Status};
use fb_core_flow::Lane;
use fb_core_log::{eq, le};

/// The privileged GPU.
pub(crate) const GPU0: usize = 0;

/// A host step's staging, one layout for `mem_budget`'s declaration and
/// `run`'s carving: an `n × n` matrix in full.
pub(crate) fn staging(n: usize) -> (Layout, Span<f64>) {
    let mut l = Layout::new();
    let square = l.span(n * n);
    (l, square)
}

/// A packed lower triangle (`(p, q)`, `q <= p`, at `p (p + 1) / 2 + q`)
/// written out as the full symmetric `n × n` matrix.
#[requires(eq!(packed.len(), n * (n + 1) / 2), "packed is the n × n lower triangle")]
#[requires(le!(n * n, full.len()), "full holds the n × n matrix")]
pub(crate) fn unpack_lower(packed: &[f64], n: usize, full: &mut [f64]) {
    for p in 0..n {
        let row = &packed[p * (p + 1) / 2..p * (p + 1) / 2 + p + 1];
        for (q, &v) in row.iter().enumerate() {
            full[p + n * q] = v;
            full[q + n * p] = v;
        }
    }
}

/// A host step's one copy: its staged bytes into `dst` on `gpu`, on the
/// step's stream there. The copy's token is dropped: the staged part is a
/// tenant of the step's lease, carved until the lease drops after the
/// wrapper's step-end sync, so its leak costs nothing. On a host-backed
/// plan, with no device behind the GPU, the bytes go straight into the
/// buffer's host memory.
///
/// # Errors
/// The enqueue's failure; `InvalidConfig` for a GPU the step has no
/// stream on and whose buffer is not host-backed.
pub(crate) fn upload(
    lane: &Lane,
    gpu: usize,
    dst: &mut DeviceBuf<f64>,
    staged: HostPart<'_, f64>,
) -> Status {
    if let Some(stream) = lane
        .gpus()
        .iter()
        .find(|g| g.gpu() == gpu)
        .and_then(|g| g.streams().first())
    {
        drop(stream.upload(dst, staged)?);
        return Ok(());
    }
    match dst.host() {
        Some(host) => {
            let n = host.len().min(staged.len());
            host[..n].copy_from_slice(&staged[..n]);
            Ok(())
        }
        None => Err(Error::config(format!(
            "a host step uploads to gpu {gpu}, which it has no stream on"
        ))),
    }
}
