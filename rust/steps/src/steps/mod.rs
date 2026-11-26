//! The steps, each in its crudest form. They and the bus are one crate, so
//! the bus names every `Output` and a step's writes are private to its
//! module.

pub mod core_hamiltonian;
pub mod eri;
pub mod guess;
pub mod metric;
pub mod orthogonalizer;
pub mod overlap;
pub mod scf;
mod scf_open;

use contracts::debug_requires;
use fb_core_budget::{DeviceBuf, HostPart, Layout, Span};
use fb_core_device::Pending;
use fb_core_error_handling::{Error, Status};
use fb_core_flow::Lane;
use fb_core_log::le;

/// The privileged GPU.
pub(crate) const GPU0: usize = 0;

/// A host step's staging, one layout for `mem_budget`'s declaration and
/// `run`'s carving: `len` doubles, at least the `n × n` matrix it uploads,
/// whose packed triangle the host integrals write at its front.
pub(crate) fn staging(len: usize) -> (Layout, Span<f64>) {
    let mut l = Layout::new();
    let span = l.span(len);
    (l, span)
}

/// The packed lower triangle at the front of `full` (`(p, q)`, `q <= p`,
/// at `p (p + 1) / 2 + q`) written out over it as the full symmetric
/// `n × n` matrix, in place.
#[debug_requires(le!(n * n, full.len()), "full holds the n × n matrix")]
pub(crate) fn unpack_lower_in_place(n: usize, full: &mut [f64]) {
    // Row p of the triangle moves from p (p + 1) / 2 to n p, the last row
    // first: its new place is at or past its old one and past every
    // earlier row's old place, so each row moves before it is overwritten.
    for p in (0..n).rev() {
        let from = p * (p + 1) / 2;
        full.copy_within(from..=from + p, n * p);
    }
    for p in 0..n {
        for q in 0..p {
            full[q * n + p] = full[p * n + q];
        }
    }
}

/// A host step's copies: its staged bytes into `dsts[gpu]` on each `gpu`
/// in turn, on the step's stream there, under one token: the first is an
/// upload, each later one joins it ([`Pending::and_upload`]), so one
/// staging serves every GPU. The token is dropped: the staged part is a
/// tenant of the step's lease, carved until the lease drops after the
/// wrapper's step-end sync, so its leak costs nothing. On a host-backed
/// plan, with no device behind the GPUs, the bytes go straight into each
/// buffer's host memory.
///
/// # Errors
/// An enqueue's failure; `InvalidConfig` for a GPU the step has no stream
/// on whose buffer is not host-backed, or one with no stream after one
/// with a stream.
pub(crate) fn upload(
    lane: &Lane,
    dsts: &mut [DeviceBuf<f64>],
    staged: HostPart<'_, f64>,
) -> Status {
    // The staged part while no copy holds it, then the copies' token.
    let mut held: std::result::Result<Pending<'_, HostPart<'_, f64>>, HostPart<'_, f64>> =
        Err(staged);
    for (gpu, dst) in dsts.iter_mut().enumerate() {
        let stream = lane
            .gpus()
            .iter()
            .find(|g| g.gpu() == gpu)
            .and_then(|g| g.streams().first());
        held = match (stream, held) {
            (Some(stream), Err(staged)) => Ok(stream.upload(dst, staged)?),
            (Some(stream), Ok(copy)) => Ok(copy.and_upload(stream, dst)?),
            (None, Err(staged)) => match dst.host() {
                Some(host) => {
                    let n = host.len().min(staged.len());
                    host[..n].copy_from_slice(&staged[..n]);
                    Err(staged)
                }
                None => {
                    return Err(Error::config(format!(
                        "a host step uploads to gpu {gpu}, which it has no stream on"
                    )));
                }
            },
            (None, Ok(_)) => {
                return Err(Error::config(format!(
                    "a host step uploads to gpu {gpu}, which it has no stream on, after a GPU it has one on"
                )));
            }
        };
    }
    drop(held);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::unpack_lower_in_place;

    /// The triangle at the front, written out in place, is the symmetric
    /// matrix element for element, for every order through 9, and with a
    /// tail past `n²`, which is left alone.
    #[test]
    #[allow(
        clippy::disallowed_macros,
        reason = "a test's matrices, outside any step's run"
    )]
    fn the_triangle_unpacks_in_place() {
        let entry = |p: usize, q: usize| (p.max(q) * 100 + p.min(q)) as f64;
        for n in 0..10 {
            let mut full = vec![-1.0; n * n + 3];
            let mut at = 0;
            for p in 0..n {
                for q in 0..=p {
                    full[at] = entry(p, q);
                    at += 1;
                }
            }
            unpack_lower_in_place(n, &mut full);
            for p in 0..n {
                for q in 0..n {
                    assert_eq!(full[p * n + q], entry(p, q), "n {n}, ({p}, {q})");
                }
            }
            assert_eq!(full[n * n..], [-1.0; 3], "n {n}: the tail");
        }
    }
}
