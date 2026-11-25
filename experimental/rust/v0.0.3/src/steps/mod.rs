//! The steps: what a batched DF J and K build needs, each in its crudest
//! form. They and the bus are one crate, so the bus names every `Output`
//! and a step's writes are private to its module.
//!
//! | Step | Runs on | Reads | `Output` | Scratch |
//! |---|---|---|---|---|
//! | [`records`] | all GPUs | | each card's primitive pair records | the shell tables, pinned and device |
//! | [`guess`] | gpu0 | | the SAD density D on gpu0 | D's pinned staging |
//! | [`eri`] | all GPUs, library | records | the metric and its factor on every card; B on the host heap | a batch of pairs, the records and work lists |
//! | [`jk`] | all GPUs, library | eri, guess | J and K on gpu0 | a batch of aux functions on each card |

pub mod eri;
pub mod guess;
pub mod jk;
pub mod records;

use fb_core_error_handling::{Error, Result};
pub(crate) use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T};

use crate::budget::bytes;

/// The privileged card.
pub(crate) const GPU0: usize = 0;

/// A step's own limit, by default, as a fraction of its card's device cap.
/// A PLACEHOLDER: the design leaves the default fraction open.
pub const DEFAULT_LIMIT_FRACTION: u64 = 8;

/// A step's own limit on `cap`: the config's, or the default fraction.
pub(crate) fn limit(configured: Option<f64>, cap: u64) -> u64 {
    match configured {
        Some(mib) => (mib * (1u64 << 20) as f64) as u64,
        None if cap == u64::MAX => u64::MAX,
        None => cap / DEFAULT_LIMIT_FRACTION,
    }
}

/// The widest batch from `least` to `most` whose `bytes_at` fit `limit`.
/// The limit only picks the width: what is charged is what the width
/// takes.
///
/// # Errors
/// `Allocation`, naming the step, its limit and its least bytes, when not
/// even the narrowest batch fits.
pub(crate) fn widest(
    step: &str,
    limit: u64,
    least: usize,
    most: usize,
    bytes_at: impl Fn(usize) -> u64,
) -> Result<usize> {
    if bytes_at(least) > limit {
        return Err(Error::allocation(format!(
            "step '{step}': its limit of {} holds no batch: the narrowest, of width {least}, \
             takes {} of device scratch per card",
            bytes(limit),
            bytes(bytes_at(least))
        )));
    }
    let (mut lo, mut hi) = (least, most.max(least));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if bytes_at(mid) <= limit {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if lo == 1 && most > 1 {
        log::warn!(
            "step '{step}': its limit of {} gives a batch width of 1",
            bytes(limit)
        );
    } else {
        log::debug!("step '{step}': batch width {lo} of {most}");
    }
    Ok(lo)
}

/// Run `card_run` once per card of a step, each on a thread of its own
/// when there is more than one, and join: whoever fans out, joins. The
/// first error, by card order.
pub(crate) fn fan_out<C: Send, R: Send>(
    cards: Vec<C>,
    card_run: impl Fn(C) -> Result<R> + Sync,
) -> Result<Vec<R>> {
    if cards.len() == 1 {
        return cards.into_iter().map(card_run).collect();
    }
    std::thread::scope(|scope| {
        let card_run = &card_run;
        let handles: Vec<_> = cards
            .into_iter()
            .map(|card| scope.spawn(move || card_run(card)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(Error::gpu("a card's inner task panicked")))
            })
            .collect()
    })
}
