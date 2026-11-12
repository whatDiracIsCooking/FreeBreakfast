//! `wwr.extension.common:abort_policy`: the opt-in members of the kit
//! (`wwr::extension::kit`), policies a consumer names when it wants the
//! common behaviour. Nothing in the core names one.

use std::marker::PhantomData;

use contracts::contract_trait;

use super::{ErrorCode, ErrorPolicy, SourceLocation};

/// Print the failure to stderr and abort: `kit::AbortPolicy<T>`. Usable in
/// every slot, destruction included.
pub struct AbortPolicy<E>(PhantomData<E>);

impl<E> AbortPolicy<E> {
    pub const fn new() -> Self {
        AbortPolicy(PhantomData)
    }
}

impl<E> Default for AbortPolicy<E> {
    fn default() -> Self {
        Self::new()
    }
}

/// wwr's `handle_error` re-checks the code against success first. Here the
/// trait's contract carries that check, so this aborts on whatever it is
/// handed.
#[contract_trait]
impl<E: ErrorCode> ErrorPolicy for AbortPolicy<E> {
    type Error = E;

    fn handle_error(&mut self, error: E, location: SourceLocation) {
        eprintln!(
            "GPU error at {}:{}: {} ({})",
            location.file(),
            location.line(),
            error.name(),
            error.description()
        );
        std::process::abort();
    }
}
