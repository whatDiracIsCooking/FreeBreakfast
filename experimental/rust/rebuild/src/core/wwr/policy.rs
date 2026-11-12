//! `free_breakfast.core.wwr:policy`: the tree's error policies for the wwr
//! RAII wrappers, which log through [`crate::core::log`].
//!
//! Not wwr's [`kit::AbortPolicy`](super::common::kit::AbortPolicy), which
//! prints to stderr: these are the ones the C++ tree binds every wrapper to.
//! Both are zero-sized, so each fits every slot at no cost.

use std::marker::PhantomData;

use contracts::contract_trait;

use super::common::{self, ErrorCode, ErrorPolicy, SourceLocation};
use super::status::gpu_status_at;
use crate::core::error_handling::check_or_abort;
use crate::core::log::{self, LogLevel};

/// Log at critical with the failing call's location, then abort:
/// `AbortPolicy<E>`.
pub struct AbortPolicy<E>(PhantomData<E>);

/// Log at error with the failing call's location, and carry on:
/// `LogPolicy<E>`.
///
/// For the destruction slots: a failed destroy runs in a `Drop`, often
/// during teardown after an earlier failure, where aborting would bury the
/// root cause under the knock-on one. The C++ one wraps its body in
/// `try`/`catch (...)`, because formatting can throw out of a `noexcept`
/// function; here a failed write is dropped inside the logger, and a failed
/// allocation aborts in Rust whatever the policy, so there is nothing to
/// catch.
pub struct LogPolicy<E>(PhantomData<E>);

macro_rules! zero_sized_policy {
    ($name:ident) => {
        impl<E> $name<E> {
            pub const fn new() -> Self {
                $name(PhantomData)
            }
        }

        impl<E> Default for $name<E> {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}
zero_sized_policy!(AbortPolicy);
zero_sized_policy!(LogPolicy);

#[contract_trait]
impl<E: ErrorCode> ErrorPolicy for AbortPolicy<E> {
    type Error = E;

    fn handle_error(&mut self, error: E, location: SourceLocation) {
        check_or_abort(gpu_status_at(error, location));
    }
}

#[contract_trait]
impl<E: ErrorCode> ErrorPolicy for LogPolicy<E> {
    type Error = E;

    fn handle_error(&mut self, error: E, location: SourceLocation) {
        if let Err(e) = gpu_status_at(error, location) {
            log::log_at(LogLevel::Error, location, format_args!("{e}"));
        }
    }
}

/// The policy for creation, allocation and device-select slots.
pub type CreatePolicy<E> = AbortPolicy<E>;

/// The policy for destruction and free slots.
pub type DestroyPolicy<E> = LogPolicy<E>;

/// [`CreatePolicy`] for alloc and [`DestroyPolicy`] for free: the tree's
/// `PolicyMap`, an implementation of wwr's [`common::PolicyMap`].
pub struct PolicyMap;

impl common::PolicyMap for PolicyMap {
    type Alloc<E: ErrorCode> = CreatePolicy<E>;
    type Free<E: ErrorCode> = DestroyPolicy<E>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use wwr_sys::wwr_status;

    #[test]
    fn the_tree_policies_are_zero_sized() {
        assert_eq!(std::mem::size_of::<CreatePolicy<wwr_status>>(), 0);
        assert_eq!(std::mem::size_of::<DestroyPolicy<wwr_status>>(), 0);
    }
}
