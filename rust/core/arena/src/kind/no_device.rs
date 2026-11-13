//! [`Device`] and [`Pinned`] without the `device` feature: still kinds, so
//! tables, rows, admission and the planner over them build and test with no
//! library linked, but a block of either of non-zero size aborts. A block of
//! zero bytes allocates nothing and makes no call here, so a plan whose rows
//! are all on a test kind still instantiates.

use std::panic::Location;

use fb_core_error_handling::{Error, Status};

use super::{Device, Pinned, mem_available, sealed};

/// Why the allocation was refused.
const REFUSED: &str = "fb-core-arena was built without its `device` feature";

impl sealed::Sealed for Device {
    const ALWAYS_FILL: bool = false;
    type Current = ();

    fn current(_: i32) {}

    fn try_current(_: i32) -> Result<(), Error> {
        Ok(())
    }

    unsafe fn alloc(_: *mut *mut u8, _: usize) -> Result<(), String> {
        Err(REFUSED.to_owned())
    }

    unsafe fn free(_: *mut u8, _: usize, _: &'static Location<'static>) {
        unreachable!("{REFUSED}: no device block was ever allocated")
    }

    unsafe fn fill(_: *mut u8, _: usize, _: &'static Location<'static>) -> Status {
        unreachable!("{REFUSED}: no device block was ever allocated")
    }

    fn free_memory() -> String {
        "free memory unknown".to_owned()
    }
}

impl sealed::Sealed for Pinned {
    const ALWAYS_FILL: bool = true;
    type Current = ();

    fn current(_: i32) {}

    fn try_current(_: i32) -> Result<(), Error> {
        Ok(())
    }

    unsafe fn alloc(_: *mut *mut u8, _: usize) -> Result<(), String> {
        Err(REFUSED.to_owned())
    }

    unsafe fn free(_: *mut u8, _: usize, _: &'static Location<'static>) {
        unreachable!("{REFUSED}: no pinned block was ever allocated")
    }

    unsafe fn fill(_: *mut u8, _: usize, _: &'static Location<'static>) -> Status {
        unreachable!("{REFUSED}: no pinned block was ever allocated")
    }

    fn free_memory() -> String {
        mem_available()
    }
}
