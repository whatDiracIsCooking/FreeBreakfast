//! The two memory kinds a block can hold, and the element types a row can.

mod sealed {
    pub trait Sealed {}
}

/// A memory kind: [`Device`] or [`Pinned`]. Sealed; the kind is in every
/// table, row and block type, so a row of one kind cannot be asked of a
/// block of the other.
pub trait Kind: sealed::Sealed + Send + Sync + 'static {
    /// How the kind is named in a table's text and an admission refusal.
    const NAME: &'static str;
}

/// Device memory on one GPU worker's card.
#[derive(Debug)]
pub enum Device {}

/// Pinned host memory owned by one GPU worker
/// (`docs/programming_model.md`, "Memory kinds").
#[derive(Debug)]
pub enum Pinned {}

impl sealed::Sealed for Device {}
impl sealed::Sealed for Pinned {}

impl Kind for Device {
    const NAME: &'static str = "device";
}

impl Kind for Pinned {
    const NAME: &'static str = "pinned";
}

/// A type a row may hold.
///
/// # Safety
/// Every bit pattern of `size_of::<Self>()` bytes is a valid value, and the
/// type has no padding and no drop glue. A block's bytes are filled with
/// `0xFF` and then handed out as `[Self]`, and a row reused by a later node
/// is read with whatever an earlier one left, so any other type would be
/// undefined behaviour.
pub unsafe trait Element: Copy + Send + Sync + 'static {}

macro_rules! element {
    ($($t:ty),*) => {$(
        // SAFETY: a primitive integer or float: every bit pattern is a
        // value, with no padding and no drop glue.
        unsafe impl Element for $t {}
    )*};
}
element!(u8, i32, u32, i64, u64, f32, f64);
