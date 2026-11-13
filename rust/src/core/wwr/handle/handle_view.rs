//! `wwr.extension.handle:handle_view`.

use std::marker::PhantomData;

/// A non-owning, copyable view of a GPU handle: `HandleView<T>`.
///
/// wwr's view can outlive its owner, and deletes `view()` on an rvalue to
/// catch the obvious case. Here the view borrows the handle it was taken
/// from for `'a`, so the borrow checker catches every case. A view of a
/// handle nothing here owns (the default stream, an interop handle) comes
/// from [`HandleView::from_raw`], at whatever lifetime the caller picks; that
/// is safe, because nothing can be done with the raw handle it yields
/// without `unsafe` code of the caller's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandleView<'a, T: Copy> {
    raw: T,
    borrow: PhantomData<&'a ()>,
}

impl<T: Copy> HandleView<'_, T> {
    /// View `raw`, which the caller keeps alive for as long as it uses it.
    pub fn from_raw(raw: T) -> Self {
        HandleView {
            raw,
            borrow: PhantomData,
        }
    }

    /// The viewed handle.
    pub fn get(&self) -> T {
        self.raw
    }
}
