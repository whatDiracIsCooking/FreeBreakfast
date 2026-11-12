//! `wwr.extension.runtime:device_handle`: the kit's ready-made device
//! handle (`wwr::extension::kit`). It lives in `kit` so the generic name
//! `DeviceHandle` stays beside the [trait](crate::core::wwr::handle::DeviceHandle)
//! it implements without taking its place.

use wwr_sys::{wwr_mem_pool, wwr_status, wwr_stream};

use super::{MemPool, Stream};
use crate::core::wwr::common::ErrorPolicy;
use crate::core::wwr::handle::{self, DeviceHandlePool, DeviceHandleStream};

/// An owned stream and memory pool on one device: `kit::DeviceHandle`, the
/// top rung of the device-handle ladder, so a buffer backed by it draws from
/// that pool on that stream.
///
/// The stream tier needs no such type: a [`Stream`] is a
/// [`DeviceHandleStream`] on its own.
///
/// wwr's `props()` is left out: `wwrDeviceProp` crosses the C ABI as a bare
/// `void*` whose layout differs between the backends.
pub struct DeviceHandle<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    index: i32,
    stream: Stream<PC, PD, PDev>,
    pool: MemPool<PC, PD, PDev>,
}

impl<PC, PD, PDev> DeviceHandle<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status> + Default,
    PD: ErrorPolicy<Error = wwr_status> + Default,
    PDev: ErrorPolicy<Error = wwr_status> + Default,
{
    /// A non-blocking stream and a default pool on device `index`, each
    /// with new default policies, as wwr default-constructs them.
    #[track_caller]
    pub fn new(index: i32) -> Self {
        DeviceHandle {
            index,
            stream: Stream::new(index, PC::default(), PD::default(), PDev::default()),
            pool: MemPool::new(index, PC::default(), PD::default(), PDev::default()),
        }
    }
}

impl<PC, PD, PDev> DeviceHandle<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// The owned stream.
    pub fn owned_stream(&self) -> &Stream<PC, PD, PDev> {
        &self.stream
    }

    /// The owned pool.
    pub fn owned_pool(&self) -> &MemPool<PC, PD, PDev> {
        &self.pool
    }
}

impl<PC, PD, PDev> handle::DeviceHandle for DeviceHandle<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// The index it was made with, as wwr's `dev_idx()` returns.
    fn dev_idx(&self) -> i32 {
        self.index
    }
}

impl<PC, PD, PDev> DeviceHandleStream for DeviceHandle<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    fn stream(&self) -> wwr_stream {
        self.stream.stream()
    }
}

impl<PC, PD, PDev> DeviceHandlePool for DeviceHandle<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// The raw pool, or null when its create failed under a policy that
    /// returns.
    fn pool(&self) -> wwr_mem_pool {
        self.pool.get().unwrap_or(std::ptr::null_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::wwr::handle::DeviceHandle as _;
    use crate::core::wwr::runtime::tests::Abort;

    /// Needs a GPU. The stream and pool land on the handle's device, and
    /// the ladder's accessors hand out the owned ones. Allocating through
    /// the handle is tested in `tests/pool_allocation.rs`.
    #[test]
    fn stream_and_pool_share_the_device() {
        let h = DeviceHandle::<Abort, Abort, Abort>::new(0);
        assert_eq!(h.dev_idx(), 0);
        assert_eq!(h.owned_stream().dev_idx(), 0);
        assert_eq!(h.owned_pool().dev_idx(), 0);
        assert_eq!(h.stream(), h.owned_stream().get().unwrap());
        assert_eq!(h.pool(), h.owned_pool().get().unwrap());
    }
}
