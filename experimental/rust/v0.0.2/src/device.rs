//! Streams and device resources: the step's own [`ForkStream`], the
//! device's one [`DeviceResources`] (a stream, the BLAS and solver handles
//! on it, and their workspace), and the copies between pinned and device
//! memory.
//!
//! The two streams are independent. A step that calls the library takes the
//! device's lock, forks from its own stream to the library's, makes its
//! calls, and joins back ([`ForkStream::library`]); those two events are the
//! only ones in the design, and they are inside a step.

use std::sync::{Arc, Mutex};

use fb_core_error_handling::{Error, Result, Status};
use fb_core_wwr::blas::BlasHandle;
use fb_core_wwr::common::{ScopedDeviceIndex, wwr_status_from_code};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut, split_at_mut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::runtime::{Event, Stream, event};
use fb_core_wwr::solver::{self, SolverHandle};
use fb_core_wwr::{gpu_status, gpu_status_at};
use fb_kernel_abi::LaunchStatus;
use wwr_sys::{wwr_memcpy_kind, wwr_solver_eig_mode, wwr_solver_fill_mode};

use crate::budget::{DeviceBuf, PinnedBuf};

const EIG_MODE: wwr_solver_eig_mode = wwr_solver_eig_mode::WWR_SOLVER_EIG_MODE_VECTOR;
const FILL_MODE: wwr_solver_fill_mode = wwr_solver_fill_mode::WWR_SOLVER_FILL_MODE_LOWER;

/// A stream with a fork and a join event: one per GPU step, made by the
/// wrapper in budget mode, synchronized by the wrapper after every `run`.
pub struct ForkStream {
    device: i32,
    stream: Stream,
    fork: Event,
    join: Event,
}

impl ForkStream {
    /// A stream on `device` with its two events.
    pub fn new(device: usize) -> ForkStream {
        let device = device as i32;
        ForkStream {
            device,
            stream: Stream::new(device),
            fork: Event::with_flags(device, event::DISABLE_TIMING),
            join: Event::with_flags(device, event::DISABLE_TIMING),
        }
    }

    /// The device the stream is on.
    pub fn device(&self) -> usize {
        self.device as usize
    }

    /// The stream: the step's own kernels and copies go on it.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Block until everything enqueued on the stream has run. The wrapper
    /// calls it after `run`; a step calls it itself before it reads a value
    /// it downloaded.
    ///
    /// # Errors
    /// A fault of the enqueued work, surfaced here.
    pub fn sync(&self) -> Status {
        gpu_status(self.stream.sync())
    }

    /// The step's library calls: take the device's lock, make the library
    /// stream wait for what this stream holds so far, run `calls` (which
    /// enqueues on the library stream), and make this stream wait for the
    /// library's work. The lock guards the host side only, two threads on
    /// one handle; the one stream orders the calls, and so protects the
    /// workspace.
    ///
    /// # Errors
    /// `calls`' error, or an event's.
    pub fn library<R>(
        &self,
        resources: &Mutex<DeviceResources>,
        calls: impl FnOnce(&mut DeviceResources) -> Result<R>,
    ) -> Result<R> {
        let mut library = resources.lock().unwrap_or_else(|e| e.into_inner());
        debug_assert_eq!(
            library.device, self.device,
            "a step's library is its device's"
        );
        gpu_status(self.fork.record(&self.stream))?;
        gpu_status(library.stream.wait_event(&self.fork, 0))?;
        let value = calls(&mut library)?;
        gpu_status(self.join.record(&library.stream))?;
        gpu_status(self.stream.wait_event(&self.join, 0))?;
        Ok(value)
    }
}

/// One per device, for the process: a stream, the BLAS and solver handles
/// bound to it, and the workspace the budget DAG's final node gives it. No
/// pool; memory is the steps'.
pub struct DeviceResources {
    device: i32,
    stream: Arc<Stream>,
    blas: BlasHandle,
    solver: SolverHandle,
    workspace: Option<DeviceBuf<f64>>,
}

impl DeviceResources {
    /// The library stream and handles of `device`. They exist before budget
    /// mode: the workspace queries need live handles.
    pub fn new(device: usize) -> DeviceResources {
        let device = device as i32;
        let _current = ScopedDeviceIndex::new(device);
        let stream = Arc::new(Stream::new(device));
        DeviceResources {
            device,
            blas: BlasHandle::new(Arc::clone(&stream)),
            solver: SolverHandle::new(Arc::clone(&stream)),
            stream,
            workspace: None,
        }
    }

    /// The library stream: every BLAS and solver call runs on it.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// The BLAS handle.
    pub fn blas(&self) -> &BlasHandle {
        &self.blas
    }

    /// syevd's workspace at order `n`, in bytes: what a step declares in
    /// `mem_budget` for each eigensolve it will make.
    ///
    /// # Errors
    /// The query's failure.
    pub fn syevd_workspace(&self, n: usize) -> Result<usize> {
        if n == 0 {
            return Ok(0);
        }
        let _current = ScopedDeviceIndex::new(self.device);
        let lwork =
            solver::syevd_lwork::<f64>(&self.solver, EIG_MODE, FILL_MODE, n as i32, n as i32)?;
        Ok(lwork * size_of::<f64>())
    }

    /// The budget DAG's final node: the max over every step's declarations,
    /// allocated once.
    pub(crate) fn bind_workspace(&mut self, bytes: usize) {
        let _current = ScopedDeviceIndex::new(self.device);
        self.workspace = Some(DeviceBuf::workspace(bytes));
    }

    /// The workspace's size, in bytes: 0 before the final node.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace
            .as_ref()
            .map_or(0, |w| w.len() * size_of::<f64>())
    }

    /// `A = V diag(w) Vᵀ` in place, the eigenvalues ascending, on the
    /// library stream, over the handle's workspace: the step never sees
    /// those bytes. `info` is the solver's flag, on the device.
    ///
    /// The call is held against its declaration: the size query is repeated
    /// here and compared with the workspace the final node allocated.
    ///
    /// # Errors
    /// `Allocation` if the workspace is smaller than this call needs, naming
    /// `step`: the step did not declare the call in its `mem_budget`.
    /// Otherwise the solver's launch error.
    pub fn syevd(
        &mut self,
        step: &str,
        n: usize,
        a: &mut impl DeviceSliceMut<f64>,
        w: &mut impl DeviceSliceMut<f64>,
        info: &mut impl DeviceSliceMut<i32>,
    ) -> Status {
        let need = self.syevd_workspace(n)? / size_of::<f64>();
        let held = self.workspace.as_ref().map_or(0, DeviceSlice::len);
        if need > held {
            return Err(Error::allocation(format!(
                "step '{step}': syevd at n = {n} needs {need} doubles of workspace, but the \
                 device's holds {held}: the call is missing from the step's mem_budget"
            )));
        }
        let workspace = self.workspace.as_mut().expect("held > 0 or need == 0");
        let (mut work, _) = split_at_mut(workspace, need);
        solver::syevd(
            &self.solver,
            EIG_MODE,
            FILL_MODE,
            n as i32,
            a,
            n as i32,
            w,
            &mut work,
            info,
        )
    }
}

/// Enqueue the copy of pinned `src` into device `dst` on `stream`.
///
/// # Errors
/// The enqueue's failure.
pub fn upload<T: Copy>(
    dst: &mut impl DeviceSliceMut<T>,
    src: &PinnedBuf<T>,
    stream: &Stream,
) -> Status {
    assert_eq!(dst.len(), src.len(), "upload: the two ends are one length");
    if src.is_empty() {
        return Ok(());
    }
    // SAFETY: both ends are `len` elements of the kinds the copy names, each
    // owned by a step's `Output`, which outlives the enqueued work: the
    // wrapper synchronizes before the step's slot is released.
    gpu_status(unsafe {
        wwr_sys::wwr_memcpy_async(
            dst.as_mut_ptr().cast(),
            src.as_ptr().cast(),
            src.len() * size_of::<T>(),
            wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
            stream.stream(),
        )
    })
}

/// Enqueue the copy of device `src` into pinned `dst` on `stream`. `dst` is
/// read only after a sync of that stream.
///
/// # Errors
/// The enqueue's failure.
pub fn download<T: Copy>(
    dst: &mut PinnedBuf<T>,
    src: &impl DeviceSlice<T>,
    stream: &Stream,
) -> Status {
    assert_eq!(
        dst.len(),
        src.len(),
        "download: the two ends are one length"
    );
    if dst.is_empty() {
        return Ok(());
    }
    // SAFETY: as `upload`'s.
    gpu_status(unsafe {
        wwr_sys::wwr_memcpy_async(
            dst.as_mut_ptr().cast(),
            src.as_ptr().cast(),
            src.len() * size_of::<T>(),
            wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
            stream.stream(),
        )
    })
}

/// A kernel launch wrapper's status as a [`Status`], naming `what`.
#[track_caller]
pub fn launched(what: &str, status: LaunchStatus) -> Status {
    fb_kernel_abi::launch_status(what, status, |code, at| {
        Some(gpu_status_at(wwr_status_from_code(code)?, at))
    })
}

/// The card's free memory, with `device` current: read once per device at
/// setup, for a defaulted cap.
///
/// # Errors
/// The runtime's failure.
pub fn free_memory(device: usize) -> Result<u64> {
    let _current = ScopedDeviceIndex::new(device as i32);
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: a plain runtime call writing two locals.
    gpu_status(unsafe { wwr_sys::wwr_mem_get_info(&mut free, &mut total) })?;
    Ok(free as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fb_core_error_handling::ErrorKind;
    use fb_core_wwr::memset;

    /// A library call against its declaration: with a workspace sized for
    /// an eigensolve at order 4, one at 4 runs and one at 64 is refused,
    /// naming the step, before anything is enqueued. Needs a GPU.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn an_undeclared_eigensolve_is_refused_naming_the_step() {
        let mut library = DeviceResources::new(0);
        let _current = ScopedDeviceIndex::new(0);
        let declared = library.syevd_workspace(4).unwrap();
        library.bind_workspace(declared);
        assert_eq!(library.workspace_bytes(), declared);

        let (mut a, mut w, mut info) = (
            DeviceBuf::<f64>::workspace(64 * 64 * 8),
            DeviceBuf::<f64>::workspace(64 * 8),
            DeviceBuf::<i32>::workspace(4),
        );
        memset::zero(&mut a, library.stream()).unwrap();
        library
            .syevd("declared", 4, &mut a, &mut w, &mut info)
            .expect("the declared call runs");
        let error = library
            .syevd("forgetful", 64, &mut a, &mut w, &mut info)
            .expect_err("the undeclared call is refused");
        assert_eq!(error.kind, ErrorKind::Allocation);
        assert!(
            error
                .message
                .starts_with("step 'forgetful': syevd at n = 64 needs"),
            "{error}"
        );
        assert!(
            error
                .message
                .ends_with("the call is missing from the step's mem_budget")
        );
        gpu_status(library.stream().sync()).unwrap();
    }
}
