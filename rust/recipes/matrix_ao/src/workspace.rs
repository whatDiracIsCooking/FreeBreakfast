//! A worker's BLAS workspace: a worker row of the arena, handed to the
//! worker's BLAS handle (`docs/gpu_layer.md`, "Layering").

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::blas::WORKSPACE_BYTES;

/// Give worker `worker`'s BLAS handle its workspace from the arena: a
/// worker row of [`WORKSPACE_BYTES`] (`blas workspace` in the plan's
/// device table), and a GPU task, `blas workspace`, that hands the row to
/// the handle every run, so no BLAS call allocates behind the table.
///
/// Add it before any task of the worker that calls BLAS: the worker's
/// tasks run in plan order, and one added earlier would run on the
/// workspace the handle allocated for itself. A plan without it still runs,
/// on that one.
///
/// The handle keeps the pointer once the run is over, so a backend runs
/// only the plan of the instance whose block holds the row, as
/// [`Device::new`](fb_core_thread_pool::Device::new) already requires.
pub fn add_blas_workspace(dag: &mut Dag<'_>, worker: usize) {
    let row = dag.device_worker_row(worker, "blas workspace", WORKSPACE_BYTES);
    dag.add("blas workspace", worker, BlasWorkspace { row });
}

/// GPU: the worker row handed to the worker's BLAS handle.
struct BlasWorkspace {
    row: RowId<Device, u8>,
}

impl Task for BlasWorkspace {
    type Rows = ();
    const GPU: bool = true;

    fn layout(&self, _: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) {}

    fn run(
        &self,
        _: &(),
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let workspace = device.take_mut(self.row);
        // SAFETY: a worker row of the block on the worker's device, aligned
        // to DEVICE_ALIGNMENT (256), live for the plan, which no task
        // declares; the handle is the worker's, used by this plan alone.
        unsafe {
            worker
                .blas()
                .set_workspace(workspace.as_mut_ptr().cast(), workspace.len())
        }
    }
}
