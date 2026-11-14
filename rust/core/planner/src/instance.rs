//! A planned DAG's blocks and its executor plan: what runs, any number of
//! times.

use std::sync::Arc;

use super::dag::{Planned, Work};
use contracts::debug_requires;
use fb_core_arena::{Block, CopyOnly, Device, Heap, Kind, Pinned, RowId, StagingKind};
#[cfg(feature = "device")]
use fb_core_error_handling::check_or_abort;
#[cfg(feature = "device")]
use fb_core_thread_pool::LaneId;
use fb_core_thread_pool::{
    self as thread_pool, ArenaSize, Backend, DeviceStream, Lane, NodeId, Plan, Record,
};

/// A planned DAG made real: each worker's two blocks and the plan's heap
/// block, allocated once, and the [`thread_pool::Plan`] whose nodes hand
/// each task its rows of them.
/// Run any number of times, one run at a time ([`Instance::run`] borrows
/// it exclusively); a run allocates nothing.
///
/// Its GPU tasks run on the `thread_pool::Device` backend `Instance::backend`
/// makes (the `device` feature): one event per GPU node of this plan, and
/// each worker's declared BLAS workspace handed to its handle.
pub struct Instance<'a, H: StagingKind = Pinned, S = DeviceStream> {
    plan: Plan<'a, S>,
    /// Per worker, its device and host blocks, each shared with the nodes
    /// that partition it and, for the device one, with a backend's handle
    /// that holds its BLAS workspace row.
    pub(super) blocks: Vec<WorkerBlocks<H>>,
    /// The plan's heap block, shared with the CPU nodes that partition it.
    heap: Arc<Block<Heap>>,
    /// Per worker, per lane, whether the lane has library handles.
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    libraries: Vec<Vec<bool>>,
    /// Per worker, its BLAS workspace rows, each with its lane, if declared.
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    pub(super) blas_workspaces: Vec<Vec<(usize, RowId<Device, u8>)>>,
}

/// A worker's device block and host block.
type WorkerBlocks<H> = (Arc<Block<Device>>, Arc<Block<H>>);

impl<'a, H: StagingKind, S: 'a> Planned<'a, H, S> {
    /// Allocate each worker's blocks, worker `w`'s on device `devices[w]`,
    /// and the plan's heap block, from the global allocator, and build the executor's plan: a node per task in rank order, after
    /// its [`predecessors`](Planned::predecessors), and for each loop its
    /// decision, closing it over its body ([`Plan::add_loop`]), or for a
    /// counted loop its close after its body's last ([`Plan::add_counted`]).
    /// Each node hands its task views for the iteration it runs in, so a
    /// counted loop's iteration reaches its own copy of the body's rows. A table of
    /// zero bytes allocates nothing, so a DAG with no device rows touches
    /// no device. A loop's body is replayed on these blocks: nothing is
    /// allocated per iteration.
    ///
    /// Aborts, as [`Block::new`] does, if a block cannot be allocated.
    #[debug_requires(
        self.n_workers() <= devices.len(),
        "a device for every worker of the plan"
    )]
    #[track_caller]
    pub fn instantiate(self, devices: &[i32]) -> Instance<'a, H, S> {
        let mut blocks = Vec::with_capacity(self.tables.len());
        for ((device, host), &dev_idx) in self.tables.into_iter().zip(devices) {
            blocks.push((
                Arc::new(Block::new(dev_idx, device)),
                Arc::new(Block::new(dev_idx, host)),
            ));
        }
        // The heap needs no device: its index is never made current.
        let heap = Arc::new(Block::new(0, self.heap));
        let mut plan = Plan::new();
        let mut ids: Vec<NodeId> = Vec::with_capacity(self.nodes.len());
        let loops = self.loops;
        for (pos, (node, deps)) in self.nodes.into_iter().zip(self.deps).enumerate() {
            let preds: Vec<NodeId> = deps.iter().map(|&u| ids[u]).collect();
            let (device, host) = &blocks[node.at.worker];
            let (device, host) = (Arc::clone(device), Arc::clone(host));
            let heap = Arc::clone(&heap);
            let placement = node.placement();
            let id = match node.work {
                Work::Gpu(run) => plan.add(node.name, placement, &preds, move |lane| {
                    let Lane::Stream {
                        stream, iteration, ..
                    } = lane
                    else {
                        unreachable!("a GPU task is placed on a stream")
                    };
                    let pinned = CopyOnly::from(host.partition(pos, iteration));
                    run(&device.partition(pos, iteration), &pinned, stream)
                }),
                Work::Cpu(run) => plan.add(node.name, placement, &preds, move |lane| {
                    let Lane::Cpu { iteration } = lane else {
                        unreachable!("a CPU task is placed on the CPU pool")
                    };
                    run(
                        &host.partition(pos, iteration),
                        &heap.partition(pos, iteration),
                    )
                }),
                Work::Decide {
                    body,
                    max_iterations,
                    decide,
                } => plan.add_loop(node.name, ids[body], &preds, max_iterations, move |i| {
                    decide(&host.partition(pos, i), &heap.partition(pos, i), i)
                }),
            };
            plan.set_dims(id, &node.dims);
            ids.push(id);
            for l in loops.iter().filter(|l| l.last == pos) {
                if let Some((trips, depth)) = l.counted {
                    plan.add_counted(ids[l.first], trips, depth);
                }
            }
        }
        Instance {
            plan,
            blocks,
            heap,
            libraries: self.libraries,
            blas_workspaces: self.blas_workspaces,
        }
    }
}

impl<'a, H: StagingKind, S> Instance<'a, H, S> {
    /// Worker `worker`'s device block.
    pub fn device_block(&self, worker: usize) -> &Block<Device> {
        &self.blocks[worker].0
    }

    /// Worker `worker`'s host block.
    pub fn host_block(&self, worker: usize) -> &Block<H> {
        &self.blocks[worker].1
    }

    /// The plan's heap block.
    pub fn heap_block(&self) -> &Block<Heap> {
        &self.heap
    }

    /// Run every task once on `n_cpu` CPU workers and `backend`'s GPU
    /// workers: [`thread_pool::run`], whose [`Record`] times each task under
    /// the name it was added with and the dimensions its
    /// [`GpuTask::dims`](super::GpuTask::dims) or
    /// [`CpuTask::dims`](super::CpuTask::dims) gave, and whose nodes are numbered
    /// as the tasks' positions. The record's
    /// [`arena`](Record::arena) is the blocks: per worker, its device block
    /// and then its host block, then the plan's heap block under worker 0,
    /// each [`Table::total`] bytes.
    ///
    /// [`Table::total`]: fb_core_arena::Table::total
    ///
    /// The instance and the backend are both borrowed exclusively, as
    /// [`thread_pool::run`] borrows the plan and the backend: one run at a
    /// time on each. The blocks' borrow flags are host-side and a GPU
    /// task's row is returned while its device work may still be in
    /// flight, so a second run of the same blocks would race on them; and
    /// a backend's events and handles serve one run. Independent (instance,
    /// backend) pairs may run on threads of their own.
    ///
    /// # Errors
    /// In [`Record::result`], as [`thread_pool::run`]: the failure lowest
    /// in rank, with `name@worker` on its context.
    pub fn run<B: Backend<Stream = S>>(&mut self, backend: &mut B, n_cpu: usize) -> Record {
        let mut record = thread_pool::run(&mut self.plan, backend, n_cpu);
        for (worker, (device, host)) in self.blocks.iter().enumerate() {
            record.arena.push(ArenaSize {
                worker,
                kind: Device::NAME.to_string(),
                bytes: device.table().total(),
            });
            record.arena.push(ArenaSize {
                worker,
                kind: H::NAME.to_string(),
                bytes: host.table().total(),
            });
        }
        record.arena.push(ArenaSize {
            worker: 0,
            kind: Heap::NAME.to_string(),
            bytes: self.heap.table().total(),
        });
        record
    }
}

#[cfg(feature = "device")]
impl<H: StagingKind> Instance<'_, H, DeviceStream> {
    /// The backend this instance's GPU tasks run on: a
    /// [`thread_pool::Device`] whose GPU worker `w` is on worker `w`'s
    /// device with the lanes its tasks name, with an event pair for every
    /// GPU node of this plan, library handles on lane 0 and on each lane a
    /// task that uses them is placed on ([`GpuTask::uses_libraries`]), and
    /// each such lane's declared BLAS workspace ([`Dag::blas_workspace`])
    /// handed to its BLAS handle. A handle holds
    /// its row as a
    /// [`HeldRow`](fb_core_arena::HeldRow) for as long as the backend
    /// lives, which keeps the block alive with it, so the backend may
    /// outlive the instance; a second backend made while the first lives
    /// would find the row held.
    ///
    /// Aborts, as a failed create does, if a handle refuses its workspace.
    ///
    /// # Panics
    /// If a lane's workspace row is already held: a backend made from this
    /// instance before still lives.
    ///
    /// [`Dag::blas_workspace`]: super::Dag::blas_workspace
    /// [`GpuTask::uses_libraries`]: super::GpuTask::uses_libraries
    #[track_caller]
    pub fn backend(&self) -> thread_pool::Device {
        let devices: Vec<i32> = self.blocks.iter().map(|(d, _)| d.dev_idx()).collect();
        let mut backend = thread_pool::Device::new(&devices, &self.plan, &self.libraries);
        for (worker, rows) in self.blas_workspaces.iter().enumerate() {
            for &(lane, row) in rows {
                let held = self.blocks[worker].0.hold(row);
                check_or_abort(backend.bind_blas_workspace(LaneId { worker, lane }, held));
            }
        }
        backend
    }
}
