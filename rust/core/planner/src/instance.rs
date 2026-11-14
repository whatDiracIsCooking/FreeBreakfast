//! A planned DAG's blocks and its executor plan: what runs, any number of
//! times.

use std::sync::Arc;

use super::dag::Planned;
use contracts::debug_requires;
use fb_core_arena::{Block, Device, HostKind, Kind, Pinned, RowId};
#[cfg(feature = "device")]
use fb_core_error_handling::check_or_abort;
use fb_core_thread_pool::{
    self as thread_pool, ArenaSize, Backend, DeviceStream, NodeId, Plan, Record,
};

/// A planned DAG made real: each worker's two blocks, allocated once, and
/// the [`thread_pool::Plan`] whose nodes hand each task its rows of them.
/// Run any number of times, one run at a time ([`Instance::run`] borrows
/// it exclusively); a run allocates nothing.
///
/// Its GPU tasks run on the `thread_pool::Device` backend `Instance::backend`
/// makes (the `device` feature): one event per GPU node of this plan, and
/// each worker's declared BLAS workspace handed to its handle.
pub struct Instance<'a, H: HostKind = Pinned, S = DeviceStream> {
    plan: Plan<'a, S>,
    /// Per worker, its device and host blocks, each shared with the nodes
    /// that partition it and, for the device one, with a backend's handle
    /// that holds its BLAS workspace row.
    pub(super) blocks: Vec<WorkerBlocks<H>>,
    /// Per worker, its BLAS workspace row, if declared.
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    pub(super) blas_workspaces: Vec<Option<RowId<Device, u8>>>,
}

/// A worker's device block and host block.
type WorkerBlocks<H> = (Arc<Block<Device>>, Arc<Block<H>>);

impl<'a, H: HostKind, S: 'a> Planned<'a, H, S> {
    /// Allocate each worker's blocks, worker `w`'s on device `devices[w]`,
    /// and build the executor's plan: a node per task in rank order, after
    /// its [`predecessors`](Planned::predecessors). A table of zero bytes
    /// allocates nothing, so a DAG with no device rows touches no device.
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
        let mut plan = Plan::new();
        let mut ids: Vec<NodeId> = Vec::with_capacity(self.nodes.len());
        for (pos, (node, deps)) in self.nodes.into_iter().zip(self.deps).enumerate() {
            let preds: Vec<NodeId> = deps.iter().map(|&u| ids[u]).collect();
            let (device, host) = &blocks[node.worker];
            let (device, host) = (Arc::clone(device), Arc::clone(host));
            let run = node.run;
            let id = plan.add(node.name, node.placement, &preds, move |lane| {
                run(&device.partition(pos), &host.partition(pos), lane)
            });
            plan.set_dims(id, &node.dims);
            ids.push(id);
        }
        Instance {
            plan,
            blocks,
            blas_workspaces: self.blas_workspaces,
        }
    }
}

impl<'a, H: HostKind, S> Instance<'a, H, S> {
    /// Worker `worker`'s device block.
    pub fn device_block(&self, worker: usize) -> &Block<Device> {
        &self.blocks[worker].0
    }

    /// Worker `worker`'s host block.
    pub fn host_block(&self, worker: usize) -> &Block<H> {
        &self.blocks[worker].1
    }

    /// Run every task once on `n_cpu` CPU workers and `backend`'s stream
    /// workers: [`thread_pool::run`], whose [`Record`] times each task under
    /// the name it was added with and the dimensions its
    /// [`Task::dims`](super::Task::dims) gave, and whose nodes are numbered
    /// as the tasks' positions. The record's
    /// [`arena`](Record::arena) is the blocks: per worker, its device block
    /// and then its host block, each [`Table::total`] bytes.
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
        record
    }
}

#[cfg(feature = "device")]
impl<H: HostKind> Instance<'_, H, DeviceStream> {
    /// The backend this instance's GPU tasks run on: a
    /// [`thread_pool::Device`] whose stream worker `w` is on worker `w`'s
    /// device, with an event pair for every GPU node of this plan, and
    /// each worker's declared BLAS workspace ([`Dag::blas_workspace`])
    /// handed to its BLAS handle. The handle holds the row as a
    /// [`HeldRow`](fb_core_arena::HeldRow) for as long as the backend
    /// lives, which keeps the block alive with it, so the backend may
    /// outlive the instance; a second backend made while the first lives
    /// would find the row held.
    ///
    /// Aborts, as a failed create does, if a handle refuses its workspace.
    ///
    /// # Panics
    /// If a worker's workspace row is already held: a backend made from
    /// this instance before still lives.
    ///
    /// [`Dag::blas_workspace`]: super::Dag::blas_workspace
    #[track_caller]
    pub fn backend(&self) -> thread_pool::Device {
        let devices: Vec<i32> = self.blocks.iter().map(|(d, _)| d.dev_idx()).collect();
        let mut backend = thread_pool::Device::new(&devices, &self.plan);
        for (worker, row) in self.blas_workspaces.iter().enumerate() {
            if let Some(row) = *row {
                let held = self.blocks[worker].0.hold(row);
                check_or_abort(backend.bind_blas_workspace(worker, held));
            }
        }
        backend
    }
}
