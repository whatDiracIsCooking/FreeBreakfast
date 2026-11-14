//! A planned DAG's blocks and its executor plan: what runs, any number of
//! times.

use std::sync::Arc;

use super::dag::Planned;
use contracts::debug_requires;
use fb_core_arena::{Block, Device, HostKind, Kind, Pinned};
use fb_core_thread_pool::{
    self as thread_pool, ArenaSize, Backend, DeviceStream, NodeId, Plan, Record,
};

/// A planned DAG made real: each worker's two blocks, allocated once, and
/// the [`thread_pool::Plan`] whose nodes hand each task its rows of them.
/// Run any number of times; a run allocates nothing.
///
/// The executor's plan is what a `thread_pool::Device` backend (the
/// `device` feature) is built over ([`Instance::plan`]), since it makes one
/// event per GPU node, once.
pub struct Instance<'a, H: HostKind = Pinned, S = DeviceStream> {
    plan: Plan<'a, S>,
    blocks: Vec<Arc<(Block<Device>, Block<H>)>>,
}

impl<'a, H: HostKind, S: 'a> Planned<'a, H, S> {
    /// Allocate each worker's blocks, worker `w`'s on device `devices[w]`,
    /// and build the executor's plan: a node per task in rank order, after
    /// its [`predecessors`](Planned::predecessors). A table of zero bytes
    /// allocates nothing, so a DAG with no device rows touches no device.
    ///
    /// Requires, checked in debug builds: `devices` has an entry per worker.
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
            blocks.push(Arc::new((
                Block::new(dev_idx, device),
                Block::new(dev_idx, host),
            )));
        }
        let mut plan = Plan::new();
        let mut ids: Vec<NodeId> = Vec::with_capacity(self.nodes.len());
        for (pos, (node, deps)) in self.nodes.into_iter().zip(self.deps).enumerate() {
            let preds: Vec<NodeId> = deps.iter().map(|&u| ids[u]).collect();
            let worker = Arc::clone(&blocks[node.worker]);
            let run = node.run;
            let id = plan.add(node.name, node.placement, &preds, move |lane| {
                let (device, host) = &*worker;
                run(&device.partition(pos), &host.partition(pos), lane)
            });
            plan.set_dims(id, &node.dims);
            ids.push(id);
        }
        Instance { plan, blocks }
    }
}

impl<'a, H: HostKind, S> Instance<'a, H, S> {
    /// The executor's plan, to build a backend over.
    pub fn plan(&self) -> &Plan<'a, S> {
        &self.plan
    }

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
    /// # Errors
    /// In [`Record::result`], as [`thread_pool::run`]: the failure lowest
    /// in rank, with `name@worker` on its context.
    pub fn run<B: Backend<Stream = S>>(&self, backend: &B, n_cpu: usize) -> Record {
        let mut record = thread_pool::run(&self.plan, backend, n_cpu);
        for (worker, blocks) in self.blocks.iter().enumerate() {
            let (device, host) = &**blocks;
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
