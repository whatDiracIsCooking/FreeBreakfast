//! Admission: a plan held to the `[memory]` caps after planning and before
//! the blocks are made (`docs/memory.md`, "Admission").

use fb_core_arena::{admit, admit_host};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Plan, Room};
use fb_core_thread_pool::DeviceStream;

use crate::MemoryCaps;

/// The floor of a device's reserve, bytes: the C++ `kDeviceReserve` of
/// 256 MiB, what a card holds outside any block, the context, the library
/// handles and the loaded modules, measured on the RTX 3080 and rounded
/// up. The reserve [`admit_plan`] takes is measured per device, the card's
/// total less its free memory once the worker's handles exist, and is
/// never less than this: lazily loaded modules grow it after the reading
/// (`docs/memory.md`, "Admission").
pub const DEVICE_RESERVE: usize = 256 << 20;

/// The largest per-thread stack of any kernel the tree launches, bytes:
/// the C++ `kKernelStackBytes`, `potential_hessian<3,3>`'s 16,912, which
/// [`admit_plan`]'s stack term is asked for
/// ([`fb_core_wwr::runtime::stack_growth`]). It bounds the Rust forks of
/// C++ tree kernels (`docs/gpu_layer.md`, "What the port changes"), which
/// the C++ ctest `core_kernel_stack_bound` holds to it, and the rust-only
/// conventional ERI units FelizJueves emits, whose largest frame is 3,856
/// bytes (`eri_2222` and the dense `eri_dj_2222` / `eri_dk_2222` in the
/// low-stack form, `sm_86` ptxas, 0 spill); nothing reads the bound off the
/// Rust libraries themselves.
pub const KERNEL_STACK_BYTES: usize = 16_912;

/// One reading of a GPU worker's card at planning, in bytes: the one
/// [`plan`] sizes the adaptive widths from and [`admit_plan`] admits the
/// device block by, so the two cannot disagree (`docs/memory.md`,
/// "Admission", "Adaptive widths: one planner routine"). The binary reads
/// each worker's card once, when planning or admission first asks for it.
///
/// [`plan`]: crate::plan
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Card {
    /// The card's free memory, read once the worker's stream and handles
    /// exist ([`fb_core_arena::device_memory`]).
    pub free: usize,
    /// The card's total memory.
    pub total: usize,
    /// What the kernels' stack can still take, read after the free memory
    /// ([`fb_core_wwr::runtime::stack_growth`] for [`KERNEL_STACK_BYTES`]).
    pub stack: usize,
}

impl Card {
    /// Read device `device`'s card, its stream and handles made for the
    /// reading, so the reserve holds them; the run's own are made with
    /// its backend. Touches the device.
    ///
    /// # Errors
    /// [`fb_core_arena::device_memory`]'s or
    /// [`fb_core_wwr::runtime::stack_growth`]'s error.
    pub fn read(device: i32) -> Result<Card> {
        let _worker = DeviceStream::new(device);
        let (free, total) = fb_core_arena::device_memory(device)?;
        let stack = fb_core_wwr::runtime::stack_growth(device, KERNEL_STACK_BYTES)?;
        Ok(Card { free, total, stack })
    }

    /// The device's reserve: the card's total less its free memory, at
    /// least [`DEVICE_RESERVE`].
    pub fn reserve(&self) -> usize {
        self.total.saturating_sub(self.free).max(DEVICE_RESERVE)
    }

    /// The worker's room under `memory`'s caps: on the device, the smaller
    /// of [`MemoryCaps::max_device_mem_per_gpu`] and the card's total, less
    /// the reserve and the stack, what [`admit_plan`] admits the device
    /// block against; pinned, [`MemoryCaps::max_pinned_mem_per_gpu`], what
    /// it admits the pinned block against.
    ///
    /// ```
    /// use fb_pipeline_config::{Card, DEVICE_RESERVE, MemoryCaps};
    ///
    /// let gib = 1usize << 30;
    /// let card = Card { free: 40 * gib, total: 48 * gib, stack: gib };
    /// let caps = MemoryCaps::default();
    /// assert_eq!(card.reserve(), 8 * gib);
    /// assert_eq!(card.room(&caps).device, 0);
    /// let caps = MemoryCaps { max_device_mem_per_gpu: 1 << 50, ..caps };
    /// assert_eq!(card.room(&caps).device, 39 * gib);
    /// assert_eq!(card.room(&caps).pinned, 2 * gib);
    /// let idle = Card { free: 48 * gib, ..card };
    /// assert_eq!(idle.reserve(), DEVICE_RESERVE);
    /// ```
    pub fn room(&self, memory: &MemoryCaps) -> Room {
        Room {
            device: memory
                .max_device_mem_per_gpu
                .min(self.total)
                .saturating_sub(self.reserve())
                .saturating_sub(self.stack),
            pinned: memory.max_pinned_mem_per_gpu,
        }
    }
}

/// Admit `planned` against `memory`'s caps, as the binary does after
/// [`plan`] and before it makes the blocks (`docs/memory.md`,
/// "Admission"). In order:
///
/// 1. The node: every GPU worker's pinned cap together must fit
///    `host_available`, the host memory the plan may take in bytes
///    ([`fb_core_arena::host_available`]: `MemAvailable` within the
///    cgroup's limit), since pinned memory is not pageable (the C++
///    `check_pinned_node`).
/// 2. Each worker's device table, [`admit`]ted against its
///    [`Card::room`]: the smaller of [`MemoryCaps::max_device_mem_per_gpu`]
///    and its card's total memory, less the reserve and the stack,
///    `block + reserve + stack <= cap`. `card(worker)` is the worker's one
///    reading of its card, the one [`plan`] sized the widths from. So a
///    cap past the card admits its free memory, and a card another
///    process holds part of shows that part in the reserve, as a smaller
///    cap. It is asked only of a worker whose device table holds a row,
///    so a plan with none touches no device.
/// 3. Each worker's pinned table, against
///    [`MemoryCaps::max_pinned_mem_per_gpu`], the pinned room.
/// 4. The plan's heap table, against [`MemoryCaps::max_host_mem`].
/// 5. The node again, by its blocks: every worker's pinned block and the
///    heap block together against `host_available`
///    ([`fb_core_arena::admit_host`]).
///
/// Each worker admitted is logged at info, its blocks and their caps, and
/// the heap block and the node's sum after them.
///
/// # Errors
/// - `InvalidConfig`, naming the key, the workers and the available
///   memory: the pinned caps together past it.
/// - `card`'s error.
/// - [`admit`]'s refusal, `Allocation`, naming the worker (`gpu<w>`), the
///   kind, the bytes over, every row over the cap alone and the rows live
///   at the peak; then the cap's terms: the key, and for a device block
///   the card's total, the reserve with its reading and its floor, and
///   the stack; for the heap block, `plan` for the worker and the key.
/// - [`admit_host`]'s refusal, `Allocation`, naming each kind's bytes and
///   the available memory.
///
/// [`plan`]: crate::plan
pub fn admit_plan(
    planned: &Plan<'_>,
    memory: &MemoryCaps,
    mut card: impl FnMut(usize) -> Result<Card>,
    host_available: usize,
) -> Status {
    let workers = planned.n_workers();
    let pinned_cap = memory.max_pinned_mem_per_gpu;
    // workers * cap > available, without the product's overflow.
    if workers != 0 && pinned_cap > host_available / workers {
        return Err(Error::config(format!(
            "{workers} GPU workers, each allowed [memory] max_pinned_mem_per_gpu = {pinned_cap} \
             bytes ({:.3} GiB) of pinned memory, can pin more than the node's MemAvailable \
             within its cgroup's limit, {host_available} bytes ({:.3} GiB); pinned memory is \
             not pageable",
            gib(pinned_cap),
            gib(host_available)
        )));
    }
    for w in 0..workers {
        let worker = format!("gpu{w}");
        let device = planned.device_table(w);
        let admitted = if device.total() == 0 {
            "no device block".to_owned()
        } else {
            let cap = memory.max_device_mem_per_gpu;
            let card = card(w)?;
            let Card { free, total, stack } = card;
            let reserve = card.reserve();
            let room = card.room(memory).device;
            admit(device, room, &worker).map_err(|mut e| {
                e.message = format!(
                    "{}; the cap is the smaller of [memory] max_device_mem_per_gpu = {cap} bytes \
                     and the card's {total} bytes, less the reserve of {reserve} bytes (the card's \
                     total less its {free} bytes free at planning, at least {DEVICE_RESERVE}) \
                     and the kernels' stack of {stack} bytes",
                    e.message
                );
                e
            })?;
            format!(
                "a device block of {} bytes against a cap of {room}",
                device.total()
            )
        };
        let host = planned.host_table(w);
        admit(host, pinned_cap, &worker).map_err(|mut e| {
            e.message = format!(
                "{}; the cap is [memory] max_pinned_mem_per_gpu = {pinned_cap} bytes",
                e.message
            );
            e
        })?;
        log::info!(
            "{worker} admitted: {admitted}, a pinned block of {} bytes against a cap of \
             {pinned_cap}",
            host.total()
        );
    }
    let heap = planned.heap_table();
    let heap_cap = memory.max_host_mem;
    admit(heap, heap_cap, "plan").map_err(|mut e| {
        e.message = format!(
            "{}; the cap is [memory] max_host_mem = {heap_cap} bytes",
            e.message
        );
        e
    })?;
    let pinned: Vec<usize> = (0..workers)
        .map(|w| planned.host_table(w).total())
        .collect();
    admit_host(&pinned, heap.total(), host_available)?;
    log::info!(
        "plan admitted: a heap block of {} bytes against a cap of {heap_cap}, and the host \
         blocks, {} bytes, against the {host_available} bytes available",
        heap.total(),
        pinned.iter().sum::<usize>() + heap.total()
    );
    Ok(())
}

/// `bytes` in GiB, for a message.
pub(crate) fn gib(bytes: usize) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}
