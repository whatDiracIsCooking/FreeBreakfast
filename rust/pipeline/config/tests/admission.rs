//! Admission on a real card (device 0), as the binary makes it: the card
//! read once by [`Card::read`], its memory by
//! [`fb_core_arena::device_memory`] with the worker's stream and handles
//! made, and the kernels' stack by [`fb_core_wwr::runtime::stack_growth`];
//! and the adaptive widths planned in that reading's room, which
//! admission takes.
//!
//! The card's free memory moves with its other tenants, so nothing here
//! asserts it equals anything: a block of a PiB is past every card, one of
//! a MiB within any card the run can use at all, and the measured reserve
//! is held to its floor from one side. A width is held to the one reading
//! it was planned from, never to a second.

use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{ErrorKind, Status};
use fb_core_planner::{Adaptive, Dag, GpuTask, Plan};
use fb_core_thread_pool::DeviceStream;
use fb_pipeline_config::{Card, DEVICE_RESERVE, MemoryCaps, admit_plan, plan, read_config};

/// A plan of one device worker row of `bytes`.
fn device_row(bytes: usize) -> Plan<'static> {
    let mut dag = Dag::new();
    let _ = dag.device_worker_row::<u8>(0, "device row", bytes);
    dag.plan()
}

/// The bytes the refusal `message` names right after `term`.
fn bytes_after(message: &str, term: &str) -> usize {
    let at = message
        .find(term)
        .unwrap_or_else(|| panic!("{term}: {message}"))
        + term.len();
    let digits: String = message[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("{term}: {message}"))
}

/// Under a cap of 2^62 bytes, past every card, a block of a PiB is refused
/// by the card, `Allocation`, naming its free memory, the measured reserve
/// and the stack; one of a MiB is admitted. Without the bound, the cap
/// alone would admit both. The reserve is at least [`DEVICE_RESERVE`] and
/// short of the card. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_cap_past_the_card_is_bounded_by_it() {
    let card = |_| Card::read(0);
    let Card { free, total, stack } = card(0).unwrap();
    println!("device 0: {free} bytes free of {total}, {stack} of stack to take");
    assert!(
        free > 0 && free < total,
        "the card has memory free and held"
    );
    let caps = MemoryCaps {
        max_device_mem_per_gpu: 1 << 62,
        ..MemoryCaps::default()
    };
    let error = admit_plan(&device_row(1 << 50), &caps, card, usize::MAX).unwrap_err();
    println!("{error}");
    assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
    assert!(error.message.contains("bytes free at planning"), "{error}");
    assert!(
        error.message.contains("over the cap alone: device row"),
        "{error}"
    );
    let reserve = bytes_after(&error.message, "the reserve of ");
    assert!(reserve >= DEVICE_RESERVE, "{error}");
    assert!(
        reserve < bytes_after(&error.message, "the card's "),
        "{error}"
    );
    let _ = bytes_after(&error.message, "the kernels' stack of ");
    admit_plan(&device_row(1 << 20), &caps, card, usize::MAX).unwrap();
}

/// Ten glycines in STO-3G over the dense conventional ERI on one card
/// (`num_gpus = 1`): `M_J` and `M_K` are 50.5 GB together, 237 Cartesian
/// AOs to the fourth twice, refused at planning by the binary's admission,
/// `Allocation`, naming the blocks, on every card of this tree's (the
/// A6000's 47.4 GiB the largest, which with its reserve and the kernels'
/// stack holds less); on two cards each holds half, admitted. The cap is
/// raised past the cards, as a run of it would raise it
/// (`[memory] max_device_mem_per_gpu`), so the card bounds the block.
/// Nothing is allocated. Needs two GPUs for the second half, which it
/// skips with a line on one.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_dense_eri_of_ten_glycines_is_refused_on_one_card() {
    let gly10 = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../scf/tests/data/gly10.xyz")
        .canonicalize()
        .unwrap();
    let card = |w: usize| Card::read(w as i32);
    let lwork = |n| {
        fb_recipes_diagonalize::syevd_lwork(
            &fb_core_wwr::solver::SolverHandle::new(std::rc::Rc::new(
                fb_core_wwr::runtime::Stream::new(0),
            )),
            n,
        )
    };
    let gpus = [1, 2];
    for n in gpus {
        if n > fb_core_wwr::runtime::device_count().unwrap() {
            println!("num_gpus = {n}: skipped, not visible");
            continue;
        }
        let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("gly10_{n}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let text = format!(
            "calculation = \"rhf\"\nnum_gpus = {n}\n[molecule]\nxyz = \"{}\"\n\
             [basis_set]\nbasis = \"sto-3g\"\n[eri]\nmode = \"conventional\"\n\
             storage = \"dense\"\n",
            gly10.display()
        );
        std::fs::write(&path, text).unwrap();
        let config = read_config(&path).unwrap();
        let planned = plan(&config, lwork, card).unwrap();
        assert_eq!(planned.n_workers(), n);
        // A cap past every card, so the card is what bounds the block: the
        // default's 8 GiB alone would refuse either count.
        let caps = MemoryCaps {
            max_device_mem_per_gpu: 1 << 62,
            ..config.memory
        };
        let admitted = admit_plan(&planned, &caps, card, usize::MAX);
        if n == 1 {
            let error = admitted.unwrap_err();
            println!("num_gpus = 1: {error}");
            assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
            assert!(error.message.contains("M_J"), "{error}");
        } else {
            admitted.unwrap();
            println!("num_gpus = {n}: admitted");
        }
    }
}

/// A part of `width` columns of [`COLUMN`] bytes on the device, staged
/// through a pinned window of two of them, up to 64 columns.
struct Columns;

/// A column's bytes: 1 MiB, a multiple of the device alignment.
const COLUMN: usize = 1 << 20;

impl Adaptive for Columns {
    fn width_max(&self) -> usize {
        64
    }

    fn bytes(&self, width: usize) -> usize {
        COLUMN * width
    }

    fn pinned_bytes(&self, width: usize) -> usize {
        2 * COLUMN * width
    }
}

/// The part's task: its columns as device scratch, its window as pinned.
struct Batch(usize);

impl GpuTask for Batch {
    type Rows = RowId<Device, u8>;

    fn layout(&self, d: &mut Layout<'_, Device>, p: &mut Layout<'_, Pinned>) -> Self::Rows {
        let _: RowId<Pinned, u8> = p.scratch("window", 2 * COLUMN * self.0);
        d.scratch("batch", COLUMN * self.0)
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        _: &DeviceStream,
    ) -> Status {
        Ok(())
    }
}

/// The fixed rows beside the part: `fixed` bytes on the device and as
/// many pinned, live for the whole plan.
fn fixed_and_batch(
    dag: &mut Dag<'static>,
    fixed: usize,
    width: impl FnOnce(&mut Dag<'static>) -> usize,
) -> usize {
    let _ = dag.device_worker_row::<u8>(0, "fixed device", fixed);
    let _ = dag.host_worker_row::<u8>(0, "fixed pinned", fixed);
    let width = width(dag);
    dag.add_gpu("batch", 0, Batch(width));
    width
}

/// The part planned at `width` by hand, for admission.
fn at(fixed: usize, width: usize) -> Plan<'static> {
    let mut dag = Dag::new();
    fixed_and_batch(&mut dag, fixed, |_| width);
    dag.plan()
}

/// `card`'s room bisected for [`Columns`] beside `fixed` bytes: the width
/// [`Dag::plan_within`] picks in [`Card::room`] under `caps` is admitted
/// by [`admit_plan`] from the same reading, and the next width up is
/// refused by it, `Allocation`, naming `kind`'s block.
fn widest_is_admitted(card: Card, caps: &MemoryCaps, fixed: usize, expected: usize, kind: &str) {
    let room = |_| Ok(card.room(caps));
    let (planned, width) = Dag::plan_within(room, |dag| {
        Ok(fixed_and_batch(dag, fixed, |dag| {
            dag.width("batch", 0, &Columns)
        }))
    })
    .unwrap();
    println!("{planned}");
    assert_eq!(width, expected);
    admit_plan(&planned, caps, |_| Ok(card), usize::MAX).unwrap();
    admit_plan(&at(fixed, width), caps, |_| Ok(card), usize::MAX).unwrap();
    let error = admit_plan(&at(fixed, width + 1), caps, |_| Ok(card), usize::MAX).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
    assert!(
        error.message.contains(&format!("gpu0: the {kind} block")),
        "{error}"
    );
}

/// Readings the case gives: under a device cap 20.5 columns above the
/// fixed rows, the reserve and the stack, the width is 20 and 21 is
/// refused; under a pinned cap 2 × 12.5 columns above the fixed pinned
/// row, with the device cap past it, the window binds, 12 and 13.
#[test]
fn the_planned_width_is_the_widest_admission_takes() {
    let gib = 1usize << 30;
    let card = Card {
        free: 40 * gib,
        total: 48 * gib,
        stack: gib,
    };
    let fixed = 64 * COLUMN;
    let caps = MemoryCaps {
        max_device_mem_per_gpu: card.reserve() + card.stack + fixed + 20 * COLUMN + COLUMN / 2,
        max_pinned_mem_per_gpu: 1 << 40,
        ..MemoryCaps::default()
    };
    widest_is_admitted(card, &caps, fixed, 20, "device");
    let caps = MemoryCaps {
        max_device_mem_per_gpu: 1 << 50,
        max_pinned_mem_per_gpu: fixed + 2 * 12 * COLUMN + COLUMN,
        ..MemoryCaps::default()
    };
    widest_is_admitted(card, &caps, fixed, 12, "pinned");
}

/// Device 0's card, read once: under a `[memory]` cap just above the
/// plan's fixed bytes, the reading's reserve and its stack (20.5 columns
/// above them), the width planned in the reading's room is 20, admitted
/// by [`admit_plan`] from that reading, and 21 is refused by it. Needs a
/// GPU with the cap's bytes, under 3 GiB past the reserve and the stack.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_width_from_the_card_is_the_widest_admission_takes() {
    let card = Card::read(0).unwrap();
    println!("device 0: {card:?}, a reserve of {}", card.reserve());
    let fixed = 64 * COLUMN;
    let cap = card.reserve() + card.stack + fixed + 20 * COLUMN + COLUMN / 2;
    assert!(
        cap < card.total,
        "the cap binds before the card: {cap} of {}",
        card.total
    );
    let caps = MemoryCaps {
        max_device_mem_per_gpu: cap,
        max_pinned_mem_per_gpu: 1 << 40,
        ..MemoryCaps::default()
    };
    widest_is_admitted(card, &caps, fixed, 20, "device");
}
