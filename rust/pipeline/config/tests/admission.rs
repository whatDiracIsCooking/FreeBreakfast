//! Admission on a real card (device 0), as the binary makes it: the card
//! read by [`fb_core_arena::device_memory`] with the worker's stream and
//! handles made, and the kernels' stack by
//! [`fb_core_wwr::runtime::stack_growth`].
//!
//! The card's free memory moves with its other tenants, so nothing here
//! asserts it equals anything: a block of a PiB is past every card, one of
//! a MiB within any card the run can use at all, and the measured reserve
//! is held to its floor from one side.

use fb_core_error_handling::ErrorKind;
use fb_core_planner::{Dag, Plan};
use fb_core_thread_pool::DeviceStream;
use fb_pipeline_config::{
    DEVICE_RESERVE, KERNEL_STACK_BYTES, MemoryCaps, admit_plan, plan, read_config,
};

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
    let card = |_| {
        let _worker = DeviceStream::new(0);
        fb_core_arena::device_memory(0)
    };
    let stack = |_| fb_core_wwr::runtime::stack_growth(0, KERNEL_STACK_BYTES);
    let (free, total) = card(0).unwrap();
    println!("device 0: {free} bytes free of {total}");
    assert!(
        free > 0 && free < total,
        "the card has memory free and held"
    );
    let caps = MemoryCaps {
        max_device_mem_per_gpu: 1 << 62,
        ..MemoryCaps::default()
    };
    let error = admit_plan(&device_row(1 << 50), &caps, card, stack, usize::MAX).unwrap_err();
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
    admit_plan(&device_row(1 << 20), &caps, card, stack, usize::MAX).unwrap();
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
    let card = |w: usize| {
        let _worker = DeviceStream::new(w as i32);
        fb_core_arena::device_memory(w as i32)
    };
    let stack = |w: usize| fb_core_wwr::runtime::stack_growth(w as i32, KERNEL_STACK_BYTES);
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
        let planned = plan(&config, lwork).unwrap();
        assert_eq!(planned.n_workers(), n);
        // A cap past every card, so the card is what bounds the block: the
        // default's 8 GiB alone would refuse either count.
        let caps = MemoryCaps {
            max_device_mem_per_gpu: 1 << 62,
            ..config.memory
        };
        let admitted = admit_plan(&planned, &caps, card, stack, usize::MAX);
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
