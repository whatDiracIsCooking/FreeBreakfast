//! The SAD guess uses exactly the threads its plan says (tier 3,
//! `docs/design.md`, "What it holds": the thread count after a device-free
//! run is compared with the plan's worker total). Formaldehyde in cc-pVDZ
//! read by path, so every stamp is an atomic SCF, on `n_cpu` CPU workers and
//! no stream worker: a probe after D sees the main thread and the `n_cpu`
//! workers and no more, so no SCF left a pool of its own behind, and after
//! the run the main thread is alone again. One run per worker count.
//!
//! The count is the `Threads:` line of `/proc/self/status`. `harness =
//! false`: libtest runs cases on threads of its own, which would be counted.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fb_atom::{Atom, Element};
use fb_core_arena::{Bufs, Device, Heap, Layout, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{HostOnly, Lane};
use fb_core_types::Double3;
use fb_molecule::Molecule;
use fb_pipeline_guess::add_sad_guess;

fn threads() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    status
        .lines()
        .find_map(|l| l.strip_prefix("Threads:"))
        .expect("a Threads: line")
        .trim()
        .parse()
        .expect("a thread count")
}

/// CPU, after D: the thread count, once it reaches `expected` (workers are
/// spawned one after another) or five seconds pass.
struct Probe {
    d: RowId<Heap, f64>,
    expected: usize,
    seen: Arc<Mutex<Option<usize>>>,
}

impl Task<Heap, ()> for Probe {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Heap>) {
        host.read(self.d);
    }

    fn run(&self, _: &(), _: &Bufs<'_, Device>, _: &Bufs<'_, Heap>, _: Lane<'_, ()>) -> Status {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut now = threads();
        while now < self.expected && Instant::now() < deadline {
            std::thread::yield_now();
            now = threads();
        }
        *self.seen.lock().unwrap() = Some(now);
        Ok(())
    }
}

fn main() {
    assert_eq!(threads(), 1, "the process starts with one thread");
    let at = |x, y, z| Double3 { x, y, z };
    let atoms = vec![
        Atom {
            position: at(0.0, 0.0, 0.0),
            element: Element::C,
        },
        Atom {
            position: at(0.0, 0.0, 2.2771),
            element: Element::O,
        },
        Atom {
            position: at(0.0, 1.7817, -1.1111),
            element: Element::H,
        },
        Atom {
            position: at(0.0, -1.7817, -1.1111),
            element: Element::H,
        },
    ];
    let path = format!(
        "{}/../../../data/basis/cc-pVDZ.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let basis = fb_io_basis::resolve_basis(&path)
        .unwrap()
        .primitive_normalized()
        .place(&Molecule::new(atoms.clone(), 0, 1))
        .unwrap();
    for n_cpu in [1, 4] {
        let (n_streams, seen) = (0, Arc::new(Mutex::new(None)));
        let expected = 1 + n_cpu + n_streams;
        let mut dag: Dag<'static, Heap, ()> = Dag::new();
        let d = add_sad_guess(&mut dag, 0, &basis, &atoms).unwrap();
        dag.add(
            "probe",
            0,
            Probe {
                d,
                expected,
                seen: Arc::clone(&seen),
            },
        );
        let mut instance = dag.plan().instantiate(&[0]);
        instance
            .run(&mut HostOnly::new(n_streams), n_cpu)
            .result
            .unwrap();
        drop(instance);
        let seen = seen.lock().unwrap().expect("the probe ran");
        assert_eq!(seen, expected, "{n_cpu} CPU workers");
        assert_eq!(threads(), 1, "a worker outlived its run");
    }
    println!("guess_threads: ok");
}
