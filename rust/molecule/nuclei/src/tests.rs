//! The nuclei's tasks for H2O and CO: the plan text, a committed golden per
//! molecule (`FB_BLESS=1` rewrites it), and the staged rows on heap blocks
//! and `HostOnly` workers (tier 3); the device rows copied back on a card
//! (tier 4, device 0).
//!
//! Every bar is bitwise: the rows hold the molecule's doubles, the charges
//! an exact conversion of a small integer, and a copy changes no bit.

use std::sync::{Arc, Mutex};

use super::*;
use fb_atom::{Atom, Element};
use fb_core_arena::Heap;
use fb_core_thread_pool::HostOnly;
use fb_core_types::Double3;
use fb_test_support::golden;

/// The two molecules, in bohr: the records' fixtures
/// (`fb_integrals_mmd`'s tests), no atom on an axis.
fn cases() -> [(&'static str, Molecule); 2] {
    let molecule = |atoms: &[(Element, [f64; 3])]| {
        Molecule::new(
            atoms
                .iter()
                .map(|&(element, [x, y, z])| Atom {
                    position: Double3 { x, y, z },
                    element,
                })
                .collect(),
            0,
            1,
        )
    };
    [
        (
            "h2o",
            molecule(&[
                (Element::O, [0.0, 0.0, 0.0]),
                (Element::H, [1.43, 0.21, -1.07]),
                (Element::H, [-1.38, 0.35, -1.12]),
            ]),
        ),
        (
            "co",
            molecule(&[
                (Element::C, [0.13, -0.21, 0.17]),
                (Element::O, [0.77, 0.86, 1.86]),
            ]),
        ),
    ]
}

/// The molecule's charges and positions as bits, in atom order: what the
/// rows must hold.
fn expected(molecule: &Molecule) -> (Vec<u64>, Vec<u64>) {
    let atoms = molecule.atoms();
    let z = atoms
        .iter()
        .map(|a| f64::from(a.element.z()).to_bits())
        .collect();
    let c = atoms
        .iter()
        .flat_map(|a| [a.position.x, a.position.y, a.position.z])
        .map(f64::to_bits)
        .collect();
    (z, c)
}

/// What [`Collect`] read: the two rows, as bits.
type Read = Arc<Mutex<(Vec<u64>, Vec<u64>)>>;

/// CPU: two host rows read into `out`, after their producer.
struct Collect<H> {
    rows: (RowId<H, f64>, RowId<H, f64>),
    out: Read,
}

impl<H: HostKind, S> Task<H, S> for Collect<H> {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, H>) {
        host.read(self.rows.0);
        host.read(self.rows.1);
    }

    fn run(&self, _: &(), _: &Bufs<'_, Device>, host: &Bufs<'_, H>, _: Lane<'_, S>) -> Status {
        let bits = |row| host.get(row).iter().map(|e: &f64| e.to_bits()).collect();
        *self.out.lock().unwrap() = (bits(self.rows.0), bits(self.rows.1));
        Ok(())
    }
}

/// The plan of each molecule's two tasks, its tables among it.
#[test]
fn plan_goldens() {
    for (stem, molecule) in cases() {
        let mut dag = Dag::new();
        let rows = add_nuclei(&mut dag, 0, &molecule);
        assert_eq!(rows.n_nuc, molecule.natoms() as i32);
        // The table's text ends an empty handoffs line in a space, which the
        // trailing-whitespace hook strips from the committed file.
        let text: String = dag
            .plan()
            .to_string()
            .lines()
            .map(|l| format!("{}\n", l.trim_end()))
            .collect();
        golden(
            env!("CARGO_MANIFEST_DIR"),
            &format!("plan_{stem}.txt"),
            &text,
        );
    }
}

/// The `nuclei` task's rows hold the molecule's charges and positions,
/// bitwise, run on heap blocks and a `HostOnly` worker.
#[test]
fn staged_rows_are_the_molecules_nuclei() {
    for (stem, molecule) in cases() {
        let mut dag: Dag<'_, Heap, ()> = Dag::new();
        let staged = dag.add(
            "nuclei",
            0,
            Stage {
                molecule: molecule.clone(),
            },
        );
        let out = Read::default();
        dag.add(
            "collect",
            0,
            Collect {
                rows: staged,
                out: Arc::clone(&out),
            },
        );
        let mut instance = dag.plan().instantiate(&[0]);
        instance.run(&mut HostOnly::new(1), 1).result.unwrap();
        assert_eq!(*out.lock().unwrap(), expected(&molecule), "{stem}");
    }
}

/// GPU: the device rows copied down into two pinned rows on the stream.
struct Download(NucleiRows);

impl Task for Download {
    type Rows = (RowId<Pinned, f64>, RowId<Pinned, f64>);
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        self.0.read(d);
        let n = self.0.n_nuc as usize;
        (h.output("down z", n), h.output("down c", 3 * n))
    }

    fn run(
        &self,
        rows: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        for (src, dst) in [(self.0.z, rows.0), (self.0.c, rows.1)] {
            fb_core_arena::memcpy_d2p(h, dst, d.get(src), stream)?;
        }
        Ok(())
    }
}

/// The device rows, copied back, equal the staged rows bitwise: the
/// molecule's charges and positions, which the tier-3 case shows the
/// staged rows hold. One plan per molecule, run once on device 0.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn device_rows_are_the_staged_rows() {
    for (stem, molecule) in cases() {
        let mut dag = Dag::new();
        let rows = add_nuclei(&mut dag, 0, &molecule);
        let down = dag.add("download", 0, Download(rows));
        let out = Read::default();
        dag.add(
            "collect",
            0,
            Collect {
                rows: down,
                out: Arc::clone(&out),
            },
        );
        let mut instance = dag.plan().instantiate(&[0]);
        let mut backend = instance.backend();
        instance.run(&mut backend, 1).result.unwrap();
        assert_eq!(*out.lock().unwrap(), expected(&molecule), "{stem}");
    }
}
