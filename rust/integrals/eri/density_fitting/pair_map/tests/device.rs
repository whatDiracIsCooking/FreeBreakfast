//! The map's tables on a card (tier 4, device 0): `fb_integrals_sys`'s
//! [`PackedTables`] filled by [`Packing::write_tables`], and its pack and
//! gather held **bitwise** to the host map's [`Packing::pack`] and
//! [`Packing::unpack`], one run per packing. Each device result is a copy,
//! or one multiply by a weight of 1 or 2, exact in fp64, so there is no
//! tolerance to set.
//!
//! One basis of s, p, d and f shells on three centres, screened so that
//! some off-diagonal pairs and one diagonal pair are dropped, over both
//! packings, Cartesian and spherical:
//!
//! - **pack D**: a symmetric density packed, `w_c D[m, n]` per column.
//! - **gather**: a batch of 3 aux functions of random columns into three
//!   square matrices, each the host's unpack of its function.
//! - **unpack J**: one packed vector gathered (`b = 1`) into the square.
//! - **unpack(pack)**: a symmetric matrix supported on the kept pairs,
//!   packed on the host and unpacked on the card, is itself.
//!
//! Beside them, the v0.0.4 layer's `fb_integrals_sys::packed_gather`, the
//! tables handed in on the host, gathers the same batch and unpacks the
//! same J, each held bitwise to the old layer's result: the same kernel on
//! the same inputs. That runs over the basis above and over water in
//! cc-pVDZ, unscreened.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]

use std::sync::{Arc, Mutex};

use fb_atom::{Atom, Element};
use fb_basis::{BasisSet, Shell};
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId};
use fb_core_budget::{Backing, Caps, Ledger};
use fb_core_device::Stream;
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_types::Double3;
use fb_integrals_eri_density_fitting_pair_map::{KeptPair, Packing, PairMap};
use fb_integrals_mmd::record_tables;
use fb_integrals_sys::{PackedTables, add_packed_tables, packed_gather};
use fb_molecule::Molecule;

/// The batch's aux functions.
const B: usize = 3;

/// CPU: one input written into a pinned row.
struct Stage {
    name: &'static str,
    data: Vec<f64>,
}

impl CpuTask for Stage {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        host.output(self.name, self.data.len())
    }

    fn run(&self, row: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        host.take_mut(*row).copy_from_slice(&self.data);
        Ok(())
    }
}

/// The device inputs and outputs of [`UnderTest`].
#[derive(Clone, Copy)]
struct Rows {
    dp: RowId<Device, f64>,
    slab: RowId<Device, f64>,
    j: RowId<Device, f64>,
    back: RowId<Device, f64>,
}

/// GPU: the pack of `d`, the gather of `batch`, and the unpacks of `jp`
/// and `sp`, each into an output of its own.
struct UnderTest {
    tables: PackedTables,
    nao: usize,
    n_columns: usize,
    d: RowId<Device, f64>,
    batch: RowId<Device, f64>,
    jp: RowId<Device, f64>,
    sp: RowId<Device, f64>,
}

impl GpuTask for UnderTest {
    type Rows = Rows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Rows {
        self.tables.read(device);
        for row in [self.d, self.batch, self.jp, self.sp] {
            device.read(row);
        }
        let square = self.nao * self.nao;
        Rows {
            dp: device.output("dp", self.n_columns),
            slab: device.output("slab", B * square),
            j: device.output("j", square),
            back: device.output("back", square),
        }
    }

    fn uses_libraries(&self) -> bool {
        false
    }

    fn run(
        &self,
        rows: &Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let t = &self.tables;
        t.pack(
            stream,
            device,
            &device.get(self.d),
            &mut device.take_mut(rows.dp),
        )?;
        t.gather(
            stream,
            device,
            B,
            &device.get(self.batch),
            &mut device.take_mut(rows.slab),
        )?;
        t.gather(
            stream,
            device,
            1,
            &device.get(self.jp),
            &mut device.take_mut(rows.j),
        )?;
        t.gather(
            stream,
            device,
            1,
            &device.get(self.sp),
            &mut device.take_mut(rows.back),
        )
    }
}

/// CPU: the downloaded results read out, in order.
struct Collect {
    rows: Vec<RowId<Pinned, f64>>,
    out: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        for &row in &self.rows {
            host.read(row);
        }
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        *self.out.lock().unwrap() = self.rows.iter().map(|&r| host.get(r).to_vec()).collect();
        Ok(())
    }
}

/// `n` values of mixed sign and magnitude, distinct, from `seed`: a
/// misplaced copy shows, and a lost bit too.
fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|i| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (x >> 11) as f64 / (1_u64 << 53) as f64;
            (u - 0.5) * 10_f64.powi((i % 7) as i32 - 3) + i as f64 * 1e-9
        })
        .collect()
}

/// A symmetric `n × n` matrix from `seed`.
fn symmetric(n: usize, seed: u64) -> Vec<f64> {
    let v = values(n * n, seed);
    let mut a = vec![0.0; n * n];
    for m in 0..n {
        for k in 0..=m {
            a[m * n + k] = v[m * n + k];
            a[k * n + m] = v[m * n + k];
        }
    }
    a
}

/// The basis's map, screened.
fn screened_map() -> PairMap {
    let at = |x: f64, z: f64| Double3 { x, y: 0.25 * x, z };
    let basis = BasisSet::new(vec![
        Shell::zipped(0, &[3.0, 0.5], &[0.6, 0.4], at(0.0, 0.0)),
        Shell::zipped(2, &[0.8], &[1.0], at(0.0, 0.0)),
        Shell::zipped(1, &[1.2], &[1.0], at(1.4, 0.3)),
        Shell::zipped(3, &[0.6], &[1.0], at(1.4, 0.3)),
        Shell::zipped(0, &[0.9], &[1.0], at(-1.1, 1.0)),
        Shell::zipped(2, &[1.5], &[1.0], at(-1.1, 1.0)),
    ]);
    let tables = record_tables(&basis);
    // Every fifth pair's bound below the threshold: off-diagonal pairs and
    // at least one diagonal one dropped.
    let bounds: Vec<f64> = (0..tables.n_pairs())
        .map(|p| if p % 5 == 2 { 1e-12 } else { 1.0 })
        .collect();
    let map = PairMap::new(&tables, Some(&bounds), 1e-10);
    let diagonal = |p: &&KeptPair| p.shells[0] == p.shells[1];
    let kept_diagonal = map.pairs().iter().filter(diagonal).count();
    assert!(kept_diagonal < 6, "no diagonal pair was screened");
    assert!(
        map.pairs().len() - kept_diagonal < tables.n_pairs() - 6,
        "no off-diagonal pair was screened"
    );
    map
}

/// Run every launch over `packing` once on device 0, and hold each result
/// to the host map's, bitwise.
fn check(packing: &Packing) {
    let (nao, n_columns) = (packing.nao(), packing.n_columns());
    let d = symmetric(nao, 1);
    let batch = values(B * n_columns, 2);
    let jp = values(n_columns, 3);
    // A symmetric matrix supported on the kept pairs: a random one's
    // unpack of its pack, on the host.
    let mut sp = vec![0.0; n_columns];
    let mut s = vec![0.0; nao * nao];
    packing.pack(&symmetric(nao, 4), &mut sp);
    packing.unpack(&sp, &mut s);
    packing.pack(&s, &mut sp);

    let mut dag = Dag::new();
    let p = packing.clone();
    let tables = add_packed_tables(&mut dag, 0, nao, n_columns, move |r, c, w| {
        p.write_tables(r, c, w)
    });
    let up = |dag: &mut Dag<'_>, name: &'static str, data: Vec<f64>| {
        let len = data.len();
        let staged = dag.add_cpu(format!("stage {name}"), 0, Stage { name, data });
        dag.upload(0, staged, name, len)
    };
    let (d_row, batch_row) = (
        up(&mut dag, "d", d.clone()),
        up(&mut dag, "batch", batch.clone()),
    );
    let (jp_row, sp_row) = (up(&mut dag, "jp", jp.clone()), up(&mut dag, "sp", sp));
    let rows = dag.add_gpu(
        "under test",
        0,
        UnderTest {
            tables,
            nao,
            n_columns,
            d: d_row,
            batch: batch_row,
            jp: jp_row,
            sp: sp_row,
        },
    );
    let square = nao * nao;
    let down = vec![
        dag.download(0, rows.dp, "dp", n_columns),
        dag.download(0, rows.slab, "slab", B * square),
        dag.download(0, rows.j, "j", square),
        dag.download(0, rows.back, "back", square),
    ];
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu(
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
    let got = std::mem::take(&mut *out.lock().unwrap());
    assert_eq!(got.len(), 4, "the collect did not run");

    let bits = |what: &str, got: &[f64], want: &[f64]| {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: element {i}, {g:e} against {w:e}"
            );
        }
    };

    // Pack D: the host's pack times each column's weight.
    let (mut r, mut c, mut w) = (vec![0; n_columns], vec![0; n_columns], vec![0.0; n_columns]);
    packing.write_tables(&mut r, &mut c, &mut w);
    let mut want = vec![0.0; n_columns];
    packing.pack(&d, &mut want);
    for (x, w) in want.iter_mut().zip(&w) {
        *x *= w;
    }
    bits("pack D", &got[0], &want);

    // Gather: each function's matrix the host's unpack of its columns.
    let mut want = vec![f64::NAN; B * square];
    for (q, m) in want.chunks_mut(square).enumerate() {
        packing.unpack(&batch[q * n_columns..(q + 1) * n_columns], m);
    }
    bits("gather", &got[1], &want);

    // Unpack J.
    let mut want = vec![f64::NAN; square];
    packing.unpack(&jp, &mut want);
    bits("unpack J", &got[2], &want);

    // Unpack of the pack of a matrix on the kept pairs: the matrix.
    bits("unpack(pack)", &got[3], &s);

    // The v0.0.4 layer's gather of the same batch and the same J: the old
    // layer's results, bitwise.
    let (gathered, j) = (
        gather_v004(packing, B, &batch),
        gather_v004(packing, 1, &jp),
    );
    bits("v0.0.4 gather", &gathered, &got[1]);
    bits("v0.0.4 unpack J", &j, &got[2]);
}

/// `packed`, `b` functions of `packing`'s columns, gathered on device 0 by
/// the v0.0.4 layer's [`packed_gather`], the tables written on the host.
fn gather_v004(packing: &Packing, b: usize, packed: &[f64]) -> Vec<f64> {
    let (nao, n_columns) = (packing.nao(), packing.n_columns());
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    let ledger = Ledger::new(caps, Backing::Real, vec![0], false);
    let budget = ledger.budget("test");
    let stream = Stream::new(0);
    let mut tables = budget.pinned::<i32>(0, 2 * n_columns);
    let mut w = budget.pinned::<f64>(0, n_columns);
    {
        let (rows, cols) = tables.split_at_mut(n_columns);
        packing.write_tables(rows, cols, &mut w);
    }
    let mut weights = budget.device::<f64>(0, n_columns);
    drop(stream.upload(&mut weights, w).unwrap().sync().unwrap());
    let mut staged = budget.pinned::<f64>(0, packed.len());
    staged.copy_from_slice(packed);
    let mut d_packed = budget.device::<f64>(0, packed.len());
    drop(
        stream
            .upload(&mut d_packed, staged)
            .unwrap()
            .sync()
            .unwrap(),
    );
    let mut tables_dev = budget.device::<i32>(0, 2 * n_columns);
    let mut slab = budget.device::<f64>(0, b * nao * nao);
    stream.fill(&mut slab, 0xFF).unwrap();
    let tables = packed_gather(
        &stream,
        (nao, n_columns, b),
        tables,
        &mut tables_dev,
        &weights,
        &d_packed,
        &mut slab,
    )
    .unwrap();
    let out = stream
        .download(budget.pinned::<f64>(0, b * nao * nao), &slab)
        .unwrap();
    drop(tables.sync().unwrap());
    out.sync().unwrap().to_vec()
}

/// Water in cc-pVDZ, every shell pair kept.
fn water_map() -> PairMap {
    let at = |x, y, z| Double3 { x, y, z };
    let molecule = Molecule::new(
        vec![
            Atom {
                position: at(0.0, 0.0, 0.0),
                element: Element::O,
            },
            Atom {
                position: at(1.43, 0.21, -1.07),
                element: Element::H,
            },
            Atom {
                position: at(-1.38, 0.35, -1.12),
                element: Element::H,
            },
        ],
        0,
        1,
    );
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
    PairMap::new(&record_tables(&basis), None, 0.0)
}

/// Needs a GPU. The Cartesian packing: the three-centre kernels' output.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_cartesian_packing_agrees_with_the_host_bitwise() {
    check(screened_map().cart());
}

/// Needs a GPU. The spherical packing: B's.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_spherical_packing_agrees_with_the_host_bitwise() {
    check(screened_map().sphe());
}

/// Needs a GPU. Water's spherical packing, unscreened: B's on water.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn waters_spherical_packing_agrees_with_the_host_bitwise() {
    check(water_map().sphe());
}
