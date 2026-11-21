//! The transform on a card (tier 4, device 0), one run: random packed
//! input over a basis reaching f, Cartesian on the orbital indices, a
//! screened map and a batch that cuts its first and last classes, staged,
//! uploaded, transformed by [`add_spherical_transform`] over the plan's U
//! row, and brought back by a task after it.
//!
//! **The reference** is the dense transform, the same approximation by
//! another route: each aux function's Cartesian matrix, symmetric and
//! random, is transformed whole, `Uᵀ X_Q U`, and packed by the map's
//! spherical packing; the input is the same matrices packed by its
//! Cartesian one. A block's spherical columns depend on its own Cartesian
//! ones alone, U being block-diagonal by shell, so the screened pairs'
//! elements, which the dense reference reads and the batch does not hold,
//! change nothing.
//!
//! **The bar, by analysis, one run.** Each output element is a sum of at
//! most `10 · 10` products in fp64 (a shell's 10 Cartesian functions on
//! each side), by two orders of summation; their difference is within
//! `2 n ε` of the sum of the terms' magnitudes, n here 100 and ε 1.1e-16,
//! and that sum is at most about ten times the largest element for random
//! terms of mixed sign, so about 2e-13 of it at worst. The test holds
//! every element within `1e-12 · max |reference|`.

use std::sync::{Arc, Mutex};

use fb_basis::{BasisSet, Shell, spherical_transform_matrix};
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_types::Double3;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_eri_density_fitting_transform::add_spherical_transform;
use fb_integrals_mmd::record_tables;
use fb_recipes_matrix_ao::add_harmonics;

/// The bar's fraction of the largest reference element.
const BAR: f64 = 1e-12;

/// The input's seed, printed with the result.
const SEED: u64 = 1688;

/// SplitMix64: a uniform double in `[-1, 1)` per call.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
    }
}

/// A basis of shells of the momenta `ls`, one primitive each, on three
/// centres in turn.
fn shells(ls: &[i32]) -> BasisSet {
    BasisSet::new(
        ls.iter()
            .enumerate()
            .map(|(i, &l)| {
                let at = Double3 {
                    x: (i % 3) as f64,
                    y: 0.0,
                    z: 0.5 * (i % 3) as f64,
                };
                Shell::zipped(l, &[0.8], &[1.0], at)
            })
            .collect(),
    )
}

/// CPU: the input written into a pinned row.
struct Stage {
    data: Vec<f64>,
}

impl CpuTask for Stage {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        host.output("X staged", self.data.len())
    }

    fn run(&self, row: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        host.take_mut(*row).copy_from_slice(&self.data);
        Ok(())
    }
}

/// GPU: a staged row copied up into a device output.
struct Upload {
    staged: RowId<Pinned, f64>,
    len: usize,
}

impl GpuTask for Upload {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.staged);
        device.output("X", self.len)
    }

    fn run(
        &self,
        row: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_p2d(&mut device.take_mut(*row), host, self.staged, stream)
    }
}

/// GPU: the output copied down into a pinned output.
struct Download {
    out: RowId<Device, f64>,
    len: usize,
}

impl GpuTask for Download {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.out);
        host.output("out down", self.len)
    }

    fn run(
        &self,
        row: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2p(host, *row, device.get(self.out), stream)
    }
}

/// CPU: the pinned copy read out.
struct Collect {
    row: RowId<Pinned, f64>,
    out: Arc<Mutex<Vec<f64>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        host.read(self.row);
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        *self.out.lock().unwrap() = host.get(self.row).to_vec();
        Ok(())
    }
}

/// A packed batch of random symmetric Cartesian matrices, one per
/// spherical aux function, over a basis through f and an aux basis
/// through g, every
/// fourth pair screened and the batch cut inside its first and last
/// classes, against the pack of the dense transform: every element within
/// `1e-12 · max |reference|`. Catches a wrong leading dimension, stride or
/// batch count in either GEMM, a U block taken at the wrong
/// shell or momentum, the bra and ket transforms swapped (a block's `a`
/// slower than `b`), and a class's columns misplaced in any layout,
/// relative to the batch or across a screened pair. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_packed_batch_matches_the_dense_transform() {
    let basis = shells(&[0, 1, 2, 3, 0, 1, 2, 3]);
    let aux = shells(&[0, 1, 2, 3, 4, 0, 2]);
    let tables = record_tables(&basis);
    let bounds: Vec<f64> = (0..tables.n_pairs())
        .map(|i| if i % 4 == 1 { 0.0 } else { 1.0 })
        .collect();
    let map = PairMap::new(&tables, Some(&bounds), 0.5);
    let (cart, sphe) = (map.cart(), map.sphe());
    let pairs = 1..map.pairs().len() - 1;
    let first = &map.classes()[0];
    let last = map.classes().last().unwrap();
    assert!(
        first.pairs.len() > 1 && last.pairs.len() > 1 && (last.la, last.lb) == (3, 3),
        "the batch cuts its first and last classes, (3, 3) among them"
    );

    let (nc, ns) = (cart.nao(), sphe.nao());
    let nas = aux.nao_sphe() as usize;
    let cols = [
        cart.blocks()[pairs.start].column..cart.blocks()[pairs.end - 1].end(),
        sphe.blocks()[pairs.start].column..sphe.blocks()[pairs.end - 1].end(),
    ];
    let mut u = vec![0.0; nc * ns];
    spherical_transform_matrix(&basis, &mut u);

    // The input, (Q, c) at Q + nas c, and the reference, the pack of each
    // Q's Uᵀ X_Q U, (Q, c) at Q + nas c.
    let mut rng = Rng(SEED);
    let mut input = vec![0.0; nas * cols[0].len()];
    let mut reference = vec![0.0; nas * cols[1].len()];
    let mut y = vec![0.0; ns * ns];
    let mut packed = vec![0.0; cart.n_columns()];
    let mut packed_sph = vec![0.0; sphe.n_columns()];
    let mut x = vec![0.0; nc * nc];
    for q in 0..nas {
        for m in 0..nc {
            for n in 0..=m {
                let v = rng.next();
                x[m * nc + n] = v;
                x[n * nc + m] = v;
            }
        }
        cart.pack(&x, &mut packed);
        for (c, &v) in packed[cols[0].clone()].iter().enumerate() {
            input[q + nas * c] = v;
        }
        // xu = X U, nc × ns; y = Uᵀ xu.
        let mut xu = vec![0.0; nc * ns];
        for m in 0..nc {
            for j in 0..ns {
                xu[m * ns + j] = (0..nc).map(|k| x[m * nc + k] * u[k + j * nc]).sum();
            }
        }
        for i in 0..ns {
            for j in 0..ns {
                y[i * ns + j] = (0..nc).map(|m| u[m + i * nc] * xu[m * ns + j]).sum();
            }
        }
        sphe.pack(&y, &mut packed_sph);
        for (c, &v) in packed_sph[cols[1].clone()].iter().enumerate() {
            reference[q + nas * c] = v;
        }
    }

    let mut dag = Dag::new();
    let len = input.len();
    let staged = dag.add_cpu("stage X", 0, Stage { data: input });
    let x = dag.add_gpu("upload X", 0, Upload { staged, len });
    let u_row = add_harmonics(&mut dag, 0, &basis, &mut None);
    let out = add_spherical_transform(&mut dag, 0, &map, pairs.clone(), pairs.len(), nas, x, u_row)
        .unwrap();
    let len = reference.len();
    let row = dag.add_gpu("download", 0, Download { out, len });
    let got = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu(
        "collect",
        0,
        Collect {
            row,
            out: Arc::clone(&got),
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    let got = got.lock().unwrap();
    assert_eq!(got.len(), reference.len());

    let largest = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut miss = 0.0_f64;
    let mut at = 0;
    for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
        // Not f64::max, which would drop a NaN.
        let d = (g - r).abs();
        if d.is_nan() || d > miss {
            (miss, at) = (d, i);
        }
    }
    println!(
        "seed {SEED}: pairs {pairs:?} of {}, {} x {} Cartesian columns -> {} x {} spherical, \
         max |reference| {largest:e}, max miss {miss:e} at (Q {}, c {}) (bar {:e})",
        map.pairs().len(),
        nas,
        cols[0].len(),
        nas,
        cols[1].len(),
        at % nas,
        at / nas,
        BAR * largest,
    );
    assert!(
        !miss.is_nan() && miss <= BAR * largest,
        "max |out - reference| = {miss:e} > {:e}",
        BAR * largest
    );
}
