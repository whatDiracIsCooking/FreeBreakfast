//! The density-fitting metric's device path: an auxiliary basis's records
//! as the density-fitting kernels read them ([`AuxTable`], checked as it is
//! built), and the two tasks that stage them with every aux shell pair and
//! launch each class of the Cartesian metric (P|Q) ([`add_metric_tasks`]).
//!
//! The tasks, added in this order:
//!
//! - **`metric tables`**, on the CPU: the aux records and the shell pairs,
//!   formed at planning, written into pinned rows, its outputs.
//! - **`metric`**, on the worker's stream: both copied into device rows,
//!   its scratch, then one [`fb_metric_class`] launch per class that has a
//!   pair, every one into the Cartesian metric, its output.
//!
//! Forming an [`AuxTable`] from a basis is the caller's; the pairs are
//! formed here, from the table alone, so every index the kernel follows is
//! one this module wrote.

use std::sync::Arc;

use contracts::debug_requires;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_h2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::common::wwr_status_from_code;
use fb_core_wwr::gpu_status_at;
use fb_core_wwr::handle::DeviceHandleStream;

use crate::{
    FB_METRIC_MAX_MOMENTUM, FB_METRIC_PAIR_FIELDS, FB_METRIC_RECORD_FIELDS,
    FB_THREE_CENTER_AUX_FIELDS, fb_metric_class,
};

/// The fields of one aux record.
const RECORD_FIELDS: usize = FB_METRIC_RECORD_FIELDS as usize;
/// The fields of one aux shell pair.
const PAIR_FIELDS: usize = FB_METRIC_PAIR_FIELDS as usize;

// The three-center integrals read the metric's aux record, field for field.
const _: () = assert!(FB_THREE_CENTER_AUX_FIELDS == FB_METRIC_RECORD_FIELDS);

/// An auxiliary basis of `n_cart` Cartesian functions as the
/// density-fitting kernels read it: each shell's momentum, its first
/// Cartesian function and its range of records, one record per primitive,
/// its centre, exponent α and weight `s = c (π/α)^{3/2}`, `c` the
/// coefficient as given ([`FB_METRIC_RECORD_FIELDS`]).
///
/// Built shell by shell ([`AuxTable::push_shell`]), which checks each
/// shell as it comes, so a table is consistent by construction: every
/// shell's block of `(l + 1)(l + 2) / 2` functions lies inside
/// `[0, n_cart)`, every record range inside the records, and `n_cart²`
/// fits the kernels' 32-bit index.
#[derive(Clone, Debug, PartialEq)]
pub struct AuxTable {
    n_cart: usize,
    momentum: Vec<i32>,
    ao_offset: Vec<i32>,
    /// Each shell's first record, `n_shells + 1` long.
    rec_begin: Vec<i32>,
    records: Vec<[f64; RECORD_FIELDS]>,
}

impl AuxTable {
    /// An empty table over `n_cart` Cartesian functions.
    ///
    /// # Panics
    /// If `n_cart²` does not fit an `i32`: the kernels index the metric in
    /// 32 bits.
    #[track_caller]
    pub fn new(n_cart: usize) -> AuxTable {
        assert!(
            n_cart
                .checked_mul(n_cart)
                .is_some_and(|n2| i32::try_from(n2).is_ok()),
            "AuxTable: {n_cart}² Cartesian functions do not fit the kernels' 32-bit index"
        );
        AuxTable {
            n_cart,
            momentum: Vec::new(),
            ao_offset: Vec::new(),
            rec_begin: vec![0],
            records: Vec::new(),
        }
    }

    /// Append a shell of momentum `l` at `centre` (bohr), its first
    /// Cartesian function `ao_offset`, over `primitives`, each an
    /// (exponent, coefficient) pair, the coefficient as the basis holds it.
    ///
    /// # Panics
    /// If `l` is negative, the shell's block `ao_offset + (l + 1)(l + 2) / 2`
    /// passes `n_cart`, or the records pass an `i32` count: each is an
    /// index the kernel follows.
    #[track_caller]
    pub fn push_shell(
        &mut self,
        l: i32,
        centre: [f64; 3],
        ao_offset: usize,
        primitives: impl IntoIterator<Item = (f64, f64)>,
    ) {
        assert!(l >= 0, "AuxTable: a shell of momentum {l}");
        let width = n_cart_of(l);
        assert!(
            ao_offset
                .checked_add(width)
                .is_some_and(|end| end <= self.n_cart),
            "AuxTable: a shell at function {ao_offset} with momentum {l} does not fit {} \
             Cartesian functions",
            self.n_cart
        );
        let [x, y, z] = centre;
        self.records.extend(
            primitives
                .into_iter()
                .map(|(alpha, c)| [x, y, z, alpha, c * (std::f64::consts::PI / alpha).powf(1.5)]),
        );
        let Ok(end) = i32::try_from(self.records.len()) else {
            panic!(
                "AuxTable: {} records, past an i32 count",
                self.records.len()
            );
        };
        self.momentum.push(l);
        // In range: ao_offset + width <= n_cart, and n_cart² fits an i32.
        self.ao_offset.push(ao_offset as i32);
        self.rec_begin.push(end);
    }

    /// The Cartesian functions, which the shells' offsets index.
    pub fn n_cart(&self) -> usize {
        self.n_cart
    }

    /// The shells pushed so far.
    pub fn n_shells(&self) -> usize {
        self.momentum.len()
    }
}

/// A shell of momentum `l >= 0`'s Cartesian functions.
fn n_cart_of(l: i32) -> usize {
    let l = l as usize;
    (l + 1) * (l + 2) / 2
}

/// What the two tasks stage, formed at planning: the records field-major,
/// and every shell pair P <= Q by shell index, the lower-momentum shell
/// first, grouped by class `(lp, lq)`, lp <= lq, each class's run
/// field-major.
#[derive(Debug)]
struct Staging {
    n_cart: usize,
    n_rec: usize,
    records: Vec<f64>,
    pairs: Vec<i32>,
    /// Each class that has a pair: its momenta, its first pair and its
    /// pair count.
    classes: Vec<(i32, i32, usize, usize)>,
}

impl Staging {
    /// The tables of `aux`: as FelizJueves's `metric_classes` stages them,
    /// a pair whose first shell has the higher momentum swapped, since the
    /// kernel stores every element with its mirror.
    fn new(aux: &AuxTable) -> Staging {
        let n_rec = aux.records.len();
        let mut records = vec![0.0; RECORD_FIELDS * n_rec];
        for (r, record) in aux.records.iter().enumerate() {
            for (f, &v) in record.iter().enumerate() {
                records[f * n_rec + r] = v;
            }
        }
        let n_l = FB_METRIC_MAX_MOMENTUM.max(aux.momentum.iter().copied().max().unwrap_or(0))
            as usize
            + 1;
        let mut by_class = vec![Vec::new(); n_l * n_l];
        let n_shells = aux.n_shells();
        for i in 0..n_shells {
            for j in i..n_shells {
                let (p, q) = if aux.momentum[i] > aux.momentum[j] {
                    (j, i)
                } else {
                    (i, j)
                };
                let class = aux.momentum[p] as usize * n_l + aux.momentum[q] as usize;
                by_class[class].push([
                    aux.rec_begin[p],
                    aux.rec_begin[p + 1],
                    aux.rec_begin[q],
                    aux.rec_begin[q + 1],
                    aux.ao_offset[p],
                    aux.ao_offset[q],
                ]);
            }
        }
        let mut pairs = Vec::with_capacity(PAIR_FIELDS * n_shells * (n_shells + 1) / 2);
        let mut classes = Vec::new();
        for (class, run) in by_class.iter().enumerate() {
            if run.is_empty() {
                continue;
            }
            classes.push((
                (class / n_l) as i32,
                (class % n_l) as i32,
                pairs.len() / PAIR_FIELDS,
                run.len(),
            ));
            for f in 0..PAIR_FIELDS {
                pairs.extend(run.iter().map(|pair| pair[f]));
            }
        }
        Staging {
            n_cart: aux.n_cart,
            n_rec,
            records,
            pairs,
            classes,
        }
    }
}

/// Add the Cartesian Coulomb metric (P|Q) of `aux` to `dag` on GPU worker
/// `worker`, and return it: an output row of `n_cart²` doubles, exactly
/// symmetric, every element written by one aux shell pair, so nothing is
/// zeroed first and a run is bit-reproducible.
///
/// `metric tables` (CPU: the records and every shell pair into pinned
/// rows), then `metric` (stream: both copied into device scratch, then one
/// [`fb_metric_class`] launch per class that has a pair). Every size is
/// the table's, fixed here at planning, and the pairs are formed here too,
/// once per plan.
///
/// A shell above the kernels' table ([`FB_METRIC_MAX_MOMENTUM`]) is the
/// launcher's refusal at run time, `Unsupported` naming the class; a
/// caller refuses one at planning.
#[debug_requires(aux.n_shells() > 0, "an aux basis of at least one shell")]
pub fn add_metric_tasks(dag: &mut Dag<'_>, worker: usize, aux: &AuxTable) -> RowId<Device, f64> {
    let staging = Arc::new(Staging::new(aux));
    let staged = dag.add(
        "metric tables",
        worker,
        Stage {
            staging: Arc::clone(&staging),
        },
    );
    dag.add("metric", worker, Metric { staging, staged }).metric
}

/// The records and the pairs, one row each, of one kind: pinned as the
/// CPU task's outputs, device as the stream task's scratch.
struct TableRows<K> {
    records: RowId<K, f64>,
    pairs: RowId<K, i32>,
}

impl<K> Clone for TableRows<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for TableRows<K> {}

/// CPU: the staged tables written into pinned rows.
struct Stage {
    staging: Arc<Staging>,
}

impl Task for Stage {
    type Rows = TableRows<Pinned>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        TableRows {
            records: host.output("aux records", self.staging.records.len()),
            pairs: host.output("aux pairs", self.staging.pairs.len()),
        }
    }

    fn run(
        &self,
        rows: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        memcpy_h2p(host, rows.records, &self.staging.records);
        memcpy_h2p(host, rows.pairs, &self.staging.pairs);
        Ok(())
    }
}

/// GPU: the tables copied to device scratch, then each class launched into
/// the Cartesian metric on the one stream.
struct Metric {
    staging: Arc<Staging>,
    staged: TableRows<Pinned>,
}

#[derive(Clone, Copy)]
struct MetricRows {
    tables: TableRows<Device>,
    metric: RowId<Device, f64>,
}

impl Task for Metric {
    type Rows = MetricRows;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> MetricRows {
        host.read(self.staged.records);
        host.read(self.staged.pairs);
        let s = &self.staging;
        MetricRows {
            tables: TableRows {
                records: device.scratch("aux records", s.records.len()),
                pairs: device.scratch("aux pairs", s.pairs.len()),
            },
            metric: device.output("M cartesian", s.n_cart * s.n_cart),
        }
    }

    fn run(
        &self,
        rows: &MetricRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let s = &self.staging;
        let records = device.take_mut(rows.tables.records);
        memcpy_p2d(&records, host, self.staged.records, worker)?;
        let pairs = device.take_mut(rows.tables.pairs);
        memcpy_p2d(&pairs, host, self.staged.pairs, worker)?;
        let metric = device.take_mut(rows.metric);
        // In range, as the casts below: n_cart² fits an i32 (AuxTable::new),
        // and so do the records (AuxTable::push_shell) and the pairs, at
        // most one per element of the metric's upper triangle.
        let (n_rec, n_aux) = (s.n_rec as i32, s.n_cart as i32);
        for &(lp, lq, first, n_pair) in &s.classes {
            // SAFETY: records is RECORD_FIELDS segments of n_rec doubles and
            // the class's pairs PAIR_FIELDS segments of n_pair int32s inside
            // the pairs row, both copied above on this stream from tables
            // Staging::new formed from an AuxTable, so every record range a
            // pair names is inside [0, n_rec) and every shell block it
            // writes, at off_p and off_q, inside [0, n_cart); metric is
            // n_cart² doubles of this task's own output; the launch only
            // enqueues on the task's stream.
            let status = unsafe {
                fb_metric_class(
                    worker.stream(),
                    lp,
                    lq,
                    n_rec,
                    records.as_ptr(),
                    n_pair as i32,
                    pairs.as_ptr().wrapping_add(PAIR_FIELDS * first),
                    n_aux,
                    metric.as_mut_ptr(),
                )
            };
            let what = format!("fb_metric_class ({lp}, {lq})");
            fb_kernel_abi::launch_status(&what, status, |code, at| {
                Some(gpu_status_at(wwr_status_from_code(code)?, at))
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A d shell of one primitive at functions 0..6, then an s of two at 6:
    /// one pair per class, (0,0), (0,2) staged swapped, and (2,2).
    fn table() -> AuxTable {
        let mut t = AuxTable::new(7);
        t.push_shell(2, [0.0, 0.0, 1.0], 0, [(2.0, 1.0)]);
        t.push_shell(0, [0.5, 0.0, 0.0], 6, [(1.0, 0.5), (0.25, 0.5)]);
        t
    }

    /// The records are field-major, the weight `c (π/α)^{3/2}`; each class's
    /// pairs field-major in the class order, and a pair whose first shell has
    /// the higher momentum staged with the lower one first.
    #[test]
    fn staging_swaps_to_the_lower_momentum_and_groups_by_class() {
        let s = Staging::new(&table());
        assert_eq!(s.n_rec, 3);
        assert_eq!(&s.records[0..3], &[0.0, 0.5, 0.5]);
        assert_eq!(&s.records[9..12], &[2.0, 1.0, 0.25]);
        assert_eq!(s.records[12], (std::f64::consts::PI / 2.0).powf(1.5));
        let classes: Vec<_> = s.classes.iter().map(|c| (c.0, c.1, c.2, c.3)).collect();
        assert_eq!(classes, [(0, 0, 0, 1), (0, 2, 1, 1), (2, 2, 2, 1)]);
        // One pair per class, so each class's run is its six fields: P's
        // record range, Q's, then off_p and off_q.
        let run = |c: usize| &s.pairs[6 * c..6 * c + 6];
        assert_eq!(run(0), [1, 3, 1, 3, 6, 6], "(s, s)");
        assert_eq!(run(1), [1, 3, 0, 1, 6, 0], "(d, s) staged as (s, d)");
        assert_eq!(run(2), [0, 1, 0, 1, 0, 0], "(d, d)");
    }

    /// A shell whose block runs past the metric: its offset is a place the
    /// kernel writes.
    #[test]
    #[should_panic(
        expected = "a shell at function 6 with momentum 1 does not fit 7 Cartesian functions"
    )]
    fn a_block_past_the_metric_panics() {
        let mut t = AuxTable::new(7);
        t.push_shell(1, [0.0; 3], 6, [(1.0, 1.0)]);
    }

    /// A negative momentum has no block.
    #[test]
    #[should_panic(expected = "a shell of momentum -1")]
    fn a_negative_momentum_panics() {
        AuxTable::new(7).push_shell(-1, [0.0; 3], 0, [(1.0, 1.0)]);
    }

    /// A metric whose square passes the kernels' 32-bit index.
    #[test]
    #[should_panic(expected = "46341² Cartesian functions do not fit the kernels' 32-bit index")]
    fn a_metric_past_the_index_panics() {
        AuxTable::new(46_341);
    }
}
