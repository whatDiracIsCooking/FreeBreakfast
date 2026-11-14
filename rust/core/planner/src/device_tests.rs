//! The BLAS workspace binding on a real card (device 0, tier 4): a unit
//! test because it reads the instance's workspace row, which no caller
//! outside the planner can name.

use std::sync::{Arc, Mutex};

use super::*;
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_thread_pool::{Backend, DeviceStream};
use fb_core_wwr::blas::gemm;
use fb_core_wwr::device_slice::DeviceSlice;
use fb_test_support::panic_message;
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_N as OP_N;

const N: usize = 16;

/// Small integers, so every product and partial sum is exact in any order.
fn operand(seed: usize) -> Vec<f64> {
    (0..N * N)
        .map(|i| ((i * 7 + seed) % 11) as f64 - 5.0)
        .collect()
}

/// `A B`, column-major, on the host.
fn product(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; N * N];
    for j in 0..N {
        for k in 0..N {
            for i in 0..N {
                c[i + j * N] += a[i + k * N] * b[k + j * N];
            }
        }
    }
    c
}

/// CPU: A and B into pinned rows.
struct Fill(usize);

impl CpuTask for Fill {
    type Rows = (RowId<Pinned, f64>, RowId<Pinned, f64>);

    fn layout(&self, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        (h.output("A", N * N), h.output("B", N * N))
    }

    fn run(&self, &(a, b): &Self::Rows, h: &Bufs<'_, Pinned>) -> Status {
        h.take_mut(a).copy_from_slice(&operand(self.0));
        h.take_mut(b).copy_from_slice(&operand(self.0 + 3));
        Ok(())
    }
}

/// GPU: A and B copied to device scratch, `C = A B` on the worker's BLAS
/// handle, C copied back to a pinned row.
struct Gemm((RowId<Pinned, f64>, RowId<Pinned, f64>));

impl GpuTask for Gemm {
    type Rows = (
        RowId<Device, f64>,
        RowId<Device, f64>,
        RowId<Device, f64>,
        RowId<Pinned, f64>,
    );

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        h.read(self.0.0);
        h.read(self.0.1);
        (
            d.scratch("dA", N * N),
            d.scratch("dB", N * N),
            d.scratch("dC", N * N),
            h.output("C", N * N),
        )
    }

    fn run(
        &self,
        &(da, db, dc, c): &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let (mut da, mut db, mut dc) = (d.take_mut(da), d.take_mut(db), d.take_mut(dc));
        memcpy_p2d(&mut da, h, self.0.0, stream)?;
        memcpy_p2d(&mut db, h, self.0.1, stream)?;
        let n = N as i32;
        gemm(
            stream.blas(),
            OP_N,
            OP_N,
            n,
            n,
            n,
            1.0,
            &da,
            n,
            &db,
            n,
            0.0,
            &mut dc,
            n,
        )?;
        memcpy_d2p(h, c, &dc, stream)
    }
}

/// CPU: C out of its pinned row.
struct Collect(RowId<Pinned, f64>, Arc<Mutex<Vec<f64>>>);

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Pinned>) {
        h.read(self.0);
    }

    fn run(&self, _: &(), h: &Bufs<'_, Pinned>) -> Status {
        *self.1.lock().unwrap() = h.get(self.0).to_vec();
        Ok(())
    }
}

/// A plan that declares worker 0's BLAS workspace and runs one GEMM on
/// operands of `seed`, collecting C into `out`.
fn gemm_instance(seed: usize, out: &Arc<Mutex<Vec<f64>>>) -> Instance<'static> {
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let ab = dag.add_cpu("fill", 0, Fill(seed));
    let (.., c) = dag.add_gpu("gemm", 0, Gemm(ab));
    dag.add_cpu("collect", 0, Collect(c, Arc::clone(out)));
    dag.plan().instantiate(&[0])
}

/// The address of the workspace lane `lane` of worker 0's BLAS handle
/// holds.
fn bound_on(backend: &fb_core_thread_pool::Device, lane: usize) -> (usize, usize) {
    let lane = fb_core_thread_pool::LaneId { worker: 0, lane };
    let workspace = backend
        .stream(lane)
        .blas()
        .workspace()
        .expect("a workspace");
    (workspace.as_ptr().addr(), workspace.len())
}

/// The address of the workspace worker 0's lane 0's BLAS handle holds.
fn bound(backend: &fb_core_thread_pool::Device) -> (usize, usize) {
    bound_on(backend, 0)
}

/// Needs a GPU. Each instance's backend hands its handle that instance's
/// own workspace row and holds it, so a second backend of the same instance
/// is refused while the first lives; the plans' GEMMs run on it to the
/// exact product, the two instances' runs interleaved; the row is the one
/// the plan declared, free again once its backend drops; and a backend
/// outlives its instance, keeping the block alive.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_backend_binds_each_instances_workspace_row() {
    let (out_a, out_b) = (Arc::default(), Arc::default());
    let mut a = gemm_instance(0, &out_a);
    let mut b = gemm_instance(5, &out_b);
    let mut backend_a = a.backend();
    let (at_a, len) = bound(&backend_a);
    assert_eq!(len, BLAS_WORKSPACE_BYTES);
    assert!(
        panic_message(|| drop(a.backend())).contains("blas workspace is already held"),
        "a second backend of one instance took its row"
    );
    let mut backend_b = b.backend();
    let (at_b, _) = bound(&backend_b);
    assert_ne!(at_a, at_b, "two instances share a workspace");

    let want_a = product(&operand(0), &operand(3));
    let want_b = product(&operand(5), &operand(8));
    for _ in 0..2 {
        a.run(&mut backend_a, 1).result.unwrap();
        assert_eq!(*out_a.lock().unwrap(), want_a);
        b.run(&mut backend_b, 1).result.unwrap();
        assert_eq!(*out_b.lock().unwrap(), want_b);
    }

    drop(backend_a);
    let (_, row) = a.blas_workspaces[0][0];
    let held = a.blocks[0].0.hold(row);
    assert_eq!((held.as_ptr().addr(), held.len()), (at_a, len));
    drop(held);

    drop(b);
    assert_eq!(bound(&backend_b), (at_b, len));
    drop(backend_b);
}

/// Needs a GPU. Two lanes of one worker, a GEMM on each from one CPU fill:
/// each lane's BLAS handle holds a workspace row of its own, the plan's
/// row for that lane, and both products are exact. A third lane, whose one
/// task is an upload, calls no library, so it has no row and the backend
/// made it no handles.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn each_lane_binds_its_own_workspace_row() {
    use fb_core_thread_pool::LaneId;
    let (out_0, out_1) = (Arc::default(), Arc::default());
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let ab = dag.add_cpu("fill", 0, Fill(0));
    let (.., c0) = dag.add_gpu("gemm 0", LaneId { worker: 0, lane: 0 }, Gemm(ab));
    let (.., c1) = dag.add_gpu("gemm 1", LaneId { worker: 0, lane: 1 }, Gemm(ab));
    dag.add_cpu("collect 0", 0, Collect(c0, Arc::clone(&out_0)));
    dag.add_cpu("collect 1", 0, Collect(c1, Arc::clone(&out_1)));
    let copy_lane = LaneId { worker: 0, lane: 2 };
    let _ = dag.upload(copy_lane, ab.0, "A", N * N);
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    let lanes = [bound_on(&backend, 0), bound_on(&backend, 1)];
    assert_eq!(
        instance.blas_workspaces[0].len(),
        2,
        "a row for the copy lane"
    );
    assert!(
        panic_message(|| {
            let _ = backend.stream(copy_lane).blas();
        })
        .contains("without library handles"),
        "the copy lane has a BLAS handle"
    );
    assert_ne!(lanes[0].0, lanes[1].0, "two lanes share a workspace");
    instance.run(&mut backend, 1).result.unwrap();
    let want = product(&operand(0), &operand(3));
    assert_eq!(*out_0.lock().unwrap(), want);
    assert_eq!(*out_1.lock().unwrap(), want);
    drop(backend);
    for (lane, &bound) in lanes.iter().enumerate() {
        let held = instance.blocks[0]
            .0
            .hold(instance.blas_workspaces[0][lane].1);
        assert_eq!((held.as_ptr().addr(), held.len()), bound, "lane {lane}");
        assert_eq!(bound.1, BLAS_WORKSPACE_BYTES);
    }
}
