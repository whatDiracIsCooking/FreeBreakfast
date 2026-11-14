//! The BLAS workspace binding on a real card (device 0, tier 4): a unit
//! test because it reads the instance's workspace row, which no caller
//! outside the planner can name.

use std::sync::{Arc, Mutex};

use super::*;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_thread_pool::{Backend, DeviceStream, Lane};
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

impl Task for Fill {
    type Rows = (RowId<Pinned, f64>, RowId<Pinned, f64>);
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        (h.output("A", N * N), h.output("B", N * N))
    }

    fn run(
        &self,
        &(a, b): &Self::Rows,
        _: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        h.take_mut(a).copy_from_slice(&operand(self.0));
        h.take_mut(b).copy_from_slice(&operand(self.0 + 3));
        Ok(())
    }
}

/// GPU: A and B copied to device scratch, `C = A B` on the worker's BLAS
/// handle, C copied back to a pinned row.
struct Gemm((RowId<Pinned, f64>, RowId<Pinned, f64>));

impl Task for Gemm {
    type Rows = (
        RowId<Device, f64>,
        RowId<Device, f64>,
        RowId<Device, f64>,
        RowId<Pinned, f64>,
    );
    const GPU: bool = true;

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
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let (da, db, mut dc) = (d.take_mut(da), d.take_mut(db), d.take_mut(dc));
        memcpy_p2d(&da, h, self.0.0, stream)?;
        memcpy_p2d(&db, h, self.0.1, stream)?;
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

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) {
        h.read(self.0);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.1.lock().unwrap() = h.get(self.0).to_vec();
        Ok(())
    }
}

/// A plan that declares worker 0's BLAS workspace and runs one GEMM on
/// operands of `seed`, collecting C into `out`.
fn gemm_instance(seed: usize, out: &Arc<Mutex<Vec<f64>>>) -> Instance<'static> {
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let ab = dag.add("fill", 0, Fill(seed));
    let (.., c) = dag.add("gemm", 0, Gemm(ab));
    dag.add("collect", 0, Collect(c, Arc::clone(out)));
    dag.plan().instantiate(&[0])
}

/// The address of the workspace stream worker 0's BLAS handle holds.
fn bound(backend: &fb_core_thread_pool::Device) -> (usize, usize) {
    let workspace = backend.stream(0).blas().workspace().expect("a workspace");
    (workspace.as_ptr().addr(), workspace.len())
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
    let row = a.blas_workspaces[0].expect("a declared workspace");
    let held = a.blocks[0].0.hold(row);
    assert_eq!((held.as_ptr().addr(), held.len()), (at_a, len));
    drop(held);

    drop(b);
    assert_eq!(bound(&backend_b), (at_b, len));
    drop(backend_b);
}
