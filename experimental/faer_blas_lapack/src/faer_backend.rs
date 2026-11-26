//! faer 0.24's `self_adjoint_evd`: Householder tridiagonalization, then
//! divide and conquer, the same shape as LAPACK's `dsyevd`. Called at the
//! low level with its scratch sized once, outside the timed region, as
//! LAPACK's workspace query is; the high-level `Mat::self_adjoint_eigen`
//! would allocate U, S and the scratch inside every call.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self, ComputeEigenvectors};
use faer::{ColMut, MatMut, MatRef, Par};

pub const NAME: &str = "faer";

pub const ROUTINES: &[&str] = &["self_adjoint_evd"];

/// The parallelism every call takes: sequential at one thread, else rayon's
/// global pool, which `main` builds with exactly `threads` workers.
fn par(threads: usize) -> Par {
    if threads == 1 {
        Par::Seq
    } else {
        Par::rayon(threads)
    }
}

pub fn describe(threads: usize) -> String {
    format!("faer 0.24, {:?}", par(threads))
}

pub struct Solver {
    n: usize,
    vectors: bool,
    par: Par,
    a: Vec<f64>,
    w: Vec<f64>,
    v: Vec<f64>,
    scratch: MemBuffer,
}

impl Solver {
    pub fn new(_routine: &str, n: usize, vectors: bool, threads: usize) -> Self {
        let par = par(threads);
        let compute = if vectors {
            ComputeEigenvectors::Yes
        } else {
            ComputeEigenvectors::No
        };
        let req = evd::self_adjoint_evd_scratch::<f64>(n, compute, par, Default::default());
        Self {
            n,
            vectors,
            par,
            a: vec![0.0; n * n],
            w: vec![0.0; n],
            v: vec![0.0; if vectors { n * n } else { 0 }],
            scratch: MemBuffer::new(req),
        }
    }

    /// Untimed: the input, column-major. faer reads the lower triangle and
    /// leaves A untouched, but it is copied anyway so both backends start
    /// each run from the same state.
    pub fn load(&mut self, a: &[f64]) {
        self.a.copy_from_slice(a);
    }

    pub fn run(&mut self) -> Result<(), String> {
        let n = self.n;
        let a = MatRef::from_column_major_slice(&self.a, n, n);
        let w = ColMut::from_slice_mut(&mut self.w).as_diagonal_mut();
        let v = self
            .vectors
            .then(|| MatMut::from_column_major_slice_mut(&mut self.v, n, n));
        evd::self_adjoint_evd(
            a,
            w,
            v,
            self.par,
            MemStack::new(&mut self.scratch),
            Default::default(),
        )
        .map_err(|e| format!("self_adjoint_evd: {e:?}"))
    }

    /// Ascending, as LAPACK's.
    pub fn values(&self) -> &[f64] {
        &self.w
    }

    pub fn vectors(&self) -> Option<&[f64]> {
        self.vectors.then_some(&self.v[..])
    }
}

/// `C = op(A) B` over column-major slices, `op(A)` m × k (`A` itself k × m
/// when `transpose_a`), `B` k × n: the accuracy checks' products.
pub fn gemm(transpose_a: bool, m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let a = if transpose_a {
        MatRef::from_column_major_slice(a, k, m).transpose()
    } else {
        MatRef::from_column_major_slice(a, m, k)
    };
    let b = MatRef::from_column_major_slice(b, k, n);
    let c = MatMut::from_column_major_slice_mut(c, m, n);
    faer::linalg::matmul::matmul(
        c,
        faer::Accum::Replace,
        a,
        b,
        1.0,
        par(rayon::current_num_threads()),
    );
}
