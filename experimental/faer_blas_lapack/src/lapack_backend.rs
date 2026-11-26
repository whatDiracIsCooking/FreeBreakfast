//! OpenBLAS's LAPACK, `dsyevd` (divide and conquer) and `dsyevr` (MRRR),
//! through hand-written Fortran bindings: the LP64 `libopenblas` of the
//! image (`Dockerfile.lapack`), 32-bit integers, trailing hidden string
//! lengths. Each workspace is queried once, outside the timed region.

use std::ffi::{c_char, c_int};

pub const NAME: &str = "lapack";

pub const ROUTINES: &[&str] = &["dsyevd", "dsyevr"];

type Len = usize;

unsafe extern "C" {
    fn dsyevd_(
        jobz: *const c_char,
        uplo: *const c_char,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        w: *mut f64,
        work: *mut f64,
        lwork: *const c_int,
        iwork: *mut c_int,
        liwork: *const c_int,
        info: *mut c_int,
        jobz_len: Len,
        uplo_len: Len,
    );
    fn dsyevr_(
        jobz: *const c_char,
        range: *const c_char,
        uplo: *const c_char,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        vl: *const f64,
        vu: *const f64,
        il: *const c_int,
        iu: *const c_int,
        abstol: *const f64,
        m: *mut c_int,
        w: *mut f64,
        z: *mut f64,
        ldz: *const c_int,
        isuppz: *mut c_int,
        work: *mut f64,
        lwork: *const c_int,
        iwork: *mut c_int,
        liwork: *const c_int,
        info: *mut c_int,
        jobz_len: Len,
        range_len: Len,
        uplo_len: Len,
    );
    fn dgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f64,
        a: *const f64,
        lda: *const c_int,
        b: *const f64,
        ldb: *const c_int,
        beta: *const f64,
        c: *mut f64,
        ldc: *const c_int,
        transa_len: Len,
        transb_len: Len,
    );
    fn openblas_set_num_threads(threads: c_int);
    fn openblas_get_num_threads() -> c_int;
    fn openblas_get_config() -> *const c_char;
}

fn ch(c: u8) -> *const c_char {
    // A pointer to a static byte: LAPACK reads one character.
    match c {
        b'N' => c"N".as_ptr(),
        b'V' => c"V".as_ptr(),
        b'L' => c"L".as_ptr(),
        b'A' => c"A".as_ptr(),
        b'T' => c"T".as_ptr(),
        _ => unreachable!(),
    }
}

fn int(n: usize) -> c_int {
    c_int::try_from(n).expect("dimension fits LP64")
}

/// Sets OpenBLAS's pool, which every call then uses.
pub fn describe(threads: usize) -> String {
    unsafe { openblas_set_num_threads(int(threads)) };
    let config = unsafe { std::ffi::CStr::from_ptr(openblas_get_config()) };
    format!("{}, {} threads", config.to_string_lossy().trim(), unsafe {
        openblas_get_num_threads()
    })
}

pub struct Solver {
    routine: &'static str,
    n: usize,
    vectors: bool,
    a: Vec<f64>,
    w: Vec<f64>,
    z: Vec<f64>,
    isuppz: Vec<c_int>,
    work: Vec<f64>,
    iwork: Vec<c_int>,
}

impl Solver {
    pub fn new(routine: &str, n: usize, vectors: bool, _threads: usize) -> Self {
        let routine = ROUTINES.iter().copied().find(|r| *r == routine).unwrap();
        let mut s = Self {
            routine,
            n,
            vectors,
            a: vec![0.0; n * n],
            w: vec![0.0; n],
            z: vec![
                0.0;
                if vectors && routine == "dsyevr" {
                    n * n
                } else {
                    1
                }
            ],
            isuppz: vec![0; 2 * n.max(1)],
            work: vec![0.0; 1],
            iwork: vec![0; 1],
        };
        // Workspace query: lwork = liwork = -1.
        let info = s.call(-1);
        assert_eq!(info, 0, "{routine} workspace query: info = {info}");
        let lwork = s.work[0] as usize;
        let liwork = s.iwork[0] as usize;
        s.work = vec![0.0; lwork.max(1)];
        s.iwork = vec![0; liwork.max(1)];
        s
    }

    fn call(&mut self, query: c_int) -> c_int {
        let n = int(self.n);
        let jobz = ch(if self.vectors { b'V' } else { b'N' });
        let (lwork, liwork) = if query == -1 {
            (-1, -1)
        } else {
            (int(self.work.len()), int(self.iwork.len()))
        };
        let mut info: c_int = 0;
        unsafe {
            if self.routine == "dsyevd" {
                dsyevd_(
                    jobz,
                    ch(b'L'),
                    &n,
                    self.a.as_mut_ptr(),
                    &n.max(1),
                    self.w.as_mut_ptr(),
                    self.work.as_mut_ptr(),
                    &lwork,
                    self.iwork.as_mut_ptr(),
                    &liwork,
                    &mut info,
                    1,
                    1,
                );
            } else {
                let mut m: c_int = 0;
                let ldz = if self.vectors { n.max(1) } else { 1 };
                dsyevr_(
                    jobz,
                    ch(b'A'),
                    ch(b'L'),
                    &n,
                    self.a.as_mut_ptr(),
                    &n.max(1),
                    &0.0,
                    &0.0,
                    &0,
                    &0,
                    // abstol = 0: LAPACK's default, eps * |T|, and MRRR's
                    // own high relative accuracy where it applies.
                    &0.0,
                    &mut m,
                    self.w.as_mut_ptr(),
                    self.z.as_mut_ptr(),
                    &ldz,
                    self.isuppz.as_mut_ptr(),
                    self.work.as_mut_ptr(),
                    &lwork,
                    self.iwork.as_mut_ptr(),
                    &liwork,
                    &mut info,
                    1,
                    1,
                    1,
                );
            }
        }
        info
    }

    /// Untimed: both routines overwrite A, so each run starts from a copy.
    pub fn load(&mut self, a: &[f64]) {
        self.a.copy_from_slice(a);
    }

    pub fn run(&mut self) -> Result<(), String> {
        match self.call(0) {
            0 => Ok(()),
            info => Err(format!("{}: info = {info}", self.routine)),
        }
    }

    /// Ascending.
    pub fn values(&self) -> &[f64] {
        &self.w
    }

    /// `dsyevd` leaves the vectors in A, `dsyevr` in Z.
    pub fn vectors(&self) -> Option<&[f64]> {
        match (self.vectors, self.routine) {
            (false, _) => None,
            (true, "dsyevd") => Some(&self.a),
            (true, _) => Some(&self.z),
        }
    }
}

/// `C = op(A) B` over column-major slices, `op(A)` m × k (`A` itself k × m
/// when `transpose_a`), `B` k × n: the accuracy checks' products.
pub fn gemm(transpose_a: bool, m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let (mi, ni, ki) = (int(m), int(n), int(k));
    let lda = if transpose_a { ki } else { mi };
    unsafe {
        dgemm_(
            ch(if transpose_a { b'T' } else { b'N' }),
            ch(b'N'),
            &mi,
            &ni,
            &ki,
            &1.0,
            a.as_ptr(),
            &lda.max(1),
            b.as_ptr(),
            &ki.max(1),
            &0.0,
            c.as_mut_ptr(),
            &mi.max(1),
            1,
            1,
        );
    }
}
