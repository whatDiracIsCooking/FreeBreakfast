//! `three_center_spherical_class`, the fully spherical packed (μν|P) over
//! `fb-core-device`'s stream and slices (tier 4).
//!
//! - One class, (p d | d), of water in cc-pVDZ against cc-pVDZ-JKFIT,
//!   launched on the inputs `experimental/rust/df_batched/single_gpu_spherical`
//!   launched its own `ssperi3` library on, and compared with what it wrote
//!   (`integrals/sys/tests/data/README.md`, "The fully spherical class"). The kernel units are
//!   the same text and the inputs the same bits, so the bar, 1e-12 of the
//!   largest element, admits only a different code generation or store
//!   order.
//! - A slice one element short, or a cell outside the slices it indexes, is
//!   refused before anything is enqueued, naming the kernel: the batch and
//!   the device cells still hold the bytes they were filled with.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use fb_core_device::{DeviceSlice, DeviceSliceMut, HostSlice, HostSliceMut, ScopedDevice, Stream};
use fb_integrals_sys::{check_library, three_center_spherical_class};
use wwr_sys::wwr_status::WWR_STATUS_SUCCESS;

/// The class the reference holds.
const CLASS: (i32, i32, i32) = (1, 2, 2);
/// The bytes a buffer is filled with to see whether anything wrote it.
const SENTINEL: u8 = 0xA5;

/// Device memory from `wwr_malloc`, freed on drop.
struct Dev<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Dev<T> {
    fn new(len: usize) -> Dev<T> {
        let _current = ScopedDevice::new(0);
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: writes one pointer through a valid local.
        let status = unsafe { wwr_sys::wwr_malloc(&mut ptr, len.max(1) * size_of::<T>()) };
        assert_eq!(status, WWR_STATUS_SUCCESS, "wwr_malloc");
        Dev {
            ptr: ptr.cast(),
            len,
        }
    }
}

impl<T> Drop for Dev<T> {
    fn drop(&mut self) {
        // SAFETY: allocated in `new`, freed once; every test syncs its
        // stream before its buffers drop.
        let status = unsafe { wwr_sys::wwr_free(self.ptr.cast()) };
        assert_eq!(status, WWR_STATUS_SUCCESS, "wwr_free");
    }
}

// SAFETY: `len` elements of device memory on device 0, live until the drop,
// which each test reaches only after syncing the stream that used it.
unsafe impl<T> DeviceSlice<T> for Dev<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr.cast_const()
    }
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above; a test hands it to one call at a time.
unsafe impl<T> DeviceSliceMut<T> for Dev<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

/// Host memory a test owns and does not touch while a copy of it is in
/// flight: it syncs the stream before reading or dropping it.
struct Host<T>(Vec<T>);

// SAFETY: the vector's elements, which the test leaves alone until the
// stream that copies them is synced.
unsafe impl<T> HostSlice<T> for Host<T> {
    fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

// SAFETY: as above.
unsafe impl<T> HostSliceMut<T> for Host<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }
}

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn npy(name: &str) -> fb_io_npy::NpyArray {
    fb_io_npy::read_npy(&data(name)).expect("the reference reads")
}

/// The reference's inputs and its output.
struct Reference {
    records: Vec<f64>,
    aux: Vec<f64>,
    cells: Vec<i32>,
    n_col: usize,
    n_aux: usize,
    eri3: Vec<f64>,
}

fn reference() -> Reference {
    let eri3 = npy("spherical_eri3.npy");
    let cells = npy("spherical_cells.npy")
        .data
        .iter()
        .map(|&v| v as i32)
        .collect();
    Reference {
        records: npy("spherical_records.npy").data,
        aux: npy("spherical_aux.npy").data,
        cells,
        n_aux: eri3.shape[0],
        n_col: eri3.shape[1],
        eri3: eri3.data,
    }
}

fn upload<T: Copy>(stream: &Stream, host: &Host<T>) -> Dev<T> {
    let mut dev = Dev::new(host.0.len());
    stream.upload(&mut dev, host).expect("the upload");
    dev
}

fn download<T: Copy + Default>(stream: &Stream, dev: &Dev<T>) -> Vec<T> {
    let mut host = Host(vec![T::default(); dev.len]);
    stream.download(&mut host, dev).expect("the download");
    stream.sync().expect("the sync");
    host.0
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn equals_the_experiments_output_on_water() {
    check_library().expect("the library is the bindings'");
    let r = reference();
    let stream = Stream::new(0);
    let records = Host(r.records.clone());
    let aux = Host(r.aux.clone());
    let cells = Host(r.cells.clone());
    let d_records = upload(&stream, &records);
    let d_aux = upload(&stream, &aux);
    let mut d_cells = Dev::new(cells.0.len());
    let mut eri3 = Dev::<f64>::new(r.n_col * r.n_aux);
    stream.zero(&mut eri3).expect("the zero");
    three_center_spherical_class(
        &stream,
        CLASS,
        r.records.len() / 11,
        &d_records,
        r.aux.len() / 5,
        &d_aux,
        r.cells.len() / 6,
        &cells,
        &mut d_cells,
        (r.n_col, r.n_aux),
        &mut eri3,
    )
    .expect("the launch");
    let got = download(&stream, &eri3);
    let max = r.eri3.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(&r.eri3)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(max > 0.1, "the reference is not empty: max {max:e}");
    assert!(
        worst <= 1e-12 * max,
        "largest difference {worst:e} over 1e-12 of max |(μν|P)| = {max:e}"
    );
}

/// The panic message of `f`, which must panic.
fn refusal(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("the call is refused");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default()
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_short_slice_or_a_stray_cell_is_refused_before_anything_is_enqueued() {
    let r = reference();
    let (n_rec, n_aux_rec, n_cell) = (r.records.len() / 11, r.aux.len() / 5, r.cells.len() / 6);
    let (n_col, n_aux) = (r.n_col, r.n_aux);
    let stream = Stream::new(0);
    let records = upload(&stream, &Host(r.records.clone()));
    let short_records = upload(&stream, &Host(r.records[1..].to_vec()));
    let aux = upload(&stream, &Host(r.aux.clone()));
    let short_aux = upload(&stream, &Host(r.aux[1..].to_vec()));
    let cells = Host(r.cells.clone());
    let short_cells = Host(r.cells[1..].to_vec());
    // The last cell's destination one element further than the batch.
    let mut stray = r.cells.clone();
    let dst = 4 * n_cell + n_cell - 1;
    stray[dst] += i32::try_from(n_col * n_aux).unwrap();
    let stray = Host(stray);
    let mut d_cells = Dev::<i32>::new(r.cells.len());
    let mut short_d_cells = Dev::<i32>::new(r.cells.len() - 1);
    let mut eri3 = Dev::<f64>::new(n_col * n_aux);
    let mut short_eri3 = Dev::<f64>::new(n_col * n_aux - 1);
    stream.fill(&mut eri3, SENTINEL).expect("the fill");
    stream.fill(&mut short_eri3, SENTINEL).expect("the fill");
    stream.fill(&mut d_cells, SENTINEL).expect("the fill");
    stream.fill(&mut short_d_cells, SENTINEL).expect("the fill");
    stream.sync().expect("the sync");

    let mut messages = Vec::new();
    messages.push(refusal(|| {
        let _ = three_center_spherical_class(
            &stream,
            CLASS,
            n_rec,
            &short_records,
            n_aux_rec,
            &aux,
            n_cell,
            &cells,
            &mut d_cells,
            (n_col, n_aux),
            &mut eri3,
        );
    }));
    messages.push(refusal(|| {
        let _ = three_center_spherical_class(
            &stream,
            CLASS,
            n_rec,
            &records,
            n_aux_rec,
            &short_aux,
            n_cell,
            &cells,
            &mut d_cells,
            (n_col, n_aux),
            &mut eri3,
        );
    }));
    messages.push(refusal(|| {
        let _ = three_center_spherical_class(
            &stream,
            CLASS,
            n_rec,
            &records,
            n_aux_rec,
            &aux,
            n_cell,
            &short_cells,
            &mut d_cells,
            (n_col, n_aux),
            &mut eri3,
        );
    }));
    messages.push(refusal(|| {
        let _ = three_center_spherical_class(
            &stream,
            CLASS,
            n_rec,
            &records,
            n_aux_rec,
            &aux,
            n_cell,
            &cells,
            &mut short_d_cells,
            (n_col, n_aux),
            &mut eri3,
        );
    }));
    messages.push(refusal(|| {
        let _ = three_center_spherical_class(
            &stream,
            CLASS,
            n_rec,
            &records,
            n_aux_rec,
            &aux,
            n_cell,
            &cells,
            &mut d_cells,
            (n_col, n_aux),
            &mut short_eri3,
        );
    }));
    messages.push(refusal(|| {
        let _ = three_center_spherical_class(
            &stream,
            CLASS,
            n_rec,
            &records,
            n_aux_rec,
            &aux,
            n_cell,
            &stray,
            &mut d_cells,
            (n_col, n_aux),
            &mut eri3,
        );
    }));
    for m in &messages {
        assert!(
            m.contains("fb_three_center_spherical_class"),
            "a refusal that does not name the kernel: {m}"
        );
    }
    assert!(
        messages[5].contains(&format!("cell {}", n_cell - 1)),
        "the stray cell is named: {}",
        messages[5]
    );

    // Nothing was enqueued: every buffer a call could write holds its fill.
    let untouched = |bytes: Vec<u8>, what: &str| {
        assert!(
            bytes.iter().all(|&b| b == SENTINEL),
            "{what} was written by a refused call"
        );
    };
    let as_bytes_f64 = |v: Vec<f64>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    let as_bytes_i32 = |v: Vec<i32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    untouched(as_bytes_f64(download(&stream, &eri3)), "eri3");
    untouched(
        as_bytes_f64(download(&stream, &short_eri3)),
        "the short eri3",
    );
    untouched(
        as_bytes_i32(download(&stream, &d_cells)),
        "the device cells",
    );
    untouched(
        as_bytes_i32(download(&stream, &short_d_cells)),
        "the short device cells",
    );
}
