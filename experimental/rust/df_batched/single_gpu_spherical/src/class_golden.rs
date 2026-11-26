//! `ssperi3-class-golden OUT_DIR [XYZ]`: one class of the fully spherical
//! packed (μν|P), (p d | d), of water in cc-pVDZ against cc-pVDZ-JKFIT,
//! through this experiment's `libfbx_ssperi3_kernels.so`
//! (`fbx_ssperi3_packed_class`), written with the inputs it was launched on,
//! so rust/'s `fb_three_center_spherical_class` can be run on the same
//! inputs and compared (rust/integrals/sys/tests/three_center_spherical.rs).
//!
//! Every input is formed on the host, as `fb_mmd_records` and `ssperi3.rs`
//! form them: each record's zeta, P, PA, PB and s = `c_a c_b (π/ζ)^{3/2}`,
//! field-major; the aux records as `ssperi3.rs`'s `Aux`; the cells of every
//! (p, d) shell pair against every d aux shell, each pair's 15 spherical
//! columns side by side. Written, as float64 `.npy`, flat unless noted:
//!
//! - `records.npy`: 11 fields of `n_rec`;
//! - `aux.npy`: 5 fields of `n_aux_rec`;
//! - `cells.npy`: 6 fields of `n_cell`, each an integer;
//! - `eri3.npy`: shape `(n_aux, n_col)`, Fortran order, so aux fastest,
//!   zero but in the d aux functions' rows, the only ones the class writes.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use fb_basis::BasisSet;
use fb_integrals_sys::fb_primitive_pair_class_view;
use wwr_sys::wwr_memcpy_kind::{WWR_MEMCPY_DEVICE_TO_HOST, WWR_MEMCPY_HOST_TO_DEVICE};
use wwr_sys::wwr_status::WWR_STATUS_SUCCESS;

#[link(name = "fbx_ssperi3_kernels")]
unsafe extern "C" {
    fn fbx_ssperi3_packed_class(
        stream: *mut c_void,
        cls: *const fb_primitive_pair_class_view,
        lp: i32,
        n_aux_rec: i32,
        aux: *const f64,
        n_cell: i32,
        cells: *const i32,
        n_col: i32,
        n_aux: i32,
        eri3: *mut f64,
    ) -> i32;
}

const LA: i32 = 1;
const LB: i32 = 2;
const LP: i32 = 2;

fn device<T: Copy>(host: &[T]) -> *mut T {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let bytes = std::mem::size_of_val(host).max(1);
    // SAFETY: allocates `bytes`, then copies `host` into it.
    unsafe {
        assert_eq!(wwr_sys::wwr_malloc(&mut ptr, bytes), WWR_STATUS_SUCCESS);
        assert_eq!(
            wwr_sys::wwr_memcpy(ptr, host.as_ptr().cast(), bytes, WWR_MEMCPY_HOST_TO_DEVICE),
            WWR_STATUS_SUCCESS
        );
    }
    ptr.cast()
}

fn bases(xyz: &Path) -> (BasisSet, BasisSet) {
    let molecule = fb_io_xyz::read_xyz(xyz, 0, 1).expect("the XYZ file reads");
    let place = |name: &str, aux: bool| {
        let stamp = if aux {
            fb_io_basis::resolve_aux_basis(name)
        } else {
            fb_io_basis::resolve_basis(name)
        };
        stamp
            .expect("a carried set")
            .primitive_normalized()
            .place(&molecule)
            .expect("the set covers the molecule")
    };
    (place("cc-pvdz", false), place("cc-pvdz-jkfit", true))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(args.get(1).expect("usage: ssperi3-class-golden OUT_DIR [XYZ]"));
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let xyz = args.get(2).map_or_else(
        || here.join("../../../../rust/examples/water.xyz"),
        PathBuf::from,
    );
    let (orbital, aux) = bases(&xyz);

    // The class's records, pair by pair, primitive pair by primitive pair.
    let shells = orbital.shells();
    let of = |l: i32| (0..shells.len()).filter(move |&s| shells[s].angular_momentum() == l);
    let mut rows: Vec<[f64; 11]> = Vec::new();
    let mut pairs: Vec<[i32; 2]> = Vec::new();
    for a in of(LA) {
        for b in of(LB) {
            let (sa, sb) = (&shells[a], &shells[b]);
            let (ca, cb) = (sa.center(), sb.center());
            let begin = rows.len() as i32;
            for pa in sa.primitives() {
                for pb in sb.primitives() {
                    let (al, be) = (pa.exponent, pb.exponent);
                    let z = al + be;
                    let p = [
                        (al * ca.x + be * cb.x) / z,
                        (al * ca.y + be * cb.y) / z,
                        (al * ca.z + be * cb.z) / z,
                    ];
                    let s = pa.coefficient * pb.coefficient * (std::f64::consts::PI / z).powf(1.5);
                    rows.push([
                        z,
                        p[0],
                        p[1],
                        p[2],
                        p[0] - ca.x,
                        p[1] - ca.y,
                        p[2] - ca.z,
                        p[0] - cb.x,
                        p[1] - cb.y,
                        p[2] - cb.z,
                        s,
                    ]);
                }
            }
            pairs.push([begin, rows.len() as i32]);
        }
    }
    let n_rec = rows.len();
    let mut records = vec![0.0; 11 * n_rec];
    for (r, row) in rows.iter().enumerate() {
        for (f, &v) in row.iter().enumerate() {
            records[f * n_rec + r] = v;
        }
    }

    // The aux records, as ssperi3.rs's Aux.
    let mut arows: Vec<[f64; 5]> = Vec::new();
    let mut rec_begin = vec![0i32];
    let mut sph = Vec::new();
    let mut n_aux = 0;
    for shell in aux.shells() {
        let c = shell.center();
        for p in shell.primitives() {
            let w = p.coefficient * (std::f64::consts::PI / p.exponent).powf(1.5);
            arows.push([c.x, c.y, c.z, p.exponent, w]);
        }
        rec_begin.push(arows.len() as i32);
        sph.push(n_aux);
        n_aux += 2 * shell.angular_momentum() as usize + 1;
    }
    let n_aux_rec = arows.len();
    let mut aux_rec = vec![0.0; 5 * n_aux_rec];
    for (r, row) in arows.iter().enumerate() {
        for (f, &v) in row.iter().enumerate() {
            aux_rec[f * n_aux_rec + r] = v;
        }
    }

    // The cells: pair outer, d aux shell inner.
    let block = ((2 * LA + 1) * (2 * LB + 1)) as usize;
    let n_col = pairs.len() * block;
    let lp_shells: Vec<usize> = (0..aux.shells().len())
        .filter(|&s| aux.shells()[s].angular_momentum() == LP)
        .collect();
    let n_cell = pairs.len() * lp_shells.len();
    let mut cells = vec![0i32; 6 * n_cell];
    let mut k = 0;
    for (place, &[r0, r1]) in pairs.iter().enumerate() {
        for &sh in &lp_shells {
            let fields = [
                r0,
                r1,
                rec_begin[sh],
                rec_begin[sh + 1],
                (place * block * n_aux + sph[sh]) as i32,
                0,
            ];
            for (f, v) in fields.into_iter().enumerate() {
                cells[f * n_cell + k] = v;
            }
            k += 1;
        }
    }

    let d_records = device(&records);
    let field = |f: usize| d_records.cast_const().wrapping_add(f * n_rec);
    let null = std::ptr::null();
    let view = fb_primitive_pair_class_view {
        la: LA,
        lb: LB,
        n_rec: n_rec as i32,
        zeta: field(0),
        p_x: field(1),
        p_y: field(2),
        p_z: field(3),
        pa_x: field(4),
        pa_y: field(5),
        pa_z: field(6),
        pb_x: field(7),
        pb_y: field(8),
        pb_z: field(9),
        s: field(10),
        ea: null,
        eb: null,
        bound: null,
        bound_ab: null,
        bound_ba: null,
        off_a: std::ptr::null(),
        off_b: std::ptr::null(),
        rank_a: std::ptr::null(),
        rank_b: std::ptr::null(),
    };
    let d_aux = device(&aux_rec);
    let d_cells = device(&cells);
    // Zeroed: a class writes only its own aux momentum's functions.
    let mut eri3 = vec![0.0; n_col * n_aux];
    let d_eri3 = device(&eri3);
    // SAFETY: every array is device memory of the length the launcher reads
    // (formed above), and every cell indexes inside them; the default
    // stream, synchronized before the copy back.
    unsafe {
        let status = fbx_ssperi3_packed_class(
            std::ptr::null_mut(),
            &view,
            LP,
            n_aux_rec as i32,
            d_aux,
            n_cell as i32,
            d_cells,
            n_col as i32,
            n_aux as i32,
            d_eri3,
        );
        assert_eq!(status, 0, "the launch");
        assert_eq!(wwr_sys::wwr_device_synchronize(), WWR_STATUS_SUCCESS);
        assert_eq!(
            wwr_sys::wwr_memcpy(
                eri3.as_mut_ptr().cast(),
                d_eri3.cast_const().cast(),
                std::mem::size_of_val(eri3.as_slice()),
                WWR_MEMCPY_DEVICE_TO_HOST
            ),
            WWR_STATUS_SUCCESS
        );
    }
    assert!(eri3.iter().all(|v| v.is_finite()), "every element finite");

    std::fs::create_dir_all(&out).expect("the output directory");
    let write = |name: &str, shape: &[usize], data: &[f64]| {
        fb_io_npy::write_npy(&out.join(name), shape, data).expect("the .npy writes");
    };
    let as_f64 = |v: &[i32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    write("records.npy", &[records.len()], &records);
    write("aux.npy", &[aux_rec.len()], &aux_rec);
    write("cells.npy", &[cells.len()], &as_f64(&cells));
    write("eri3.npy", &[n_aux, n_col], &eri3);
    let max = eri3.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    println!(
        "({LA}, {LB}, {LP}): n_rec {n_rec}, n_aux_rec {n_aux_rec}, n_cell {n_cell}, n_col {n_col}, \
         n_aux {n_aux}, max |(μν|P)| {max:.6e}"
    );
}
