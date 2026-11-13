//! `test/io/npy_host.cpp`, one test per case, each named after it, and the
//! SAD stamps of `data/sad/STO-3G/` read and written back.
//!
//! An integration test rather than unit tests, for `CARGO_TARGET_TMPDIR`,
//! the scratch directory the disk cases write into.

use std::path::{Path, PathBuf};

use fb_core_error_handling::ErrorKind;
use fb_io_npy::{NpyArray, element_count, npy_to_bytes, parse_npy, read_npy, write_npy};

/// Hand-assemble a `.npy` document: any version, any header dict, any
/// payload, for the reader's refusals, which the writer cannot make:
/// `make_npy`.
fn make_npy(dict: &str, payload: &[u8], major: u8) -> Vec<u8> {
    let mut header = dict.as_bytes().to_vec();
    header.push(b'\n');
    let mut out = b"\x93NUMPY".to_vec();
    out.extend_from_slice(&[major, 0]);
    let len_bytes = if major == 1 { 2 } else { 4 };
    out.extend_from_slice(&(header.len() as u32).to_le_bytes()[..len_bytes]);
    out.extend_from_slice(&header);
    out.extend_from_slice(payload);
    out
}

/// The elements' little-endian bytes, `doubles_as_bytes`.
fn doubles_as_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `expect_io_error`: `bytes` is refused as `Io`, the message holding
/// `needle`.
#[track_caller]
fn expect_io_error(bytes: &[u8], needle: &str) {
    let error = parse_npy(bytes).expect_err(&format!("accepted, expected '{needle}'"));
    assert_eq!(error.kind, ErrorKind::Io);
    assert!(error.message.contains(needle), "{}", error.message);
}

/// A fresh scratch directory of this binary's, `ScratchDir`.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// --- element_count ---------------------------------------------------------

/// `IoNpy.ElementCountOfEmptyShapeIsOneScalar`.
#[test]
fn element_count_of_empty_shape_is_one_scalar() {
    assert_eq!(element_count(&[]).unwrap(), 1);
}

/// `IoNpy.ElementCountIsTheProductOfDimensions`.
#[test]
fn element_count_is_the_product_of_dimensions() {
    assert_eq!(element_count(&[7]).unwrap(), 7);
    assert_eq!(element_count(&[2, 3, 4]).unwrap(), 24);
    assert_eq!(element_count(&[5, 0, 3]).unwrap(), 0);
}

/// `IoNpy.ElementCountOverflowIsInvalidConfig`.
#[test]
fn element_count_overflow_is_invalid_config() {
    let error = element_count(&[usize::MAX / 2, 3]).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
}

// --- Writer ----------------------------------------------------------------

/// `IoNpy.HeaderMatchesNumpySaveByteForByte`: the 128-byte headers numpy
/// 2.5.4's `np.save` writes for a float64 Fortran-order array of each
/// shape, captured from real `np.save` output by the C++ case.
#[test]
fn header_matches_numpy_save_byte_for_byte() {
    let golden: [(&[usize], &str); 7] = [
        (
            &[],
            "{'descr': '<f8', 'fortran_order': False, 'shape': (), }",
        ),
        (
            &[3],
            "{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }",
        ),
        (
            &[0],
            "{'descr': '<f8', 'fortran_order': False, 'shape': (0,), }",
        ),
        (
            &[2, 3],
            "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }",
        ),
        (
            &[1, 3],
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1, 3), }",
        ),
        (
            &[0, 3],
            "{'descr': '<f8', 'fortran_order': False, 'shape': (0, 3), }",
        ),
        (
            &[2, 1, 2],
            "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 1, 2), }",
        ),
    ];
    for (shape, dict) in golden {
        let array = NpyArray {
            shape: shape.to_vec(),
            data: vec![0.0; element_count(shape).unwrap()],
        };
        let bytes = npy_to_bytes(&array).unwrap();

        let mut expected = b"\x93NUMPY\x01\x00\x76\x00".to_vec(); // v1.0, header_len 118
        expected.extend_from_slice(dict.as_bytes());
        expected.resize(127, b' ');
        expected.push(b'\n');
        assert!(bytes.len() >= 128);
        assert_eq!(&bytes[..128], expected.as_slice(), "{dict}");
        assert_eq!(bytes.len(), 128 + array.data.len() * 8);
    }
}

/// `IoNpy.HeaderIsSixtyFourByteAlignedForHighRank`: a rank-40 shape's tuple
/// alone outgrows the usual 128-byte header.
#[test]
fn header_is_sixty_four_byte_aligned_for_high_rank() {
    let bytes = npy_to_bytes(&NpyArray {
        shape: vec![1; 40],
        data: vec![42.0],
    })
    .unwrap();
    let data_offset = bytes.len() - 8;
    assert!(data_offset > 128);
    assert_eq!(data_offset % 64, 0);
    assert_eq!(bytes[data_offset - 1], b'\n');
}

/// `IoNpy.ShapeDataMismatchIsInvalidConfig`.
#[test]
fn shape_data_mismatch_is_invalid_config() {
    let error = npy_to_bytes(&NpyArray {
        shape: vec![2, 2],
        data: vec![1.0, 2.0, 3.0],
    })
    .unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert!(error.message.contains("(2, 2)"), "{}", error.message);
}

// --- Round trip ------------------------------------------------------------

/// `IoNpy.RoundTripsEveryRankAndPreservesBitPatterns`. Bits are compared,
/// not values: NaN != NaN and -0.0 == 0.0 would both hide a corrupted
/// payload.
#[test]
fn round_trips_every_rank_and_preserves_bit_patterns() {
    let arrays = [
        NpyArray {
            shape: vec![],
            data: vec![-0.0],
        },
        NpyArray {
            shape: vec![0],
            data: vec![],
        },
        NpyArray {
            shape: vec![4],
            data: vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::from_bits(1), // denorm_min
            ],
        },
        NpyArray {
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        NpyArray {
            shape: vec![2, 1, 2],
            data: vec![0.1, 0.2, 0.3, 1.0 / 3.0],
        },
    ];
    for array in &arrays {
        let back = parse_npy(&npy_to_bytes(array).unwrap()).unwrap();
        assert_eq!(back.shape, array.shape);
        assert_eq!(doubles_as_bytes(&back.data), doubles_as_bytes(&array.data));
    }
}

/// `IoNpy.WriteThenReadRoundTripsOnDisk`.
#[test]
fn write_then_read_round_trips_on_disk() {
    let dir = scratch("write_then_read_round_trips_on_disk");
    let array = NpyArray {
        shape: vec![3, 2],
        data: vec![1.5, -2.5, 3.25, 0.0, 1e300, -1e-300],
    };
    write_npy(&dir.join("a.npy"), &array).unwrap();
    let back = read_npy(&dir.join("a.npy")).unwrap();
    assert_eq!(back.shape, array.shape);
    assert_eq!(back.data, array.data);
}

/// `IoNpy.WriteOfMismatchedArrayLeavesNoFile`.
#[test]
fn write_of_mismatched_array_leaves_no_file() {
    let dir = scratch("write_of_mismatched_array_leaves_no_file");
    let bad = NpyArray {
        shape: vec![3],
        data: vec![1.0],
    };
    assert!(write_npy(&dir.join("bad.npy"), &bad).is_err());
    assert!(!dir.join("bad.npy").exists());
}

// --- Reader ----------------------------------------------------------------

/// `IoNpy.AcceptsFormatVersionsTwoAndThree`.
#[test]
fn accepts_format_versions_two_and_three() {
    let values = doubles_as_bytes(&[1.0, 2.0]);
    for major in [2, 3] {
        let array = parse_npy(&make_npy(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }",
            &values,
            major,
        ))
        .unwrap_or_else(|e| panic!("v{major}: {}", e.message));
        assert_eq!(array.data, [1.0, 2.0]);
    }
}

/// `IoNpy.ToleratesKeyOrderAndSpacing`.
#[test]
fn tolerates_key_order_and_spacing() {
    let array = parse_npy(&make_npy(
        "{'shape':(2,2),'fortran_order':False,'descr':'<f8'}",
        &doubles_as_bytes(&[1.0, 2.0, 3.0, 4.0]),
        1,
    ))
    .unwrap();
    assert_eq!(array.shape, [2, 2]);
}

/// `IoNpy.RejectsBadMagic`.
#[test]
fn rejects_bad_magic() {
    expect_io_error(b"not an npy file at all", "bad magic");
    expect_io_error(b"", "bad magic");
}

/// `IoNpy.RejectsUnsupportedVersion`.
#[test]
fn rejects_unsupported_version() {
    expect_io_error(&make_npy("{}", b"", 4), "unsupported NPY format version 4");
    expect_io_error(&make_npy("{}", b"", 0), "unsupported NPY format version 0");
}

/// `IoNpy.RejectsTruncation`: the length field cut short, then a header
/// length claiming more bytes than the file holds.
#[test]
fn rejects_truncation() {
    expect_io_error(b"\x93NUMPY\x02\x00\x10", "truncated NPY header length");
    expect_io_error(b"\x93NUMPY\x01\x00\xff\x00{'descr'", "truncated NPY header");
}

/// `IoNpy.RejectsOtherDtypes`.
#[test]
fn rejects_other_dtypes() {
    for descr in ["<f4", ">f8", "<i8", "|b1"] {
        expect_io_error(
            &make_npy(
                &format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': (1,), }}"),
                &[0; 8],
                1,
            ),
            &format!("unsupported NPY dtype '{descr}'"),
        );
    }
}

/// `IoNpy.WritesTheColumnMajorPayloadAsIs`.
#[test]
fn writes_the_column_major_payload_as_is() {
    let array = NpyArray {
        shape: vec![2, 3],
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    };
    let bytes = npy_to_bytes(&array).unwrap();
    assert_eq!(&bytes[128..], doubles_as_bytes(&array.data).as_slice());
}

/// `IoNpy.ReadsFortranOrderPayloadAsIs`:
/// `np.asfortranarray([[1, 2, 3], [4, 5, 6]])`, the columns contiguous.
#[test]
fn reads_fortran_order_payload_as_is() {
    let array = parse_npy(&make_npy(
        "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }",
        &doubles_as_bytes(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        1,
    ))
    .unwrap();
    assert_eq!(array.shape, [2, 3]);
    assert_eq!(array.data, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

/// `IoNpy.ReordersCOrderMatrixIntoColumnMajor`:
/// `np.array([[1, 2, 3], [4, 5, 6]])` saved C-order, the rows contiguous.
#[test]
fn reorders_c_order_matrix_into_column_major() {
    let array = parse_npy(&make_npy(
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }",
        &doubles_as_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        1,
    ))
    .unwrap();
    assert_eq!(array.shape, [2, 3]);
    assert_eq!(array.data, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

/// `IoNpy.ReordersCOrderRankThreeIntoColumnMajor`:
/// `np.arange(24.0).reshape(2, 3, 4)`, element (i, j, k) `12i + 4j + k`,
/// which column-major puts at `i + 2j + 6k`.
#[test]
fn reorders_c_order_rank_three_into_column_major() {
    let c_order: Vec<f64> = (0..24).map(f64::from).collect();
    let array = parse_npy(&make_npy(
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3, 4), }",
        &doubles_as_bytes(&c_order),
        1,
    ))
    .unwrap();
    assert_eq!(array.data.len(), 24);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(
                    array.data[i + 2 * j + 6 * k],
                    (12 * i + 4 * j + k) as f64,
                    "{i}{j}{k}"
                );
            }
        }
    }
}

/// `IoNpy.ReadsEmptyCOrderMatrix`.
#[test]
fn reads_empty_c_order_matrix() {
    let array = parse_npy(&make_npy(
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0, 3), }",
        b"",
        1,
    ))
    .unwrap();
    assert_eq!(array.shape, [0, 3]);
    assert!(array.data.is_empty());
}

/// `IoNpy.RejectsMalformedHeaderFields`.
#[test]
fn rejects_malformed_header_fields() {
    expect_io_error(
        &make_npy("{'fortran_order': False, 'shape': (1,), }", b"", 1),
        "quoted 'descr'",
    );
    expect_io_error(
        &make_npy("{'descr': '<f8', 'shape': (1,), }", b"", 1),
        "boolean 'fortran_order'",
    );
    expect_io_error(
        &make_npy("{'descr': '<f8', 'fortran_order': False, }", b"", 1),
        "shape tuple",
    );
    expect_io_error(
        &make_npy(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (2, x), }",
            b"",
            1,
        ),
        "not a non-negative integer",
    );
    expect_io_error(
        &make_npy(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (2, ",
            b"",
            1,
        ),
        "unterminated",
    );
}

/// `IoNpy.RejectsPayloadLengthMismatch`.
#[test]
fn rejects_payload_length_mismatch() {
    let dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }";
    expect_io_error(
        &make_npy(dict, &[0; 8], 1),
        "payload is 8 bytes but the shape needs 16",
    );
    expect_io_error(&make_npy(dict, &[0; 24], 1), "payload is 24 bytes");
}

/// `IoNpy.ReadOfMissingFileIsIoError`.
#[test]
fn read_of_missing_file_is_io_error() {
    let dir = scratch("read_of_missing_file_is_io_error");
    let error = read_npy(&dir.join("missing.npy")).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
}

// --- The SAD stamps --------------------------------------------------------

/// Every stamp under `data/sad/STO-3G/` parses to a square float64 matrix,
/// and writing it back gives the file's bytes. The C++ generator wrote
/// them through `npy_to_bytes`, so this is the writer agreeing with it
/// byte for byte as well as the reader reading what the tree holds.
#[test]
fn sad_stamps_parse_square_and_write_back_byte_for_byte() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/sad/STO-3G");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "npy"))
        .collect();
    paths.sort();
    // H through Kr, the elements STO-3G carries.
    assert_eq!(paths.len(), 36, "{}", dir.display());
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        let array =
            parse_npy(&bytes).unwrap_or_else(|e| panic!("{}: {}", path.display(), e.message));
        let [rows, cols] = array.shape[..] else {
            panic!(
                "{}: shape {:?} is not a matrix",
                path.display(),
                array.shape
            );
        };
        assert!(
            rows > 0 && rows == cols,
            "{}: {rows} x {cols}",
            path.display()
        );
        assert_eq!(npy_to_bytes(&array).unwrap(), bytes, "{}", path.display());
    }
}
