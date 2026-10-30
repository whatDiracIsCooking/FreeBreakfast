# io/npy

The NumPy `.npy` binary array format
([spec](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)),
restricted to the slice this project needs: **float64 (`'<f8'`), held
column-major** -- the layout of every matrix in this tree, so a device buffer
copied to the host is already an `NpyArray`'s `data`. This is the interop
bridge to the Python tooling layer — a file written here loads with a plain
`np.load`, and an `np.save`d float64 array of either layout reads back here.

* `NpyArray` (in `:shared`) is the exchange type: a shape vector plus the
  elements flattened in Fortran order (first axis fastest). `{}` is a NumPy
  scalar, `{n}` a vector, `{rows, cols}` a matrix with (i, j) at
  `data[i + j * rows]`.
* The **writer** always emits format version 1.0 with NumPy's own canonical
  header (same dict spelling, same 64-byte alignment padding), and the
  payload as is. `'fortran_order'` is `True` exactly when np.save would say
  so for a Fortran-contiguous array: when the C and Fortran layouts actually
  differ (non-empty, two or more axes longer than one), `False` otherwise —
  a scalar, a vector, `(1, n)` or `(0, n)` is both. For every ordinary shape
  that is byte-for-byte `np.save(path, np.asfortranarray(a))` output —
  `test/io/npy_host.cpp` pins the 128-byte header numpy 2.5.4 writes for `()`,
  `(3,)`, `(2, 3)`, `(1, 3)`, `(0, 3)` and `(2, 1, 2)`. It is not identical
  *in general*: modern `np.save` also reserves up to 21 spaces so the first
  axis can grow in place, which this writer does not, so for a shape tuple
  long enough to straddle a 64-byte boundary the padding differs. Either way
  the file is valid and loads with `np.load`.
* The **reader** accepts versions 1.0–3.0 (they differ only in the header
  length field) and both layouts: a Fortran-order payload is taken as is, a
  C-order one is reordered into column-major. It rejects any dtype other
  than `'<f8'` as a classified `ErrorKind::IO` — never a silent
  misinterpretation.

The payload is memcpy'd, so the module `static_assert`s a little-endian
IEEE-754 host.

## Files

| File | Role |
|------|------|
| `npy.cppm` | Primary interface `free_breakfast.io.npy`; `export import`s the partitions |
| `npy_shared.cppm` | `:shared` — `NpyArray` and `element_count` |
| `npy_reader.cppm` | `:reader` — `parse_npy(bytes)` / `read_npy(path)` |
| `npy_writer.cppm` | `:writer` — `npy_to_bytes(array)` / `write_npy(path, array)`, and `npy_header(shape)`, the header alone, for a writer that streams the payload |
| `npy_shared.cpp`, `npy_reader.cpp`, `npy_writer.cpp` | Implementations |
| `CMakeLists.txt` | The `free_breakfast.io.npy` module target |
