/// @file
/// @brief Aggregated interface for the NumPy .npy format module.
///
/// `import free_breakfast.io.npy;` gives @ref free_breakfast::NpyArray
/// and both directions of the NumPy binary array format
/// (https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html),
/// restricted to the slice this project needs: float64 (`'<f8'`), held in
/// column-major order. A file written here loads in NumPy with a plain
/// `np.load`, and an `np.save`d float64 array of either layout reads back
/// here -- the interop bridge to the Python tooling layer.

export module free_breakfast.io.npy;

export import :shared;
export import :reader;
export import :writer;
