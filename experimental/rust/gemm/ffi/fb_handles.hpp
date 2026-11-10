/// @file fb_handles.hpp
/// @brief What fb_gemm.h's opaque handles hold, for the C ABIs built on them
/// (rust/gemm's own, and rust/s_overlap's). C++ only, and internal: Rust sees
/// the handles only as pointers.
///
/// Include it after fb_gemm.h and after importing free_breakfast.core.wwr and
/// free_breakfast.core.resources: a plain header cannot import the modules
/// its types come from, so it relies on its includer having done so.
#ifndef FREE_BREAKFAST_EXPERIMENTAL_RUST_GEMM_FB_HANDLES_HPP
#define FREE_BREAKFAST_EXPERIMENTAL_RUST_GEMM_FB_HANDLES_HPP

struct fb_resources {
  free_breakfast::DeviceResources res;
};

struct fb_buffer {
  free_breakfast::Buffers::device<double> data;
};

/// Record `message` for fb_last_error() on this thread, and return FB_FAILED.
fb_status fb_fail(std::string message) noexcept;

#endif // FREE_BREAKFAST_EXPERIMENTAL_RUST_GEMM_FB_HANDLES_HPP
