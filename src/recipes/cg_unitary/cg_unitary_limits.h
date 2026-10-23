/// @file
/// @brief kCgMaxSamples, the line searches' sample cap, for the module and
/// its device code alike.
///
/// Its own header, without the bridge's runtime include, so the module
/// interface can GMF-include it, as src/core/kernel/kernel.cppm does
/// block_params.h: wwr's runtime header in an interface's GMF reaches every
/// importer and collides there with the named wwr.runtime_api module.

#pragma once

#include "integral_types.h"

namespace free_breakfast {

/// @brief The most derivative samples one line search takes.
///
/// The kernels size their arrays by it, and Table 2's Fourier polynomial has
/// degree samples - 1, which polynomial_roots solves one lane per root. So it
/// must stay at most kWarpSize + 1; 32 leaves the DFT 31 samples, the paper's
/// own N_DFT at K = 3, N_T = 10.
inline constexpr int32_t kCgMaxSamples = 32;

} // namespace free_breakfast
