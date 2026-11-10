/// @file
/// @brief The workspaces a test's DeviceResources or NodeResources holds
/// when the routine under test brings no footprint of its own.
///
/// Lives in test/support/ (target free_breakfast::test::support), the one home
/// of the helpers every gtest tree shares; consumers include it by bare name.
///
/// Include it AFTER `import free_breakfast.core;` (or
/// `free_breakfast.core.footprint`, which `free_breakfast.core.resources`
/// re-exports), like pool_footprint.h: it names `free_breakfast::MemoryFootprint`
/// through the including TU's imports and includes nothing itself.
///
/// A DeviceResources' workspaces are sized at construction and never grow
/// (src/core/resources/README.md), so a test builds one sized for what it
/// runs, in one line:
///
/// ```cpp
/// fb::DeviceResources res{0, fb::test::kWorkspaces};           // most suites
/// fb::NodeResources node{{.workspaces = fb::test::kWorkspaces}};
/// fb::DeviceResources res{0, fp};  // a footprint test: the routine's own
/// ```
///
/// A footprint test sizes it with the routine's footprint instead, so the
/// workspace it runs in is exactly the one the footprint states, and a capped
/// one too, since the workspace counts against the pool's cap.

#pragma once

namespace free_breakfast::test {

/// @brief The device and pinned workspaces of a test DeviceResources: 256 MiB
/// and 4 MiB. The fast set's largest borrows, measured on gfx1200 before the
/// workspaces were fixed, were 100 MiB of device workspace (a benzene CIS
/// with the RIS preconditioner) and under 1 MiB pinned. A case past them is
/// refused at its borrow as Unsupported, naming the routine and both sizes,
/// and passes its own footprint or a larger constant.
inline constexpr MemoryFootprint kWorkspaces{.device = {.workspace = std::size_t{256} << 20},
                                             .pinned = {.workspace = std::size_t{4} << 20}};

} // namespace free_breakfast::test
