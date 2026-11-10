// Compile-time checks of DeviceResources' workspaces (#1123): their sizes are
// the constructor's, which takes them as a MemoryFootprint and has no form
// without them, and no member grows them -- each borrow and carve names its
// caller and returns a Result, the refusal of a request past the size, so the
// pre-#1123 growing forms, a bare byte or element count, no longer compile.
// NodeResources passes its config's workspaces to every device.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;

namespace {

namespace fb = free_breakfast;

// ---------------------------------------------------------------------------
// The constructor takes the sizes
// ---------------------------------------------------------------------------

static_assert(
    std::is_constructible_v<fb::DeviceResources, std::int32_t, const fb::MemoryFootprint &>);
static_assert(
    std::is_constructible_v<fb::DeviceResources, std::int32_t, const fb::MemoryFootprint &,
                            std::uint64_t, std::size_t, std::uint64_t>);
static_assert(!std::is_default_constructible_v<fb::DeviceResources>);
static_assert(!std::is_constructible_v<fb::DeviceResources, std::int32_t>);
// The pre-#1123 (dev_idx, release threshold, ...) forms: no integer stands in
// for the sizes.
static_assert(!std::is_constructible_v<fb::DeviceResources, std::int32_t, std::uint64_t>);
static_assert(!std::is_constructible_v<fb::DeviceResources, std::int32_t, std::uint64_t,
                                       std::size_t, std::uint64_t>);
static_assert(!std::is_convertible_v<std::int32_t, fb::DeviceResources>);

static_assert(std::same_as<decltype(fb::NodeResourcesConfig::workspaces), fb::MemoryFootprint>);

// ---------------------------------------------------------------------------
// Borrows, never growth
// ---------------------------------------------------------------------------

using Borrow = fb::Result<std::span<std::byte>>;
static_assert(std::same_as<decltype(std::declval<fb::DeviceResources &>().device_workspace(
                               std::string_view{}, std::size_t{})),
                           Borrow>);
static_assert(std::same_as<decltype(std::declval<fb::DeviceResources &>().pinned_workspace(
                               std::string_view{}, std::size_t{})),
                           Borrow>);
static_assert(std::same_as<
              decltype(std::declval<fb::DeviceResources &>().device_regions<double, std::int32_t>(
                  std::string_view{}, std::size_t{}, std::size_t{})),
              fb::Result<std::tuple<double *, std::int32_t *>>>);
static_assert(std::same_as<decltype(std::declval<fb::DeviceResources &>().pinned_regions<double>(
                               std::string_view{}, std::size_t{}, std::size_t{})),
                           fb::Result<std::tuple<double *, double *>>>);

// No caller, no borrow: the growing forms are gone.
template <class R>
concept GrowsByBytes = requires(R &res) { res.device_workspace(std::size_t{}); };
template <class R>
concept CarvesUnnamed = requires(R &res) { res.template device_regions<double>(std::size_t{}); };
static_assert(!GrowsByBytes<fb::DeviceResources>);
static_assert(!CarvesUnnamed<fb::DeviceResources>);

// The sizes, and the largest borrows, read back as what the constructor took.
static_assert(std::same_as<decltype(std::declval<const fb::DeviceResources &>().workspaces()),
                           fb::MemoryFootprint>);
static_assert(
    std::same_as<decltype(std::declval<const fb::DeviceResources &>().workspace_high_water()),
                 fb::MemoryFootprint>);

} // namespace
