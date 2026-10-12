module;

#include <toml++/toml.hpp>

export module free_breakfast.third_party.tomlplusplus;

export namespace free_breakfast::toml {

using ::toml::array;
using ::toml::key;
using ::toml::node;
using ::toml::node_view;
using ::toml::parse;
using ::toml::parse_error;
using ::toml::table;
using ::toml::value;
using ::toml::value_flags;
inline constexpr auto preserve_source_value_flags = ::toml::preserve_source_value_flags;

} // namespace free_breakfast::toml

export namespace free_breakfast::toml::impl {

using ::toml::impl::table_init_pair;
using ::toml::impl::make_node;
using ::toml::impl::node_ptr;

} // namespace free_breakfast::toml::impl
