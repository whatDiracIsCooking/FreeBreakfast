module;

#include <nlohmann/json.hpp>

export module free_breakfast.third_party.nlohmann;

export namespace free_breakfast::nlohmann {
using ::nlohmann::json;
using ::nlohmann::ordered_json;
} // namespace free_breakfast::nlohmann
