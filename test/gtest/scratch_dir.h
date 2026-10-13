/// @file
/// @brief A per-test scratch directory for the I/O tests that touch disk.
///
/// Include it AFTER `import std;`, like diagnostics.h: it includes no standard
/// header itself and names `std::` only through the including TU's import, so
/// the header build and the module build never overlap.

#pragma once

namespace free_breakfast::test {

/// @brief A fresh, empty directory under the system temp dir, removed (with
/// everything in it) on destruction.
///
/// Named after the running gtest case plus a per-process counter, so parallel
/// ctest entries -- one process per case -- never share a directory.
class ScratchDir {
public:
  explicit ScratchDir(std::string_view tag) {
    static std::int32_t counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            std::format("free_breakfast-{}-{}-{}", tag,
                        std::chrono::steady_clock::now().time_since_epoch().count(), counter++);
    std::filesystem::create_directories(path_);
  }
  ~ScratchDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  ScratchDir(const ScratchDir &) = delete;
  ScratchDir &operator=(const ScratchDir &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const { return path_; }
  [[nodiscard]] std::filesystem::path operator/(std::string_view name) const {
    return path_ / name;
  }

private:
  std::filesystem::path path_;
};

} // namespace free_breakfast::test
