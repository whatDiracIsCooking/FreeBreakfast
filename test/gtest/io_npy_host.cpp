// Host-only cases for io/npy: the float64 C-order slice of NumPy's .npy
// format, both directions.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.npy;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

/// @brief Hand-assemble a .npy document: any version, any header dict, any
/// payload -- for the reader's rejection cases, which the writer cannot make.
std::string make_npy(std::string_view dict, std::string_view payload, unsigned major = 1) {
  std::string header{dict};
  header.push_back('\n');
  std::string out{"\x93NUMPY", 6};
  out.push_back(static_cast<char>(major));
  out.push_back('\0');
  const std::size_t len_bytes = major == 1 ? 2 : 4;
  for (std::size_t i = 0; i < len_bytes; ++i) {
    out.push_back(static_cast<char>((header.size() >> (8 * i)) & 0xff));
  }
  return out + header + std::string{payload};
}

std::string doubles_as_bytes(std::span<const double> values) {
  return {reinterpret_cast<const char *>(values.data()), values.size_bytes()};
}

void expect_io_error(std::string_view bytes, std::string_view needle) {
  auto array = fb::parse_npy(bytes);
  ASSERT_FALSE(array.has_value()) << "accepted, expected '" << needle << "'";
  EXPECT_EQ(array.error().kind, fb::ErrorKind::IO);
  EXPECT_NE(array.error().message.find(needle), std::string::npos) << array.error().message;
}

// ---------------------------------------------------------------------------
// element_count
// ---------------------------------------------------------------------------

TEST(IoNpy, ElementCountOfEmptyShapeIsOneScalar) {
  EXPECT_EQ(fb::element_count({}).value(), 1U);
}

TEST(IoNpy, ElementCountIsTheProductOfDimensions) {
  EXPECT_EQ(fb::element_count(std::vector<std::size_t>{7}).value(), 7U);
  EXPECT_EQ(fb::element_count(std::vector<std::size_t>{2, 3, 4}).value(), 24U);
  EXPECT_EQ(fb::element_count(std::vector<std::size_t>{5, 0, 3}).value(), 0U);
}

TEST(IoNpy, ElementCountOverflowIsInvalidConfig) {
  const auto big = std::numeric_limits<std::size_t>::max() / 2;
  auto count = fb::element_count(std::vector<std::size_t>{big, 3});
  ASSERT_FALSE(count.has_value());
  EXPECT_EQ(count.error().kind, fb::ErrorKind::InvalidConfig);
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

// The 128-byte headers numpy 2.5.4's np.save writes for a float64 C-order
// array of each shape, captured from real np.save output. The payload follows
// at byte 128 in every case.
TEST(IoNpy, HeaderMatchesNumpySaveByteForByte) {
  const std::array<std::pair<std::vector<std::size_t>, std::string_view>, 4> golden{{
      {{}, "{'descr': '<f8', 'fortran_order': False, 'shape': (), }"},
      {{3}, "{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }"},
      {{2, 3}, "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }"},
      {{0}, "{'descr': '<f8', 'fortran_order': False, 'shape': (0,), }"},
  }};
  for (const auto &[shape, dict] : golden) {
    fb::NpyArray array{shape, std::vector<double>(fb::element_count(shape).value(), 0.0)};
    auto bytes = fb::npy_to_bytes(array);
    ASSERT_TRUE(bytes.has_value()) << bytes.error().message;

    std::string expected{"\x93NUMPY\x01\x00\x76\x00", 10}; // v1.0, header_len 118
    expected += dict;
    expected.append(118 - dict.size() - 1, ' ');
    expected.push_back('\n');
    ASSERT_GE(bytes->size(), 128U);
    EXPECT_EQ(bytes->substr(0, 128), expected) << dict;
    EXPECT_EQ(bytes->size(), 128 + array.data.size() * sizeof(double));
  }
}

TEST(IoNpy, HeaderIsSixtyFourByteAlignedForHighRank) {
  // A rank-40 shape: its tuple alone outgrows the usual 128-byte header.
  const std::vector<std::size_t> shape(40, 1);
  auto bytes = fb::npy_to_bytes({shape, {42.0}});
  ASSERT_TRUE(bytes.has_value());
  const std::size_t data_offset = bytes->size() - sizeof(double);
  EXPECT_GT(data_offset, 128U);
  EXPECT_EQ(data_offset % 64, 0U);
  EXPECT_EQ((*bytes)[data_offset - 1], '\n');
}

TEST(IoNpy, ShapeDataMismatchIsInvalidConfig) {
  auto bytes = fb::npy_to_bytes({{2, 2}, {1.0, 2.0, 3.0}});
  ASSERT_FALSE(bytes.has_value());
  EXPECT_EQ(bytes.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(bytes.error().message.find("(2, 2)"), std::string::npos) << bytes.error().message;
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

TEST(IoNpy, RoundTripsEveryRankAndPreservesBitPatterns) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::array arrays{
      fb::NpyArray{{}, {-0.0}},
      fb::NpyArray{{0}, {}},
      fb::NpyArray{{4}, {nan, inf, -inf, std::numeric_limits<double>::denorm_min()}},
      fb::NpyArray{{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}},
      fb::NpyArray{{2, 1, 2}, {0.1, 0.2, 0.3, 1.0 / 3.0}},
  };
  for (const auto &array : arrays) {
    auto bytes = fb::npy_to_bytes(array);
    ASSERT_TRUE(bytes.has_value()) << bytes.error().message;
    auto back = fb::parse_npy(*bytes);
    ASSERT_TRUE(back.has_value()) << back.error().message;
    EXPECT_EQ(back->shape, array.shape);
    // Compare bits, not values: NaN != NaN and -0.0 == 0.0 would both hide a
    // corrupted payload.
    EXPECT_EQ(doubles_as_bytes(back->data), doubles_as_bytes(array.data));
  }
}

TEST(IoNpy, WriteThenReadRoundTripsOnDisk) {
  const fb::test::ScratchDir dir("io-npy");
  const fb::NpyArray array{{3, 2}, {1.5, -2.5, 3.25, 0.0, 1e300, -1e-300}};
  ASSERT_TRUE(fb::write_npy(dir / "a.npy", array).has_value());
  auto back = fb::read_npy(dir / "a.npy");
  ASSERT_TRUE(back.has_value()) << back.error().message;
  EXPECT_EQ(back->shape, array.shape);
  EXPECT_EQ(back->data, array.data);
}

TEST(IoNpy, WriteOfMismatchedArrayLeavesNoFile) {
  const fb::test::ScratchDir dir("io-npy");
  ASSERT_FALSE(fb::write_npy(dir / "bad.npy", {{3}, {1.0}}).has_value());
  EXPECT_FALSE(std::filesystem::exists(dir / "bad.npy"));
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

TEST(IoNpy, AcceptsFormatVersionsTwoAndThree) {
  const std::array values{1.0, 2.0};
  for (const unsigned major : {2U, 3U}) {
    auto array = fb::parse_npy(make_npy("{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }",
                                        doubles_as_bytes(values), major));
    ASSERT_TRUE(array.has_value()) << "v" << major << ": " << array.error().message;
    EXPECT_EQ(array->data, (std::vector<double>{1.0, 2.0}));
  }
}

TEST(IoNpy, ToleratesKeyOrderAndSpacing) {
  const std::array values{1.0, 2.0, 3.0, 4.0};
  auto array = fb::parse_npy(
      make_npy("{'shape':(2,2),'fortran_order':False,'descr':'<f8'}", doubles_as_bytes(values)));
  ASSERT_TRUE(array.has_value()) << array.error().message;
  EXPECT_EQ(array->shape, (std::vector<std::size_t>{2, 2}));
}

TEST(IoNpy, RejectsBadMagic) {
  expect_io_error("not an npy file at all", "bad magic");
  expect_io_error("", "bad magic");
}

TEST(IoNpy, RejectsUnsupportedVersion) {
  expect_io_error(make_npy("{}", "", 4), "unsupported NPY format version 4");
  expect_io_error(make_npy("{}", "", 0), "unsupported NPY format version 0");
}

TEST(IoNpy, RejectsTruncation) {
  expect_io_error(std::string{"\x93NUMPY\x02\x00\x10", 9}, "truncated NPY header length");
  // Header length claims more bytes than the file holds.
  expect_io_error(std::string{"\x93NUMPY\x01\x00\xff\x00{'descr'", 17}, "truncated NPY header");
}

TEST(IoNpy, RejectsOtherDtypes) {
  for (const std::string_view descr : {"<f4", ">f8", "<i8", "|b1"}) {
    expect_io_error(
        make_npy(std::format("{{'descr': '{}', 'fortran_order': False, 'shape': (1,), }}", descr),
                 std::string(8, '\0')),
        std::format("unsupported NPY dtype '{}'", descr));
  }
}

TEST(IoNpy, RejectsFortranOrder) {
  expect_io_error(
      make_npy("{'descr': '<f8', 'fortran_order': True, 'shape': (1,), }", std::string(8, '\0')),
      "Fortran-order");
}

TEST(IoNpy, RejectsMalformedHeaderFields) {
  expect_io_error(make_npy("{'fortran_order': False, 'shape': (1,), }", ""), "quoted 'descr'");
  expect_io_error(make_npy("{'descr': '<f8', 'shape': (1,), }", ""), "boolean 'fortran_order'");
  expect_io_error(make_npy("{'descr': '<f8', 'fortran_order': False, }", ""), "shape tuple");
  expect_io_error(make_npy("{'descr': '<f8', 'fortran_order': False, 'shape': (2, x), }", ""),
                  "not a non-negative integer");
  expect_io_error(make_npy("{'descr': '<f8', 'fortran_order': False, 'shape': (2, ", ""),
                  "unterminated");
}

TEST(IoNpy, RejectsPayloadLengthMismatch) {
  const auto dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }";
  expect_io_error(make_npy(dict, std::string(8, '\0')),
                  "payload is 8 bytes but the shape needs 16");
  expect_io_error(make_npy(dict, std::string(24, '\0')), "payload is 24 bytes");
}

TEST(IoNpy, ReadOfMissingFileIsIoError) {
  const fb::test::ScratchDir dir("io-npy");
  auto array = fb::read_npy(dir / "missing.npy");
  ASSERT_FALSE(array.has_value());
  EXPECT_EQ(array.error().kind, fb::ErrorKind::IO);
}

} // namespace
