#include <gtest/gtest.h>

#include <stdexcept>

#include "cpuprobe/util.hpp"

using namespace cpuprobe;

TEST(ParseSize, Suffixes) {
    EXPECT_EQ(parse_size("4K"), 4096u);
    EXPECT_EQ(parse_size("4k"), 4096u);
    EXPECT_EQ(parse_size("512M"), 536870912u);
    EXPECT_EQ(parse_size("1G"), 1073741824u);
    EXPECT_EQ(parse_size("100"), 100u);
    EXPECT_EQ(parse_size("32768K"), 32u << 20);  // this is how /sys writes it
}

TEST(ParseSize, BadInputThrows) {
    EXPECT_THROW(parse_size(""), std::invalid_argument);
    EXPECT_THROW(parse_size("K"), std::invalid_argument);
    EXPECT_THROW(parse_size("4X"), std::invalid_argument);
}

TEST(FormatSize, RoundTrip) {
    EXPECT_EQ(format_size(4096), "4K");
    EXPECT_EQ(format_size(6144), "6K");
    EXPECT_EQ(format_size(1u << 20), "1M");
}

TEST(Median, OddEvenAndEmpty) {
    EXPECT_DOUBLE_EQ(median({3, 1, 2}), 2.0);
    EXPECT_DOUBLE_EQ(median({4, 1, 3, 2}), 2.5);
    EXPECT_THROW(median({}), std::invalid_argument);
}
