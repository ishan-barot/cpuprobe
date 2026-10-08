#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "cpuprobe/latency.hpp"

using namespace cpuprobe;

// if you start at 0 and keep following p you should hit every index once and end up back at 0
TEST(Sattolo, FormsSingleCycle) {
    for (std::size_t n : {2u, 3u, 10u, 1000u, 65536u}) {
        std::mt19937_64 rng(123);
        auto p = sattolo(n, rng);
        std::vector<bool> seen(n, false);
        std::size_t cur = 0, steps = 0;
        do {
            ASSERT_FALSE(seen[cur]) << "revisited " << cur << " for n=" << n;
            seen[cur] = true;
            cur = p[cur];
            ++steps;
        } while (cur != 0);
        EXPECT_EQ(steps, n);
    }
}

TEST(Sattolo, NoFixedPoints) {
    std::mt19937_64 rng(7);
    auto p = sattolo(5000, rng);
    for (std::size_t i = 0; i < p.size(); ++i) EXPECT_NE(p[i], i);
}

TEST(Node, IsOneCacheLineAndAligned) {
    EXPECT_EQ(sizeof(Node), 64u);
    EXPECT_EQ(alignof(Node), 64u);
}

TEST(Latency, ReturnsPositive) {
    auto r = measure_latency(16 * 1024, 3);
    EXPECT_GT(r.ns_per_access, 0.0);
}
