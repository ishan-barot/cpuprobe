#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "cpuprobe/kernels.hpp"

using namespace cpuprobe;

// the good and bad versions have to give the same answer or the speed comparison means nothing

TEST(Kernels, RowsAndColsAgree) {
    std::size_t n = 257;  // odd size on purpose
    std::vector<int> m(n * n);
    for (std::size_t i = 0; i < m.size(); ++i) m[i] = static_cast<int>(i % 97);
    EXPECT_EQ(sum_rows(m, n), sum_cols(m, n));
}

TEST(Kernels, BranchyAndBranchlessAgree) {
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> d(0, 255);
    std::vector<int> v(10000);
    for (auto& x : v) x = d(rng);
    EXPECT_EQ(sum_branchy(v), sum_branchless(v));
    std::sort(v.begin(), v.end());
    EXPECT_EQ(sum_branchy(v), sum_branchless(v));
}

TEST(Kernels, AosAndSoaAgree) {
    std::vector<Particle> aos(1000);
    std::vector<float> xs(1000);
    for (int i = 0; i < 1000; ++i) {
        aos[static_cast<std::size_t>(i)] = Particle{static_cast<float>(i), 0, 0, 0, 0, 0, 0, 0};
        xs[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    EXPECT_DOUBLE_EQ(sum_x_aos(aos), sum_x_soa(xs));
}

TEST(Kernels, ParticleIs32Bytes) { EXPECT_EQ(sizeof(Particle), 32u); }

TEST(Kernels, FalseShareCountsEverything) {
    for (bool padded : {false, true}) {
        auto total = run_falseshare(padded, 2, 10000, {});
        EXPECT_EQ(total, 20000u) << (padded ? "padded" : "packed");
    }
}

TEST(Kernels, FalseShareRejectsBadThreadCount) {
    EXPECT_THROW(run_falseshare(true, 0, 10, {}), std::invalid_argument);
    EXPECT_THROW(run_falseshare(true, 9, 10, {}), std::invalid_argument);
}
