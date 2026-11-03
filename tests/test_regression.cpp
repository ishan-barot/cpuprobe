#include <gtest/gtest.h>

#include <sstream>

#include "cpuprobe/regression.hpp"
#include "cpuprobe/stats.hpp"

using namespace cpuprobe;

TEST(Stats, PercentileInterpolates) {
    EXPECT_DOUBLE_EQ(percentile({1, 2, 3, 4, 5}, 50), 3.0);
    EXPECT_DOUBLE_EQ(percentile({1, 2, 3, 4, 5}, 0), 1.0);
    EXPECT_DOUBLE_EQ(percentile({1, 2, 3, 4, 5}, 100), 5.0);
    EXPECT_DOUBLE_EQ(percentile({10, 20}, 25), 12.5);
    EXPECT_THROW(percentile({}, 50), std::invalid_argument);
    EXPECT_THROW(percentile({1}, 101), std::invalid_argument);
}

TEST(Stats, SummaryOrderDoesntMatter) {
    auto s = summarize({5, 1, 4, 2, 3});
    EXPECT_EQ(s.n, 5);
    EXPECT_DOUBLE_EQ(s.median, 3);
    EXPECT_DOUBLE_EQ(s.min, 1);
    EXPECT_DOUBLE_EQ(s.max, 5);
    EXPECT_LE(s.p05, s.median);
    EXPECT_GE(s.p95, s.median);
}

static BenchRow row(const std::string& v, double med, double p05, double p95) {
    BenchRow r;
    r.benchmark = "transpose";
    r.variant = v;
    r.size = 4096;
    r.reps = 30;
    r.median_ms = med;
    r.p05_ms = p05;
    r.p95_ms = p95;
    return r;
}

TEST(BenchCsv, RoundTripKeepsEverything) {
    BenchFile f;
    f.meta["gpu"] = "NVIDIA GeForce RTX 4070";
    f.meta["commit"] = "abc1234";
    f.rows.push_back(row("naive", 1.5, 1.4, 1.7));
    f.rows.back().gbps = 123.25;
    std::stringstream ss;
    write_bench_csv(ss, f);

    auto g = read_bench_csv(ss);
    EXPECT_EQ(g.meta["gpu"], "NVIDIA GeForce RTX 4070");
    EXPECT_EQ(g.meta["commit"], "abc1234");
    ASSERT_EQ(g.rows.size(), 1u);
    EXPECT_EQ(g.rows[0].variant, "naive");
    EXPECT_DOUBLE_EQ(g.rows[0].median_ms, 1.5);
    EXPECT_DOUBLE_EQ(g.rows[0].gbps, 123.25);
}

TEST(BenchCsv, BadFilesThrow) {
    std::stringstream empty;
    EXPECT_THROW(read_bench_csv(empty), std::runtime_error);
    std::stringstream wrong_header("a,b,c\n");
    EXPECT_THROW(read_bench_csv(wrong_header), std::runtime_error);
    std::stringstream short_row(
        "benchmark,variant,size,reps,median_ms,p05_ms,p95_ms,gbps,gflops\nx,y,1\n");
    EXPECT_THROW(read_bench_csv(short_row), std::runtime_error);
}

TEST(Compare, ClearSlowdownIsRegression) {
    BenchFile base, cur;
    base.rows.push_back(row("tiled_padded", 1.00, 0.98, 1.02));
    cur.rows.push_back(row("tiled_padded", 1.30, 1.27, 1.33));  // 30% slower, no overlap
    auto rep = compare(base, cur, 10);
    ASSERT_EQ(rep.rows.size(), 1u);
    EXPECT_EQ(rep.rows[0].verdict, Verdict::Regression);
    EXPECT_NEAR(rep.rows[0].change_pct, 30.0, 1e-9);
    EXPECT_TRUE(rep.has_regression());
}

TEST(Compare, OverlappingSpreadIsOnlyNoisy) {
    BenchFile base, cur;
    base.rows.push_back(row("naive", 1.00, 0.80, 1.40));  // jittery baseline
    cur.rows.push_back(row("naive", 1.15, 0.90, 1.50));   // slower median, but it overlaps
    auto rep = compare(base, cur, 10);
    EXPECT_EQ(rep.rows[0].verdict, Verdict::Noisy);
    EXPECT_FALSE(rep.has_regression());
}

TEST(Compare, SmallChangeIsOkAndFasterIsImproved) {
    BenchFile base, cur;
    base.rows.push_back(row("copy", 1.00, 0.99, 1.01));
    base.rows.push_back(row("tiled", 2.00, 1.98, 2.02));
    cur.rows.push_back(row("copy", 1.03, 1.02, 1.04));
    cur.rows.push_back(row("tiled", 1.00, 0.99, 1.01));
    auto rep = compare(base, cur, 10);
    EXPECT_EQ(rep.rows[0].verdict, Verdict::Ok);
    EXPECT_EQ(rep.rows[1].verdict, Verdict::Improved);
}

TEST(Compare, WarnsOnDifferentGpuAndMissingRows) {
    BenchFile base, cur;
    base.meta["gpu"] = "RTX 4070";
    cur.meta["gpu"] = "Tesla T4";
    base.rows.push_back(row("copy", 1, 1, 1));
    base.rows.push_back(row("naive", 1, 1, 1));
    cur.rows.push_back(row("copy", 1, 1, 1));
    auto rep = compare(base, cur, 10);
    EXPECT_EQ(rep.warnings.size(), 2u);  // gpu differs + naive missing
}
