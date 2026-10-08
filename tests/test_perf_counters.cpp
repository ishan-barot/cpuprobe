#include <gtest/gtest.h>

#include "cpuprobe/perf_counters.hpp"

using namespace cpuprobe;

// ci runners and wsl are vms with no pmu, so these skip there instead of failing
#define SKIP_IF_NO_PMU() \
    if (!PerfGroup::available()) GTEST_SKIP() << "no hardware counters on this machine"

TEST(PerfGroup, CountsATrivialLoop) {
    SKIP_IF_NO_PMU();
    PerfGroup pg({Event::Cycles, Event::Instructions});
    pg.start();
    volatile std::uint64_t x = 0;
    for (int i = 0; i < 1'000'000; ++i) x = x + 1;
    auto c = pg.stop();
    EXPECT_GT(c.get(Event::Cycles), 0u);
    EXPECT_GT(c.get(Event::Instructions), 1'000'000u);  // at least one instruction per loop
    EXPECT_GT(c.ipc(), 0.0);
}

TEST(PerfGroup, CanStartAndStopTwice) {
    SKIP_IF_NO_PMU();
    PerfGroup pg({Event::Instructions});
    pg.start();
    auto a = pg.stop();
    pg.start();
    volatile int x = 0;
    for (int i = 0; i < 100'000; ++i) x = x + 1;
    auto b = pg.stop();
    // reset should wipe the first run, so the second one counts only its own loop
    EXPECT_GT(b.get(Event::Instructions), a.get(Event::Instructions));
}

TEST(PerfGroup, DefaultEventsOpen) {
    SKIP_IF_NO_PMU();
    PerfGroup pg(PerfGroup::default_events());
    pg.start();
    auto c = pg.stop();
    EXPECT_TRUE(c.has(Event::Cycles));
}

TEST(Counts, IpcAndAdd) {
    Counts a;
    a.values[Event::Cycles] = 100;
    a.values[Event::Instructions] = 250;
    EXPECT_DOUBLE_EQ(a.ipc(), 2.5);
    Counts b;
    b.values[Event::Cycles] = 100;
    b.multiplexed = true;
    a += b;
    EXPECT_EQ(a.get(Event::Cycles), 200u);
    EXPECT_TRUE(a.multiplexed);
    EXPECT_EQ(a.get(Event::BranchMisses), 0u);
}

TEST(Counts, IpcIsZeroWithoutCycles) {
    Counts c;
    EXPECT_DOUBLE_EQ(c.ipc(), 0.0);
}
