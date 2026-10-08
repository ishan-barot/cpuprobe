#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cpuprobe {

enum class Event {
    Cycles,
    Instructions,
    CacheMisses,  // last level cache misses, maps to different raw events on amd vs intel
    BranchMisses,
    L1DMisses,  // l1 data cache read misses
};

std::string event_name(Event e);

struct Counts {
    std::map<Event, std::uint64_t> values;
    bool multiplexed = false;  // true if the kernel had to share counters and we scaled

    bool has(Event e) const { return values.count(e) != 0; }
    std::uint64_t get(Event e) const;  // returns 0 if the event wasnt counted
    double ipc() const;
    Counts& operator+=(const Counts& other);
};

// wraps a group of perf_event_open counters. the first event is the group leader,
// so all of them start and stop at the exact same moment.
class PerfGroup {
public:
    explicit PerfGroup(std::vector<Event> events);
    ~PerfGroup();
    PerfGroup(const PerfGroup&) = delete;
    PerfGroup& operator=(const PerfGroup&) = delete;

    void start();
    Counts stop();

    // events the cpu or kernel didnt support get skipped instead of failing everything
    const std::vector<Event>& skipped() const { return skipped_; }

    // quick check so tests and ci can bail out on machines without a pmu (vms, wsl)
    static bool available();

    // the set we use for the kernels, kept to 5 so zen 3's 6 counters dont multiplex
    static std::vector<Event> default_events();

private:
    std::vector<int> fds_;
    std::vector<Event> events_;
    std::vector<Event> skipped_;
};

}  // namespace cpuprobe
