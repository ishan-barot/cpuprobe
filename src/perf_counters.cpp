#include "cpuprobe/perf_counters.hpp"

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace cpuprobe {

std::string event_name(Event e) {
    switch (e) {
        case Event::Cycles:
            return "cycles";
        case Event::Instructions:
            return "instructions";
        case Event::CacheMisses:
            return "llc_misses";
        case Event::BranchMisses:
            return "branch_misses";
        case Event::L1DMisses:
            return "l1d_misses";
    }
    return "unknown";
}

std::uint64_t Counts::get(Event e) const {
    auto it = values.find(e);
    return it == values.end() ? 0 : it->second;
}

double Counts::ipc() const {
    auto c = get(Event::Cycles);
    return c ? static_cast<double>(get(Event::Instructions)) / static_cast<double>(c) : 0.0;
}

Counts& Counts::operator+=(const Counts& other) {
    for (auto& [e, v] : other.values) values[e] += v;
    multiplexed = multiplexed || other.multiplexed;
    return *this;
}

namespace {

perf_event_attr make_attr(Event e, bool leader) {
    perf_event_attr a{};
    a.size = sizeof(a);
    switch (e) {
        case Event::Cycles:
            a.type = PERF_TYPE_HARDWARE;
            a.config = PERF_COUNT_HW_CPU_CYCLES;
            break;
        case Event::Instructions:
            a.type = PERF_TYPE_HARDWARE;
            a.config = PERF_COUNT_HW_INSTRUCTIONS;
            break;
        case Event::CacheMisses:
            a.type = PERF_TYPE_HARDWARE;
            a.config = PERF_COUNT_HW_CACHE_MISSES;
            break;
        case Event::BranchMisses:
            a.type = PERF_TYPE_HARDWARE;
            a.config = PERF_COUNT_HW_BRANCH_MISSES;
            break;
        case Event::L1DMisses:
            a.type = PERF_TYPE_HW_CACHE;
            a.config = PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                       (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            break;
    }
    a.disabled = leader ? 1 : 0;  // only the leader starts off, the rest follow it
    a.exclude_kernel = 1;         // user space only, works with paranoid <= 2
    a.exclude_hv = 1;
    a.read_format =
        PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return a;
}

int perf_open(perf_event_attr* a, int group_fd) {
    // pid 0 means this thread, cpu -1 means whatever cpu it runs on
    return static_cast<int>(syscall(SYS_perf_event_open, a, 0, -1, group_fd, 0));
}

std::string explain(int err) {
    switch (err) {
        case EACCES:
        case EPERM:
            return "permission denied, run: sudo sysctl kernel.perf_event_paranoid=1";
        case ENOENT:
        case EOPNOTSUPP:
        case ENODEV:
            return "no hardware counters here (are you in a vm or wsl?)";
        case ENOSYS:
            return "perf_event_open isnt supported by this kernel";
        default:
            return std::strerror(err);
    }
}

bool is_unsupported(int err) {
    return err == ENOENT || err == EOPNOTSUPP || err == EINVAL || err == ENODEV;
}

}  // namespace

PerfGroup::PerfGroup(std::vector<Event> events) {
    if (events.empty()) throw std::invalid_argument("PerfGroup needs at least one event");
    for (std::size_t i = 0; i < events.size(); ++i) {
        bool leader = fds_.empty();
        auto attr = make_attr(events[i], leader);
        int fd = perf_open(&attr, leader ? -1 : fds_.front());
        if (fd < 0) {
            int err = errno;
            // if the leader fails nothing works, but a follower failing just means
            // this cpu doesnt have that event so we skip it
            if (leader || !is_unsupported(err)) {
                for (int f : fds_) close(f);
                throw std::runtime_error("perf_event_open(" + event_name(events[i]) +
                                         ") failed: " + explain(err));
            }
            skipped_.push_back(events[i]);
            continue;
        }
        fds_.push_back(fd);
        events_.push_back(events[i]);
    }
}

PerfGroup::~PerfGroup() {
    for (int fd : fds_) close(fd);
}

void PerfGroup::start() {
    ioctl(fds_.front(), PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(fds_.front(), PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

Counts PerfGroup::stop() {
    ioctl(fds_.front(), PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

    // layout is: nr, time_enabled, time_running, then one value per event
    std::vector<std::uint64_t> buf(3 + events_.size());
    auto want = static_cast<ssize_t>(buf.size() * sizeof(std::uint64_t));
    if (read(fds_.front(), buf.data(), static_cast<std::size_t>(want)) != want)
        throw std::runtime_error("failed to read perf counters");

    std::uint64_t enabled = buf[1], running = buf[2];
    Counts c;
    double scale = 1.0;
    if (running > 0 && running < enabled) {
        // the kernel only had our counters on for part of the time, so scale up
        scale = static_cast<double>(enabled) / static_cast<double>(running);
        c.multiplexed = true;
    }
    for (std::size_t i = 0; i < events_.size(); ++i)
        c.values[events_[i]] = static_cast<std::uint64_t>(static_cast<double>(buf[3 + i]) * scale);
    return c;
}

bool PerfGroup::available() {
    auto attr = make_attr(Event::Cycles, true);
    int fd = perf_open(&attr, -1);
    if (fd < 0) return false;
    close(fd);
    return true;
}

std::vector<Event> PerfGroup::default_events() {
    return {Event::Cycles, Event::Instructions, Event::L1DMisses, Event::CacheMisses,
            Event::BranchMisses};
}

}  // namespace cpuprobe
