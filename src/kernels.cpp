#include "cpuprobe/kernels.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <random>
#include <stdexcept>
#include <thread>

#include "cpuprobe/util.hpp"

namespace cpuprobe {

// ---------- the kernels themselves ----------

std::uint64_t sum_rows(const std::vector<int>& m, std::size_t n) {
    std::uint64_t s = 0;
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c) s += static_cast<std::uint64_t>(m[r * n + c]);
    return s;
}

std::uint64_t sum_cols(const std::vector<int>& m, std::size_t n) {
    std::uint64_t s = 0;
    // same math, but each step jumps a whole row ahead (16kb for a 4096 wide matrix)
    for (std::size_t c = 0; c < n; ++c)
        for (std::size_t r = 0; r < n; ++r) s += static_cast<std::uint64_t>(m[r * n + c]);
    return s;
}

std::uint64_t sum_branchy(const std::vector<int>& data) {
    std::uint64_t s = 0;
    for (int x : data) {
        if (x >= 128) {
            s += static_cast<std::uint64_t>(x);
            // empty asm stops gcc and clang from turning this into a cmov or simd,
            // without it the optimizer quietly makes this branchless and the comparison is
            // pointless
            asm volatile("");
        }
    }
    return s;
}

std::uint64_t sum_branchless(const std::vector<int>& data) {
    std::uint64_t s = 0;
    for (int x : data) s += static_cast<std::uint64_t>((x >= 128) * x);
    return s;
}

double sum_x_aos(const std::vector<Particle>& p) {
    double s = 0;
    for (const auto& q : p) s += q.x;  // drags in all 32 bytes of each particle to use 4
    return s;
}

double sum_x_soa(const std::vector<float>& x) {
    double s = 0;
    for (float v : x) s += v;  // every byte loaded is a byte we actually use
    return s;
}

namespace {

// packed: all counters share one cache line
struct Packed {
    std::atomic<std::uint64_t> c[8];
};
static_assert(sizeof(Packed) <= 64);

// padded: each counter gets a whole line to itself
struct alignas(64) PaddedSlot {
    std::atomic<std::uint64_t> c;
};

template <class Slot>
void bump(Slot& slot, std::uint64_t iters) {
    // plain load + store instead of fetch_add so theres no lock prefix,
    // that way the slowdown is purely the cache line bouncing between cores
    for (std::uint64_t i = 0; i < iters; ++i)
        slot.store(slot.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

}  // namespace

std::uint64_t run_falseshare(bool padded, int threads, std::uint64_t iters,
                             const std::vector<int>& cpus, Counts* counts_out) {
    if (threads < 1 || threads > 8) throw std::invalid_argument("threads must be 1 to 8");

    Packed packed{};
    std::vector<PaddedSlot> pads(static_cast<std::size_t>(threads));
    for (auto& p : pads) p.c.store(0);
    for (auto& c : packed.c) c.store(0);

    std::atomic<bool> go{false};
    std::vector<Counts> per_thread(static_cast<std::size_t>(threads));
    std::vector<std::thread> pool;

    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            if (!cpus.empty()) pin_to_cpu(cpus[static_cast<std::size_t>(t) % cpus.size()]);
            // perf counters only see the thread that opened them, so each worker
            // opens its own group and we add them up after
            std::optional<PerfGroup> pg;
            if (counts_out) pg.emplace(PerfGroup::default_events());
            while (!go.load(std::memory_order_acquire)) {}
            if (pg) pg->start();
            if (padded)
                bump(pads[static_cast<std::size_t>(t)].c, iters);
            else
                bump(packed.c[t], iters);
            if (pg) per_thread[static_cast<std::size_t>(t)] = pg->stop();
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& th : pool) th.join();

    if (counts_out) {
        *counts_out = Counts{};
        for (auto& c : per_thread) *counts_out += c;
    }
    std::uint64_t total = 0;
    for (int t = 0; t < threads; ++t)
        total += padded ? pads[static_cast<std::size_t>(t)].c.load() : packed.c[t].load();
    return total;
}

// ---------- shared timing harness ----------

namespace {

using clk = std::chrono::steady_clock;

// runs fn `reps` times and keeps the run with the median time,
// counters come from that same run so time and counts line up
KernelResult measure(const std::string& kernel, const std::string& variant, const KernelOptions& o,
                     const std::function<std::uint64_t()>& fn) {
    std::vector<KernelResult> runs;
    std::optional<PerfGroup> pg;
    if (o.counters) pg.emplace(PerfGroup::default_events());

    fn();  // warm up so the first rep isnt paying for page faults
    for (int r = 0; r < o.reps; ++r) {
        KernelResult kr{kernel, variant, 0, std::nullopt, 0};
        if (pg) pg->start();
        auto t0 = clk::now();
        kr.checksum = fn();
        auto t1 = clk::now();
        if (pg) kr.counts = pg->stop();
        do_not_optimize(kr.checksum);
        kr.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        runs.push_back(kr);
    }
    std::sort(runs.begin(), runs.end(),
              [](const KernelResult& a, const KernelResult& b) { return a.ms < b.ms; });
    return runs[runs.size() / 2];
}

std::vector<int> random_bytes(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> d(0, 255);
    std::vector<int> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

}  // namespace

std::vector<KernelResult> bench_rowcol(const KernelOptions& o) {
    std::size_t n = o.size;
    std::vector<int> m(n * n);
    for (std::size_t i = 0; i < m.size(); ++i) m[i] = static_cast<int>(i & 0xff);
    return {
        measure("rowcol", "row_major", o, [&] { return sum_rows(m, n); }),
        measure("rowcol", "col_major", o, [&] { return sum_cols(m, n); }),
    };
}

std::vector<KernelResult> bench_branch(const KernelOptions& o) {
    auto random = random_bytes(o.n, 1);
    auto sorted = random;
    std::sort(sorted.begin(), sorted.end());
    return {
        measure("branch", "branchy_random", o, [&] { return sum_branchy(random); }),
        measure("branch", "branchy_sorted", o, [&] { return sum_branchy(sorted); }),
        measure("branch", "branchless_random", o, [&] { return sum_branchless(random); }),
        measure("branch", "branchless_sorted", o, [&] { return sum_branchless(sorted); }),
    };
}

std::vector<KernelResult> bench_layout(const KernelOptions& o) {
    std::vector<Particle> aos(o.n);
    std::vector<float> xs(o.n);
    for (std::size_t i = 0; i < o.n; ++i) {
        float v = static_cast<float>(i % 1000) * 0.001f;
        aos[i] = Particle{v, 1, 2, 3, 4, 5, 6, 7};
        xs[i] = v;
    }
    // doubles summed in a float loop can differ in the last bits, so the checksum is rounded
    auto as_u64 = [](double d) { return static_cast<std::uint64_t>(d); };
    return {
        measure("layout", "aos", o, [&] { return as_u64(sum_x_aos(aos)); }),
        measure("layout", "soa", o, [&] { return as_u64(sum_x_soa(xs)); }),
    };
}

std::vector<KernelResult> bench_falseshare(const KernelOptions& o) {
    std::vector<KernelResult> out;
    for (bool padded : {false, true}) {
        std::vector<KernelResult> runs;
        for (int r = 0; r < o.reps; ++r) {
            KernelResult kr{"falseshare", padded ? "padded" : "packed", 0, std::nullopt, 0};
            Counts c;
            auto t0 = clk::now();
            kr.checksum =
                run_falseshare(padded, o.threads, o.iters, o.cpus, o.counters ? &c : nullptr);
            auto t1 = clk::now();
            if (o.counters) kr.counts = c;
            kr.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            runs.push_back(kr);
        }
        std::sort(runs.begin(), runs.end(),
                  [](const KernelResult& a, const KernelResult& b) { return a.ms < b.ms; });
        out.push_back(runs[runs.size() / 2]);
    }
    return out;
}

}  // namespace cpuprobe
