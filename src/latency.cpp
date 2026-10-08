#include "cpuprobe/latency.hpp"
#include "cpuprobe/util.hpp"

#include <chrono>
#include <cstdlib>
#include <memory>
#include <new>
#include <numeric>

namespace cpuprobe {

std::vector<std::size_t> sattolo(std::size_t n, std::mt19937_64& rng) {
    std::vector<std::size_t> p(n);
    std::iota(p.begin(), p.end(), 0);
    for (std::size_t i = n - 1; i > 0; --i) {
        std::uniform_int_distribution<std::size_t> d(0, i - 1);
        std::swap(p[i], p[d(rng)]);
    }
    return p;
}

namespace {
struct FreeDeleter { void operator()(void* p) const { std::free(p); } };
using clk = std::chrono::steady_clock;

double chase_ns(Node* start, std::size_t iters) {
    Node* cur = start;
    auto t0 = clk::now();
    for (std::size_t i = 0; i < iters; ++i) cur = cur->next;
    auto t1 = clk::now();
    do_not_optimize(cur);
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}
}  // namespace

LatencyResult measure_latency(std::size_t bytes, int reps, std::uint64_t seed) {
    std::size_t n = bytes / sizeof(Node);
    if (n < 2) n = 2;

    std::unique_ptr<Node, FreeDeleter> buf(
        static_cast<Node*>(std::aligned_alloc(64, n * sizeof(Node))));
    if (!buf) throw std::bad_alloc();
    Node* nodes = buf.get();

    std::mt19937_64 rng(seed);
    auto p = sattolo(n, rng);
    for (std::size_t i = 0; i < n; ++i) nodes[i].next = &nodes[p[i]];

    // do one full lap first so the first timed run isnt cold
    chase_ns(nodes, n);

    // figure out how many hops it takes for one run to last around 100ms
    std::size_t iters = 1 << 20;
    double t = chase_ns(nodes, iters);
    if (t > 0) iters = static_cast<std::size_t>(iters * (100e6 / t));
    if (iters < n) iters = n;

    std::vector<double> samples;
    for (int r = 0; r < reps; ++r) samples.push_back(chase_ns(nodes, iters) / iters);
    return {bytes, median(samples)};
}

}  // namespace cpuprobe
