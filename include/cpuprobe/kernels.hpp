#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "cpuprobe/perf_counters.hpp"

namespace cpuprobe {

// one measured run of one variant of a kernel
struct KernelResult {
    std::string kernel;
    std::string variant;
    double ms = 0;
    std::optional<Counts> counts;
    std::uint64_t checksum = 0;  // printed so the compiler cant throw the work away
};

struct KernelOptions {
    std::size_t size = 4096;            // matrix side for rowcol
    std::size_t n = 10'000'000;         // element count for branch and layout
    int threads = 2;                    // falseshare thread count
    std::uint64_t iters = 100'000'000;  // falseshare increments per thread
    std::vector<int> cpus{2, 4};        // 2 and 4 are always different physical cores
    int reps = 5;
    bool counters = false;
};

// these are the raw kernels, exposed so tests can check good and bad give the same answer
std::uint64_t sum_rows(const std::vector<int>& m, std::size_t n);
std::uint64_t sum_cols(const std::vector<int>& m, std::size_t n);

std::uint64_t sum_branchy(const std::vector<int>& data);
std::uint64_t sum_branchless(const std::vector<int>& data);

struct Particle {
    float x, y, z, vx, vy, vz, mass, charge;
};
double sum_x_aos(const std::vector<Particle>& p);
double sum_x_soa(const std::vector<float>& x);

// runs `threads` threads that each bump their own counter `iters` times.
// padded puts each counter on its own cache line, packed squeezes them into one.
// returns the total of all counters (should be threads * iters).
std::uint64_t run_falseshare(bool padded, int threads, std::uint64_t iters,
                             const std::vector<int>& cpus, Counts* counts_out = nullptr);

// the full benchmark drivers, each returns one result per variant
std::vector<KernelResult> bench_rowcol(const KernelOptions& o);
std::vector<KernelResult> bench_branch(const KernelOptions& o);
std::vector<KernelResult> bench_layout(const KernelOptions& o);
std::vector<KernelResult> bench_falseshare(const KernelOptions& o);

}  // namespace cpuprobe
