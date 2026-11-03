// gpu version of the cpu pointer chase. one thread follows a random ring through global
// memory so every load waits on the one before it, which exposes raw load latency at each
// level: l1 (per sm), l2 (shared by the whole chip), then gddr dram
#include <algorithm>
#include <random>

#include "cpuprobe/gpu/gpu.hpp"
#include "cpuprobe/latency.hpp"
#include "cpuprobe/util.hpp"
#include "cuda_check.cuh"

namespace cpuprobe::gpu {

namespace {

// 128 bytes is an l1 line on nvidia, so one node per line like the cpu version does per 64
constexpr std::size_t kNodeBytes = 128;
constexpr std::size_t kStride = kNodeBytes / sizeof(unsigned);

__global__ void chase(const unsigned* next, unsigned long long iters, unsigned* sink,
                      long long* cycles) {
    unsigned p = 0;
    long long t0 = clock64();
    // __ldca asks for the load to be cached in l1 as well as l2
    for (unsigned long long i = 0; i < iters; ++i) p = __ldca(&next[p]);
    long long t1 = clock64();
    *sink = p;  // keeps the compiler from throwing the loop away
    *cycles = t1 - t0;
}

}  // namespace

LatencyPoint measure_gpu_latency(std::size_t bytes, int reps, std::uint64_t seed) {
    std::size_t nodes = std::max<std::size_t>(bytes / kNodeBytes, 2);
    std::vector<unsigned> host(nodes * kStride, 0);
    std::mt19937_64 rng(seed);
    auto perm = sattolo(nodes, rng);  // same one big cycle shuffle the cpu benchmark uses
    for (std::size_t i = 0; i < nodes; ++i)
        host[i * kStride] = static_cast<unsigned>(perm[i] * kStride);

    detail::DeviceBuffer<unsigned> next(host.size());
    next.upload(host);
    detail::DeviceBuffer<unsigned> sink(1);
    detail::DeviceBuffer<long long> cyc(1);

    // ask for as much l1 as possible instead of shared memory, we dont use any shared memory
    CUDA_CHECK(cudaFuncSetAttribute(chase, cudaFuncAttributePreferredSharedMemoryCarveout, 0));

    // windows kills gpu work that runs for more than ~2s (tdr), and wsl goes through the
    // windows driver, so keep every launch well under a second even at dram latency
    const unsigned long long warm = std::min<std::size_t>(nodes, 1u << 20);
    const unsigned long long iters = 1u << 19;

    chase<<<1, 1>>>(next.get(), warm, sink.get(), cyc.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> ns, cycles;
    auto ms = detail::time_launches(
        [&] { chase<<<1, 1>>>(next.get(), iters, sink.get(), cyc.get()); }, 0, reps);
    for (double m : ms) ns.push_back(m * 1e6 / static_cast<double>(iters));
    // the cycle counter only holds the last rep, good enough next to the event median
    auto c = cyc.download();
    cycles.push_back(static_cast<double>(c[0]) / static_cast<double>(iters));

    LatencyPoint lp;
    lp.bytes = bytes;
    lp.ns_per_access = median(ns);
    lp.cycles_per_access = median(cycles);
    return lp;
}

}  // namespace cpuprobe::gpu
