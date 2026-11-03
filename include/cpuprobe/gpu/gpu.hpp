#pragma once
// nvidia gpu backend. this header has no cuda types in it on purpose, so plain g++ code
// (the cli, the tests) can call into it without dragging the cuda headers along
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "cpuprobe/regression.hpp"

namespace cpuprobe::gpu {

struct DeviceInfo {
    std::string name;
    int cc_major = 0, cc_minor = 0;  // compute capability, 8.9 on ada (rtx 40 series)
    int sm_count = 0;
    std::size_t global_mem = 0;
    std::size_t l2_bytes = 0;
    std::size_t smem_per_sm = 0;
    int driver_version = 0;   // e.g. 13040 means 13.4
    int runtime_version = 0;  // the cuda toolkit we built against
};

// 0 when there is no driver or no gpu, never throws
int device_count();
DeviceInfo device_info(int dev = 0);

// gpu name, driver, runtime, git commit. goes at the top of every results file
std::map<std::string, std::string> metadata();

struct BenchOptions {
    std::size_t size = 4096;  // matrix is size x size floats
    int reps = 30;
    int warmup = 3;
    // swaps the padded transpose for the unpadded one while keeping the padded name,
    // the same thing a bad commit would do. only there to prove `compare` catches it
    bool inject_regression = false;
};

// copy (the bandwidth ceiling), naive, tiled and tiled_padded transpose
std::vector<BenchRow> bench_transpose(const BenchOptions& o);

// naive, smem_tiled, blocktiled_2d (own kernels) and cublas as the reference.
// size must be a multiple of 128 for blocktiled_2d
std::vector<BenchRow> bench_sgemm(const BenchOptions& o);

// largest abs difference vs a plain cpu loop. used by the tests
double transpose_max_error(std::size_t rows, std::size_t cols, const std::string& variant);
double sgemm_max_error(int m, int n, int k, const std::string& variant);

// single thread pointer chase in gpu global memory, same idea as the cpu latency benchmark
struct LatencyPoint {
    std::size_t bytes = 0;
    double ns_per_access = 0;
    double cycles_per_access = 0;  // sm clock cycles from clock64()
};
LatencyPoint measure_gpu_latency(std::size_t bytes, int reps, std::uint64_t seed = 42);

// a small realistic pipeline for the tracer to look at: pageable and pinned uploads, a memset,
// every transpose kernel `iters` times, then a download. blocks until the gpu is done
void run_trace_workload(std::size_t n, int iters);

// entry point for `cpuprobe gpu ...`
int gpu_main(int argc, char** argv);

}  // namespace cpuprobe::gpu
