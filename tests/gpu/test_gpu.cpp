// these only run on a machine with an nvidia gpu, everywhere else they skip themselves
#include <gtest/gtest.h>

#include "cpuprobe/gpu/gpu.hpp"
#include "cpuprobe/gpu/tracer.hpp"

using namespace cpuprobe;
using namespace cpuprobe::gpu;

#define NEED_GPU() \
    if (device_count() == 0) GTEST_SKIP() << "no cuda gpu here"

TEST(Gpu, DeviceInfoLooksSane) {
    NEED_GPU();
    auto d = device_info(0);
    EXPECT_FALSE(d.name.empty());
    EXPECT_GT(d.sm_count, 0);
    EXPECT_GT(d.l2_bytes, 0u);
}

// odd sizes on purpose: not multiples of 32, so the edge tiles and bounds checks get exercised
TEST(Gpu, TransposeMatchesCpuOnOddShapes) {
    NEED_GPU();
    for (auto [r, c] :
         {std::pair<std::size_t, std::size_t>{64, 64}, {1000, 37}, {33, 1025}, {1, 1}})
        for (auto v : {"copy", "naive", "tiled", "tiled_padded"})
            EXPECT_EQ(transpose_max_error(r, c, v), 0.0) << v << " " << r << "x" << c;
}

TEST(Gpu, SgemmMatchesCpu) {
    NEED_GPU();
    // general shapes for the kernels that handle edges
    for (auto v : {"naive", "smem_tiled", "cublas"})
        EXPECT_LT(sgemm_max_error(70, 50, 33, v), 1e-3) << v;
    // the register blocked kernel only takes multiples of its tile
    EXPECT_LT(sgemm_max_error(256, 128, 64, "blocktiled_2d"), 1e-3);
    EXPECT_THROW(sgemm_max_error(100, 128, 64, "blocktiled_2d"), std::invalid_argument);
}

TEST(Gpu, LatencyIsPositiveAndGrows) {
    NEED_GPU();
    auto small = measure_gpu_latency(16 << 10, 3);
    auto big = measure_gpu_latency(256u << 20, 3);
    EXPECT_GT(small.ns_per_access, 0);
    EXPECT_GT(big.ns_per_access, small.ns_per_access);  // dram has to be slower than l1
}

TEST(Gpu, TracerCapturesKernelsCopiesAndLinks) {
    NEED_GPU();
    CuptiTracer t;
    t.start();
    run_trace_workload(256, 1);
    t.stop();
    auto ev = t.take_events();

    int kernels = 0, copies = 0, api = 0;
    for (auto& e : ev) {
        if (e.category == "kernel") ++kernels;
        if (e.category == "memcpy") ++copies;
        if (e.category == "cuda_api") ++api;
    }
    EXPECT_EQ(kernels, 4);  // copy, naive, tiled, tiled_padded once each
    EXPECT_GE(copies, 3);   // pageable up, pinned up, pinned down
    EXPECT_GT(api, 0);
    EXPECT_GE(count_correlated(ev), 4u);  // every kernel traced back to its launch call
    EXPECT_EQ(t.dropped_records(), 0u);
}
