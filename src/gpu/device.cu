#include "cpuprobe/gpu/gpu.hpp"
#include "cuda_check.cuh"

#ifndef CPUPROBE_GIT_COMMIT
#define CPUPROBE_GIT_COMMIT "unknown"
#endif

namespace cpuprobe::gpu {

int device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();  // clear it so later calls dont trip over it
        return 0;
    }
    return n;
}

DeviceInfo device_info(int dev) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    DeviceInfo d;
    d.name = p.name;
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    d.sm_count = p.multiProcessorCount;
    d.global_mem = p.totalGlobalMem;
    d.l2_bytes = static_cast<std::size_t>(p.l2CacheSize);
    d.smem_per_sm = p.sharedMemPerMultiprocessor;
    CUDA_CHECK(cudaDriverGetVersion(&d.driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&d.runtime_version));
    return d;
}

static std::string version_str(int v) {
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

std::map<std::string, std::string> metadata() {
    auto d = device_info(0);
    return {
        {"gpu", d.name},
        {"compute_capability", std::to_string(d.cc_major) + "." + std::to_string(d.cc_minor)},
        {"driver", version_str(d.driver_version)},
        {"cuda_runtime", version_str(d.runtime_version)},
        {"commit", CPUPROBE_GIT_COMMIT},
    };
}

}  // namespace cpuprobe::gpu
