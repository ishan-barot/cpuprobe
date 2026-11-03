#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

// every cuda call goes through this so a failure says what and where instead of failing later
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess)                                                                   \
            throw std::runtime_error(std::string(#call) + " failed: " + cudaGetErrorString(err_) + \
                                     " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")");      \
    } while (0)

namespace cpuprobe::gpu::detail {

// small raii owner for device memory so an exception mid benchmark doesnt leak vram
template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) : n_(n) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(p_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return p_; }
    std::size_t size() const { return n_; }
    void upload(const std::vector<T>& h) {
        CUDA_CHECK(cudaMemcpy(p_, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> h(n_);
        CUDA_CHECK(cudaMemcpy(h.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }

private:
    T* p_ = nullptr;
    std::size_t n_;
};

// times one launch per rep with cuda events on the default stream, after a few untimed
// warmups. returns milliseconds per rep. the launch lambda must only enqueue gpu work
template <class F>
std::vector<double> time_launches(F&& launch, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(a));
        launch();
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        CUDA_CHECK(cudaGetLastError());
        float t = 0;
        CUDA_CHECK(cudaEventElapsedTime(&t, a, b));
        ms.push_back(t);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms;
}

}  // namespace cpuprobe::gpu::detail
