// matrix transpose, the classic "same math, very different memory traffic" gpu experiment.
// a warp is 32 threads and global memory is happiest when those 32 threads touch one
// contiguous chunk (coalesced). transpose reads rows and writes columns, so one side is
// always strided unless you bounce it through shared memory first.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

#include "cpuprobe/gpu/gpu.hpp"
#include "cpuprobe/stats.hpp"
#include "cuda_check.cuh"

namespace cpuprobe::gpu {

namespace {

constexpr int kTile = 32;  // one warp wide
constexpr int kRows = 8;   // each block is 32x8 threads and every thread handles 4 elements

// same access pattern as transpose minus the transpose, so this is the speed limit
__global__ void copy_kernel(const float* in, float* out, int rows, int cols) {
    int x = blockIdx.x * kTile + threadIdx.x;
    int y = blockIdx.y * kTile + threadIdx.y;
    for (int j = 0; j < kTile; j += kRows)
        if (x < cols && y + j < rows)
            out[static_cast<size_t>(y + j) * cols + x] = in[static_cast<size_t>(y + j) * cols + x];
}

// reads are coalesced, writes jump a whole row apart, so every warp store hits 32 lines
__global__ void transpose_naive(const float* in, float* out, int rows, int cols) {
    int x = blockIdx.x * kTile + threadIdx.x;
    int y = blockIdx.y * kTile + threadIdx.y;
    for (int j = 0; j < kTile; j += kRows)
        if (x < cols && y + j < rows)
            out[static_cast<size_t>(x) * rows + (y + j)] =
                in[static_cast<size_t>(y + j) * cols + x];
}

// load a 32x32 tile with coalesced reads, flip it inside shared memory, write it out coalesced.
// with pad 0 a column read of the tile hits the same shared memory bank 32 times (32 floats
// per row, 32 banks), so the warp gets serialized. pad 1 shifts each row by one bank
template <int Pad>
__global__ void transpose_tiled(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[kTile][kTile + Pad];
    int x = blockIdx.x * kTile + threadIdx.x;
    int y = blockIdx.y * kTile + threadIdx.y;
    for (int j = 0; j < kTile; j += kRows)
        if (x < cols && y + j < rows)
            tile[threadIdx.y + j][threadIdx.x] = in[static_cast<size_t>(y + j) * cols + x];
    __syncthreads();

    // swap the block coordinates, the output is cols x rows
    x = blockIdx.y * kTile + threadIdx.x;
    y = blockIdx.x * kTile + threadIdx.y;
    for (int j = 0; j < kTile; j += kRows)
        if (x < rows && y + j < cols)
            out[static_cast<size_t>(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

void launch(const std::string& variant, const float* in, float* out, int rows, int cols,
            cudaStream_t s = 0) {
    dim3 block(kTile, kRows);
    dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    if (variant == "copy")
        copy_kernel<<<grid, block, 0, s>>>(in, out, rows, cols);
    else if (variant == "naive")
        transpose_naive<<<grid, block, 0, s>>>(in, out, rows, cols);
    else if (variant == "tiled")
        transpose_tiled<0><<<grid, block, 0, s>>>(in, out, rows, cols);
    else if (variant == "tiled_padded")
        transpose_tiled<1><<<grid, block, 0, s>>>(in, out, rows, cols);
    else
        throw std::invalid_argument("unknown transpose variant: " + variant);
}

std::vector<float> random_matrix(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

}  // namespace

double transpose_max_error(std::size_t rows, std::size_t cols, const std::string& variant) {
    auto h = random_matrix(rows * cols, 7);
    detail::DeviceBuffer<float> in(rows * cols), out(rows * cols);
    in.upload(h);
    CUDA_CHECK(cudaMemset(out.get(), 0, rows * cols * sizeof(float)));
    launch(variant, in.get(), out.get(), static_cast<int>(rows), static_cast<int>(cols));
    CUDA_CHECK(cudaGetLastError());
    auto g = out.download();

    double worst = 0;
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) {
            // copy keeps the layout, everything else should be the transpose
            float want = h[r * cols + c];
            float got = variant == "copy" ? g[r * cols + c] : g[c * rows + r];
            worst = std::max(worst, static_cast<double>(std::fabs(want - got)));
        }
    return worst;
}

std::vector<BenchRow> bench_transpose(const BenchOptions& o) {
    const std::size_t n = o.size, elems = n * n;
    auto h = random_matrix(elems, 1);
    detail::DeviceBuffer<float> in(elems), out(elems);
    in.upload(h);

    std::vector<BenchRow> rows;
    for (std::string v : {"copy", "naive", "tiled", "tiled_padded"}) {
        // the injected regression runs the bank conflicted kernel under the good name
        std::string real = (o.inject_regression && v == "tiled_padded") ? "tiled" : v;
        if (real != "copy" && transpose_max_error(67, 45, real) != 0.0)
            throw std::runtime_error("transpose " + real + " gave a wrong answer");

        auto ms = detail::time_launches(
            [&] { launch(real, in.get(), out.get(), static_cast<int>(n), static_cast<int>(n)); },
            o.warmup, o.reps);
        auto s = summarize(ms);
        BenchRow r;
        r.benchmark = "transpose";
        r.variant = v;
        r.size = n;
        r.reps = o.reps;
        r.median_ms = s.median;
        r.p05_ms = s.p05;
        r.p95_ms = s.p95;
        // every element is read once and written once
        r.gbps = 2.0 * elems * sizeof(float) / (s.median * 1e-3) / 1e9;
        rows.push_back(r);
    }
    return rows;
}

void run_trace_workload(std::size_t n, int iters) {
    const std::size_t elems = n * n, bytes = elems * sizeof(float);
    auto pageable = random_matrix(elems, 3);
    float* pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned), bytes));
    std::copy(pageable.begin(), pageable.end(), pinned);

    try {
        detail::DeviceBuffer<float> in(elems), out(elems);
        // same bytes twice: pageable has to be staged through a hidden pinned buffer by the
        // driver, pinned can be dma'd straight across pcie. the trace shows the difference
        CUDA_CHECK(cudaMemcpy(in.get(), pageable.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(in.get(), pinned, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(out.get(), 0, bytes));
        for (int i = 0; i < iters; ++i)
            for (std::string v : {"copy", "naive", "tiled", "tiled_padded"})
                launch(v, in.get(), out.get(), static_cast<int>(n), static_cast<int>(n));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(pinned, out.get(), bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        cudaFreeHost(pinned);
        throw;
    }
    cudaFreeHost(pinned);
}

}  // namespace cpuprobe::gpu
