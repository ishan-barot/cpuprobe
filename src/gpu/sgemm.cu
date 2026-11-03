// single precision matrix multiply, c = a * b, all row major. three of my own kernels, each
// fixing the main bottleneck of the one before, with cublas as the "how good can it get" bar
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "cpuprobe/gpu/gpu.hpp"
#include "cpuprobe/stats.hpp"
#include "cuda_check.cuh"

#define CUBLAS_CHECK(call)                                                                \
    do {                                                                                  \
        cublasStatus_t st_ = (call);                                                      \
        if (st_ != CUBLAS_STATUS_SUCCESS)                                                 \
            throw std::runtime_error(std::string(#call) + " failed with cublas status " + \
                                     std::to_string(static_cast<int>(st_)));              \
    } while (0)

namespace cpuprobe::gpu {

namespace {

// one thread per output element. neighbouring threads share a row of a (broadcast) and read
// neighbouring columns of b (coalesced), but every value gets pulled from global memory k times
__global__ void sgemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
    C[static_cast<size_t>(row) * N + col] = acc;
}

// classic shared memory tiling: the block stages a 32x32 tile of a and of b, then every value
// loaded from global memory gets reused 32 times out of shared memory
constexpr int kT = 32;
__global__ void sgemm_smem(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[kT][kT];
    __shared__ float Bs[kT][kT];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * kT + ty, col = blockIdx.x * kT + tx;
    float acc = 0.f;
    for (int t = 0; t < K; t += kT) {
        As[ty][tx] = (row < M && t + tx < K) ? A[static_cast<size_t>(row) * K + t + tx] : 0.f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[static_cast<size_t>(t + ty) * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < kT; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = acc;
}

// 2d register blocking. the smem kernel does 2 shared loads per fma, so shared memory becomes
// the wall. here each thread owns an 8x8 patch of c in registers and does 64 fmas for every
// 16 shared loads. block tile is 128x128 with a k step of 8, 256 threads per block
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int kThreads2d = (BM * BN) / (TM * TN);  // 256

__global__ void __launch_bounds__(kThreads2d)
    sgemm_2d(int /*rows, the grid already covers them*/, int N, int K, const float* A,
             const float* B, float* C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tid = threadIdx.x;
    const int thread_col = tid % (BN / TN);  // which 8 wide column strip of the tile
    const int thread_row = tid / (BN / TN);  // which 8 tall row strip

    // move the pointers to this block's corner
    A += static_cast<size_t>(blockIdx.y) * BM * K;
    B += static_cast<size_t>(blockIdx.x) * BN;
    C += static_cast<size_t>(blockIdx.y) * BM * N + static_cast<size_t>(blockIdx.x) * BN;

    // how each thread helps load the tiles: the a tile is 128x8, the b tile is 8x128
    const int a_row = tid / BK, a_col = tid % BK, a_stride = kThreads2d / BK;
    const int b_row = tid / BN, b_col = tid % BN, b_stride = kThreads2d / BN;

    float acc[TM * TN] = {0.f};
    float reg_m[TM];
    float reg_n[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int off = 0; off < BM; off += a_stride)
            As[(a_row + off) * BK + a_col] = A[static_cast<size_t>(a_row + off) * K + a_col];
        for (int off = 0; off < BK; off += b_stride)
            Bs[(b_row + off) * BN + b_col] = B[static_cast<size_t>(b_row + off) * N + b_col];
        __syncthreads();
        A += BK;
        B += static_cast<size_t>(BK) * N;

        for (int d = 0; d < BK; ++d) {
            // pull a column of a and a row of b into registers, then outer product them
            for (int i = 0; i < TM; ++i) reg_m[i] = As[(thread_row * TM + i) * BK + d];
            for (int j = 0; j < TN; ++j) reg_n[j] = Bs[d * BN + thread_col * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += reg_m[i] * reg_n[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[static_cast<size_t>(thread_row * TM + i) * N + thread_col * TN + j] = acc[i * TN + j];
}

class Cublas {
public:
    Cublas() { CUBLAS_CHECK(cublasCreate(&h_)); }
    ~Cublas() { cublasDestroy(h_); }
    Cublas(const Cublas&) = delete;
    Cublas& operator=(const Cublas&) = delete;

    // cublas is column major. a row major matrix is the transpose of a column major one, so
    // c = a*b row major is the same as c^t = b^t * a^t column major, so swap a/b and m/n
    void sgemm(int M, int N, int K, const float* A, const float* B, float* C) {
        const float one = 1.f, zero = 0.f;
        CUBLAS_CHECK(
            cublasSgemm(h_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N));
    }

private:
    cublasHandle_t h_{};
};

void launch(const std::string& v, int M, int N, int K, const float* A, const float* B, float* C,
            Cublas* cb) {
    if (v == "naive") {
        dim3 block(32, 8);
        dim3 grid((N + 31) / 32, (M + 7) / 8);
        sgemm_naive<<<grid, block>>>(M, N, K, A, B, C);
    } else if (v == "smem_tiled") {
        dim3 block(kT, kT);
        dim3 grid((N + kT - 1) / kT, (M + kT - 1) / kT);
        sgemm_smem<<<grid, block>>>(M, N, K, A, B, C);
    } else if (v == "blocktiled_2d") {
        if (M % BM || N % BN || K % BK)
            throw std::invalid_argument("blocktiled_2d needs M,N multiples of 128 and K of 8");
        dim3 grid(N / BN, M / BM);
        sgemm_2d<<<grid, kThreads2d>>>(M, N, K, A, B, C);
    } else if (v == "cublas") {
        cb->sgemm(M, N, K, A, B, C);
    } else {
        throw std::invalid_argument("unknown sgemm variant: " + v);
    }
}

std::vector<float> random_matrix(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

}  // namespace

double sgemm_max_error(int m, int n, int k, const std::string& variant) {
    auto a = random_matrix(static_cast<std::size_t>(m) * k, 11);
    auto b = random_matrix(static_cast<std::size_t>(k) * n, 12);
    detail::DeviceBuffer<float> da(a.size()), db(b.size()), dc(static_cast<std::size_t>(m) * n);
    da.upload(a);
    db.upload(b);
    Cublas cb;
    launch(variant, m, n, k, da.get(), db.get(), dc.get(), &cb);
    CUDA_CHECK(cudaGetLastError());
    auto c = dc.download();

    // double precision reference on the cpu
    double worst = 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double want = 0;
            for (int p = 0; p < k; ++p)
                want += static_cast<double>(a[static_cast<std::size_t>(i) * k + p]) *
                        b[static_cast<std::size_t>(p) * n + j];
            worst = std::max(worst, std::fabs(want - c[static_cast<std::size_t>(i) * n + j]));
        }
    return worst;
}

std::vector<BenchRow> bench_sgemm(const BenchOptions& o) {
    const int n = static_cast<int>(o.size);
    if (n % 128 != 0) throw std::invalid_argument("sgemm --size must be a multiple of 128");
    const std::size_t elems = static_cast<std::size_t>(n) * n;
    auto a = random_matrix(elems, 21), b = random_matrix(elems, 22);
    detail::DeviceBuffer<float> da(elems), db(elems), dc(elems), dref(elems);
    da.upload(a);
    db.upload(b);
    Cublas cb;

    // cublas answer is the reference the full size results get checked against
    cb.sgemm(n, n, n, da.get(), db.get(), dref.get());
    auto ref = dref.download();
    double ref_max = 0;
    for (float x : ref) ref_max = std::max(ref_max, static_cast<double>(std::fabs(x)));

    std::vector<BenchRow> rows;
    for (std::string v : {"naive", "smem_tiled", "blocktiled_2d", "cublas"}) {
        auto ms = detail::time_launches(
            [&] { launch(v, n, n, n, da.get(), db.get(), dc.get(), &cb); }, o.warmup, o.reps);

        auto c = dc.download();
        double worst = 0;
        for (std::size_t i = 0; i < elems; ++i)
            worst = std::max(worst, static_cast<double>(std::fabs(c[i] - ref[i])));
        if (worst > 1e-3 * ref_max)
            throw std::runtime_error("sgemm " + v + " disagrees with cublas (max diff " +
                                     std::to_string(worst) + ")");

        auto s = summarize(ms);
        BenchRow r;
        r.benchmark = "sgemm";
        r.variant = v;
        r.size = o.size;
        r.reps = o.reps;
        r.median_ms = s.median;
        r.p05_ms = s.p05;
        r.p95_ms = s.p95;
        r.gflops = 2.0 * n * static_cast<double>(n) * n / (s.median * 1e-3) / 1e9;
        rows.push_back(r);
    }
    return rows;
}

}  // namespace cpuprobe::gpu
