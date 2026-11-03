# GPU backend: how it works

Optional NVIDIA backend, built with `-DCPUPROBE_ENABLE_CUDA=ON`. The CPU-only build is unchanged and is still what the gcc/clang CI jobs build.

## Commands
```bash
cpuprobe gpu info                                   # name, SMs, L2 size, driver/runtime
cpuprobe gpu bench transpose --csv results/gpu/bench.csv
cpuprobe gpu bench all --csv out.csv --trace out.json   # benchmarks + CUPTI timeline
cpuprobe gpu latency --csv results/gpu/latency.csv  # pointer chase, 4K to 256M
cpuprobe gpu trace --out results/gpu/trace.json     # traced pipeline + tracing overhead
cpuprobe compare --baseline old.csv --current new.csv   # exits 3 on a confirmed regression
```

## CUPTI tracer (`src/gpu/cupti_tracer.cpp`)
- Uses the CUPTI Activity API and records four kinds of activity: runtime API calls, concurrent kernels, memcpys and memsets.
- CUPTI asks for empty buffers through a callback and gives them back full on its own thread. The completion callback stays short: it parses the records, frees the buffer, and adds the events to a vector under a mutex.
- `cuptiActivityFlushAll(FORCED)` on stop, so partly filled buffers also get delivered.
- Dropped records are counted with `cuptiActivityGetNumDroppedRecords` and printed. A trace that silently loses events is worse than having no trace.
- **Correlation.** Every runtime call and the GPU work it launches share a correlation ID. The exporter draws a flow arrow from each `cudaLaunchKernel`/`cudaMemcpy` on the CPU track to its kernel or copy on the GPU track.
- Output is Chrome trace-event JSON, so you can drag it into [ui.perfetto.dev](https://ui.perfetto.dev).
- Kernel names are demangled, and the struct version (`CUpti_ActivityKernel11` on CUDA 13.1+) is picked from `CUDA_VERSION`, because NVIDIA bumps it when it adds fields.
- `gpu trace` runs the same workload 7 times untraced and 7 times traced, then reports the median overhead.

## Benchmarks
**Transpose (4096x4096 fp32, 64 MB in and out).** A warp reads 32 consecutive floats, which is coalesced. A naive transpose then writes them 16 KB apart, so each warp store touches 32 different memory segments. The tiled version stages a 32x32 tile in shared memory so that both the read and the write are coalesced. With a 32x32 tile every column of the tile falls in the same shared-memory bank, so the warp's 32 accesses get serialized. Padding the tile to 32x33 shifts each row by one bank. `copy` has the same memory traffic with no transpose, so it serves as the ceiling. Each variant is first checked against a CPU transpose on an odd shape (67x45).

**SGEMM (fp32, row-major).** Each kernel fixes the main limit of the previous one.
1. `naive`: one thread per output element. Every input value gets loaded from global memory K times.
2. `smem_tiled`: 32x32 shared-memory tiles, so each global load is reused 32 times. Now shared-memory bandwidth is the limit, at 2 shared loads per FMA.
3. `blocktiled_2d`: each thread keeps an 8x8 patch of C in registers and does 64 FMAs per 16 shared loads.
4. `cublas`: the reference. cuBLAS is column-major, so row-major C = AB is computed as C^T = B^T A^T.
Every result is checked against cuBLAS (max abs diff < 1e-3 x max|C|). The tests also compare against a double-precision CPU loop.

**Latency ladder.** One thread chases a Sattolo ring (the same shuffle as the CPU benchmark) with one node per 128-byte line. `__ldca` keeps loads cached in L1, and the carveout is set to give L1 as much space as possible. Expect a step near the per-SM L1 size, another at the L2 size reported by `cudaDeviceProp`, and then DRAM.

## Methodology
- Timing uses CUDA events on the same stream around a single launch. There are 3 warmup launches, then 30 reps (10 for sgemm). The CSV records the median, p05 and p95.
- Allocation, uploads and correctness checks all happen outside the timed window.
- Every results file starts with the GPU, compute capability, driver, CUDA runtime and git commit.
- WSL2 runs on the Windows driver, which kills GPU work that runs longer than ~2 s (TDR). Every launch is kept well under that.

## Regression detection
`compare` matches rows by (benchmark, variant, size). A row is a **regression** only if its median slowed by more than `--threshold` (10% by default) **and** the new p05 is above the old p95, which means the two distributions don't overlap. A slower median with overlapping spreads is reported as `noisy` and doesn't fail. If the baseline came from a different GPU, driver or runtime, it prints a warning. `gpu bench --inject-regression` swaps the bank-conflicted kernel in under the padded kernel's name, the same thing a bad commit would do, so you can see `compare` catch it.

## Limitations
- GitHub's free runners have no GPU, so CI only checks that the CUDA build compiles and links against CUDA 13.0 and that the GPU tests skip cleanly. The GPU tests run locally.
- The activity trace gives timing, not hardware metrics. Claims about memory-boundedness or bank conflicts are backed up with Nsight Compute, not by the tracer.
