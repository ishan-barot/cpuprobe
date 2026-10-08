# cpuprobe
[![ci](https://github.com/ishan-barot/cpuprobe/actions/workflows/ci.yml/badge.svg)](https://github.com/ishan-barot/cpuprobe/actions/workflows/ci.yml)

A C++ tool that measures how your CPU's memory hierarchy actually behaves (cache sizes, latencies, miss rates, IPC) using microbenchmarks and Linux hardware performance counters, with Python automation to sweep configs and plot results.

![latency staircase](results/wsl/latency.png)

## Key findings
Measured on an AMD Ryzen 9 5900X (Zen 3), Ubuntu 24.04 under WSL2, DDR4 at 2400 MT/s, gcc 13 `-O2`.

**Memory latency** (pointer chase, one core, median of 7 runs)

| Level | Size | Latency |
|---|---|---|
| L1d | 32K | 0.75 ns |
| L2 | 512K | 2.3 ns |
| L3 | 32M per CCD | 8 to 11 ns (1M to 4M) |
| DRAM | | ~105 ns |

The steps land right where `/sys` says each cache ends. Latency creeps up between 8M and 16M before the real L3 edge, which lines up with Zen 3's L2 TLB running out (2048 entries x 4K pages = 8M of coverage). Past that point every load also pays for a page walk.

**Good vs bad code, with hardware counters**

| Kernel | Slowdown | What the counters show |
|---|---|---|
| Column-major vs row-major matrix sum (64MB) | **22.7x** | L1d misses go from 1.05M to 16.9M, IPC drops from 2.99 to 0.28 |
| Branchy loop on random vs sorted data | **6.9x** | Branch misses go from 47 to 5,000,005, IPC drops from 3.73 to 0.51 |
| Array of structs vs struct of arrays | **1.8x** | L1d misses go from 625K to 5.0M |
| False sharing: two cores writing to one cache line vs separate lines | **11.7x** | IPC drops from 2.90 to 0.22, 32M lines bounced between cores |

![kernel timings](results/wsl/kernels_time.png)
![kernel counters](results/wsl/kernels_counters.png)

**The counters matched first-principles predictions.** Before running, each miss count can be worked out on paper:
- Row-major reads 64MB in 64-byte lines, so 64MB / 64B = 1,048,576 L1d misses. Measured: 1,048,973.
- Column-major jumps 16KB per step, so every one of the 16.7M loads should miss. Measured: 16.9M.
- 10M random bytes against a `>= 128` check means the predictor is right half the time, so ~5M misses. Measured: 5,000,005.
- AoS drags 320MB through the cache (5M lines) to use 40MB; SoA reads just the 40MB (625K lines). Measured: 5.0M vs 625K.

## How it works
- **Pointer chasing.** Each load depends on the one before it (`p = p->next`), so the CPU can't overlap them, and the random order defeats the hardware prefetcher. What's left is raw load-to-use latency.
- **Sattolo's algorithm.** A shuffle that always produces one single cycle, so the chase visits every node instead of getting stuck in a small loop that stays in cache.
- **One node per 64-byte cache line**, so every hop touches a new line.
- **`perf_event_open`.** An RAII `PerfGroup` class opens a group of hardware counters (cycles, instructions, L1d misses, cache misses, branch misses) with the first one as group leader, so they all start and stop at the same instant. If the kernel has to multiplex counters, the values get scaled by `time_enabled / time_running` and flagged.
- **Keeping the comparisons honest.** At `-O2`, gcc and clang quietly turn the branchy loop into a branchless `cmov`, which made the first version of that benchmark show no difference at all. An empty `asm volatile("")` inside the branch keeps it a real branch. Every good/bad pair also has a unit test proving both versions compute the same result.
- **False sharing.** Each thread pins itself to its own physical core and opens its own counter group, since perf counters only see the thread that opened them. The results get summed afterwards.

## Methodology
- Release build (`-O2`), never benchmarked with sanitizers on
- Pinned to one core with `sched_setaffinity` (false sharing uses cores 2 and 4, which are separate physical cores)
- One warm-up pass before timing
- Latency: each measurement runs ~100 ms and the reported value is the median of 7 runs
- Kernels: median of 5 runs, with counters taken from that same median run
- `scripts/run_sweep.py` records CPU model, kernel version, compiler, governor and git commit next to every result set (see `results/wsl/meta.json`)

## Build and run
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cpuprobe_tests

./build/cpuprobe info                      # cache sizes from /sys
./build/cpuprobe latency --min 4K --max 512M --csv out.csv
./build/cpuprobe kernel all --counters     # or rowcol, branch, layout, falseshare

# full sweep + charts
python3 scripts/run_sweep.py --tag mybox --counters
python3 scripts/plot.py results/mybox/latency.csv -o results/mybox/latency.png
python3 scripts/plot_kernels.py results/mybox/kernels.csv -o results/mybox
```
If counters fail with a permission error, run `sudo sysctl kernel.perf_event_paranoid=1` (or `scripts/setup.sh`).

## Testing and CI
- 19 GoogleTest tests: Sattolo forms exactly one cycle, size parsing, stats, alignment, good/bad kernel pairs agree, counter sanity checks
- Counter tests skip themselves with `GTEST_SKIP` on machines without a PMU (like GitHub's runners)
- CI on every push: gcc and clang, Debug build with AddressSanitizer + UndefinedBehaviorSanitizer, all tests, a smoke run of every command, and a `clang-format` check

## Limitations
- These numbers come from WSL2, which is a Hyper-V VM. Counters work there on this machine, but page walks are nested (guest + host page tables), so TLB-heavy results are likely a bit worse than bare metal.
- RAM was running at 2400 MT/s with DOCP off; the kit is rated for 3600, so the DRAM numbers are pessimistic.
- Generic perf events map to different raw events on AMD and Intel. On Zen 3 the generic `cache-misses` event doesn't cleanly mean "L3 misses", so treat the `llc_misses` column as approximate. The L1d and branch counts line up with predictions almost exactly.
- No frequency control under WSL (no `cpupower`), so boost clocks can add some noise.

## What I learned
The biggest takeaway was that the compiler is part of the experiment. My first branch benchmark showed sorted and random data running at the same speed, which made no sense until I looked at the assembly and saw `-O2` had already removed the branch. I also learned to predict a counter value before measuring it. When 5,000,005 branch misses showed up after I'd expected 5M, I knew the tool was measuring what I thought it was.

## Roadmap
- [x] Pointer-chase latency benchmark, cache info, plots
- [x] Hardware counters via `perf_event_open`
- [x] Comparison kernels: row vs column, branchy vs branchless, AoS vs SoA, false sharing
- [x] CI on gcc and clang with sanitizers
- [ ] Native Linux run with DOCP on, compared against WSL2
- [ ] Core-to-core latency heatmap (same CCD vs across CCDs)
- [ ] Huge pages to show the TLB effect shrinking
