# cpuprobe

A C++ tool that measures how your CPU's memory hierarchy actually behaves (cache sizes, latencies, miss rates, IPC) using microbenchmarks and Linux hardware performance counters, with Python automation to sweep configs and plot results.

![latency](results/latency.png)

## Key findings (AMD Ryzen 9 5900X, Zen 3)
- L1d: __ ns · L2: __ ns · L3: __ ns · DRAM: __ ns
- _(fill in after running)_

## How it works
- **Pointer chasing:** each load depends on the previous (`p = p->next`), so the CPU can't overlap loads. Random order defeats the prefetcher.
- **Sattolo's algorithm:** random permutation guaranteed to be one big cycle, so the chase visits every cache line.
- One `Node` per 64-byte cache line.

## Methodology
- Release build (`-O2`), pinned to one core, 1 warm-up lap, ~100 ms per measurement, median of 7 reps.

## Build & run
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cpuprobe_tests
./build/cpuprobe info
./build/cpuprobe latency --min 4K --max 512M --csv results/latency.csv
python3 scripts/plot.py results/latency.csv -o results/latency.png
```

## Limitations
- WSL2/VMs don't expose the PMU, so hardware counters need native Linux.
