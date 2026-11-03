// `cpuprobe gpu ...` subcommands. plain c++, all the cuda lives behind gpu.hpp
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpuprobe/gpu/gpu.hpp"
#include "cpuprobe/gpu/tracer.hpp"
#include "cpuprobe/stats.hpp"
#include "cpuprobe/util.hpp"

namespace cpuprobe::gpu {

namespace {

void usage() {
    std::cerr << "usage:\n"
                 "  cpuprobe gpu info\n"
                 "  cpuprobe gpu bench <transpose|sgemm|all> [--size 4096] [--reps 30]\n"
                 "                     [--csv FILE] [--trace FILE] [--inject-regression]\n"
                 "  cpuprobe gpu latency [--min 4K] [--max 256M] [--reps 5] [--csv FILE]\n"
                 "  cpuprobe gpu trace [--size 2048] [--iters 3] [--out trace.json]\n";
}

struct Args {
    std::vector<std::string> pos;
    std::size_t size = 0, min = 4 << 10, max = 256u << 20;
    int reps = 0, iters = 3;
    std::string csv, trace, out = "trace.json";
    bool inject = false;
};

Args parse(int argc, char** argv, int from) {
    Args a;
    for (int i = from; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + s);
            return argv[++i];
        };
        if (s == "--size")
            a.size = parse_size(next());
        else if (s == "--reps")
            a.reps = std::stoi(next());
        else if (s == "--iters")
            a.iters = std::stoi(next());
        else if (s == "--min")
            a.min = parse_size(next());
        else if (s == "--max")
            a.max = parse_size(next());
        else if (s == "--csv")
            a.csv = next();
        else if (s == "--trace")
            a.trace = next();
        else if (s == "--out")
            a.out = next();
        else if (s == "--inject-regression")
            a.inject = true;
        else if (!s.empty() && s[0] == '-')
            throw std::invalid_argument("unknown flag " + s);
        else
            a.pos.push_back(s);
    }
    return a;
}

void require_gpu() {
    if (device_count() == 0)
        throw std::runtime_error(
            "no cuda gpu found. on wsl, check that nvidia-smi works inside ubuntu and that you "
            "did NOT install a linux nvidia driver (only the toolkit)");
}

int cmd_info() {
    require_gpu();
    auto d = device_info(0);
    auto meta = metadata();
    std::printf("GPU:                %s\n", d.name.c_str());
    std::printf("compute capability: %d.%d\n", d.cc_major, d.cc_minor);
    std::printf("SMs:                %d\n", d.sm_count);
    std::printf("memory:             %.1f GiB\n", static_cast<double>(d.global_mem) / (1u << 30));
    std::printf("L2 cache:           %s\n", format_size(d.l2_bytes).c_str());
    std::printf("shared mem per SM:  %s (shared with L1)\n", format_size(d.smem_per_sm).c_str());
    std::printf("driver / runtime:   %s / %s\n", meta["driver"].c_str(),
                meta["cuda_runtime"].c_str());
    return 0;
}

void write_trace(const std::string& path, const std::vector<TraceEvent>& ev,
                 std::uint64_t dropped) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cant write " + path);
    write_chrome_trace(f, ev);
    std::printf("wrote %zu events (%zu cpu->gpu links, %llu dropped) to %s\n", ev.size(),
                count_correlated(ev), static_cast<unsigned long long>(dropped), path.c_str());
    std::printf("open it at https://ui.perfetto.dev (drag the file in)\n");
}

void print_rows(const std::vector<BenchRow>& rows) {
    std::printf("%-10s %-14s %6s %10s %10s %10s %10s\n", "bench", "variant", "size", "median_ms",
                "p95_ms", "GB/s", "GFLOP/s");
    double cublas = 0, copy = 0;
    for (auto& r : rows) {
        if (r.variant == "cublas") cublas = r.gflops;
        if (r.variant == "copy") copy = r.gbps;
    }
    for (auto& r : rows) {
        std::printf("%-10s %-14s %6zu %10.3f %10.3f %10.1f %10.1f", r.benchmark.c_str(),
                    r.variant.c_str(), r.size, r.median_ms, r.p95_ms, r.gbps, r.gflops);
        // the number worth putting in a readme: how close each kernel gets to the ceiling
        if (cublas > 0 && r.gflops > 0)
            std::printf("   %5.1f%% of cublas", 100 * r.gflops / cublas);
        if (copy > 0 && r.gbps > 0 && r.variant != "copy")
            std::printf("   %5.1f%% of copy", 100 * r.gbps / copy);
        std::printf("\n");
    }
    std::printf("\n");
}

int cmd_bench(const Args& a) {
    require_gpu();
    if (a.pos.size() < 3) {
        usage();
        return 2;
    }
    std::string which = a.pos[2];

    BenchFile file;
    file.meta = metadata();
    if (a.inject) file.meta["note"] = "inject-regression";

    std::unique_ptr<CuptiTracer> tracer;
    if (!a.trace.empty()) {
        tracer = std::make_unique<CuptiTracer>();
        tracer->start();
    }

    if (which == "transpose" || which == "all") {
        BenchOptions o;
        if (a.size) o.size = a.size;
        o.reps = a.reps ? a.reps : 30;
        o.inject_regression = a.inject;
        auto rows = bench_transpose(o);
        print_rows(rows);
        file.rows.insert(file.rows.end(), rows.begin(), rows.end());
    }
    if (which == "sgemm" || which == "all") {
        BenchOptions o;
        if (a.size) o.size = a.size;
        o.reps = a.reps ? a.reps : 10;
        auto rows = bench_sgemm(o);
        print_rows(rows);
        file.rows.insert(file.rows.end(), rows.begin(), rows.end());
    }
    if (file.rows.empty()) {
        usage();
        return 2;
    }

    if (tracer) {
        tracer->stop();
        write_trace(a.trace, tracer->take_events(), tracer->dropped_records());
    }
    if (!a.csv.empty()) {
        std::ofstream f(a.csv);
        if (!f) throw std::runtime_error("cant write " + a.csv);
        write_bench_csv(f, file);
        std::printf("wrote %s\n", a.csv.c_str());
    }
    return 0;
}

int cmd_latency(const Args& a) {
    require_gpu();
    int reps = a.reps ? a.reps : 5;
    auto d = device_info(0);
    std::printf("%s, L2 = %s\n", d.name.c_str(), format_size(d.l2_bytes).c_str());

    std::ofstream out;
    if (!a.csv.empty()) {
        out.open(a.csv);
        out << "# gpu=" << d.name << "\n# l2_bytes=" << d.l2_bytes << "\n";
        out << "bytes,ns_per_access,cycles_per_access\n";
    }
    std::printf("%10s %12s %12s\n", "size", "ns/access", "cycles");
    for (std::size_t s = a.min; s <= a.max; s *= 2) {
        for (std::size_t sz : {s, s + s / 2}) {
            if (sz > a.max) continue;
            auto p = measure_gpu_latency(sz, reps);
            std::printf("%10s %12.1f %12.1f\n", format_size(sz).c_str(), p.ns_per_access,
                        p.cycles_per_access);
            std::fflush(stdout);
            if (out) out << p.bytes << "," << p.ns_per_access << "," << p.cycles_per_access << "\n";
        }
    }
    return 0;
}

// runs the demo workload without and then with the tracer, so we can report what tracing costs
int cmd_trace(const Args& a) {
    require_gpu();
    std::size_t n = a.size ? a.size : 2048;
    const int runs = 7;
    run_trace_workload(n, a.iters);  // warm up: context creation, module loading

    auto time_runs = [&] {
        std::vector<double> ms;
        for (int i = 0; i < runs; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            run_trace_workload(n, a.iters);
            auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return median(ms);
    };

    double plain = time_runs();
    CuptiTracer tracer;
    tracer.start();
    double traced = time_runs();
    tracer.stop();
    auto ev = tracer.take_events();

    std::printf("workload: %zux%zu floats, %d iterations of 4 kernels, median of %d runs\n", n, n,
                a.iters, runs);
    std::printf("  untraced %.2f ms, traced %.2f ms, overhead %+.1f%%\n\n", plain, traced,
                plain > 0 ? (traced / plain - 1) * 100 : 0.0);

    std::printf("%-10s %-36s %6s %10s\n", "category", "name", "count", "total_ms");
    for (auto& r : summarize_trace(ev))
        std::printf("%-10s %-36s %6d %10.3f\n", r.category.c_str(), r.name.c_str(), r.count,
                    r.total_ms);
    std::printf("\n");
    write_trace(a.out, ev, tracer.dropped_records());
    return 0;
}

}  // namespace

int gpu_main(int argc, char** argv) {
    // argv is: cpuprobe gpu <sub> ...
    if (argc < 3) {
        usage();
        return 2;
    }
    Args a = parse(argc, argv, 1);
    std::string sub = a.pos.size() > 1 ? a.pos[1] : "";
    if (sub == "info") return cmd_info();
    if (sub == "bench") return cmd_bench(a);
    if (sub == "latency") return cmd_latency(a);
    if (sub == "trace") return cmd_trace(a);
    usage();
    return 2;
}

}  // namespace cpuprobe::gpu
