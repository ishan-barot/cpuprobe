#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cpuprobe/cpu_info.hpp"
#include "cpuprobe/kernels.hpp"
#include "cpuprobe/latency.hpp"
#include "cpuprobe/util.hpp"

using namespace cpuprobe;

static void usage() {
    std::cerr
        << "usage:\n"
           "  cpuprobe info\n"
           "  cpuprobe latency [--min 4K] [--max 512M] [--reps 7] [--cpu 2] [--quick] [--csv "
           "FILE]\n"
           "  cpuprobe kernel <rowcol|branch|layout|falseshare|all> [--size 4096] [--n 10M]\n"
           "                  [--threads 2] [--iters 100M] [--cpus 2,4] [--cpu 2] [--reps 5]\n"
           "                  [--counters] [--quick] [--csv FILE]\n";
}

static int cmd_info() {
    std::cout << "CPU: " << cpu_model() << "\n\n";
    std::printf("%-6s %-12s %10s %6s  %s\n", "Level", "Type", "Size", "Line", "Shared CPUs");
    for (auto& c : cache_info())
        std::printf("L%-5d %-12s %10s %6d  %s\n", c.level, c.type.c_str(),
                    format_size(c.size).c_str(), c.line_size, c.shared_cpus.c_str());
    return 0;
}

static int cmd_latency(int argc, char** argv) {
    std::size_t lo = parse_size("4K"), hi = parse_size("512M");
    int reps = 7, cpu = 2;
    bool quick = false;
    std::string csv;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--min")
            lo = parse_size(next());
        else if (a == "--max")
            hi = parse_size(next());
        else if (a == "--reps")
            reps = std::stoi(next());
        else if (a == "--cpu")
            cpu = std::stoi(next());
        else if (a == "--csv")
            csv = next();
        else if (a == "--quick")
            quick = true;
        else {
            usage();
            return 2;
        }
    }
    if (quick) {
        hi = std::min(hi, parse_size("8M"));
        reps = 3;
    }

    if (!pin_to_cpu(cpu)) std::cerr << "warning: could not pin to cpu " << cpu << "\n";

    // powers of 2 plus the 1.5x point in between (4k, 6k, 8k, 12k...) so the curve looks smoother
    std::vector<std::size_t> sizes;
    for (std::size_t s = lo; s <= hi; s *= 2) {
        sizes.push_back(s);
        if (!quick && s + s / 2 <= hi) sizes.push_back(s + s / 2);
    }

    std::ofstream out;
    if (!csv.empty()) {
        out.open(csv);
        out << "bytes,ns_per_access\n";
    }

    std::printf("%10s %14s\n", "size", "ns/access");
    for (auto s : sizes) {
        auto r = measure_latency(s, reps);
        std::printf("%10s %14.2f\n", format_size(r.bytes).c_str(), r.ns_per_access);
        std::fflush(stdout);
        if (out) out << r.bytes << "," << r.ns_per_access << "\n";
    }
    return 0;
}

static std::vector<int> parse_cpus(const std::string& s) {
    std::vector<int> out;
    std::size_t start = 0;
    while (start <= s.size()) {
        auto comma = s.find(',', start);
        auto part = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!part.empty()) out.push_back(std::stoi(part));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (out.empty()) throw std::invalid_argument("--cpus needs something like 2,4");
    return out;
}

static void print_results(const std::vector<KernelResult>& rs, bool counters) {
    if (counters)
        std::printf("%-11s %-18s %10s %8s %14s %14s %14s %14s\n", "kernel", "variant", "ms", "ipc",
                    "instructions", "l1d_misses", "llc_misses", "branch_misses");
    else
        std::printf("%-11s %-18s %10s\n", "kernel", "variant", "ms");
    for (auto& r : rs) {
        if (counters && r.counts) {
            auto& c = *r.counts;
            std::printf("%-11s %-18s %10.2f %8.2f %14llu %14llu %14llu %14llu%s\n",
                        r.kernel.c_str(), r.variant.c_str(), r.ms, c.ipc(),
                        static_cast<unsigned long long>(c.get(Event::Instructions)),
                        static_cast<unsigned long long>(c.get(Event::L1DMisses)),
                        static_cast<unsigned long long>(c.get(Event::CacheMisses)),
                        static_cast<unsigned long long>(c.get(Event::BranchMisses)),
                        c.multiplexed ? "  (scaled)" : "");
        } else {
            std::printf("%-11s %-18s %10.2f\n", r.kernel.c_str(), r.variant.c_str(), r.ms);
        }
    }
    // the speedup line is the number people actually care about
    if (rs.size() >= 2 && rs[0].ms > 0 && rs[1].ms > 0) {
        double a = rs[0].ms, b = rs[1].ms;
        auto& slow = a > b ? rs[0] : rs[1];
        auto& fast = a > b ? rs[1] : rs[0];
        std::printf("  -> %s is %.1fx slower than %s\n", slow.variant.c_str(), slow.ms / fast.ms,
                    fast.variant.c_str());
    }
    std::printf("\n");
}

static int cmd_kernel(int argc, char** argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    std::string which = argv[2];
    KernelOptions o;
    int cpu = 2;
    std::string csv;
    bool quick = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--size")
            o.size = parse_size(next());
        else if (a == "--n")
            o.n = parse_size(next());
        else if (a == "--threads")
            o.threads = std::stoi(next());
        else if (a == "--iters")
            o.iters = parse_size(next());
        else if (a == "--cpus")
            o.cpus = parse_cpus(next());
        else if (a == "--cpu")
            cpu = std::stoi(next());
        else if (a == "--reps")
            o.reps = std::stoi(next());
        else if (a == "--counters")
            o.counters = true;
        else if (a == "--quick")
            quick = true;
        else if (a == "--csv")
            csv = next();
        else {
            usage();
            return 2;
        }
    }
    if (quick) {
        // tiny sizes so ci finishes in a couple seconds
        o.size = 512;
        o.n = 1'000'000;
        o.iters = 1'000'000;
        o.reps = 3;
    }

    if (o.counters && !PerfGroup::available()) {
        std::cerr << "no hardware counters available here (vm or wsl?), running without them\n";
        o.counters = false;
    }

    std::vector<std::string> todo;
    if (which == "all")
        todo = {"rowcol", "branch", "layout", "falseshare"};
    else
        todo = {which};

    std::ofstream out;
    if (!csv.empty()) {
        out.open(csv);
        out << "kernel,variant,ms,cycles,instructions,ipc,l1d_misses,llc_misses,branch_misses,"
               "multiplexed\n";
    }

    for (auto& k : todo) {
        // single threaded kernels get pinned, falseshare pins its own threads
        if (k != "falseshare" && !pin_to_cpu(cpu))
            std::cerr << "warning: could not pin to cpu " << cpu << "\n";

        std::vector<KernelResult> rs;
        if (k == "rowcol")
            rs = bench_rowcol(o);
        else if (k == "branch")
            rs = bench_branch(o);
        else if (k == "layout")
            rs = bench_layout(o);
        else if (k == "falseshare")
            rs = bench_falseshare(o);
        else {
            std::cerr << "unknown kernel: " << k << "\n";
            return 2;
        }

        // for branch the speedup line compares the first two, branchy random vs sorted
        print_results(rs, o.counters);

        if (out) {
            for (auto& r : rs) {
                Counts c = r.counts.value_or(Counts{});
                out << r.kernel << "," << r.variant << "," << r.ms << "," << c.get(Event::Cycles)
                    << "," << c.get(Event::Instructions) << "," << c.ipc() << ","
                    << c.get(Event::L1DMisses) << "," << c.get(Event::CacheMisses) << ","
                    << c.get(Event::BranchMisses) << "," << (c.multiplexed ? 1 : 0) << "\n";
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    std::string cmd = argv[1];
    try {
        if (cmd == "info") return cmd_info();
        if (cmd == "latency") return cmd_latency(argc, argv);
        if (cmd == "kernel") return cmd_kernel(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    usage();
    return 2;
}
