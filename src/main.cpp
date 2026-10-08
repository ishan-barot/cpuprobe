#include "cpuprobe/cpu_info.hpp"
#include "cpuprobe/latency.hpp"
#include "cpuprobe/util.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace cpuprobe;

static void usage() {
    std::cerr << "usage:\n"
                 "  cpuprobe info\n"
                 "  cpuprobe latency [--min 4K] [--max 512M] [--reps 7] [--cpu 2] [--quick] [--csv FILE]\n";
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
        if (a == "--min") lo = parse_size(next());
        else if (a == "--max") hi = parse_size(next());
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--cpu") cpu = std::stoi(next());
        else if (a == "--csv") csv = next();
        else if (a == "--quick") quick = true;
        else { usage(); return 2; }
    }
    if (quick) { hi = std::min(hi, parse_size("8M")); reps = 3; }

    if (!pin_to_cpu(cpu)) std::cerr << "warning: could not pin to cpu " << cpu << "\n";

    // powers of 2 plus the 1.5x point in between (4K, 6K, 8K, 12K...) so the curve looks smoother
    std::vector<std::size_t> sizes;
    for (std::size_t s = lo; s <= hi; s *= 2) {
        sizes.push_back(s);
        if (!quick && s + s / 2 <= hi) sizes.push_back(s + s / 2);
    }

    std::ofstream out;
    if (!csv.empty()) { out.open(csv); out << "bytes,ns_per_access\n"; }

    std::printf("%10s %14s\n", "size", "ns/access");
    for (auto s : sizes) {
        auto r = measure_latency(s, reps);
        std::printf("%10s %14.2f\n", format_size(r.bytes).c_str(), r.ns_per_access);
        std::fflush(stdout);
        if (out) out << r.bytes << "," << r.ns_per_access << "\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    std::string cmd = argv[1];
    try {
        if (cmd == "info") return cmd_info();
        if (cmd == "latency") return cmd_latency(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    usage();
    return 2;
}
