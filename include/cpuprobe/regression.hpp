#pragma once
#include <cstddef>
#include <istream>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace cpuprobe {

// one line of a benchmark results file
struct BenchRow {
    std::string benchmark;  // e.g. transpose, sgemm
    std::string variant;    // e.g. naive, tiled_padded, cublas
    std::size_t size = 0;
    int reps = 0;
    double median_ms = 0, p05_ms = 0, p95_ms = 0;
    double gbps = 0;    // effective bandwidth, 0 if it doesnt apply
    double gflops = 0;  // 0 if it doesnt apply
};

// a whole results file: "# key=value" lines up top (gpu, driver, commit...) then a csv table
struct BenchFile {
    std::map<std::string, std::string> meta;
    std::vector<BenchRow> rows;
};

void write_bench_csv(std::ostream& out, const BenchFile& f);
BenchFile read_bench_csv(std::istream& in);  // throws runtime_error with the line number

enum class Verdict { Ok, Improved, Noisy, Regression };
const char* to_string(Verdict v);

struct Comparison {
    BenchRow base, cur;
    double change_pct = 0;  // positive means slower
    Verdict verdict = Verdict::Ok;
};

struct CompareReport {
    std::vector<Comparison> rows;
    std::vector<std::string> warnings;
    bool has_regression() const;
};

// a row only counts as a regression if its median got slower by more than threshold_pct
// and the new runs dont overlap the old ones (new p05 is above old p95). if it got slower
// but the spreads overlap it gets marked noisy instead, so random jitter doesnt fail ci
CompareReport compare(const BenchFile& base, const BenchFile& cur, double threshold_pct);

}  // namespace cpuprobe
