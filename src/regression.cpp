#include "cpuprobe/regression.hpp"

#include <sstream>
#include <stdexcept>

namespace cpuprobe {

static const char* kHeader = "benchmark,variant,size,reps,median_ms,p05_ms,p95_ms,gbps,gflops";

void write_bench_csv(std::ostream& out, const BenchFile& f) {
    for (auto& [k, v] : f.meta) out << "# " << k << "=" << v << "\n";
    out << kHeader << "\n";
    for (auto& r : f.rows) {
        out << r.benchmark << "," << r.variant << "," << r.size << "," << r.reps << ","
            << r.median_ms << "," << r.p05_ms << "," << r.p95_ms << "," << r.gbps << "," << r.gflops
            << "\n";
    }
}

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream ss(s);
    while (std::getline(ss, cur, sep)) out.push_back(cur);
    if (!s.empty() && s.back() == sep) out.push_back("");
    return out;
}

BenchFile read_bench_csv(std::istream& in) {
    BenchFile f;
    std::string line;
    int lineno = 0;
    bool saw_header = false;
    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();  // files touched on windows
        if (line.empty()) continue;
        if (line[0] == '#') {
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            auto key = line.substr(1, eq - 1);
            key.erase(0, key.find_first_not_of(' '));
            f.meta[key] = line.substr(eq + 1);
            continue;
        }
        if (!saw_header) {
            if (line != kHeader)
                throw std::runtime_error("line " + std::to_string(lineno) +
                                         ": not a cpuprobe bench file (bad header)");
            saw_header = true;
            continue;
        }
        auto c = split(line, ',');
        if (c.size() != 9)
            throw std::runtime_error("line " + std::to_string(lineno) + ": expected 9 columns");
        try {
            BenchRow r;
            r.benchmark = c[0];
            r.variant = c[1];
            r.size = std::stoull(c[2]);
            r.reps = std::stoi(c[3]);
            r.median_ms = std::stod(c[4]);
            r.p05_ms = std::stod(c[5]);
            r.p95_ms = std::stod(c[6]);
            r.gbps = std::stod(c[7]);
            r.gflops = std::stod(c[8]);
            f.rows.push_back(r);
        } catch (const std::logic_error&) {
            throw std::runtime_error("line " + std::to_string(lineno) + ": bad number");
        }
    }
    if (!saw_header) throw std::runtime_error("empty or headerless bench file");
    return f;
}

const char* to_string(Verdict v) {
    switch (v) {
        case Verdict::Ok:
            return "ok";
        case Verdict::Improved:
            return "improved";
        case Verdict::Noisy:
            return "noisy";
        case Verdict::Regression:
            return "REGRESSION";
    }
    return "?";
}

bool CompareReport::has_regression() const {
    for (auto& r : rows)
        if (r.verdict == Verdict::Regression) return true;
    return false;
}

static std::string key_of(const BenchRow& r) {
    return r.benchmark + "/" + r.variant + "/" + std::to_string(r.size);
}

CompareReport compare(const BenchFile& base, const BenchFile& cur, double threshold_pct) {
    if (threshold_pct < 0) throw std::invalid_argument("threshold cant be negative");
    CompareReport rep;

    // comparing runs from two different gpus or drivers is apples to oranges, so say so
    for (const char* k : {"gpu", "driver", "cuda_runtime"}) {
        auto a = base.meta.find(k), b = cur.meta.find(k);
        if (a != base.meta.end() && b != cur.meta.end() && a->second != b->second)
            rep.warnings.push_back(std::string(k) + " differs: baseline '" + a->second +
                                   "' vs current '" + b->second + "'");
    }

    std::map<std::string, const BenchRow*> base_rows;
    for (auto& r : base.rows) base_rows[key_of(r)] = &r;

    for (auto& r : cur.rows) {
        auto it = base_rows.find(key_of(r));
        if (it == base_rows.end()) {
            rep.warnings.push_back("no baseline for " + key_of(r));
            continue;
        }
        const BenchRow& b = *it->second;
        Comparison c;
        c.base = b;
        c.cur = r;
        c.change_pct = b.median_ms > 0 ? (r.median_ms / b.median_ms - 1.0) * 100.0 : 0.0;
        if (c.change_pct > threshold_pct)
            c.verdict = r.p05_ms > b.p95_ms ? Verdict::Regression : Verdict::Noisy;
        else if (c.change_pct < -threshold_pct && r.p95_ms < b.p05_ms)
            c.verdict = Verdict::Improved;
        rep.rows.push_back(c);
        base_rows.erase(it);
    }
    for (auto& [k, _] : base_rows) rep.warnings.push_back("missing from current run: " + k);
    return rep;
}

}  // namespace cpuprobe
