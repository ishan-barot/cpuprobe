#pragma once
#include <vector>

namespace cpuprobe {

// percentile with linear interpolation between neighbours, p goes 0 to 100. throws if v is empty
double percentile(std::vector<double> v, double p);

struct Summary {
    int n = 0;
    double median = 0, p05 = 0, p95 = 0, min = 0, max = 0;
};

// the numbers we keep for every timed benchmark
Summary summarize(const std::vector<double>& v);

}  // namespace cpuprobe
