#include "cpuprobe/stats.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cpuprobe {

double percentile(std::vector<double> v, double p) {
    if (v.empty()) throw std::invalid_argument("percentile of nothing");
    if (p < 0 || p > 100) throw std::invalid_argument("percentile must be 0 to 100");
    std::sort(v.begin(), v.end());
    double pos = (p / 100.0) * static_cast<double>(v.size() - 1);
    auto lo = static_cast<std::size_t>(std::floor(pos));
    auto hi = static_cast<std::size_t>(std::ceil(pos));
    double frac = pos - static_cast<double>(lo);
    return v[lo] + (v[hi] - v[lo]) * frac;
}

Summary summarize(const std::vector<double>& v) {
    if (v.empty()) throw std::invalid_argument("summarize of nothing");
    Summary s;
    s.n = static_cast<int>(v.size());
    s.median = percentile(v, 50);
    s.p05 = percentile(v, 5);
    s.p95 = percentile(v, 95);
    auto [mn, mx] = std::minmax_element(v.begin(), v.end());
    s.min = *mn;
    s.max = *mx;
    return s;
}

}  // namespace cpuprobe
