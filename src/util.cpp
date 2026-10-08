#include "cpuprobe/util.hpp"

#include <algorithm>
#include <cctype>
#include <sched.h>
#include <stdexcept>

namespace cpuprobe {

std::size_t parse_size(const std::string& s) {
    if (s.empty()) throw std::invalid_argument("empty size");
    std::size_t i = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
    if (i == 0) throw std::invalid_argument("size must start with a number: " + s);
    std::size_t num = std::stoull(s.substr(0, i));
    std::string suf = s.substr(i);
    std::transform(suf.begin(), suf.end(), suf.begin(), ::toupper);
    if (suf == "" || suf == "B") return num;
    if (suf == "K" || suf == "KB" || suf == "KIB") return num << 10;
    if (suf == "M" || suf == "MB" || suf == "MIB") return num << 20;
    if (suf == "G" || suf == "GB" || suf == "GIB") return num << 30;
    throw std::invalid_argument("bad size suffix: " + s);
}

std::string format_size(std::size_t b) {
    if (b >= (1ULL << 30) && b % (1ULL << 30) == 0) return std::to_string(b >> 30) + "G";
    if (b >= (1ULL << 20) && b % (1ULL << 20) == 0) return std::to_string(b >> 20) + "M";
    if (b >= (1ULL << 10) && b % (1ULL << 10) == 0) return std::to_string(b >> 10) + "K";
    return std::to_string(b) + "B";
}

double median(std::vector<double> v) {
    if (v.empty()) throw std::invalid_argument("median of empty vector");
    std::sort(v.begin(), v.end());
    std::size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

bool pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

}  // namespace cpuprobe
