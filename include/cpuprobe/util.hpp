#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace cpuprobe {

// turns "4K" into 4096, "512M" into 536870912 etc, throws invalid_argument on junk
std::size_t parse_size(const std::string& s);

// the other way around, 4096 becomes "4K"
std::string format_size(std::size_t bytes);

// median of v (takes a copy so the original isnt sorted), throws if its empty
double median(std::vector<double> v);

// locks this thread onto one cpu so it doesnt get moved around mid run
bool pin_to_cpu(int cpu);

// stops the compiler from deleting code whose result we never use (same trick google benchmark uses)
template <class T>
inline void do_not_optimize(T const& v) {
    asm volatile("" : : "r,m"(v) : "memory");
}

}  // namespace cpuprobe
