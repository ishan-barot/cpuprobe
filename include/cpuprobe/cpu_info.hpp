#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace cpuprobe {

struct CacheInfo {
    int level;
    std::string type;   // data, instruction or unified
    std::size_t size;   // bytes
    int line_size;
    std::string shared_cpus;
};

std::string cpu_model();                 // reads it out of /proc/cpuinfo
std::vector<CacheInfo> cache_info(int cpu = 0);  // reads /sys/devices/system/cpu/cpuN/cache

}  // namespace cpuprobe
