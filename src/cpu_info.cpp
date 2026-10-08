#include "cpuprobe/cpu_info.hpp"

#include <filesystem>
#include <fstream>

#include "cpuprobe/util.hpp"

namespace cpuprobe {

static std::string read_line(const std::filesystem::path& p) {
    std::ifstream f(p);
    std::string s;
    std::getline(f, s);
    return s;
}

std::string cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) return line.substr(pos + 2);
        }
    }
    return "unknown";
}

std::vector<CacheInfo> cache_info(int cpu) {
    namespace fs = std::filesystem;
    std::vector<CacheInfo> out;
    fs::path base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache";
    for (int i = 0;; ++i) {
        fs::path d = base / ("index" + std::to_string(i));
        if (!fs::exists(d)) break;
        CacheInfo c;
        c.level = std::stoi(read_line(d / "level"));
        c.type = read_line(d / "type");
        c.size = parse_size(read_line(d / "size"));
        c.line_size = std::stoi(read_line(d / "coherency_line_size"));
        c.shared_cpus = read_line(d / "shared_cpu_list");
        out.push_back(c);
    }
    return out;
}

}  // namespace cpuprobe
