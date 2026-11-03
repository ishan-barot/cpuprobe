#include "cpuprobe/trace_json.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>

namespace cpuprobe {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\r':
                out += "\\r";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string short_kernel_name(std::string s) {
    // drop the argument list: walk back from the last ')' to the '(' that matches it.
    // a plain find('(') is wrong because "(anonymous namespace)" comes first
    if (!s.empty() && s.back() == ')') {
        int depth = 0;
        for (std::size_t i = s.size(); i-- > 0;) {
            if (s[i] == ')') {
                ++depth;
            } else if (s[i] == '(' && --depth == 0) {
                s.erase(i);
                break;
            }
        }
    }
    // cut after the last "::" or space that isnt inside template brackets, which removes
    // namespaces and a leading "void "
    int angle = 0;
    std::size_t cut = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '<') {
            ++angle;
        } else if (s[i] == '>') {
            --angle;
        } else if (angle == 0 && s[i] == ' ') {
            cut = i + 1;
        } else if (angle == 0 && s[i] == ':' && i + 1 < s.size() && s[i + 1] == ':') {
            cut = i + 2;
        }
    }
    s = s.substr(cut);
    return s.empty() ? "unknown kernel" : s;
}

namespace {
constexpr int kCpuPid = 1;
constexpr int kGpuPid = 2;

// chrome trace wants microseconds, keep sub-microsecond detail as decimals
std::string us(std::uint64_t ns, std::uint64_t t0) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", static_cast<double>(ns - t0) / 1000.0);
    return buf;
}

// api calls that can create gpu work, used to pick which correlation ids get an arrow
std::unordered_map<std::uint32_t, const TraceEvent*> api_by_correlation(
    const std::vector<TraceEvent>& events) {
    std::unordered_map<std::uint32_t, const TraceEvent*> m;
    for (auto& e : events)
        if (!e.on_gpu && e.correlation != 0) m[e.correlation] = &e;
    return m;
}
}  // namespace

std::size_t count_correlated(const std::vector<TraceEvent>& events) {
    auto api = api_by_correlation(events);
    std::size_t n = 0;
    for (auto& e : events)
        if (e.on_gpu && e.correlation != 0 && api.count(e.correlation)) ++n;
    return n;
}

void write_chrome_trace(std::ostream& out, const std::vector<TraceEvent>& events) {
    std::uint64_t t0 = std::numeric_limits<std::uint64_t>::max();
    for (auto& e : events) t0 = std::min(t0, e.start_ns);
    if (events.empty()) t0 = 0;

    out << "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n";
    bool first = true;
    auto sep = [&]() {
        if (!first) out << ",\n";
        first = false;
    };

    // name the two processes and every track so the viewer shows something readable
    sep();
    out << R"js({"ph":"M","name":"process_name","pid":1,"args":{"name":"CPU (CUDA API calls)"}})js";
    sep();
    out << R"js({"ph":"M","name":"process_name","pid":2,"args":{"name":"GPU"}})js";
    std::set<std::pair<int, std::uint32_t>> tracks;
    for (auto& e : events) tracks.insert({e.on_gpu ? kGpuPid : kCpuPid, e.tid});
    for (auto& [pid, tid] : tracks) {
        sep();
        out << "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":" << pid << ",\"tid\":" << tid
            << ",\"args\":{\"name\":\""
            << (pid == kGpuPid ? "stream " : "thread ") + std::to_string(tid) << "\"}}";
    }

    for (auto& e : events) {
        sep();
        std::uint64_t dur = e.end_ns > e.start_ns ? e.end_ns - e.start_ns : 0;
        out << "{\"ph\":\"X\",\"name\":\"" << json_escape(e.name) << "\",\"cat\":\""
            << json_escape(e.category) << "\",\"pid\":" << (e.on_gpu ? kGpuPid : kCpuPid)
            << ",\"tid\":" << e.tid << ",\"ts\":" << us(e.start_ns, t0)
            << ",\"dur\":" << us(t0 + dur, t0) << ",\"args\":{\"correlation\":" << e.correlation;
        for (auto& [k, v] : e.args)
            out << ",\"" << json_escape(k) << "\":\"" << json_escape(v) << "\"";
        out << "}}";
    }

    // flow arrows: start inside the api call, end on the gpu work it caused
    auto api = api_by_correlation(events);
    for (auto& e : events) {
        if (!e.on_gpu || e.correlation == 0) continue;
        auto it = api.find(e.correlation);
        if (it == api.end()) continue;
        const TraceEvent& a = *it->second;
        sep();
        out << "{\"ph\":\"s\",\"name\":\"launch\",\"cat\":\"flow\",\"id\":" << e.correlation
            << ",\"pid\":" << kCpuPid << ",\"tid\":" << a.tid << ",\"ts\":" << us(a.start_ns, t0)
            << "}";
        sep();
        out << "{\"ph\":\"f\",\"bp\":\"e\",\"name\":\"launch\",\"cat\":\"flow\",\"id\":"
            << e.correlation << ",\"pid\":" << kGpuPid << ",\"tid\":" << e.tid
            << ",\"ts\":" << us(e.start_ns, t0) << "}";
    }
    out << "\n]}\n";
}

std::vector<TraceSummaryRow> summarize_trace(const std::vector<TraceEvent>& events) {
    std::map<std::pair<std::string, std::string>, TraceSummaryRow> m;
    for (auto& e : events) {
        auto& r = m[{e.category, e.name}];
        r.category = e.category;
        r.name = e.name;
        r.count++;
        if (e.end_ns > e.start_ns) r.total_ms += static_cast<double>(e.end_ns - e.start_ns) / 1e6;
    }
    std::vector<TraceSummaryRow> out;
    for (auto& [_, r] : m) out.push_back(r);
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.total_ms > b.total_ms; });
    return out;
}

}  // namespace cpuprobe
