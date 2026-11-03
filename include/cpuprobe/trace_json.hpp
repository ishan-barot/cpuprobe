#pragma once
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace cpuprobe {

// one thing that happened on the timeline, either a cuda api call on the cpu or work on the gpu
struct TraceEvent {
    std::string name;
    std::string category;  // cuda_api, kernel, memcpy, memset
    std::uint64_t start_ns = 0, end_ns = 0;
    bool on_gpu = false;
    std::uint32_t tid = 0;          // cpu thread id, or the stream id for gpu work
    std::uint32_t correlation = 0;  // links a gpu event back to the api call that launched it
    std::vector<std::pair<std::string, std::string>> args;
};

std::string json_escape(const std::string& s);

// turns a demangled kernel name like
// "void cpuprobe::gpu::(anonymous namespace)::transpose_tiled<1>(float const*, float*, int, int)"
// into "transpose_tiled<1>". keeps template args, drops the return type, namespaces and arguments
std::string short_kernel_name(std::string s);

// writes chrome trace event format json, which opens in ui.perfetto.dev or chrome://tracing.
// cpu calls and gpu work go on separate tracks and an arrow joins each launch to its kernel
void write_chrome_trace(std::ostream& out, const std::vector<TraceEvent>& events);

// how many api->gpu links write_chrome_trace would draw
std::size_t count_correlated(const std::vector<TraceEvent>& events);

struct TraceSummaryRow {
    std::string category, name;
    int count = 0;
    double total_ms = 0;
};

// totals per (category, name), biggest first. answers "where did the time go"
std::vector<TraceSummaryRow> summarize_trace(const std::vector<TraceEvent>& events);

}  // namespace cpuprobe
