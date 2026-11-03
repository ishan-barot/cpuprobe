#pragma once
#include <cstdint>
#include <vector>

#include "cpuprobe/trace_json.hpp"

namespace cpuprobe::gpu {

// records cuda runtime calls, kernels, memcpys and memsets with nvidia's cupti activity api.
// cupti hands us filled buffers from its own thread, so only one tracer can exist at a time
class CuptiTracer {
public:
    CuptiTracer();
    ~CuptiTracer();
    CuptiTracer(const CuptiTracer&) = delete;
    CuptiTracer& operator=(const CuptiTracer&) = delete;

    void start();
    void stop();  // flushes every buffer so all records are in before it returns

    std::vector<TraceEvent> take_events();
    std::uint64_t dropped_records() const;  // records cupti had to throw away (buffers full)
};

}  // namespace cpuprobe::gpu
