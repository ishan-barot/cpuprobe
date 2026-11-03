#include <cuda.h>
#include <cupti.h>
#include <cxxabi.h>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>

#include "cpuprobe/gpu/tracer.hpp"

// the kernel record struct gets a new version number every so often as nvidia adds fields.
// the old fields keep their offsets, so any version that exists in our headers works
#if CUDA_VERSION >= 13010
using KernelRecord = CUpti_ActivityKernel11;
#elif CUDA_VERSION >= 13000
using KernelRecord = CUpti_ActivityKernel10;
#else
using KernelRecord = CUpti_ActivityKernel9;
#endif
#if CUDA_VERSION >= 12080
using MemcpyRecord = CUpti_ActivityMemcpy6;
#else
using MemcpyRecord = CUpti_ActivityMemcpy5;
#endif
using MemsetRecord = CUpti_ActivityMemset4;

#define CUPTI_CHECK(call)                                                                  \
    do {                                                                                   \
        CUptiResult r_ = (call);                                                           \
        if (r_ != CUPTI_SUCCESS) {                                                         \
            const char* msg_ = nullptr;                                                    \
            cuptiGetResultString(r_, &msg_);                                               \
            throw std::runtime_error(std::string(#call) +                                  \
                                     " failed: " + (msg_ ? msg_ : "unknown cupti error")); \
        }                                                                                  \
    } while (0)

namespace cpuprobe::gpu {

namespace {

constexpr std::size_t kBufferBytes = 8u << 20;  // 8mb per buffer, cupti asks for more as needed

// state shared with the cupti callbacks, which are plain c functions with no user pointer
struct State {
    std::mutex mu;
    std::vector<TraceEvent> events;
    std::atomic<std::uint64_t> dropped{0};
    bool callbacks_registered = false;
    bool tracer_alive = false;
};
State& state() {
    static State s;
    return s;
}

std::string demangle(const char* name) {
    if (!name) return "unknown kernel";
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> out(abi::__cxa_demangle(name, nullptr, nullptr, &status),
                                               std::free);
    return short_kernel_name((status == 0 && out) ? out.get() : name);
}

// runtime api names come back with a _v7000 style version suffix, which is just noise
std::string api_name(std::uint32_t cbid) {
    const char* name = nullptr;
    if (cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, cbid, &name) != CUPTI_SUCCESS || !name)
        return "cuda api " + std::to_string(cbid);
    static const std::regex suffix("_v[0-9]+$");
    return std::regex_replace(std::string(name), suffix, "");
}

const char* copy_kind(std::uint8_t k) {
    switch (k) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            return "memcpy HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            return "memcpy DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            return "memcpy DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            return "memcpy HtoH";
        default:
            return "memcpy";
    }
}

const char* host_mem_kind(std::uint8_t k) {
    switch (k) {
        case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
            return "pageable";
        case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
            return "pinned";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
            return "device";
        default:
            return "other";
    }
}

std::string gbps(std::uint64_t bytes, std::uint64_t ns) {
    if (ns == 0) return "0";
    return std::to_string(static_cast<double>(bytes) / static_cast<double>(ns));  // b/ns == gb/s
}

void handle_record(CUpti_Activity* rec, std::vector<TraceEvent>& out) {
    TraceEvent e;
    switch (rec->kind) {
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
            auto* a = reinterpret_cast<CUpti_ActivityAPI*>(rec);
            e.name = api_name(a->cbid);
            e.category = "cuda_api";
            e.start_ns = a->start;
            e.end_ns = a->end;
            e.tid = a->threadId;
            e.correlation = a->correlationId;
            break;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto* k = reinterpret_cast<KernelRecord*>(rec);
            e.name = demangle(k->name);
            e.category = "kernel";
            e.start_ns = k->start;
            e.end_ns = k->end;
            e.on_gpu = true;
            e.tid = k->streamId;
            e.correlation = k->correlationId;
            e.args = {{"grid", std::to_string(k->gridX) + "x" + std::to_string(k->gridY) + "x" +
                                   std::to_string(k->gridZ)},
                      {"block", std::to_string(k->blockX) + "x" + std::to_string(k->blockY) + "x" +
                                    std::to_string(k->blockZ)},
                      {"static_smem_bytes", std::to_string(k->staticSharedMemory)},
                      {"registers_per_thread", std::to_string(k->registersPerThread)}};
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto* m = reinterpret_cast<MemcpyRecord*>(rec);
            e.name = copy_kind(m->copyKind);
            e.category = "memcpy";
            e.start_ns = m->start;
            e.end_ns = m->end;
            e.on_gpu = true;
            e.tid = m->streamId;
            e.correlation = m->correlationId;
            std::uint8_t host_side =
                m->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD ? m->srcKind : m->dstKind;
            e.args = {{"bytes", std::to_string(m->bytes)},
                      {"host_memory", host_mem_kind(host_side)},
                      {"gb_per_s", gbps(m->bytes, m->end - m->start)}};
            // pageable vs pinned is the whole point of the demo, so put it in the label too
            if (m->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD ||
                m->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOH)
                e.name += std::string(" (") + host_mem_kind(host_side) + ")";
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            auto* m = reinterpret_cast<MemsetRecord*>(rec);
            e.name = "memset";
            e.category = "memset";
            e.start_ns = m->start;
            e.end_ns = m->end;
            e.on_gpu = true;
            e.tid = m->streamId;
            e.correlation = m->correlationId;
            e.args = {{"bytes", std::to_string(m->bytes)}};
            break;
        }
        default:
            return;  // other kinds arent enabled, ignore anything unexpected
    }
    out.push_back(std::move(e));
}

// cupti calls this when it needs an empty buffer. records must be 8 byte aligned
void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_records) {
    *buffer = static_cast<uint8_t*>(std::aligned_alloc(8, kBufferBytes));
    *size = *buffer ? kBufferBytes : 0;
    *max_records = 0;  // as many as fit
}

// cupti calls this (from its own thread) with a full buffer. keep it short: parse and free
void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t stream_id, uint8_t* buffer, size_t,
                               size_t valid_size) {
    std::vector<TraceEvent> local;
    CUpti_Activity* rec = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &rec) == CUPTI_SUCCESS)
        handle_record(rec, local);

    size_t dropped = 0;
    if (cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped) == CUPTI_SUCCESS)
        state().dropped += dropped;
    std::free(buffer);

    std::lock_guard<std::mutex> lk(state().mu);
    auto& ev = state().events;
    ev.insert(ev.end(), std::make_move_iterator(local.begin()),
              std::make_move_iterator(local.end()));
}

constexpr CUpti_ActivityKind kKinds[] = {
    CUPTI_ACTIVITY_KIND_RUNTIME,
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
    CUPTI_ACTIVITY_KIND_MEMCPY,
    CUPTI_ACTIVITY_KIND_MEMSET,
};

}  // namespace

CuptiTracer::CuptiTracer() {
    auto& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.tracer_alive) throw std::logic_error("only one CuptiTracer at a time");
    s.tracer_alive = true;
    if (!s.callbacks_registered) {
        CUPTI_CHECK(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
        s.callbacks_registered = true;
    }
}

CuptiTracer::~CuptiTracer() {
    try {
        stop();
    } catch (...) {
        // destructors dont throw, a failed flush just means a shorter trace
    }
    std::lock_guard<std::mutex> lk(state().mu);
    state().tracer_alive = false;
}

void CuptiTracer::start() {
    {
        std::lock_guard<std::mutex> lk(state().mu);
        state().events.clear();
    }
    state().dropped = 0;
    for (auto k : kKinds) CUPTI_CHECK(cuptiActivityEnable(k));
}

void CuptiTracer::stop() {
    // forced flush also hands over buffers that arent full yet
    CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    for (auto k : kKinds) cuptiActivityDisable(k);
}

std::vector<TraceEvent> CuptiTracer::take_events() {
    std::lock_guard<std::mutex> lk(state().mu);
    return std::move(state().events);
}

std::uint64_t CuptiTracer::dropped_records() const { return state().dropped; }

}  // namespace cpuprobe::gpu
