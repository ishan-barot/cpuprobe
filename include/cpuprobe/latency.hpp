#pragma once
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace cpuprobe {

// each node takes up a whole 64 byte cache line so every hop lands on a new line
struct alignas(64) Node {
    Node* next;
    char pad[64 - sizeof(Node*)];
};
static_assert(sizeof(Node) == 64);

// sattolo's shuffle, gives a random order thats guaranteed to be one big loop
std::vector<std::size_t> sattolo(std::size_t n, std::mt19937_64& rng);

struct LatencyResult {
    std::size_t bytes;
    double ns_per_access;  // median across all the runs
};

// builds a pointer chase ring of this many bytes and gives back the median ns per access
LatencyResult measure_latency(std::size_t bytes, int reps, std::uint64_t seed = 42);

}  // namespace cpuprobe
