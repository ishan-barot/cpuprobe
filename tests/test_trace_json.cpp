#include <gtest/gtest.h>

#include <sstream>

#include "cpuprobe/trace_json.hpp"

using namespace cpuprobe;

static std::vector<TraceEvent> sample() {
    TraceEvent launch;
    launch.name = "cudaLaunchKernel";
    launch.category = "cuda_api";
    launch.start_ns = 1000;
    launch.end_ns = 6000;
    launch.tid = 42;
    launch.correlation = 7;

    TraceEvent kernel;
    kernel.name = "transpose_tiled<1>";
    kernel.category = "kernel";
    kernel.start_ns = 9000;
    kernel.end_ns = 59000;
    kernel.on_gpu = true;
    kernel.tid = 7;  // stream
    kernel.correlation = 7;

    TraceEvent orphan = kernel;  // gpu work whose api call wasnt captured, gets no arrow
    orphan.correlation = 99;
    orphan.start_ns = 60000;
    orphan.end_ns = 61000;
    return {launch, kernel, orphan};
}

TEST(TraceJson, EscapesQuotesAndControlChars) {
    EXPECT_EQ(json_escape("a\"b"), "a\\\"b");
    EXPECT_EQ(json_escape("back\\slash"), "back\\\\slash");
    EXPECT_EQ(json_escape("line\nbreak"), "line\\nbreak");
    EXPECT_EQ(json_escape(std::string(1, '\x01')), "\\u0001");
}

TEST(TraceJson, LinksApiCallToKernelOnly) {
    auto ev = sample();
    EXPECT_EQ(count_correlated(ev), 1u);

    std::stringstream ss;
    write_chrome_trace(ss, ev);
    auto s = ss.str();
    // timestamps are relative to the first event and in microseconds
    EXPECT_NE(s.find("\"ts\":0.000"), std::string::npos);
    EXPECT_NE(s.find("\"dur\":50.000"), std::string::npos);
    // exactly one flow start and one flow end
    EXPECT_EQ(s.find("\"ph\":\"s\""), s.rfind("\"ph\":\"s\""));
    EXPECT_NE(s.find("\"ph\":\"f\""), std::string::npos);
    EXPECT_NE(s.find("transpose_tiled<1>"), std::string::npos);
    EXPECT_NE(s.find("stream 7"), std::string::npos);
}

TEST(TraceJson, SummaryAddsUpPerName) {
    auto rows = summarize_trace(sample());
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].name, "transpose_tiled<1>");  // biggest total first
    EXPECT_EQ(rows[0].count, 2);
    EXPECT_NEAR(rows[0].total_ms, 0.051, 1e-9);
}

TEST(TraceJson, EmptyTraceIsStillValid) {
    std::stringstream ss;
    write_chrome_trace(ss, {});
    EXPECT_NE(ss.str().find("traceEvents"), std::string::npos);
}

// this one bit me: the first version cut at the first '(' which is the one in
// "(anonymous namespace)", so every kernel came out with an empty name
TEST(TraceJson, ShortKernelNames) {
    EXPECT_EQ(short_kernel_name("cpuprobe::gpu::(anonymous namespace)::copy_kernel(float const*, "
                                "float*, int, int)"),
              "copy_kernel");
    EXPECT_EQ(short_kernel_name("void cpuprobe::gpu::(anonymous namespace)::transpose_tiled<1>("
                                "float const*, float*, int, int)"),
              "transpose_tiled<1>");
    EXPECT_EQ(short_kernel_name("void ns::foo<ns::Bar, 3>(int)"), "foo<ns::Bar, 3>");
    EXPECT_EQ(short_kernel_name("plain_kernel"), "plain_kernel");
    EXPECT_EQ(short_kernel_name(""), "unknown kernel");
}
