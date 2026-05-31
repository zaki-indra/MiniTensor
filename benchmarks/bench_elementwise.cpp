// benchmarks/bench_elementwise.cpp
//
// Benchmarks for the ArrayIterator-backed elementwise paths:
//   - binary  (a + b)      : two-input kernel
//   - scalar  (a * 2.0)    : one-input kernel + captured scalar
//   - unary   (sin(a))     : the "promote through double" path
//
// Each fixes inputs outside the timed loop and only measures the op.
// `DoNotOptimize` on the result's data pointer prevents the compiler from
// eliding the allocation + compute.

#include "helper.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <minitensor/minitensor.hpp>

namespace
{

// ---------------------------------------------------------
// Binary: a + b
// ---------------------------------------------------------
void BM_Add_f32(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    mt::Array         a = mt::randn({n}, mt::DataType::f32);
    mt::Array         b = mt::randn({n}, mt::DataType::f32);
    for (auto _ : state) {
        mt::Array c = a + b;
        benchmark::DoNotOptimize(c.data<float>());
    }
    mtbench::set_throughput(state, n, sizeof(float));
}
BENCHMARK(BM_Add_f32)->Apply(mtbench::elementwise_sizes);

void BM_Add_i32(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    mt::Array         a = mt::randn({n}, mt::DataType::i32);
    mt::Array         b = mt::randn({n}, mt::DataType::i32);
    for (auto _ : state) {
        mt::Array c = a + b;
        benchmark::DoNotOptimize(c.data<std::int32_t>());
    }
    mtbench::set_throughput(state, n, sizeof(std::int32_t));
}
BENCHMARK(BM_Add_i32)->Apply(mtbench::elementwise_sizes);

// ---------------------------------------------------------
// Scalar: a * 2.0
// ---------------------------------------------------------
void BM_ScalarMul_f32(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    mt::Array         a = mt::randn({n}, mt::DataType::f32);
    for (auto _ : state) {
        mt::Array c = a * 2.0;
        benchmark::DoNotOptimize(c.data<float>());
    }
    mtbench::set_throughput(state, n, sizeof(float));
}
BENCHMARK(BM_ScalarMul_f32)->Apply(mtbench::elementwise_sizes);

// ---------------------------------------------------------
// Unary math: sin(a). Computed in double then cast back to T, so the
// f32 and i32 variants share the same transcendental cost — the contrast
// isolates the cast overhead, not the math.
// ---------------------------------------------------------
void BM_Sin_f32(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    mt::Array         a = mt::randn({n}, mt::DataType::f32);
    for (auto _ : state) {
        mt::Array c = mt::sin(a);
        benchmark::DoNotOptimize(c.data<float>());
    }
    mtbench::set_throughput(state, n, sizeof(float));
}
BENCHMARK(BM_Sin_f32)->Apply(mtbench::elementwise_sizes);

} // namespace
