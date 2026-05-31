// benchmarks/bench_core.cpp
//
// Benchmarks for core Array machinery outside the elementwise path:
//   - reshape : should be ~constant in n (shares storage, no copy)
//   - cast    : f32 -> f64, exercises the nested src x dst dtype dispatch
//   - zeros   : Storage allocation + fill
//   - randn   : Storage allocation + RNG fill (RNG-dominated)

#include "helper.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <minitensor/minitensor.hpp>

namespace
{

// ---------------------------------------------------------
// reshape: a view operation. Timing should stay flat as n grows — that
// flatness IS the result (it proves reshape doesn't copy the buffer).
// ---------------------------------------------------------
void BM_Reshape(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    mt::Array         a = mt::randn({n}, mt::EDataType::f32);
    for (auto _ : state) {
        mt::Array r = a.reshape({1, n});
        benchmark::DoNotOptimize(r.data<float>());
    }
    mtbench::set_throughput(state, n, sizeof(float));
}
BENCHMARK(BM_Reshape)->Apply(mtbench::elementwise_sizes);

// ---------------------------------------------------------
// cast: f32 -> f64. Nested dispatch on (source dtype, dest dtype), one
// converting copy. Scales with n.
// ---------------------------------------------------------
void BM_Cast_f32_to_f64(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    mt::Array         a = mt::randn({n}, mt::EDataType::f32);
    for (auto _ : state) {
        mt::Array c = a.cast(mt::EDataType::f64);
        benchmark::DoNotOptimize(c.data<double>());
    }
    // Reads n*4 bytes, writes n*8 — report on the source element size.
    mtbench::set_throughput(state, n, sizeof(float));
}
BENCHMARK(BM_Cast_f32_to_f64)->Apply(mtbench::elementwise_sizes);

// ---------------------------------------------------------
// zeros: allocation + zero-fill through the Storage allocator.
// ---------------------------------------------------------
void BM_Zeros_f32(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        mt::Array a = mt::zeros({n}, mt::EDataType::f32);
        benchmark::DoNotOptimize(a.data<float>());
    }
    mtbench::set_throughput(state, n, sizeof(float));
}
BENCHMARK(BM_Zeros_f32)->Apply(mtbench::elementwise_sizes);

// ---------------------------------------------------------
// randn: allocation + per-element RNG fill. The RNG dominates; this is the
// baseline to beat if randn is ever vectorized.
// ---------------------------------------------------------
void BM_Randn_f32(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        mt::Array a = mt::randn({n}, mt::EDataType::f32);
        benchmark::DoNotOptimize(a.data<float>());
    }
    mtbench::set_throughput(state, n, sizeof(float));
}
BENCHMARK(BM_Randn_f32)->Apply(mtbench::elementwise_sizes);

} // namespace
