// benchmarks/helper.hpp
//
// Shared utilities for the MiniTensor benchmark suite.

#pragma once

#include <benchmark/benchmark.h>
#include <cstddef>
#include <minitensor/minitensor.hpp>

namespace mtbench
{

// Standard element-count sweep for elementwise / per-element benchmarks.
// Spans roughly L1-resident (64 elts) up to clearly RAM-bound (~1M elts)
// so the report shows the cache-hierarchy transitions. RangeMultiplier(8)
// keeps the number of points small while covering four orders of magnitude.
inline void elementwise_sizes(benchmark::internal::Benchmark* b) {
    b->RangeMultiplier(8)->Range(1 << 6, 1 << 20);
}

// Record throughput in both elements/s and bytes/s for a run that
// processed `n` elements of type-size `elem_size` per iteration.
inline void set_throughput(benchmark::State& state, std::size_t n, std::size_t elem_size) {
    const auto iters = static_cast<std::size_t>(state.iterations());
    state.SetItemsProcessed(static_cast<std::int64_t>(iters * n));
    state.SetBytesProcessed(static_cast<std::int64_t>(iters * n * elem_size));
}

} // namespace mtbench
