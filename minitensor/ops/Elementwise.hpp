// minitensor/ops/Elementwise.hpp
//
// Shared helpers for elementwise operations. Each one wraps ArrayIterator
// with the common "allocate output, configure operands, run kernel" pattern
// so op call sites stay focused on the math.
//
// This is a private/internal header (not installed). Include via
// "ops/Elementwise.hpp" using the library's PRIVATE include path.

#pragma once

#include "core/ArrayIterator.hpp"

#include <minitensor/minitensor.hpp>
#include <utility>

namespace mt::ops
{

// Apply a scalar kernel `T -> T` to every element of `a`.
template <class Kernel>
[[nodiscard]] Array elementwise_unary(const Array& a, Kernel&& k) {
    Array r(a.shape(), a.dtype(), a.device());
    ArrayIteratorConfig().add_output(r).add_input(a).build().for_each(std::forward<Kernel>(k));
    return r;
}

// Apply a unary kernel that promotes through `double`. Common for
// transcendentals where std::sin / std::log / ... are defined over
// floating-point only. The cast back to T is lossy for integer dtypes;
// callers accept that trade-off.
template <class FnDouble>
[[nodiscard]] Array elementwise_unary_via_double(const Array& a, FnDouble&& fn) {
    return elementwise_unary(a, [&fn]<typename T>(T x) -> T {
        return static_cast<T>(fn(static_cast<double>(x)));
    });
}

// Apply a binary kernel `(T, T) -> T` pairwise across `a` and `b`.
template <class Kernel>
[[nodiscard]] Array elementwise_binary(const Array& a, const Array& b, Kernel&& k) {
    Array r(a.shape(), a.dtype(), a.device());
    ArrayIteratorConfig()
      .add_output(r)
      .add_input(a)
      .add_input(b)
      .build()
      .for_each(std::forward<Kernel>(k));
    return r;
}

// Apply a binary op where one operand is a scalar broadcast across every
// position. The scalar is captured by value as `double` and cast to T
// inside the kernel; callers pass a binary `(T, T) -> T` op (typically
// std::plus<>{} / std::minus<>{} / etc.).
template <class Op>
[[nodiscard]] Array elementwise_scalar(const Array& a, double s, Op&& op) {
    Array r(a.shape(), a.dtype(), a.device());
    ArrayIteratorConfig().add_output(r).add_input(a).build().for_each(
      [op = std::forward<Op>(op), s]<typename T>(T x) -> T { return op(x, static_cast<T>(s)); });
    return r;
}

} // namespace mt::ops
