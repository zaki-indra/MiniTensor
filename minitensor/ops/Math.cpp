// minitensor/ops/Math.cpp
//
// Free math functions: transcendentals (sin, cos, tan, exp, log), sqrt
// (with integer-domain check), and divide (with division-by-zero
// detection for integer dtypes).

#include "Elementwise.hpp"
#include "core/ArrayIterator.hpp"
#include "core/ErrorMacros.hpp"

#include <cmath>
#include <minitensor/minitensor.hpp>
#include <type_traits>

namespace mt
{

// ---------------------------------------------------------
// Transcendentals (computed in double, cast back to T)
// ---------------------------------------------------------
Array sin(const Array& a) {
    return ops::elementwise_unary_via_double(a, [](double x) { return std::sin(x); });
}
Array cos(const Array& a) {
    return ops::elementwise_unary_via_double(a, [](double x) { return std::cos(x); });
}
Array tan(const Array& a) {
    return ops::elementwise_unary_via_double(a, [](double x) { return std::tan(x); });
}
Array exp(const Array& a) {
    return ops::elementwise_unary_via_double(a, [](double x) { return std::exp(x); });
}
Array log(const Array& a) {
    return ops::elementwise_unary_via_double(a, [](double x) { return std::log(x); });
}

// ---------------------------------------------------------
// sqrt: integer dtypes throw on negative input
// ---------------------------------------------------------
Array sqrt(const Array& arr) {
    Array r(arr.shape(), arr.dtype(), arr.device());
    ArrayIteratorConfig().add_output(r).add_input(arr).build().for_each([]<typename T>(T x) -> T {
        if constexpr (std::is_integral_v<T>) {
            if (x < 0) {
                MT_THROW(mt::ArithmeticError, "Square root of negative integer " << x << " is undefined.");
            }
        }
        return static_cast<T>(std::sqrt(static_cast<double>(x)));
    });
    return r;
}

// ---------------------------------------------------------
// divide: integer division-by-zero is detected and surfaced after the
// loop completes; float division-by-zero produces inf/nan per IEEE-754.
// ---------------------------------------------------------
Array divide(const Array& a, const Array& b) {
    Array r(a.shape(), a.dtype(), a.device());
    bool  zero_division = false;
    ArrayIteratorConfig().add_output(r).add_input(a).add_input(b).build().for_each(
      [&zero_division]<typename T>(T x, T y) -> T {
          if (y == T{0}) {
              if constexpr (std::is_integral_v<T>) {
                  zero_division = true;
                  return T{0};
              }
          }
          return x / y;
      });
    if (zero_division) {
        throw mt::ArithmeticError("Division by zero encountered.");
    }
    return r;
}

} // namespace mt
