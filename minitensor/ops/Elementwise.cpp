// minitensor/ops/Elementwise.cpp
//
// Definitions of Array's binary and scalar elementwise operators. Member
// functions of Array, but defined here so that adding new ops doesn't
// grow Array.cpp.

#include "Elementwise.hpp"

#include <functional>
#include <minitensor/minitensor.hpp>

namespace mt
{

// ---------------------------------------------------------
// Binary operators (Array <op> Array)
// ---------------------------------------------------------
Array Array::operator+(const Array& o) const {
    return ops::elementwise_binary(*this, o, std::plus<>{});
}
Array Array::operator-(const Array& o) const {
    return ops::elementwise_binary(*this, o, std::minus<>{});
}
Array Array::operator*(const Array& o) const {
    return ops::elementwise_binary(*this, o, std::multiplies<>{});
}

// ---------------------------------------------------------
// Scalar operators (Array <op> double)
// ---------------------------------------------------------
Array Array::operator+(double s) const {
    return ops::elementwise_scalar(*this, s, std::plus<>{});
}
Array Array::operator-(double s) const {
    return ops::elementwise_scalar(*this, s, std::minus<>{});
}
Array Array::operator*(double s) const {
    return ops::elementwise_scalar(*this, s, std::multiplies<>{});
}
Array Array::operator/(double s) const {
    return ops::elementwise_scalar(*this, s, std::divides<>{});
}

} // namespace mt
