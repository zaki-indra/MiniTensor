// minitensor/ops/Initializers.cpp
//
// Array factories: zeros, ones, full, randn. Both the static-member form
// (Array::zeros) and the free-function form (mt::zeros) live here.
//
// These don't use ArrayIterator: they have no input operands, just a
// dtype-dispatched write into freshly allocated storage.

#include "core/Macros.hpp"

#include <algorithm>
#include <minitensor/minitensor.hpp>
#include <random>

namespace mt
{
namespace
{

template <class T>
Array filled(const Shape& shape, T v, EDataType dtype, EDeviceType device) {
    Array r(shape, dtype, device);
    MT_DISPATCH_ALL_TYPES(dtype, U, [&]() {
        U* p = r.data<U>();
        std::fill(p, p + r.numel(), static_cast<U>(v));
    });
    return r;
}

} // namespace

// ---------------------------------------------------------
// Array static initializers
// ---------------------------------------------------------
Array Array::zeros(const Shape& s, EDataType d, EDeviceType dev) {
    return filled(s, 0, d, dev);
}
Array Array::ones(const Shape& s, EDataType d, EDeviceType dev) {
    return filled(s, 1, d, dev);
}

template <class T>
Array Array::full(const Shape& s, T v, EDataType d, EDeviceType dev) {
    return filled(s, v, d, dev);
}

Array Array::randn(const Shape& shape, EDataType dtype, EDeviceType device) {
    Array              r(shape, dtype, device);
    std::random_device rd;
    std::mt19937       gen(rd());

    MT_DISPATCH_ALL_TYPES(dtype, T, [&]<typename U = T>() {
        U*                       ptr = r.data<U>();
        std::normal_distribution dis(0.0, 1.0);
        for (std::size_t i = 0; i < r.numel(); ++i)
            ptr[i] = dis(gen);
    });
    return r;
}

// ---------------------------------------------------------
// Free factory functions (delegate to Array statics)
// ---------------------------------------------------------
Array zeros(const Shape& shape, EDataType dtype, EDeviceType device) {
    return Array::zeros(shape, dtype, device);
}
Array ones(const Shape& shape, EDataType dtype, EDeviceType device) {
    return Array::ones(shape, dtype, device);
}
Array randn(const Shape& shape, EDataType dtype, EDeviceType device) {
    return Array::randn(shape, dtype, device);
}
template <typename T>
Array full(const Shape& shape, T fill_value, EDataType dtype, EDeviceType device) {
    return Array::full(shape, fill_value, dtype, device);
}

// ---------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------
template Array Array::full<float>(const Shape& shape, float fill_value, EDataType dtype, EDeviceType device);
template Array Array::full<double>(const Shape& shape, double fill_value, EDataType dtype, EDeviceType device);
template Array Array::full<int>(const Shape& shape, int fill_value, EDataType dtype, EDeviceType device);
template Array Array::full<long>(const Shape& shape, long fill_value, EDataType dtype, EDeviceType device);
template Array Array::full<long long>(const Shape& shape, long long fill_value, EDataType dtype, EDeviceType device);

template Array full<float>(const Shape& shape, float fill_value, EDataType dtype, EDeviceType device);
template Array full<double>(const Shape& shape, double fill_value, EDataType dtype, EDeviceType device);
template Array full<int>(const Shape& shape, int fill_value, EDataType dtype, EDeviceType device);
template Array full<long>(const Shape& shape, long fill_value, EDataType dtype, EDeviceType device);
template Array full<long long>(const Shape& shape, long long fill_value, EDataType dtype, EDeviceType device);

} // namespace mt
