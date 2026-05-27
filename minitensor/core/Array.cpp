// minitensor/core/Array.cpp

#include "ArrayIterator.hpp"
#include "ErrorMacros.hpp"
#include "Macros.hpp"
#include "Storage.hpp"

#include <cstring>
#include <functional>
#include <minitensor/minitensor.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mt
{

// ---------------------------------------------------------
// Private Helpers
// ---------------------------------------------------------
void Array::allocate_storage() {
    data_ = std::make_shared<Storage>(numel_, dtype_, device_);
}

void* Array::raw_data() noexcept {
    return data_ ? data_->data() : nullptr;
}

const void* Array::raw_data() const noexcept {
    return data_ ? data_->data() : nullptr;
}

void Array::compute_strides() {
    this->strides_.resize(shape_.size());
    std::size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        this->strides_[i] = stride;
        stride *= shape_[i];
    }
}

std::size_t Array::shape_product(const Shape& s) noexcept {
    return std::accumulate(s.begin(), s.end(), std::size_t{1}, std::multiplies<>{});
}

bool Array::shapes_equal(const Shape& a, const Shape& b) noexcept {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

// ---------------------------------------------------------
// Constructors
// ---------------------------------------------------------

Array::Array(Shape shape, DataType dtype, DeviceType device)
    : defined_(true), shape_(std::move(shape)), dtype_(dtype), device_(device) {
    numel_ = shape_product(shape_);
    compute_strides();
    allocate_storage();
}

Array::Array(const void* data, Shape shape, DataType dtype, DeviceType device)
    : defined_(true), shape_(std::move(shape)), dtype_(dtype), device_(device) {
    numel_ = shape_product(shape_);
    compute_strides();
    allocate_storage();
    std::memcpy(raw_data(), data, numel_ * element_size(dtype));
}

// ---------------------------------------------------------
// Clone, cast, and move device
// ---------------------------------------------------------
Array Array::clone() const {
    if (!defined())
        return Array();
    Array cloned(shape_, dtype_, device_);
    std::memcpy(cloned.raw_data(), this->raw_data(), numel_ * element_size(dtype_));
    return cloned;
}

Array Array::cast(DataType new_dtype) const {
    if (!defined())
        return Array();
    Array casted(shape_, new_dtype, device_);
    MT_DISPATCH_ALL_TYPES(this->dtype(), SrcT, [&]() {
        MT_DISPATCH_ALL_TYPES(new_dtype, DstT, [&]() {
            const SrcT* src_ptr = static_cast<const SrcT*>(this->raw_data());
            DstT*       dst_ptr = static_cast<DstT*>(casted.raw_data());
            for (std::size_t i = 0; i < numel_; ++i)
                dst_ptr[i] = static_cast<DstT>(src_ptr[i]);
        });
    });
    return casted;
}

Array Array::to(DeviceType target_device) const {
    if (!defined())
        return Array();
    if (this->device() == target_device) {
        return *this;
    }
    if (target_device == DeviceType::cuda) {
        throw DeviceError("CUDA device is not supported on this platform.");
    }
    Array copied(shape_, dtype_, target_device);
    std::memcpy(copied.raw_data(), this->raw_data(), numel_ * element_size(dtype_));
    return copied;
}

Array Array::reshape(const Shape& new_shape) const {
    MT_CHECK_DEFINED(*this);

    Shape       target_shape  = new_shape;
    std::size_t neg_one_idx   = static_cast<std::size_t>(-1);
    std::size_t neg_one_count = 0;
    std::size_t product       = 1;

    for (std::size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == static_cast<std::size_t>(-1)) {
            neg_one_idx = i;
            neg_one_count++;
        } else {
            product *= target_shape[i];
        }
    }

    MT_CHECK(neg_one_count <= 1, mt::ShapeError, "Only one dimension can be -1 in reshape, but got " << neg_one_count);

    if (neg_one_count == 1) {
        MT_CHECK(product > 0 && numel_ % product == 0, mt::ShapeError,
                 "Invalid shape for reshape with -1. Total elements "
                   << numel_ << " not divisible by product of other dimensions " << product);
        target_shape[neg_one_idx] = numel_ / product;
        product                   = numel_;
    }

    MT_CHECK(product == numel_, mt::ShapeError,
             "Shape mismatch in reshape: number of elements must remain the same. Original size: "
               << numel_ << ", new shape size: " << product);

    // Create a new Array sharing the same storage
    Array reshaped(*this);
    reshaped.shape_ = target_shape;
    reshaped.compute_strides();
    return reshaped;
}

Array Array::flatten(std::size_t start_dim, std::size_t end_dim) const {
    MT_CHECK_DEFINED(*this);

    std::size_t nd = ndim();
    if (nd == 0) {
        return reshape({1});
    }

    if (end_dim == static_cast<std::size_t>(-1)) {
        end_dim = nd - 1;
    }

    MT_CHECK(start_dim <= end_dim && end_dim < nd, mt::ShapeError,
             "Invalid dimensions for flatten: start_dim (" << start_dim << ") must be <= end_dim (" << end_dim
                                                           << ") and end_dim must be < ndim (" << nd << ")");

    Shape new_shape;
    for (std::size_t i = 0; i < start_dim; ++i) {
        new_shape.push_back(shape_[i]);
    }

    std::size_t flattened_dim = 1;
    for (std::size_t i = start_dim; i <= end_dim; ++i) {
        flattened_dim *= shape_[i];
    }
    new_shape.push_back(flattened_dim);

    for (std::size_t i = end_dim + 1; i < nd; ++i) {
        new_shape.push_back(shape_[i]);
    }

    return reshape(new_shape);
}

// ---------------------------------------------------------
// Helper Templates for Operations
// ---------------------------------------------------------
namespace
{

template <class F>
Array elementwise_unary(const Array& a, F&& fn) {
    MT_CHECK_DEFINED(a);
    Array r(a.shape(), a.dtype(), a.device());
    ArrayIteratorConfig()
      .add_output(r)
      .add_input(a)
      .build()
      .for_each([&fn]<typename T>(T x) -> T { return static_cast<T>(fn(static_cast<double>(x))); });
    return r;
}

template <class Op>
Array elementwise_binary(const Array& a, const Array& b, Op&& op) {
    Array r(a.shape(), a.dtype(), a.device());
    ArrayIteratorConfig()
      .add_output(r)
      .add_input(a)
      .add_input(b)
      .build()
      .for_each([&op]<typename T>(T x, T y) -> T { return op(x, y); });
    return r;
}

template <class Op>
Array elementwise_scalar(const Array& a, double s, Op&& op) {
    Array r(a.shape(), a.dtype(), a.device());
    ArrayIteratorConfig()
      .add_output(r)
      .add_input(a)
      .build()
      .for_each([&op, s]<typename T>(T x) -> T { return op(x, static_cast<T>(s)); });
    return r;
}

template <class T>
Array filled(const Shape& shape, T v, DataType dtype, DeviceType device) {
    Array r(shape, dtype, device);
    MT_DISPATCH_ALL_TYPES(dtype, U, [&]() {
        U* p = r.data<U>();
        std::fill(p, p + r.numel(), static_cast<U>(v));
    });
    return r;
}

} // namespace

// ---------------------------------------------------------
// Initializers
// ---------------------------------------------------------
Array Array::zeros(const Shape& s, DataType d, DeviceType dev) {
    return filled(s, 0, d, dev);
}
Array Array::ones(const Shape& s, DataType d, DeviceType dev) {
    return filled(s, 1, d, dev);
}
template <class T>
Array Array::full(const Shape& s, T v, DataType d, DeviceType dev) {
    return filled(s, v, d, dev);
}

Array Array::randn(const Shape& shape, DataType dtype, DeviceType device) {
    Array              r(shape, dtype, device);
    std::random_device rd;
    std::mt19937       gen(rd());

    MT_DISPATCH_ALL_TYPES(dtype, T, [&]<typename U = T>() {
        U* ptr = r.data<U>();

        std::normal_distribution dis(0.0, 1.0);
        for (std::size_t i = 0; i < r.numel(); ++i)
            ptr[i] = dis(gen);
    });
    return r;
}

Array zeros(const Shape& shape, DataType dtype, DeviceType device) {
    return Array::zeros(shape, dtype, device);
}
Array ones(const Shape& shape, DataType dtype, DeviceType device) {
    return Array::ones(shape, dtype, device);
}
Array randn(const Shape& shape, DataType dtype, DeviceType device) {
    return Array::randn(shape, dtype, device);
}
template <typename T>
Array full(const Shape& shape, T fill_value, DataType dtype, DeviceType device) {
    return Array::full(shape, fill_value, dtype, device);
}

// ---------------------------------------------------------
// Explicit Template Instantiations
// ---------------------------------------------------------

template Array full<float>(const Shape& shape, float fill_value, DataType dtype, DeviceType device);
template Array full<double>(const Shape& shape, double fill_value, DataType dtype, DeviceType device);
template Array full<int>(const Shape& shape, int fill_value, DataType dtype, DeviceType device);
template Array full<long>(const Shape& shape, long fill_value, DataType dtype, DeviceType device);
template Array full<long long>(const Shape& shape, long long fill_value, DataType dtype, DeviceType device);

// ---------------------------------------------------------
// Mathematical functions
// ---------------------------------------------------------
Array sin(const Array& a) {
    return elementwise_unary(a, [](double x) { return std::sin(x); });
}
Array cos(const Array& a) {
    return elementwise_unary(a, [](double x) { return std::cos(x); });
}
Array tan(const Array& a) {
    return elementwise_unary(a, [](double x) { return std::tan(x); });
}
Array exp(const Array& a) {
    return elementwise_unary(a, [](double x) { return std::exp(x); });
}
Array log(const Array& a) {
    return elementwise_unary(a, [](double x) { return std::log(x); });
}

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

// ---------------------------------------------------------
// Element-wise operations
// ---------------------------------------------------------
Array Array::operator+(const Array& o) const {
    return elementwise_binary(*this, o, std::plus<>{});
}
Array Array::operator-(const Array& o) const {
    return elementwise_binary(*this, o, std::minus<>{});
}
Array Array::operator*(const Array& o) const {
    return elementwise_binary(*this, o, std::multiplies<>{});
}

// ---------------------------------------------------------
// Scalar operations
// ---------------------------------------------------------
Array Array::operator+(double s) const {
    return elementwise_scalar(*this, s, std::plus<>{});
}
Array Array::operator-(double s) const {
    return elementwise_scalar(*this, s, std::minus<>{});
}
Array Array::operator*(double s) const {
    return elementwise_scalar(*this, s, std::multiplies<>{});
}
Array Array::operator/(double s) const {
    return elementwise_scalar(*this, s, std::divides<>{});
}

// ---------------------------------------------------------
// Explicit Template Instantiations
// ---------------------------------------------------------

template Array Array::full<float>(const Shape& shape, float fill_value, DataType dtype, DeviceType device);
template Array Array::full<double>(const Shape& shape, double fill_value, DataType dtype, DeviceType device);
template Array Array::full<int>(const Shape& shape, int fill_value, DataType dtype, DeviceType device);
template Array Array::full<long>(const Shape& shape, long fill_value, DataType dtype, DeviceType device);
template Array Array::full<long long>(const Shape& shape, long long fill_value, DataType dtype, DeviceType device);

} // namespace mt