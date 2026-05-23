// minitensor/core/Array.cpp

#include "Storage.hpp"

#include <cstring>
#include <minitensor/minitensor.hpp>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>

// ---------------------------------------------------------
// Runtime Type Dispatcher
// ---------------------------------------------------------
#define MT_DISPATCH_ALL_TYPES(TYPE, TYPE_NAME, ...)                                                                    \
    [&]() {                                                                                                            \
        switch (TYPE) {                                                                                                \
        case mt::DataType::f32: {                                                                                      \
            using TYPE_NAME = float;                                                                                   \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case mt::DataType::f64: {                                                                                      \
            using TYPE_NAME = double;                                                                                  \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case mt::DataType::i32: {                                                                                      \
            using TYPE_NAME = int32_t;                                                                                 \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case mt::DataType::i64: {                                                                                      \
            using TYPE_NAME = int64_t;                                                                                 \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        default:                                                                                                       \
            throw std::runtime_error("Unsupported DataType in MT_DISPATCH_ALL_TYPES");                                 \
        }                                                                                                              \
    }()

namespace mt
{

// ---------------------------------------------------------
// Private Bridge Helpers
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

std::size_t Array::shape_product(const Shape& s) noexcept {
    std::size_t res = 1;
    for (auto dim : s)
        res *= dim;
    return res;
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

void Array::compute_strides() {
    strides_.resize(shape_.size());
    std::size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

// ---------------------------------------------------------
// Constructors
// ---------------------------------------------------------
Array::Array() noexcept = default;

Array::Array(const Array& other)
    : defined_(other.defined_), is_view_(other.is_view_), numel_(other.numel_), shape_(other.shape_),
      strides_(other.strides_), offsets_(other.offsets_), dtype_(other.dtype_), device_(other.device_),
      data_(other.data_) {
}

Array::Array(Array&& other) noexcept
    : defined_(other.defined_), is_view_(other.is_view_), numel_(other.numel_), shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)), offsets_(std::move(other.offsets_)), dtype_(other.dtype_),
      device_(other.device_), data_(std::move(other.data_)) {
    other.defined_ = false;
    other.numel_   = 0;
}

Array::Array(Shape shape, DataType dtype, DeviceType device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device), defined_(true) {
    numel_ = shape_product(shape_);
    compute_strides();
    allocate_storage();
}

Array::Array(const void* data, Shape shape, DataType dtype, DeviceType device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device), defined_(true) {
    numel_ = shape_product(shape_);
    compute_strides();
    allocate_storage();
    std::memcpy(raw_data(), data, numel_ * element_size(dtype));
}

// ---------------------------------------------------------
// Metadata Getter
// ---------------------------------------------------------
bool Array::defined() const noexcept {
    return this->defined_;
}
const Shape& Array::shape() const noexcept {
    return this->shape_;
}
const Shape& Array::strides() const noexcept {
    return this->strides_;
}
std::size_t Array::ndim() const noexcept {
    return this->shape_.size();
}
std::size_t Array::numel() const noexcept {
    return this->numel_;
}
DataType Array::dtype() const noexcept {
    return this->dtype_;
}
DeviceType Array::device() const noexcept {
    return this->device_;
}

Array Array::clone() const {
    if (!defined())
        return Array();
    Array cloned(shape_, dtype_, device_);
    std::memcpy(cloned.raw_data(), this->raw_data(), numel_ * element_size(dtype_));
    return cloned;
}

// ---------------------------------------------------------
// Initializers
// ---------------------------------------------------------
Array Array::zeros(const Shape& shape, DataType dtype, DeviceType device) {
    Array r(shape, dtype, device);
    MT_DISPATCH_ALL_TYPES(dtype, T, [&]() {
        T* ptr = static_cast<T*>(r.raw_data());
        std::fill(ptr, ptr + r.numel(), static_cast<T>(0));
    });
    return r;
}

Array Array::ones(const Shape& shape, DataType dtype, DeviceType device) {
    Array r(shape, dtype, device);
    MT_DISPATCH_ALL_TYPES(dtype, T, [&]() {
        T* ptr = static_cast<T*>(r.raw_data());
        std::fill(ptr, ptr + r.numel(), static_cast<T>(1));
    });
    return r;
}

Array Array::randn(const Shape& shape, DataType dtype, DeviceType device) {
    Array              r(shape, dtype, device);
    std::random_device rd;
    std::mt19937       gen(rd());

    MT_DISPATCH_ALL_TYPES(dtype, T, [&]<typename U = T>() {
        U* ptr = static_cast<U*>(r.raw_data());

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

// ---------------------------------------------------------
// Element-wise operations
// ---------------------------------------------------------
Array Array::operator+(const Array& other) const {
    if (!defined() || !other.defined() || !shapes_equal(this->shape(), other.shape()) ||
        this->dtype() != other.dtype() || this->device() != other.device()) {
        return Array();
    }
    Array r(this->shape(), this->dtype(), this->device());
    MT_DISPATCH_ALL_TYPES(this->dtype(), T, [&]() {
        const T* a_ptr = static_cast<const T*>(this->raw_data());
        const T* b_ptr = static_cast<const T*>(other.raw_data());
        T*       r_ptr = static_cast<T*>(r.raw_data());
        for (std::size_t i = 0; i < this->numel(); ++i)
            r_ptr[i] = a_ptr[i] + b_ptr[i];
    });
    return r;
}

Array Array::operator-(const Array& other) const {
    if (!defined() || !other.defined() || !shapes_equal(this->shape(), other.shape()) || this->dtype() != other.dtype())
        return Array();
    Array r(this->shape(), this->dtype(), this->device());
    MT_DISPATCH_ALL_TYPES(this->dtype(), T, [&]() {
        const T* a_ptr = static_cast<const T*>(this->raw_data());
        const T* b_ptr = static_cast<const T*>(other.raw_data());
        T*       r_ptr = static_cast<T*>(r.raw_data());
        for (std::size_t i = 0; i < this->numel(); ++i)
            r_ptr[i] = a_ptr[i] - b_ptr[i];
    });
    return r;
}

Array Array::operator*(const Array& other) const {
    if (!defined() || !other.defined() || !shapes_equal(this->shape(), other.shape()) || this->dtype() != other.dtype())
        return Array();
    Array r(this->shape(), this->dtype(), this->device());
    MT_DISPATCH_ALL_TYPES(this->dtype(), T, [&]() {
        const T* a_ptr = static_cast<const T*>(this->raw_data());
        const T* b_ptr = static_cast<const T*>(other.raw_data());
        T*       r_ptr = static_cast<T*>(r.raw_data());
        for (std::size_t i = 0; i < this->numel(); ++i)
            r_ptr[i] = a_ptr[i] * b_ptr[i];
    });
    return r;
}

// ---------------------------------------------------------
// Scalar operations
// ---------------------------------------------------------
Array Array::operator+(double scalar) const {
    if (!defined())
        return Array();
    Array r(this->shape(), this->dtype(), this->device());
    MT_DISPATCH_ALL_TYPES(this->dtype(), T, [&]() {
        const T* a_ptr = static_cast<const T*>(this->raw_data());
        T*       r_ptr = static_cast<T*>(r.raw_data());
        T        s     = static_cast<T>(scalar);
        for (std::size_t i = 0; i < this->numel(); ++i)
            r_ptr[i] = a_ptr[i] + s;
    });
    return r;
}

Array Array::operator-(double scalar) const {
    if (!defined())
        return Array();
    Array r(this->shape(), this->dtype(), this->device());
    MT_DISPATCH_ALL_TYPES(this->dtype(), T, [&]() {
        const T* a_ptr = static_cast<const T*>(this->raw_data());
        T*       r_ptr = static_cast<T*>(r.raw_data());
        T        s     = static_cast<T>(scalar);
        for (std::size_t i = 0; i < this->numel(); ++i)
            r_ptr[i] = a_ptr[i] - s;
    });
    return r;
}

Array Array::operator*(double scalar) const {
    if (!defined())
        return Array();
    Array r(this->shape(), this->dtype(), this->device());
    MT_DISPATCH_ALL_TYPES(this->dtype(), T, [&]() {
        const T* a_ptr = static_cast<const T*>(this->raw_data());
        T*       r_ptr = static_cast<T*>(r.raw_data());
        T        s     = static_cast<T>(scalar);
        for (std::size_t i = 0; i < this->numel(); ++i)
            r_ptr[i] = a_ptr[i] * s;
    });
    return r;
}

Array Array::operator/(double scalar) const {
    if (!defined())
        return Array();
    Array r(this->shape(), this->dtype(), this->device());
    MT_DISPATCH_ALL_TYPES(this->dtype(), T, [&]() {
        const T* a_ptr = static_cast<const T*>(this->raw_data());
        T*       r_ptr = static_cast<T*>(r.raw_data());
        T        s     = static_cast<T>(scalar);
        for (std::size_t i = 0; i < this->numel(); ++i)
            r_ptr[i] = a_ptr[i] / s;
    });
    return r;
}

} // namespace mt