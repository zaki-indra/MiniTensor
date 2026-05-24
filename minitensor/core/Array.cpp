// minitensor/core/Array.cpp

#include "Macros.hpp"
#include "Storage.hpp"

#include <cstring>
#include <minitensor/minitensor.hpp>
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
    data_ = new Storage(numel_, dtype_, device_);
}

void* Array::raw_data() noexcept {
    return data_ ? data_->data() : nullptr;
}

const void* Array::raw_data() const noexcept {
    return data_ ? data_->data() : nullptr;
}

// ---------------------------------------------------------
// Constructors & Destructors
// ---------------------------------------------------------
Array::Array() noexcept = default;

Array::Array(const Array& other)
    : defined_(other.defined_), is_view_(other.is_view_), numel_(other.numel_), shape_(other.shape_),
      strides_(other.strides_), offsets_(other.offsets_), dtype_(other.dtype_), device_(other.device_),
      data_(other.data_) {
    if (data_) {
        data_->retain();
    }
}

Array::Array(Array&& other) noexcept
    : defined_(other.defined_), is_view_(other.is_view_), numel_(other.numel_), shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)), offsets_(std::move(other.offsets_)), dtype_(other.dtype_),
      device_(other.device_), data_(other.data_) {
    other.data_    = nullptr;
    other.defined_ = false;
    other.numel_   = 0;
}

Array::~Array() {
    if (data_) {
        data_->release();
    }
}

// ---------------------------------------------------------
// Assignment Operators
// ---------------------------------------------------------
Array& Array::operator=(const Array& other) {
    if (this != &other) {
        if (data_) {
            data_->release();
        }
        defined_ = other.defined_;
        is_view_ = other.is_view_;
        numel_   = other.numel_;
        shape_   = other.shape_;
        strides_ = other.strides_;
        offsets_ = other.offsets_;
        dtype_   = other.dtype_;
        device_  = other.device_;
        data_    = other.data_;
        if (data_) {
            data_->retain();
        }
    }
    return *this;
}

Array& Array::operator=(Array&& other) noexcept {
    if (this != &other) {
        if (data_) {
            data_->release();
        }
        defined_ = other.defined_;
        is_view_ = other.is_view_;
        numel_   = other.numel_;
        shape_   = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        offsets_ = std::move(other.offsets_);
        dtype_   = other.dtype_;
        device_  = other.device_;
        data_    = other.data_;

        other.data_    = nullptr;
        other.defined_ = false;
        other.numel_   = 0;
    }
    return *this;
}

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
        throw std::runtime_error("CUDA device is not supported on this platform.");
    }
    Array copied(shape_, dtype_, target_device);
    std::memcpy(copied.raw_data(), this->raw_data(), numel_ * element_size(dtype_));
    return copied;
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

template <typename T>
Array Array::full(const Shape& shape, T fill_value, DataType dtype, DeviceType device) {
    Array r(shape, dtype, device);
    MT_DISPATCH_ALL_TYPES(dtype, U, [&]() {
        U* ptr = static_cast<U*>(r.raw_data());
        std::fill(ptr, ptr + r.numel(), static_cast<U>(fill_value));
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

Array Array::reshape(const Shape& new_shape) const {
    if (!defined())
        throw std::runtime_error("Cannot reshape undefined Array.");

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

    if (neg_one_count > 1) {
        throw std::runtime_error("Only one dimension can be -1 in reshape.");
    }

    if (neg_one_count == 1) {
        if (product == 0 || numel_ % product != 0) {
            throw std::runtime_error("Invalid shape for reshape with -1.");
        }
        target_shape[neg_one_idx] = numel_ / product;
        product                   = numel_;
    }

    if (product != numel_) {
        throw std::runtime_error("Shape mismatch in reshape: number of elements must remain the same.");
    }

    // Create a new Array sharing the same storage
    Array reshaped(*this);
    reshaped.shape_ = target_shape;
    reshaped.compute_strides();
    return reshaped;
}

Array Array::flatten(std::size_t start_dim, std::size_t end_dim) const {
    if (!defined())
        throw std::runtime_error("Cannot flatten undefined Array.");

    std::size_t nd = ndim();
    if (nd == 0) {
        return reshape({1});
    }

    if (end_dim == static_cast<std::size_t>(-1)) {
        end_dim = nd - 1;
    }

    if (start_dim > end_dim || end_dim >= nd) {
        throw std::runtime_error("Invalid dimensions for flatten.");
    }

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
// Unary Mathematical Functions
// ---------------------------------------------------------

Array sin(const Array& arr) {
    if (!arr.defined())
        return Array();
    Array r(arr.shape(), arr.dtype(), arr.device());
    MT_DISPATCH_ALL_TYPES(arr.dtype(), T, [&]() {
        const T* src = arr.data<T>();
        T*       dst = r.data<T>();
        for (std::size_t i = 0; i < arr.numel(); ++i) {
            dst[i] = static_cast<T>(std::sin(static_cast<double>(src[i])));
        }
    });
    return r;
}

Array cos(const Array& arr) {
    if (!arr.defined())
        return Array();
    Array r(arr.shape(), arr.dtype(), arr.device());
    MT_DISPATCH_ALL_TYPES(arr.dtype(), T, [&]() {
        const T* src = arr.data<T>();
        T*       dst = r.data<T>();
        for (std::size_t i = 0; i < arr.numel(); ++i) {
            dst[i] = static_cast<T>(std::cos(static_cast<double>(src[i])));
        }
    });
    return r;
}

Array tan(const Array& arr) {
    if (!arr.defined())
        return Array();
    Array r(arr.shape(), arr.dtype(), arr.device());
    MT_DISPATCH_ALL_TYPES(arr.dtype(), T, [&]() {
        const T* src = arr.data<T>();
        T*       dst = r.data<T>();
        for (std::size_t i = 0; i < arr.numel(); ++i) {
            dst[i] = static_cast<T>(std::tan(static_cast<double>(src[i])));
        }
    });
    return r;
}

Array exp(const Array& arr) {
    if (!arr.defined())
        return Array();
    Array r(arr.shape(), arr.dtype(), arr.device());
    MT_DISPATCH_ALL_TYPES(arr.dtype(), T, [&]() {
        const T* src = arr.data<T>();
        T*       dst = r.data<T>();
        for (std::size_t i = 0; i < arr.numel(); ++i) {
            dst[i] = static_cast<T>(std::exp(static_cast<double>(src[i])));
        }
    });
    return r;
}

Array log(const Array& arr) {
    if (!arr.defined())
        return Array();
    Array r(arr.shape(), arr.dtype(), arr.device());
    MT_DISPATCH_ALL_TYPES(arr.dtype(), T, [&]() {
        const T* src = arr.data<T>();
        T*       dst = r.data<T>();
        for (std::size_t i = 0; i < arr.numel(); ++i) {
            dst[i] = static_cast<T>(std::log(static_cast<double>(src[i])));
        }
    });
    return r;
}

Array sqrt(const Array& arr) {
    if (!arr.defined())
        return Array();
    Array r(arr.shape(), arr.dtype(), arr.device());
    MT_DISPATCH_ALL_TYPES(arr.dtype(), T, [&]() {
        const T* src = arr.data<T>();
        T*       dst = r.data<T>();
        for (std::size_t i = 0; i < arr.numel(); ++i) {
            if constexpr (std::is_integral_v<T>) {
                if (src[i] < 0) {
                    throw std::runtime_error("Square root of negative integer is undefined.");
                }
            }
            dst[i] = static_cast<T>(std::sqrt(static_cast<double>(src[i])));
        }
    });
    return r;
}

// ---------------------------------------------------------
// Binary Element-Wise Operations
// ---------------------------------------------------------

Array add(const Array& a, const Array& b) {
    return a + b;
}

Array subtract(const Array& a, const Array& b) {
    return a - b;
}

Array multiply(const Array& a, const Array& b) {
    return a * b;
}

Array divide(const Array& a, const Array& b) {
    if (!a.defined() || !b.defined() || a.shape() != b.shape() || a.dtype() != b.dtype() || a.device() != b.device()) {
        return Array();
    }
    Array r(a.shape(), a.dtype(), a.device());
    MT_DISPATCH_ALL_TYPES(a.dtype(), T, [&]() {
        const T* a_ptr = a.data<T>();
        const T* b_ptr = b.data<T>();
        T*       r_ptr = r.data<T>();
        for (std::size_t i = 0; i < a.numel(); ++i) {
            if (b_ptr[i] == 0) {
                if constexpr (std::is_integral_v<T>) {
                    throw std::runtime_error("Division by zero in integer division.");
                }
            }
            r_ptr[i] = a_ptr[i] / b_ptr[i];
        }
    });
    return r;
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