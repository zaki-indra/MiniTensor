// include/minitensor/core.hpp
#pragma once

#include "Error.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mt
{
// ---------------------------------------------------------
// Internal helpers (public header — kept minimal on purpose).
// The full error-reporting machinery (MT_CHECK / MT_THROW) lives in the
// private header minitensor/core/ErrorMacros.hpp and must not appear in
// any public header. These helpers exist only so the public template
// methods below can throw without dragging in macros.
// ---------------------------------------------------------
namespace detail
{
template <typename... Args>
inline std::string format_msg(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    return oss.str();
}
} // namespace detail

// ---------------------------------------------------------
// Basic Types
// ---------------------------------------------------------
enum class EDataType {
    i32,
    i64,
    f32,
    f64,
};

enum class EDeviceType {
    cpu,
    cuda,
};

// ---------------------------------------------------------
// Device Tag Types
// ---------------------------------------------------------
// Empty tag types used as the compile-time alias target inside device
// dispatch macros. Kernel authors can specialize per device with
//   if constexpr (std::is_same_v<Dev, mt::CpuDevice>) { ... }
// while staying inside a single dispatch site.
struct CpuDevice {};
struct CudaDevice {};

// ---------------------------------------------------------
// DataType Type Traits
// ---------------------------------------------------------
template <EDataType D>
struct DataTypeToType;

template <>
struct DataTypeToType<EDataType::i32> {
    using type = int32_t;
};
template <>
struct DataTypeToType<EDataType::i64> {
    using type = int64_t;
};
template <>
struct DataTypeToType<EDataType::f32> {
    using type = float;
};
template <>
struct DataTypeToType<EDataType::f64> {
    using type = double;
};

template <EDataType D>
using DataTypeToType_t = typename DataTypeToType<D>::type;

template <typename T>
struct TypeToDataType;

template <>
struct TypeToDataType<int> {
    static constexpr EDataType value = EDataType::i32;
};
template <>
struct TypeToDataType<long> {
    static constexpr EDataType value = sizeof(long) == 8 ? EDataType::i64 : EDataType::i32;
};
template <>
struct TypeToDataType<long long> {
    static constexpr EDataType value = EDataType::i64;
};
template <>
struct TypeToDataType<float> {
    static constexpr EDataType value = EDataType::f32;
};
template <>
struct TypeToDataType<double> {
    static constexpr EDataType value = EDataType::f64;
};

constexpr std::size_t element_size(EDataType dtype) noexcept {
    switch (dtype) {
    case EDataType::i32:
        return 4;
    case EDataType::i64:
        return 8;
    case EDataType::f32:
        return 4;
    case EDataType::f64:
        return 8;
    }
    return 0;
}

using Shape = std::vector<std::size_t>;

std::ostream& operator<<(std::ostream& os, EDataType dtype);
std::ostream& operator<<(std::ostream& os, EDeviceType device);
std::ostream& operator<<(std::ostream& os, const Shape& shape);

// ---------------------------------------------------------
// Forward Declarations
// ---------------------------------------------------------
class Storage;

// ---------------------------------------------------------
// Core Array Types
// ---------------------------------------------------------
class Array {
  private:
    // ---------------------------------------------------------
    // Metadata
    // ---------------------------------------------------------
    bool       defined_ = false;
    size_t     numel_   = 0;
    Shape      shape_;
    Shape      strides_;
    // Defaulted so default-constructed (undefined) Arrays have valid
    // metadata reads. Without this, code that derives a child Array via
    // `Array r(a.shape(), a.dtype(), a.device())` would propagate
    // indeterminate enum values into the Storage allocator.
    EDataType   dtype_  = EDataType::f32;
    EDeviceType device_ = EDeviceType::cpu;

    // ---------------------------------------------------------
    // Data Storage
    // ---------------------------------------------------------
    std::shared_ptr<Storage> data_;

    // ---------------------------------------------------------
    // Private Helpers
    // ---------------------------------------------------------
    void               compute_strides();
    static std::size_t shape_product(const Shape& s) noexcept;
    static bool        shapes_equal(const Shape& a, const Shape& b) noexcept;

    // Private functions to manage storage
    void        allocate_storage();
    void*       raw_data() noexcept;
    const void* raw_data() const noexcept;

  public:
    // ---------------------------------------------------------
    // Constructors & Destructors
    // ---------------------------------------------------------
    Array() noexcept          = default;
    Array(const Array& other) = default;
    Array(Array&& other) noexcept
        : defined_(std::exchange(other.defined_, false)), numel_(std::exchange(other.numel_, 0)),
          shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), dtype_(other.dtype_),
          device_(other.device_), data_(std::move(other.data_)) {
    }
    ~Array() = default;

    // Assignment Operators
    Array& operator=(const Array& other) = default;
    Array& operator=(Array&& other) noexcept {
        if (this != &other) {
            defined_ = std::exchange(other.defined_, false);
            numel_   = std::exchange(other.numel_, 0);
            shape_   = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            dtype_   = other.dtype_;
            device_  = other.device_;
            data_    = std::move(other.data_);
        }
        return *this;
    }

    // Internal generic constructor
    Array(Shape shape, EDataType dtype = EDataType::f32, EDeviceType device = EDeviceType::cpu);
    Array(const void* data, Shape shape, EDataType dtype, EDeviceType device = EDeviceType::cpu);

    // Type-Safe Templated Constructors
    template <typename T>
    Array(std::span<const T> data, Shape shape, EDeviceType device = EDeviceType::cpu)
        : defined_(true), shape_(std::move(shape)), dtype_(TypeToDataType<T>::value), device_(device) {
        numel_ = shape_product(shape_);
        compute_strides();
        allocate_storage();
        std::memcpy(raw_data(), data.data(), numel_ * sizeof(T));
    }

    template <typename T>
    explicit Array(const std::vector<T>& data, EDeviceType device = EDeviceType::cpu)
        : Array(std::span<const T>(data), {data.size()}, device) {
    }

    // ---------------------------------------------------------
    // Metadata Getter
    // ---------------------------------------------------------
    [[nodiscard]] bool defined() const noexcept {
        return defined_;
    }
    [[nodiscard]] const Shape& shape() const noexcept {
        return shape_;
    }
    [[nodiscard]] const Shape& strides() const noexcept {
        return strides_;
    }
    [[nodiscard]] std::size_t ndim() const noexcept {
        return shape_.size();
    }
    [[nodiscard]] std::size_t numel() const noexcept {
        return numel_;
    }
    [[nodiscard]] EDataType dtype() const noexcept {
        return dtype_;
    }
    [[nodiscard]] EDeviceType device() const noexcept {
        return device_;
    }

    // ---------------------------------------------------------
    // Type-Safe Indexing & Accessors
    // ---------------------------------------------------------
    template <typename T>
    [[nodiscard]] T* data() {
        if (dtype_ != TypeToDataType<T>::value)
            throw mt::TypeError("Type mismatch in data(): requested C++ type does not match dynamic tensor DataType.");
        return static_cast<T*>(raw_data());
    }

    template <typename T>
    [[nodiscard]] const T* data() const {
        if (dtype_ != TypeToDataType<T>::value)
            throw mt::TypeError("Type mismatch in data(): requested C++ type does not match dynamic tensor DataType.");
        return static_cast<const T*>(raw_data());
    }

    template <typename T>
    [[nodiscard]] T& at(std::initializer_list<std::size_t> indices) {
        if (dtype_ != TypeToDataType<T>::value)
            throw mt::TypeError("Type mismatch in at(): requested C++ type does not match dynamic tensor DataType.");
        if (indices.size() != shape_.size())
            throw mt::ShapeError(detail::format_msg("Dimension mismatch in at(): expected ", shape_.size(),
                                                    " indices, got ", indices.size()));
        std::size_t pos = 0, i = 0;
        for (auto idx : indices) {
            if (idx >= shape_[i])
                throw mt::IndexError(
                  detail::format_msg("Index ", idx, " is out of bounds for dimension ", i, " with size ", shape_[i]));
            pos += idx * strides_[i++];
        }
        return static_cast<T*>(raw_data())[pos];
    }

    template <typename T>
    [[nodiscard]] const T& at(std::initializer_list<std::size_t> indices) const {
        if (dtype_ != TypeToDataType<T>::value)
            throw mt::TypeError("Type mismatch in at(): requested C++ type does not match dynamic tensor DataType.");
        if (indices.size() != shape_.size())
            throw mt::ShapeError(detail::format_msg("Dimension mismatch in at(): expected ", shape_.size(),
                                                    " indices, got ", indices.size()));
        std::size_t pos = 0, i = 0;
        for (auto idx : indices) {
            if (idx >= shape_[i])
                throw mt::IndexError(
                  detail::format_msg("Index ", idx, " is out of bounds for dimension ", i, " with size ", shape_[i]));
            pos += idx * strides_[i++];
        }
        return static_cast<const T*>(raw_data())[pos];
    }

    template <typename T>
    [[nodiscard]] T item() const {
        if (!defined())
            throw mt::UndefinedError("Tensor is not defined (has no storage).");
        if (numel() != 1)
            throw mt::ShapeError(
              detail::format_msg("item() is only valid for 1-element arrays, but array has ", numel(), " elements."));
        if (dtype_ != TypeToDataType<T>::value)
            throw mt::TypeError("Type mismatch in item(): requested C++ type does not match dynamic tensor DataType.");
        return static_cast<const T*>(raw_data())[0];
    }

    // ---------------------------------------------------------
    // Clone, cast, and move device
    // ---------------------------------------------------------
    [[nodiscard]] Array clone() const;
    [[nodiscard]] Array cast(EDataType dtype) const;
    [[nodiscard]] Array to(EDeviceType device) const;

    // ---------------------------------------------------------
    // Static Initializers
    // ---------------------------------------------------------
    [[nodiscard]] static Array zeros(const Shape& shape, EDataType dtype = EDataType::f32,
                                     EDeviceType device = EDeviceType::cpu);
    [[nodiscard]] static Array ones(const Shape& shape, EDataType dtype = EDataType::f32,
                                    EDeviceType device = EDeviceType::cpu);
    [[nodiscard]] static Array randn(const Shape& shape, EDataType dtype = EDataType::f32,
                                     EDeviceType device = EDeviceType::cpu);
    template <typename T>
    [[nodiscard]] static Array full(const Shape& shape, T fill_value, EDataType dtype = TypeToDataType<T>::value,
                                    EDeviceType device = EDeviceType::cpu);

    // ---------------------------------------------------------
    // Array Manipulation
    // ---------------------------------------------------------
    [[nodiscard]] Array reshape(const Shape& new_shape) const;
    [[nodiscard]] Array flatten(std::size_t start_dim = 0, std::size_t end_dim = static_cast<std::size_t>(-1)) const;

    // ---------------------------------------------------------
    // Element-wise operations
    // ---------------------------------------------------------
    [[nodiscard]] Array operator+(const Array& other) const;
    [[nodiscard]] Array operator-(const Array& other) const;
    [[nodiscard]] Array operator*(const Array& other) const;

    // ---------------------------------------------------------
    // Scalar operations
    // ---------------------------------------------------------
    [[nodiscard]] Array operator+(double scalar) const;
    [[nodiscard]] Array operator-(double scalar) const;
    [[nodiscard]] Array operator*(double scalar) const;
    [[nodiscard]] Array operator/(double scalar) const;
};

// Scalar-to-Array commutative operations
[[nodiscard]] inline Array operator+(double scalar, const Array& arr) {
    return arr + scalar;
}
[[nodiscard]] inline Array operator*(double scalar, const Array& arr) {
    return arr * scalar;
}
[[nodiscard]] inline Array operator-(double scalar, const Array& arr) {
    return (arr * -1.0) + scalar;
}

// ---------------------------------------------------------
// Mathematical functions
// ---------------------------------------------------------
[[nodiscard]] Array sin(const Array& arr);
[[nodiscard]] Array cos(const Array& arr);
[[nodiscard]] Array tan(const Array& arr);

[[nodiscard]] Array exp(const Array& arr);
[[nodiscard]] Array log(const Array& arr);
[[nodiscard]] Array sqrt(const Array& arr);
[[nodiscard]] Array divide(const Array& a, const Array& b);

// Free factory functions
[[nodiscard]] Array zeros(const Shape& shape, EDataType dtype = EDataType::f32, EDeviceType device = EDeviceType::cpu);
[[nodiscard]] Array ones(const Shape& shape, EDataType dtype = EDataType::f32, EDeviceType device = EDeviceType::cpu);
[[nodiscard]] Array randn(const Shape& shape, EDataType dtype = EDataType::f32, EDeviceType device = EDeviceType::cpu);
template <typename T>
[[nodiscard]] Array full(const Shape& shape, T fill_value, EDataType dtype = TypeToDataType<T>::value,
                         EDeviceType device = EDeviceType::cpu);

} // namespace mt
