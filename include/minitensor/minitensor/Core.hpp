// include/minitensor/minitensor/Core.hpp

#pragma once

#ifndef _MINITENSOR_HPP_
#error "include minitensor/minitensor.hpp in your application, **not** minitensor/minitensor/Core.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace mt
{
// ---------------------------------------------------------
// Basic Types
// ---------------------------------------------------------
enum class DataType {
    i32,
    i64,
    f32,
    f64,
};

enum class DeviceType {
    cpu,
    cuda,
};

// ---------------------------------------------------------
// DataType Type Traits
// ---------------------------------------------------------
template <DataType D>
struct DataTypeToType;

template <>
struct DataTypeToType<DataType::i32> {
    using type = int32_t;
};
template <>
struct DataTypeToType<DataType::i64> {
    using type = int64_t;
};
template <>
struct DataTypeToType<DataType::f32> {
    using type = float;
};
template <>
struct DataTypeToType<DataType::f64> {
    using type = double;
};

template <DataType D>
using DataTypeToType_t = typename DataTypeToType<D>::type;

template <typename T>
struct TypeToDataType;

template <>
struct TypeToDataType<int> {
    static constexpr DataType value = DataType::i32;
};
template <>
struct TypeToDataType<long> {
    static constexpr DataType value = sizeof(long) == 8 ? DataType::i64 : DataType::i32;
};
template <>
struct TypeToDataType<long long> {
    static constexpr DataType value = DataType::i64;
};
template <>
struct TypeToDataType<float> {
    static constexpr DataType value = DataType::f32;
};
template <>
struct TypeToDataType<double> {
    static constexpr DataType value = DataType::f64;
};

constexpr std::size_t element_size(DataType dtype) noexcept {
    switch (dtype) {
    case DataType::i32:
        return 4;
    case DataType::i64:
        return 8;
    case DataType::f32:
        return 4;
    case DataType::f64:
        return 8;
    }
    return 0;
}

using Shape = std::vector<std::size_t>;

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
    bool       is_view_ = false;
    size_t     numel_   = 0;
    Shape      shape_;
    Shape      strides_;
    Shape      offsets_;
    DataType   dtype_;
    DeviceType device_;

    // ---------------------------------------------------------
    // Data Storage
    // ---------------------------------------------------------
    Storage* data_ = nullptr;

    // ---------------------------------------------------------
    // Private Helpers
    // ---------------------------------------------------------
    void               compute_strides();
    static std::size_t shape_product(const Shape& s) noexcept;
    static bool        shapes_equal(const Shape& a, const Shape& b) noexcept;

    // Broadcasting
    static Shape broadcast_shapes(const Shape& a, const Shape& b);
    Array        broadcast_to(const Shape& target_shape) const;

    // Private functions to manage storage
    void        allocate_storage();
    void*       raw_data() noexcept;
    const void* raw_data() const noexcept;

  public:
    // ---------------------------------------------------------
    // Constructors & Destructors
    // ---------------------------------------------------------
    Array() noexcept;
    Array(const Array& other);
    Array(Array&& other) noexcept;
    ~Array();

    // Assignment Operators
    Array& operator=(const Array& other);
    Array& operator=(Array&& other) noexcept;

    // Internal generic constructor
    Array(Shape shape, DataType dtype = DataType::f32, DeviceType device = DeviceType::cpu);
    Array(const void* data, Shape shape, DataType dtype, DeviceType device = DeviceType::cpu);

    // Type-Safe Templated Constructors
    template <typename T>
    Array(std::span<const T> data, Shape shape, DeviceType device = DeviceType::cpu)
        : defined_(true), shape_(std::move(shape)), dtype_(TypeToDataType<T>::value), device_(device) {
        numel_ = shape_product(shape_);
        compute_strides();
        allocate_storage();
        std::memcpy(raw_data(), data.data(), numel_ * sizeof(T));
    }

    template <typename T>
    explicit Array(const std::vector<T>& data, DeviceType device = DeviceType::cpu)
        : Array(std::span<const T>(data), {data.size()}, device) {
    }

    // ---------------------------------------------------------
    // Metadata Getter
    // ---------------------------------------------------------
    [[nodiscard]] bool         defined() const noexcept;
    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] const Shape& strides() const noexcept;
    [[nodiscard]] std::size_t  ndim() const noexcept;
    [[nodiscard]] std::size_t  numel() const noexcept;
    [[nodiscard]] DataType     dtype() const noexcept;
    [[nodiscard]] DeviceType   device() const noexcept;

    // ---------------------------------------------------------
    // Type-Safe Indexing & Accessors
    // ---------------------------------------------------------
    template <typename T>
    [[nodiscard]] T* data() {
        if (dtype_ != TypeToDataType<T>::value)
            throw std::runtime_error("Type mismatch in data()");
        return static_cast<T*>(raw_data());
    }

    template <typename T>
    [[nodiscard]] const T* data() const {
        if (dtype_ != TypeToDataType<T>::value)
            throw std::runtime_error("Type mismatch in data()");
        return static_cast<const T*>(raw_data());
    }

    template <typename T>
    [[nodiscard]] T& at(std::initializer_list<std::size_t> indices) {
        if (dtype_ != TypeToDataType<T>::value)
            throw std::runtime_error("Type mismatch in at()");
        if (indices.size() != shape_.size())
            throw std::out_of_range("Dimension mismatch in at()");
        std::size_t pos = 0, i = 0;
        for (auto idx : indices) {
            if (idx >= shape_[i])
                throw std::out_of_range("Index out of bounds in at()");
            pos += idx * strides_[i++];
        }
        return static_cast<T*>(raw_data())[pos];
    }

    template <typename T>
    [[nodiscard]] const T& at(std::initializer_list<std::size_t> indices) const {
        if (dtype_ != TypeToDataType<T>::value)
            throw std::runtime_error("Type mismatch in at()");
        if (indices.size() != shape_.size())
            throw std::out_of_range("Dimension mismatch in at()");
        std::size_t pos = 0, i = 0;
        for (auto idx : indices) {
            if (idx >= shape_[i])
                throw std::out_of_range("Index out of bounds in at()");
            pos += idx * strides_[i++];
        }
        return static_cast<const T*>(raw_data())[pos];
    }

    template <typename T>
    [[nodiscard]] T item() const {
        if (!defined())
            throw std::runtime_error("Cannot call item() on undefined Array.");
        if (numel() != 1)
            throw std::runtime_error("item() is only valid for 1-element arrays.");
        if (dtype_ != TypeToDataType<T>::value)
            throw std::runtime_error("Type mismatch in item()");
        return static_cast<const T*>(raw_data())[0];
    }

    // ---------------------------------------------------------
    // Clone, cast, and move device
    // ---------------------------------------------------------
    [[nodiscard]] Array clone() const;
    [[nodiscard]] Array cast(DataType dtype) const;
    [[nodiscard]] Array to(DeviceType device) const;

    // ---------------------------------------------------------
    // Static Initializers
    // ---------------------------------------------------------
    [[nodiscard]] static Array zeros(const Shape& shape, DataType dtype = DataType::f32,
                                     DeviceType device = DeviceType::cpu);
    [[nodiscard]] static Array ones(const Shape& shape, DataType dtype = DataType::f32,
                                    DeviceType device = DeviceType::cpu);
    [[nodiscard]] static Array randn(const Shape& shape, DataType dtype = DataType::f32,
                                     DeviceType device = DeviceType::cpu);
    template <typename T>
    [[nodiscard]] static Array full(const Shape& shape, T fill_value, DataType dtype = TypeToDataType<T>::value,
                                    DeviceType device = DeviceType::cpu);

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

// Convenience free functions
[[nodiscard]] Array zeros(const Shape& shape, DataType dtype = DataType::f32, DeviceType device = DeviceType::cpu);
[[nodiscard]] Array ones(const Shape& shape, DataType dtype = DataType::f32, DeviceType device = DeviceType::cpu);
[[nodiscard]] Array randn(const Shape& shape, DataType dtype = DataType::f32, DeviceType device = DeviceType::cpu);
template <typename T>
[[nodiscard]] inline Array full(const Shape& shape, T fill_value, DataType dtype = TypeToDataType<T>::value,
                                DeviceType device = DeviceType::cpu) {
    return Array::full(shape, fill_value, dtype, device);
}

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

[[nodiscard]] Array add(const Array& a, const Array& b);
[[nodiscard]] Array subtract(const Array& a, const Array& b);
[[nodiscard]] Array multiply(const Array& a, const Array& b);
[[nodiscard]] Array divide(const Array& a, const Array& b);

} // namespace mt