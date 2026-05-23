// include/minitensor/minitensor/Core.hpp

#pragma once

#ifndef _MINITENSOR_HPP_
#error "include minitensor/minitensor.hpp in your application, **not** minitensor/minitensor/Core.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
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

enum class DeviceType {
    cpu,
    cuda,
};

using Shape = std::vector<std::size_t>;

template <class T>
concept Index = std::convertible_to<T, std::size_t>;

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
    DataType   dtype_  = DataType::f32;
    DeviceType device_ = DeviceType::cpu;

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

    // Broadcasting: computes the output shape and returns per-dim broadcast
    // factors for both operands. Throws std::invalid_argument if shapes are
    // not broadcast-compatible.
    static Shape broadcast_shapes(const Shape& a, const Shape& b);
    Array        broadcast_to(const Shape& target_shape) const;

    void set_item_from_float(std::size_t index, float value);

  public:
    // ---------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------
    Array() noexcept;
    Array(const Array& other);
    Array(Array&& other) noexcept;
    explicit Array(Shape shape);
    explicit Array(std::vector<float> data);
    Array(std::vector<float> data, Shape shape, DataType dtype = DataType::f32, DeviceType device = DeviceType::cpu);
    Array(const void* data, Shape shape, DataType dtype = DataType::f32, DeviceType device = DeviceType::cpu);

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
    // Raw Data Pointer
    // ---------------------------------------------------------
    [[nodiscard]] float*       data() noexcept;
    [[nodiscard]] const float* data() const noexcept;
    [[nodiscard]] float        get_item_as_float(std::size_t index) const;

    // ---------------------------------------------------------
    // Indexing
    // ---------------------------------------------------------
    [[nodiscard]] float&       at(std::initializer_list<std::size_t> indices);
    [[nodiscard]] const float& at(std::initializer_list<std::size_t> indices) const;

    // ---------------------------------------------------------
    // Static Initializers
    // ---------------------------------------------------------
    [[nodiscard]] static Array zeros(const Shape& shape, DataType dtype = DataType::f32,
                                     DeviceType device = DeviceType::cpu);
    [[nodiscard]] static Array ones(const Shape& shape, DataType dtype = DataType::f32,
                                    DeviceType device = DeviceType::cpu);
    [[nodiscard]] static Array randn(const Shape& shape, DataType dtype = DataType::f32,
                                     DeviceType device = DeviceType::cpu);
    [[nodiscard]] static Array empty(const Shape& shape);
    [[nodiscard]] static Array full(const Shape& shape, float fill_value);
    [[nodiscard]] static Array eye(std::size_t n);
    [[nodiscard]] static Array arange(float start, float stop, float step = 1.0f);
    [[nodiscard]] static Array linspace(float start, float stop, std::size_t num);

    // ---------------------------------------------------------
    // Array utilities
    // ---------------------------------------------------------
    [[nodiscard]] float item() const;
    [[nodiscard]] Array clone() const;

    // ---------------------------------------------------------
    // Array Manipulation
    // ---------------------------------------------------------
    [[nodiscard]] Array reshape(const Shape& new_shape) const;
    [[nodiscard]] Array flatten(std::size_t start_dim = 0, std::size_t end_dim = static_cast<std::size_t>(-1)) const;
    [[nodiscard]] Array transpose(std::size_t dim0, std::size_t dim1) const;
    [[nodiscard]] Array unsqueeze(std::size_t dim) const;
    [[nodiscard]] Array squeeze(std::optional<std::size_t> dim = std::nullopt) const;

    // ---------------------------------------------------------
    // Element-wise operations (broadcasting supported)
    // ---------------------------------------------------------
    [[nodiscard]] Array operator+(const Array& other) const;
    [[nodiscard]] Array operator-(const Array& other) const;
    [[nodiscard]] Array operator*(const Array& other) const;
    [[nodiscard]] Array operator/(const Array& other) const;
    [[nodiscard]] Array operator-() const;

    // ---------------------------------------------------------
    // Scalar operations
    // ---------------------------------------------------------
    [[nodiscard]] Array operator+(float scalar) const;
    [[nodiscard]] Array operator-(float scalar) const;
    [[nodiscard]] Array operator*(float scalar) const;
    [[nodiscard]] Array operator/(float scalar) const;

    // ---------------------------------------------------------
    // In-place operations
    // ---------------------------------------------------------
    Array& operator+=(const Array& other);
    Array& operator-=(const Array& other);
    Array& operator*=(const Array& other);
    Array& operator/=(const Array& other);
    Array& operator+=(float scalar);
    Array& operator-=(float scalar);
    Array& operator*=(float scalar);
    Array& operator/=(float scalar);
    Array& fill_(float value);

    // ---------------------------------------------------------
    // Comparison operators (return float 0.0 / 1.0)
    // ---------------------------------------------------------
    [[nodiscard]] Array operator==(const Array& other) const;
    [[nodiscard]] Array operator!=(const Array& other) const;
    [[nodiscard]] Array operator<(const Array& other) const;
    [[nodiscard]] Array operator<=(const Array& other) const;
    [[nodiscard]] Array operator>(const Array& other) const;
    [[nodiscard]] Array operator>=(const Array& other) const;

    [[nodiscard]] Array matmul(const Array& other) const;

    // ---------------------------------------------------------
    // Unary math
    // ---------------------------------------------------------
    [[nodiscard]] Array relu() const;
    [[nodiscard]] Array sigmoid() const;
    [[nodiscard]] Array tanh() const;
    [[nodiscard]] Array exp() const;
    [[nodiscard]] Array log() const;
    [[nodiscard]] Array abs() const;
    [[nodiscard]] Array sqrt() const;
    [[nodiscard]] Array pow(float exponent) const;
    [[nodiscard]] Array clip(float min, float max) const;

    // ---------------------------------------------------------
    // Reductions
    // ---------------------------------------------------------
    [[nodiscard]] Array sum(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Array mean(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Array max(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Array min(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Array var(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Array std(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Array norm(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Array cumsum(std::size_t dim) const;
    [[nodiscard]] Array argmax(std::size_t dim) const;
    [[nodiscard]] Array argmin(std::size_t dim) const;
    [[nodiscard]] Array all(std::optional<std::size_t> dim = std::nullopt) const;
    [[nodiscard]] Array any(std::optional<std::size_t> dim = std::nullopt) const;

    // ---------------------------------------------------------
    // Activations
    // ---------------------------------------------------------
    [[nodiscard]] Array softmax(std::size_t dim) const;
    [[nodiscard]] Array log_softmax(std::size_t dim) const;
};

// Convenience free functions
[[nodiscard]] Array zeros(const Shape& shape, DataType dtype = DataType::f32, DeviceType device = DeviceType::cpu);
[[nodiscard]] Array ones(const Shape& shape, DataType dtype = DataType::f32, DeviceType device = DeviceType::cpu);
[[nodiscard]] Array randn(const Shape& shape, DataType dtype = DataType::f32, DeviceType device = DeviceType::cpu);
[[nodiscard]] Array empty(const Shape& shape);
[[nodiscard]] Array full(const Shape& shape, float fill_value);
[[nodiscard]] Array eye(std::size_t n);
[[nodiscard]] Array arange(float start, float stop, float step = 1.0f);
[[nodiscard]] Array linspace(float start, float stop, std::size_t num);

[[nodiscard]] Array reshape(const Array& input, const Shape& shape);

[[nodiscard]] Array cat(std::span<const Array> arrays, std::size_t dim = 0);
[[nodiscard]] Array stack(std::span<const Array> arrays, std::size_t dim = 0);
[[nodiscard]] float dot(const Array& a, const Array& b);

// Scalar-to-Array commutative operations
[[nodiscard]] inline Array operator+(float scalar, const Array& arr) {
    return arr + scalar;
}
[[nodiscard]] inline Array operator-(float scalar, const Array& arr) {
    return (arr * -1.0f) + scalar;
}
[[nodiscard]] inline Array operator*(float scalar, const Array& arr) {
    return arr * scalar;
}
[[nodiscard]] inline Array operator/(float scalar, const Array& arr) {
    return arr.pow(-1.0f) * scalar;
}

// Printing stream support
std::ostream& operator<<(std::ostream& os, const Array& tensor);
} // namespace mt