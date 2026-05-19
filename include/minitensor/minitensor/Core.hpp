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
#include <vector>

namespace mt
{
// ---------------------------------------------------------
// Basic Types
// ---------------------------------------------------------
enum class DataType {
    f16,
    f32,
    f64,
    bf16,
};

enum class DeviceType {
    cpu,
    cuda,
};

using Shape = std::vector<std::size_t>;

template <class T>
concept Index = std::convertible_to<T, std::size_t>;

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
    //
    // For a minimal library, a flat vector is sufficient for CPU.
    // To support CUDA later, we need to replace this with a
    // custom `Storage` struct that wraps a void* and a custom
    // device allocator.
    std::vector<float> data_;

    // ---------------------------------------------------------
    // Private Helpers
    // ---------------------------------------------------------
    void               compute_strides();
    static std::size_t shape_product(const Shape& s) noexcept;

  public:
    // ---------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------
    Array() noexcept;
    Array(const Array& other);
    Array(Array&& other) noexcept;
    explicit Array(Shape shape);
    explicit Array(std::vector<float> data);
    Array(std::vector<float> data, Shape shape);
    Array(std::vector<float> data, Shape shape, DataType dtype);
    Array(std::vector<float> data, Shape shape, DataType dtype, DeviceType device);

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

    // ---------------------------------------------------------
    // Indexing
    // ---------------------------------------------------------
    [[nodiscard]] float&       at(std::initializer_list<std::size_t> indices);
    [[nodiscard]] const float& at(std::initializer_list<std::size_t> indices) const;

    // ---------------------------------------------------------
    // Static Initializers
    // ---------------------------------------------------------
    static Array zeros(const Shape& shape);
    static Array ones(const Shape& shape);
    static Array randn(const Shape& shape);

    [[nodiscard]] Array grad() const;
    void                zero_grad();
    void                backward();
    void                backward(const Array& grad_output);

    [[nodiscard]] Array detach() const;

    [[nodiscard]] float item() const;

    [[nodiscard]] Array reshape(const Shape& new_shape) const;
    [[nodiscard]] Array flatten(std::size_t start_dim = 0, std::size_t end_dim = static_cast<std::size_t>(-1)) const;
    [[nodiscard]] Array transpose(std::size_t dim0, std::size_t dim1) const;
    [[nodiscard]] Array unsqueeze(std::size_t dim) const;
    [[nodiscard]] Array squeeze(std::optional<std::size_t> dim = std::nullopt) const;

    // Tensor-to-Tensor operations
    [[nodiscard]] Array operator+(const Array& other) const;
    [[nodiscard]] Array operator-(const Array& other) const;
    [[nodiscard]] Array operator*(const Array& other) const;
    [[nodiscard]] Array operator/(const Array& other) const;
    [[nodiscard]] Array operator-() const;

    // Tensor-to-Scalar operations
    [[nodiscard]] Array operator+(float scalar) const;
    [[nodiscard]] Array operator-(float scalar) const;
    [[nodiscard]] Array operator*(float scalar) const;
    [[nodiscard]] Array operator/(float scalar) const;

    [[nodiscard]] Array matmul(const Array& other) const;

    [[nodiscard]] Array relu() const;
    [[nodiscard]] Array sigmoid() const;
    [[nodiscard]] Array tanh() const;
    [[nodiscard]] Array exp() const;
    [[nodiscard]] Array log() const;

    [[nodiscard]] Array sum(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Array mean(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Array argmax(std::size_t dim) const;
    [[nodiscard]] Array softmax(std::size_t dim) const;
    [[nodiscard]] Array log_softmax(std::size_t dim) const;
};

// Convenience free functions
[[nodiscard]] Array zeros(const Shape& shape);
[[nodiscard]] Array ones(const Shape& shape);
[[nodiscard]] Array randn(const Shape& shape);
[[nodiscard]] Array reshape(const Array& input, const Shape& shape);

// Scalar-to-Tensor commutative operations
[[nodiscard]] inline Array operator+(float scalar, const Array& tensor) {
    return tensor + scalar;
}
[[nodiscard]] inline Array operator*(float scalar, const Array& tensor) {
    return tensor * scalar;
}

// Printing stream support
std::ostream& operator<<(std::ostream& os, const Array& tensor);
} // namespace mt