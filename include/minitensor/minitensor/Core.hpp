// include/minitensor/minitensor/Tensor.hpp

#pragma once

#ifndef _MINITENSOR_HPP_
#error "include minitensor/minitensor.h in your application, **not** minitensor/minitensor/Core.h"
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
  public:
    // ---------------------------------------------------------
    // Data Storage
    // ---------------------------------------------------------
    // For a minimal library, a flat vector is sufficient for CPU.
    // To support CUDA later, you would replace this with a custom `Storage`
    // struct that wraps a void* and a custom device allocator.
    std::vector<float> data_;

    // ---------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------
    Array() noexcept;
    explicit Array(Shape shape);
    Array(std::vector<float> data, Shape shape);
    Array(std::initializer_list<float> values);

    // ---------------------------------------------------------
    // Indexing
    // ---------------------------------------------------------
    [[nodiscard]] float&       at(std::initializer_list<std::size_t> indices);
    [[nodiscard]] const float& at(std::initializer_list<std::size_t> indices) const;

    // ---------------------------------------------------------
    // Metadata Getter
    // ---------------------------------------------------------
    [[nodiscard]] bool         defined() const noexcept;
    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] std::size_t  ndim() const noexcept;
    [[nodiscard]] std::size_t  numel() const noexcept;
    [[nodiscard]] DataType     dtype() const noexcept;
    [[nodiscard]] DeviceType   device() const noexcept;

    static Array zeros(const Shape& shape);
    static Array ones(const Shape& shape);
    static Array randn(const Shape& shape);

    [[nodiscard]] float*       data() noexcept;
    [[nodiscard]] const float* data() const noexcept;

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

  private:
    // ---------------------------------------------------------
    // Metadata
    // ---------------------------------------------------------
    bool       defined_ = false;
    size_t     numel_;
    Shape      shape_;
    Shape      strides_;
    DataType   dtype_  = DataType::f32;
    DeviceType device_ = DeviceType::cpu;

    void compute_strides();
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