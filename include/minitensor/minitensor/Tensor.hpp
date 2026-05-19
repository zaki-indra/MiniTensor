// include/minitensor/minitensor/Tensor.hpp

#pragma once

#ifndef _MINITENSOR_HPP_
#error "include minitensor/minitensor.h in your application, **not** minitensor/minitensor/Tensor.h"
#endif

#include "Types.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace mt
{

struct TensorImpl;

class Tensor {
  public:
    Tensor() noexcept;
    explicit Tensor(Shape shape, bool requires_grad = false);
    Tensor(std::vector<float> data, Shape shape, bool requires_grad = false);
    Tensor(std::initializer_list<float> values);

    template <mt::Index... Indices>
    [[nodiscard]] float& at(Indices... indices) {
        constexpr std::size_t num_indices = sizeof...(Indices);
        if constexpr (num_indices == 0) {
            return _at_impl(nullptr, 0);
        } else {
            // Unpack the parameter pack into a standard array to pass across the bridge
            std::size_t idxs[] = {static_cast<std::size_t>(indices)...};
            return _at_impl(idxs, num_indices);
        }
    }
    template <mt::Index... Indices>
    [[nodiscard]] const float& at(Indices... indices) const {
        constexpr std::size_t num_indices = sizeof...(Indices);
        if constexpr (num_indices == 0) {
            return _at_impl(nullptr, 0);
        } else {
            std::size_t idxs[] = {static_cast<std::size_t>(indices)...};
            return _at_impl(idxs, num_indices);
        }
    }

    template <Index... Indices>
    [[nodiscard]] float& at1(Indices... indices);

    static Tensor zeros(const Shape& shape, bool requires_grad = false);
    static Tensor ones(const Shape& shape, bool requires_grad = false);
    static Tensor randn(const Shape& shape, bool requires_grad = false);

    [[nodiscard]] bool         defined() const noexcept;
    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] std::size_t  ndim() const noexcept;
    [[nodiscard]] std::size_t  numel() const noexcept;
    [[nodiscard]] bool         requires_grad() const noexcept;

    [[nodiscard]] DataType   dtype() const noexcept;
    [[nodiscard]] DeviceType device() const noexcept;

    [[nodiscard]] float*       data() noexcept;
    [[nodiscard]] const float* data() const noexcept;

    [[nodiscard]] Tensor grad() const;
    void                 zero_grad();
    void                 backward();
    void                 backward(const Tensor& grad_output);

    [[nodiscard]] Tensor detach() const;
    void                 set_requires_grad(bool enabled);

    [[nodiscard]] float item() const;

    [[nodiscard]] Tensor reshape(const Shape& new_shape) const;
    [[nodiscard]] Tensor flatten(std::size_t start_dim = 0, std::size_t end_dim = static_cast<std::size_t>(-1)) const;
    [[nodiscard]] Tensor transpose(std::size_t dim0, std::size_t dim1) const;
    [[nodiscard]] Tensor unsqueeze(std::size_t dim) const;
    [[nodiscard]] Tensor squeeze(std::optional<std::size_t> dim = std::nullopt) const;

    // Tensor-to-Tensor operations
    [[nodiscard]] Tensor operator+(const Tensor& other) const;
    [[nodiscard]] Tensor operator-(const Tensor& other) const;
    [[nodiscard]] Tensor operator*(const Tensor& other) const;
    [[nodiscard]] Tensor operator/(const Tensor& other) const;
    [[nodiscard]] Tensor operator-() const;

    // Tensor-to-Scalar operations
    [[nodiscard]] Tensor operator+(float scalar) const;
    [[nodiscard]] Tensor operator-(float scalar) const;
    [[nodiscard]] Tensor operator*(float scalar) const;
    [[nodiscard]] Tensor operator/(float scalar) const;

    [[nodiscard]] Tensor matmul(const Tensor& other) const;

    [[nodiscard]] Tensor relu() const;
    [[nodiscard]] Tensor sigmoid() const;
    [[nodiscard]] Tensor tanh() const;
    [[nodiscard]] Tensor exp() const;
    [[nodiscard]] Tensor log() const;

    [[nodiscard]] Tensor sum(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Tensor mean(std::optional<std::size_t> dim = std::nullopt, bool keepdim = false) const;
    [[nodiscard]] Tensor argmax(std::size_t dim) const;
    [[nodiscard]] Tensor softmax(std::size_t dim) const;
    [[nodiscard]] Tensor log_softmax(std::size_t dim) const;

  private:
    std::shared_ptr<TensorImpl> impl_;

    float&       _at_impl(const std::size_t* indices, std::size_t num_indices);
    const float& _at_impl(const std::size_t* indices, std::size_t num_indices) const;
};

// Convenience free functions
[[nodiscard]] Tensor zeros(const Shape& shape, bool requires_grad = false);
[[nodiscard]] Tensor ones(const Shape& shape, bool requires_grad = false);
[[nodiscard]] Tensor randn(const Shape& shape, bool requires_grad = false);
[[nodiscard]] Tensor reshape(const Tensor& input, const Shape& shape);

// Scalar-to-Tensor commutative operations
[[nodiscard]] inline Tensor operator+(float scalar, const Tensor& tensor) {
    return tensor + scalar;
}
[[nodiscard]] inline Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

// Printing stream support
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
} // namespace mt