#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace mt
{

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

struct TensorImpl;

using Shape = std::vector<std::size_t>;

class Tensor {
  public:
    Tensor() noexcept;
    explicit Tensor(Shape shape, bool requires_grad = false);
    Tensor(std::vector<float> data, Shape shape, bool requires_grad = false);
    Tensor(std::initializer_list<float> values);

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

// ---------------------------------------------------------
// Neural network modules
// ---------------------------------------------------------
namespace nn
{

class Module {
  public:
    virtual ~Module() = default;

    [[nodiscard]] virtual Tensor forward(const Tensor& input) const = 0;
    [[nodiscard]] Tensor         operator()(const Tensor& input) const {
        return forward(input);
    }

    [[nodiscard]] virtual std::vector<Tensor> parameters() const = 0;

    virtual void train() noexcept {
        training_ = true;
    }
    virtual void eval() noexcept {
        training_ = false;
    }
    [[nodiscard]] bool training() const noexcept {
        return training_;
    }

  protected:
    bool training_ = true;
};

class Linear final : public Module {
  public:
    Linear(std::size_t in_features, std::size_t out_features, bool bias = true);

    [[nodiscard]] Tensor              forward(const Tensor& input) const override;
    [[nodiscard]] std::vector<Tensor> parameters() const override;

    [[nodiscard]] const Tensor& weight() const noexcept;
    [[nodiscard]] const Tensor& bias() const noexcept;

  private:
    Tensor weight_;
    Tensor bias_;
    bool   has_bias_ = true;
};

class LossFunction {
  public:
    virtual ~LossFunction() = default;

    [[nodiscard]] virtual Tensor forward(const Tensor& input, const Tensor& target) const = 0;
    [[nodiscard]] Tensor         operator()(const Tensor& input, const Tensor& target) const {
        return forward(input, target);
    }
};

class MSELoss final : public LossFunction {
  public:
    MSELoss() = default;
    [[nodiscard]] Tensor forward(const Tensor& input, const Tensor& target) const override;
};

class CrossEntropyLoss final : public LossFunction {
  public:
    CrossEntropyLoss() = default;
    [[nodiscard]] Tensor forward(const Tensor& logits, const Tensor& target) const override;
};

} // namespace nn

// ---------------------------------------------------------
// Optimizers
// ---------------------------------------------------------
namespace optim
{

class SGD {
  public:
    SGD(std::vector<Tensor> parameters, float learning_rate);

    void zero_grad();
    void step();

    [[nodiscard]] float learning_rate() const noexcept;
    void                set_learning_rate(float lr) noexcept;

  private:
    std::vector<Tensor> parameters_;
    float               learning_rate_ = 1e-2f;
};

} // namespace optim

} // namespace mt