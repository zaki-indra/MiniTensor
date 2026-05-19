#pragma once

#ifndef _MINITENSOR_HPP_
#error "include minitensor/minitensor.h in your application, **not** minitensor/minitensor/Modules.h"
#endif

#include "Tensor.hpp"

#include <vector>

namespace mt
{
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
} // namespace mt