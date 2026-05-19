#pragma once

#ifndef _MINITENSOR_HPP_
#error "include minitensor/minitensor.h in your application, **not** minitensor/minitensor/Optim.h"
#endif

#include "Tensor.hpp"

#include <vector>

namespace mt
{
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