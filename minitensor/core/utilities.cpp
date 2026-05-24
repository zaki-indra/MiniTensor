// minitensor/core/utilities.cpp

#include <minitensor/minitensor.hpp>
#include <numeric>

namespace mt
{

void Array::compute_strides() {
    this->strides_.resize(shape_.size());
    std::size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        this->strides_[i] = stride;
        stride *= shape_[i];
    }
}

std::size_t Array::shape_product(const Shape& s) noexcept {
    return std::accumulate(s.begin(), s.end(), std::size_t{1}, std::multiplies<>{});
}

bool Array::shapes_equal(const Shape& a, const Shape& b) noexcept {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i])
            return false;
    }
    return true;
}
} // namespace mt