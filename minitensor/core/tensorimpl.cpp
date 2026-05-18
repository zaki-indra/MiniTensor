
#include "tensorimpl.hpp"

#include <minitensor/types.hpp>

float& mt::TensorImpl::at(const std::size_t* indices, std::size_t num_indices) {
    std::size_t index = 0;
    // TODO: assert indices length equal to stride length
    for (size_t i = 0; i < num_indices; ++i) {
        index += indices[i] * strides_[i];
    }
    return data_[index];
}