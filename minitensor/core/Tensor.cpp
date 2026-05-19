// minitensor/core/minitensor.cpp

#include "TensorImpl.hpp"

#include <minitensor/minitensor.hpp>

mt::Tensor::Tensor() noexcept = default;

mt::Tensor::Tensor(Shape shape, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(std::move(shape), requires_grad)) {
}

mt::Tensor::Tensor(std::vector<float> data, Shape shape, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(std::move(data), std::move(shape), requires_grad)) {
}

mt::Tensor::Tensor(std::initializer_list<float> values)
    : impl_(std::make_shared<TensorImpl>(std::vector<float>(values), Shape{values.size()}, false)) {
}

float& mt::Tensor::_at_impl(const std::size_t* indices, std::size_t num_indices) {
    return impl_.get()->at(indices, num_indices);
}

mt::Tensor mt::Tensor::zeros(const Shape& shape, bool requires_grad) {
    // TensorImpl's Shape constructor inherently allocates and zero-initializes the data
    return mt::Tensor(shape, requires_grad);
}

mt::Tensor mt::zeros(const Shape& shape, bool requires_grad) {
    return mt::Tensor::zeros(shape, requires_grad);
}

bool mt::Tensor::defined() const noexcept {
    return impl_ != nullptr;
}

const mt::Shape& mt::Tensor::shape() const noexcept {
    return impl_->shape_;
}

std::size_t mt::Tensor::numel() const noexcept {
    return impl_->data_.size();
}

float* mt::Tensor::data() noexcept {
    return impl_->data_.data();
}

const float* mt::Tensor::data() const noexcept {
    return impl_->data_.data();
}

mt::DataType mt::Tensor::dtype() const noexcept {
    return impl_->dtype_;
}

mt::DeviceType mt::Tensor::device() const noexcept {
    return impl_->device_;
}

bool mt::Tensor::requires_grad() const noexcept {
    return impl_->requires_grad_;
}