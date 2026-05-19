// minitensor/core/minitensor.cpp

#include "TensorImpl.hpp"

#include <minitensor/minitensor.hpp>

mt::Tensor::Tensor() noexcept = default;

mt::Tensor::Tensor(Shape shape, bool requires_grad) : shape_(std::move(shape)) {
    std::size_t size = 1;
    for (auto s : shape_)
        size *= s;
    this->data_.resize(size, 0.0f);
    compute_strides();
    this->numel_   = size;
    this->defined_ = true;
}

void mt::Tensor::compute_strides() {
    this->strides_.resize(shape_.size());
    std::size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        this->strides_[i] = stride;
        stride *= shape_[i];
    }
}

mt::Tensor::Tensor(std::vector<float> data, Shape shape, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(std::move(data), std::move(shape), requires_grad)) {
}

mt::Tensor::Tensor(std::initializer_list<float> values)
    : impl_(std::make_shared<TensorImpl>(std::vector<float>(values), Shape{values.size()}, false)) {
}

float& mt::Tensor::at(std::initializer_list<std::size_t> indices) {
    std::size_t pos = 0, i = 0;
    for (auto idx : indices) {
        pos += idx * this->strides_[i++];
    }
    return this->data_[pos];
}

const float& mt::Tensor::at(std::initializer_list<std::size_t> indices) const {
    return std::as_const(const_cast<Tensor&>(*this).at(indices));
}

mt::Tensor mt::Tensor::zeros(const Shape& shape, bool requires_grad) {
    return mt::Tensor(shape, requires_grad);
}

mt::Tensor mt::zeros(const Shape& shape, bool requires_grad) {
    return mt::Tensor::zeros(shape, requires_grad);
}

bool mt::Tensor::defined() const noexcept {
    return this->defined_;
}

const mt::Shape& mt::Tensor::shape() const noexcept {
    return this->shape_;
}

std::size_t mt::Tensor::ndim() const noexcept {
    return this->shape_.size();
}

std::size_t mt::Tensor::numel() const noexcept {
    return this->numel_;
}

mt::DataType mt::Tensor::dtype() const noexcept {
    return this->dtype_;
}

mt::DeviceType mt::Tensor::device() const noexcept {
    return this->device_;
}