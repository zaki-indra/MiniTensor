// minitensor/core/minitensor.cpp

#include <minitensor/minitensor.hpp>

namespace mt
{
// ---------------------------------------------------------
// Constructors
// ---------------------------------------------------------
Array::Array() noexcept = default;

Array::Array(Shape shape) : shape_(std::move(shape)) {
    std::size_t size = 1;
    for (auto s : shape_)
        size *= s;
    this->data_.resize(size, 0.0f);
    compute_strides();
    this->numel_   = size;
    this->defined_ = true;
}

Array::Array(std::vector<float> data, Shape shape) : data_(std::move(data)), shape_(std::move(shape)) {
    compute_strides();
    this->numel_   = this->data_.size();
    this->defined_ = true;
}

void Array::compute_strides() {
    this->strides_.resize(shape_.size());
    std::size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        this->strides_[i] = stride;
        stride *= shape_[i];
    }
}

// ---------------------------------------------------------
// Metadata Getter
// ---------------------------------------------------------
bool Array::defined() const noexcept {
    return this->defined_;
}

const Shape& Array::shape() const noexcept {
    return this->shape_;
}

const Shape& Array::strides() const noexcept {
    return this->strides_;
}

std::size_t Array::ndim() const noexcept {
    return this->shape_.size();
}

std::size_t Array::numel() const noexcept {
    return this->numel_;
}

DataType Array::dtype() const noexcept {
    return this->dtype_;
}

DeviceType Array::device() const noexcept {
    return this->device_;
}

// ---------------------------------------------------------
// Indexing
// ---------------------------------------------------------
float& Array::at(std::initializer_list<std::size_t> indices) {
    std::size_t pos = 0, i = 0;
    for (auto idx : indices) {
        pos += idx * this->strides_[i++];
    }
    return this->data_[pos];
}

const float& Array::at(std::initializer_list<std::size_t> indices) const {
    return std::as_const(const_cast<Array&>(*this).at(indices));
}

// ---------------------------------------------------------
// Initializers
// ---------------------------------------------------------
Array Array::zeros(const Shape& shape) {
    return Array(shape);
}

Array Array::ones(const Shape& shape) {
    std::size_t numel = 1;
    for (auto s : shape)
        numel *= s;
    std::vector<float> data(numel, 1.0f);
    return Array(data, shape);
}

Array zeros(const Shape& shape) {
    return Array::zeros(shape);
}

Array ones(const Shape& shape) {
    return Array::ones(shape);
}
} // namespace mt