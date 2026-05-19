// minitensor/core/minitensor.cpp

#include <minitensor/minitensor.hpp>
#include <numeric>
#include <utility>

namespace mt
{
// ---------------------------------------------------------
// Constructors
// ---------------------------------------------------------
Array::Array() noexcept = default;

Array::Array(const Array& other)
    : data_(other.data_), shape_(other.shape_), dtype_(other.dtype_), device_(other.device_), defined_(other.defined_),
      numel_(other.numel_) {
    compute_strides();
}

Array::Array(Array&& other) noexcept
    : data_(std::move(other.data_)), shape_(std::move(other.shape_)), dtype_(other.dtype_), device_(other.device_),
      defined_(other.defined_), numel_(other.numel_) {
    compute_strides();
    other.defined_ = false;
    other.numel_   = 0;
}

Array::Array(Shape shape)
    : Array(std::vector<float>(shape_product(shape), 0.0f), std::move(shape), DataType::f32, DeviceType::cpu) {
}

Array::Array(std::vector<float> data)
    : data_(std::move(data)), shape_({data.size()}), dtype_(DataType::f32), device_(DeviceType::cpu), defined_(true) {
    numel_ = data_.size();
    compute_strides();
}

Array::Array(std::vector<float> data, Shape shape)
    : Array(std::move(data), std::move(shape), DataType::f32, DeviceType::cpu) {
}

Array::Array(std::vector<float> data, Shape shape, DataType dtype)
    : Array(std::move(data), std::move(shape), dtype, DeviceType::cpu) {
}

// Canonical constructor
Array::Array(std::vector<float> data, Shape shape, DataType dtype, DeviceType device)
    : data_(std::move(data)), shape_(std::move(shape)), dtype_(dtype), device_(device), defined_(true) {
    numel_ = data_.size();
    compute_strides();
}

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