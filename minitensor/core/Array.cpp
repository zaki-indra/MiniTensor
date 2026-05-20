// minitensor/core/minitensor.cpp

#include <minitensor/minitensor.hpp>
#include <random>
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
Array Array::zeros(const Shape& shape, DataType dtype, DeviceType device) {
    std::size_t numel = 1;
    for (auto s : shape)
        numel *= s;
    std::vector<float> data(numel, 0.0f);
    return Array(data, shape, dtype, device);
}

Array Array::zeros(const Shape& shape, DataType dtype) {
    return Array::zeros(shape, dtype, DeviceType::cpu);
}

Array Array::zeros(const Shape& shape) {
    return Array::zeros(shape, DataType::f32, DeviceType::cpu);
}

Array Array::ones(const Shape& shape, DataType dtype, DeviceType device) {
    std::size_t numel = 1;
    for (auto s : shape)
        numel *= s;
    std::vector<float> data(numel, 1.0f);
    return Array(data, shape, dtype, device);
}

Array Array::ones(const Shape& shape, DataType dtype) {
    return Array::ones(shape, dtype, DeviceType::cpu);
}

Array Array::ones(const Shape& shape) {
    return Array::ones(shape, DataType::f32, DeviceType::cpu);
}

Array Array::randn(const Shape& shape, DataType dtype, DeviceType device) {
    std::size_t numel = 1;
    for (auto s : shape)
        numel *= s;
    std::vector<float>              data(numel);
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (auto& x : data) {
        x = dis(gen);
    }
    return Array(data, shape, dtype, device);
}

Array Array::randn(const Shape& shape, DataType dtype) {
    return Array::randn(shape, dtype, DeviceType::cpu);
}

Array Array::randn(const Shape& shape) {
    return Array::randn(shape, DataType::f32, DeviceType::cpu);
}

Array zeros(const Shape& shape, DataType dtype, DeviceType device) {
    return Array::zeros(shape, dtype, device);
}

Array zeros(const Shape& shape, DataType dtype) {
    return Array::zeros(shape, dtype);
}

Array zeros(const Shape& shape) {
    return Array::zeros(shape);
}

Array ones(const Shape& shape, DataType dtype, DeviceType device) {
    return Array::ones(shape, dtype, device);
}

Array ones(const Shape& shape, DataType dtype) {
    return Array::ones(shape, dtype);
}

Array ones(const Shape& shape) {
    return Array::ones(shape);
}

Array randn(const Shape& shape, DataType dtype, DeviceType device) {
    return Array::randn(shape, dtype, device);
}

Array randn(const Shape& shape, DataType dtype) {
    return Array::randn(shape, dtype);
}

Array randn(const Shape& shape) {
    return Array::randn(shape);
}

// ---------------------------------------------------------
// Element-wise operations
// ---------------------------------------------------------
Array Array::operator+(const Array& other) const {
    if (!defined() || !other.defined()) {
        return Array();
    }
    if (!shapes_equal(this->shape(), other.shape())) {
        return Array();
    }
    if (this->dtype() != other.dtype() || this->device() != other.device()) {
        return Array();
    }

    const Shape& shape = this->shape();
    auto         r     = Array::zeros(shape, this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.data_[i] = this->data_[i] + other.data_[i];
    }
    return r;
}

Array Array::operator-(const Array& other) const {
    if (!defined() || !other.defined()) {
        return Array();
    }
    if (!shapes_equal(this->shape(), other.shape())) {
        return Array();
    }
    if (this->dtype() != other.dtype() || this->device() != other.device()) {
        return Array();
    }

    const Shape& shape = this->shape();
    auto         r     = Array::zeros(shape, this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.data_[i] = this->data_[i] - other.data_[i];
    }
    return r;
}

Array Array::operator*(const Array& other) const {
    if (!defined() || !other.defined()) {
        return Array();
    }
    if (!shapes_equal(this->shape(), other.shape())) {
        return Array();
    }
    if (this->dtype() != other.dtype() || this->device() != other.device()) {
        return Array();
    }

    const Shape& shape = this->shape();
    auto         r     = Array::zeros(shape, this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.data_[i] = this->data_[i] * other.data_[i];
    }
    return r;
}

Array Array::operator/(const Array& other) const {
    if (!defined() || !other.defined()) {
        return Array();
    }
    if (!shapes_equal(this->shape(), other.shape())) {
        return Array();
    }
    if (this->dtype() != other.dtype() || this->device() != other.device()) {
        return Array();
    }

    const Shape& shape = this->shape();
    auto         r     = Array::zeros(shape, this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.data_[i] = this->data_[i] / other.data_[i];
    }
    return r;
}

// ---------------------------------------------------------
// Scalar operations
// ---------------------------------------------------------
Array Array::operator+(float scalar) const {
    if (!defined()) {
        return Array();
    }
    auto r = Array::zeros(this->shape(), this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.data_[i] = this->data_[i] + scalar;
    }
    return r;
}

Array Array::operator-(float scalar) const {
    if (!defined()) {
        return Array();
    }
    auto r = Array::zeros(this->shape(), this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.data_[i] = this->data_[i] - scalar;
    }
    return r;
}

Array Array::operator*(float scalar) const {
    if (!defined()) {
        return Array();
    }
    auto r = Array::zeros(this->shape(), this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.data_[i] = this->data_[i] * scalar;
    }
    return r;
}

Array Array::operator/(float scalar) const {
    if (!defined()) {
        return Array();
    }
    auto r = Array::zeros(this->shape(), this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.data_[i] = this->data_[i] / scalar;
    }
    return r;
}
} // namespace mt