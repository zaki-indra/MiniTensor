// minitensor/core/minitensor.cpp

#include "Storage.hpp"

#include <cstring>
#include <minitensor/minitensor.hpp>
#include <random>
#include <stdexcept>
#include <utility>

namespace mt
{
namespace
{
uint16_t float_to_bf16(float f) {
    uint32_t val;
    std::memcpy(&val, &f, sizeof(float));
    // Round to nearest even to prevent systematic bias
    uint32_t rounding_bias = 0x00007FFF + ((val >> 16) & 1);
    val += rounding_bias;
    return static_cast<uint16_t>(val >> 16);
}

float bf16_to_float(uint16_t val) {
    uint32_t res = static_cast<uint32_t>(val) << 16;
    float    f;
    std::memcpy(&f, &res, sizeof(float));
    return f;
}

uint16_t float_to_f16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    uint32_t sign     = (u >> 16) & 0x8000;
    int32_t  exponent = ((u >> 23) & 0xff) - 127;
    uint32_t mantissa = u & 0x7fffff;

    if (exponent <= -15) {
        if (exponent < -24) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000;
        uint32_t shift = static_cast<uint32_t>(-14 - exponent);
        mantissa >>= shift;
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    } else if (exponent >= 16) {
        // Overflow / Inf
        return static_cast<uint16_t>(sign | 0x7c00);
    }

    exponent += 15;
    return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

float f16_to_float(uint16_t val) {
    uint32_t sign     = (val & 0x8000) << 16;
    uint32_t exponent = (val & 0x7c00) >> 10;
    uint32_t mantissa = val & 0x03ff;

    if (exponent == 0) {
        if (mantissa == 0) {
            float f;
            std::memcpy(&f, &sign, sizeof(float));
            return f;
        }
        while ((mantissa & 0x0400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= ~0x0400;
    } else if (exponent == 31) {
        uint32_t res = sign | 0x7f800000 | (mantissa << 13);
        float    f;
        std::memcpy(&f, &res, sizeof(float));
        return f;
    }

    exponent     = (exponent - 15 + 127) & 0xff;
    uint32_t res = sign | (exponent << 23) | (mantissa << 13);
    float    f;
    std::memcpy(&f, &res, sizeof(float));
    return f;
}
} // namespace

// ---------------------------------------------------------
// Private Item Helpers
// ---------------------------------------------------------
float Array::get_item_as_float(std::size_t index) const {
    if (!data_ || !data_->data())
        return 0.0f;
    void* ptr = data_->data();
    switch (dtype_) {
    case DataType::f32:
        return static_cast<float*>(ptr)[index];
    case DataType::f64:
        return static_cast<float>(static_cast<double*>(ptr)[index]);
    case DataType::f16:
        return f16_to_float(static_cast<uint16_t*>(ptr)[index]);
    case DataType::bf16:
        return bf16_to_float(static_cast<uint16_t*>(ptr)[index]);
    }
    return 0.0f;
}

void Array::set_item_from_float(std::size_t index, float value) {
    if (!data_ || !data_->data())
        return;
    void* ptr = data_->data();
    switch (dtype_) {
    case DataType::f32:
        static_cast<float*>(ptr)[index] = value;
        break;
    case DataType::f64:
        static_cast<double*>(ptr)[index] = static_cast<double>(value);
        break;
    case DataType::f16:
        static_cast<uint16_t*>(ptr)[index] = float_to_f16(value);
        break;
    case DataType::bf16:
        static_cast<uint16_t*>(ptr)[index] = float_to_bf16(value);
        break;
    }
}

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

Array::Array(Shape shape) : shape_(std::move(shape)), dtype_(DataType::f32), device_(DeviceType::cpu), defined_(true) {
    numel_ = shape_product(shape_);
    compute_strides();
    data_ = std::make_shared<Storage>(numel_, dtype_, device_);
    for (std::size_t i = 0; i < numel_; ++i) {
        set_item_from_float(i, 0.0f);
    }
}

Array::Array(std::vector<float> data)
    : shape_({data.size()}), dtype_(DataType::f32), device_(DeviceType::cpu), defined_(true) {
    numel_ = shape_[0];
    compute_strides();
    data_ = std::make_shared<Storage>(numel_, dtype_, device_);
    for (std::size_t i = 0; i < numel_; ++i) {
        set_item_from_float(i, data[i]);
    }
}

Array::Array(std::vector<float> data, Shape shape)
    : Array(std::move(data), std::move(shape), DataType::f32, DeviceType::cpu) {
}

Array::Array(std::vector<float> data, Shape shape, DataType dtype)
    : Array(std::move(data), std::move(shape), dtype, DeviceType::cpu) {
}

// Canonical constructor
Array::Array(std::vector<float> data, Shape shape, DataType dtype, DeviceType device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device), defined_(true) {
    numel_ = shape_product(shape_);
    compute_strides();
    data_ = std::make_shared<Storage>(numel_, dtype_, device_);
    for (std::size_t i = 0; i < numel_; ++i) {
        float val = (i < data.size()) ? data[i] : 0.0f;
        set_item_from_float(i, val);
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
// Indexing & Raw Data Pointer
// ---------------------------------------------------------
float* Array::data() noexcept {
    if (!data_ || dtype_ != DataType::f32) {
        return nullptr;
    }
    return static_cast<float*>(data_->data());
}

const float* Array::data() const noexcept {
    if (!data_ || dtype_ != DataType::f32) {
        return nullptr;
    }
    return static_cast<const float*>(data_->data());
}

float& Array::at(std::initializer_list<std::size_t> indices) {
    std::size_t pos = 0, i = 0;
    for (auto idx : indices) {
        pos += idx * this->strides_[i++];
    }
    if (!data_ || dtype_ != DataType::f32) {
        throw std::runtime_error("at() reference access is only valid for defined f32 arrays.");
    }
    return static_cast<float*>(this->data_->data())[pos];
}

const float& Array::at(std::initializer_list<std::size_t> indices) const {
    std::size_t pos = 0, i = 0;
    for (auto idx : indices) {
        pos += idx * this->strides_[i++];
    }
    if (!data_ || dtype_ != DataType::f32) {
        throw std::runtime_error("at() reference access is only valid for defined f32 arrays.");
    }
    return static_cast<const float*>(this->data_->data())[pos];
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
// Array Manipulation
// ---------------------------------------------------------

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
        r.set_item_from_float(i, this->get_item_as_float(i) + other.get_item_as_float(i));
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
        r.set_item_from_float(i, this->get_item_as_float(i) - other.get_item_as_float(i));
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
        r.set_item_from_float(i, this->get_item_as_float(i) * other.get_item_as_float(i));
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
        r.set_item_from_float(i, this->get_item_as_float(i) / other.get_item_as_float(i));
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
        r.set_item_from_float(i, this->get_item_as_float(i) + scalar);
    }
    return r;
}

Array Array::operator-(float scalar) const {
    if (!defined()) {
        return Array();
    }
    auto r = Array::zeros(this->shape(), this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.set_item_from_float(i, this->get_item_as_float(i) - scalar);
    }
    return r;
}

Array Array::operator*(float scalar) const {
    if (!defined()) {
        return Array();
    }
    auto r = Array::zeros(this->shape(), this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.set_item_from_float(i, this->get_item_as_float(i) * scalar);
    }
    return r;
}

Array Array::operator/(float scalar) const {
    if (!defined()) {
        return Array();
    }
    auto r = Array::zeros(this->shape(), this->dtype(), this->device());
    for (std::size_t i = 0; i < this->numel(); ++i) {
        r.set_item_from_float(i, this->get_item_as_float(i) / scalar);
    }
    return r;
}

float Array::item() const {
    if (!defined()) {
        throw std::runtime_error("Cannot call item() on undefined Array.");
    }
    if (numel() != 1) {
        throw std::runtime_error("item() is only valid for 1-element arrays.");
    }
    return get_item_as_float(0);
}

Array Array::clone() const {
    if (!defined()) {
        return Array();
    }
    auto cloned = Array::zeros(shape_, dtype_, device_);
    for (std::size_t i = 0; i < numel_; ++i) {
        cloned.set_item_from_float(i, get_item_as_float(i));
    }
    return cloned;
}
} // namespace mt