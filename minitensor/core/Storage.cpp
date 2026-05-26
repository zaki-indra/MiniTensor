// minitensor/core/Storage.cpp

#include "Storage.hpp"

#include <cstddef>
#include <cstdlib>
#include <minitensor/minitensor.hpp>
#include <utility>

namespace mt
{
Storage::Storage(std::size_t size, DataType dtype, DeviceType device) : size_(size), dtype_(dtype), device_(device) {
    std::size_t type_size = element_size(dtype);
    std::size_t bytes     = size_ * type_size;
    this->data_           = (bytes == 0) ? nullptr : ::operator new(bytes, std::align_val_t{64});
}

Storage::Storage(Storage&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)), dtype_(other.dtype_),
      device_(other.device_) {
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        if (data_)
            ::operator delete(data_, std::align_val_t{64});
        data_   = std::exchange(other.data_, nullptr);
        size_   = std::exchange(other.size_, 0);
        dtype_  = other.dtype_;
        device_ = other.device_;
    }
    return *this;
}

Storage::~Storage() {
    if (this->data_) {
        ::operator delete(this->data_, std::align_val_t{64});
        this->data_ = nullptr;
    }
}

void* Storage::data() noexcept {
    return this->data_;
}
const void* Storage::data() const noexcept {
    return this->data_;
}
std::size_t Storage::size() const noexcept {
    return this->size_;
}
DataType Storage::dtype() const noexcept {
    return this->dtype_;
}
DeviceType Storage::device() const noexcept {
    return this->device_;
}
} // namespace mt