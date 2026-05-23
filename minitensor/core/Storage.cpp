// minitensor/core/Storage.cpp

#include "Storage.hpp"

#include "Allocator.hpp"

#include <cstddef>
#include <cstdlib>
#include <minitensor/minitensor.hpp>

namespace mt
{
Storage::Storage(std::size_t size, DataType dtype, DeviceType device) : size_(size), dtype_(dtype), device_(device) {
    std::size_t type_size = element_size(dtype);
    this->data_           = DefaultAllocator::allocate(size_, type_size);
}

Storage::Storage(Storage&& other) noexcept
    : data_(std::move(other.data_)), size_(other.size_), dtype_(other.dtype_), device_(other.device_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

Storage::~Storage() {
    if (this->data_) {
        DefaultAllocator::deallocate(this->data_);
        this->data_ = nullptr;
    }
}

void Storage::retain() noexcept {
    ++ref_count_;
}

void Storage::release() noexcept {
    if (--ref_count_ == 0) {
        delete this;
    }
}

int Storage::ref_count() const noexcept {
    return ref_count_;
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