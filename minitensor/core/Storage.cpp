#include "Storage.hpp"

#include "Allocator.hpp"

#include <cstddef>
#include <cstdlib>
#include <minitensor/minitensor.hpp>

namespace mt
{
Storage::Storage(std::size_t numel, DataType dtype, DeviceType device) : numel_(numel), dtype_(dtype), device_(device) {
    std::size_t type_size = element_size(dtype);
    this->data_           = DefaultAllocator::allocate(numel, type_size);
}

Storage::~Storage() {
    if (this->data_) {
        DefaultAllocator::deallocate(this->data_);
        this->data_ = nullptr;
    }
}

void* Storage::data() noexcept {
    return this->data_;
}

const void* Storage::data() const noexcept {
    return this->data_;
}

std::size_t Storage::numel() const noexcept {
    return this->numel_;
}

DataType Storage::dtype() const noexcept {
    return this->dtype_;
}

DeviceType Storage::device() const noexcept {
    return this->device_;
}
} // namespace mt