// minitensor/core/Storage.cpp

#include "Storage.hpp"

#include <minitensor/minitensor.hpp>
#include <new>
#include <utility>

namespace mt
{

namespace
{

// 64-byte alignment matches a typical cache line and the AVX-512 vector
// width — chosen so SIMD elementwise kernels can rely on it. When CUDA
// arrives, the CUDA allocator will pick its own alignment.
constexpr std::size_t kCpuAlignment = 64;

class CpuAllocator final : public AllocatorInterface {
  public:
    void* allocate(std::size_t bytes, std::size_t alignment) override {
        if (bytes == 0)
            return nullptr;
        return ::operator new(bytes, std::align_val_t{alignment});
    }
    void deallocate(void* ptr, std::size_t /*bytes*/, std::size_t alignment) override {
        if (ptr)
            ::operator delete(ptr, std::align_val_t{alignment});
    }
};

} // namespace

AllocatorInterface& default_allocator(EDeviceType device) {
    static CpuAllocator cpu;
    switch (device) {
    case EDeviceType::cpu:
        return cpu;
    case EDeviceType::cuda:
        throw mt::DeviceError("CUDA allocator is not available on this build.");
    }
    throw mt::DispatchError("Unknown device in default_allocator");
}

// ---------------------------------------------------------
// Storage
// ---------------------------------------------------------
Storage::Storage(std::size_t numel, EDataType dtype, EDeviceType device) : size_(numel), dtype_(dtype), device_(device) {
    std::size_t bytes = size_ * element_size(dtype_);
    data_             = default_allocator(device_).allocate(bytes, kCpuAlignment);
}

Storage::Storage(Storage&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)), dtype_(other.dtype_),
      device_(other.device_) {
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        if (data_) {
            std::size_t bytes = size_ * element_size(dtype_);
            default_allocator(device_).deallocate(data_, bytes, kCpuAlignment);
        }
        data_   = std::exchange(other.data_, nullptr);
        size_   = std::exchange(other.size_, 0);
        dtype_  = other.dtype_;
        device_ = other.device_;
    }
    return *this;
}

Storage::~Storage() {
    if (data_) {
        std::size_t bytes = size_ * element_size(dtype_);
        default_allocator(device_).deallocate(data_, bytes, kCpuAlignment);
    }
}

void* Storage::data() noexcept {
    return data_;
}
const void* Storage::data() const noexcept {
    return data_;
}
std::size_t Storage::size() const noexcept {
    return size_;
}
EDataType Storage::dtype() const noexcept {
    return dtype_;
}
EDeviceType Storage::device() const noexcept {
    return device_;
}

} // namespace mt
