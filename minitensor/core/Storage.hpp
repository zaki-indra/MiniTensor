// minitensor/core/Storage.hpp
#pragma once

#include <minitensor/minitensor.hpp>

namespace mt
{

// ---------------------------------------------------------
// Allocator Interface
// ---------------------------------------------------------
// Backend-pluggable byte allocator. Each device has a default
// implementation registered in Storage.cpp; the factory below picks one.
//
// Signature is `(bytes, alignment)` rather than `(numel, dtype, device)` —
// device routing is the factory's job, and computing byte size from numel
// belongs above the allocator so every backend doesn't re-derive it.
class AllocatorInterface {
  public:
    virtual ~AllocatorInterface()                                                 = default;
    virtual void* allocate(std::size_t bytes, std::size_t alignment)              = 0;
    virtual void  deallocate(void* ptr, std::size_t bytes, std::size_t alignment) = 0;
};

// Returns the default allocator for the given device. The returned
// reference is to a function-local static and is valid for the program
// lifetime. Concrete implementations are translation-unit-local in
// Storage.cpp.
AllocatorInterface& default_allocator(DeviceType device);

// ---------------------------------------------------------
// Storage
// ---------------------------------------------------------
class Storage {
  private:
    void*       data_   = nullptr;
    std::size_t size_   = 0;
    DataType    dtype_  = DataType::f32;
    DeviceType  device_ = DeviceType::cpu;

  public:
    Storage()                          = delete;
    Storage(const Storage&)            = delete;
    Storage& operator=(const Storage&) = delete;

    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;

    Storage(std::size_t numel, DataType dtype, DeviceType device);

    ~Storage();

    [[nodiscard]] void*       data() noexcept;
    [[nodiscard]] const void* data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] DataType    dtype() const noexcept;
    [[nodiscard]] DeviceType  device() const noexcept;
};

} // namespace mt
