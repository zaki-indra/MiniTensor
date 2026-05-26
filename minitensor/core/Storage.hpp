// minitensor/core/Storage.hpp
#pragma once

#include <minitensor/minitensor.hpp>

namespace mt
{
class Storage {
  private:
    void*       data_   = nullptr;
    std::size_t size_   = 0;
    DataType    dtype_  = DataType::f32;
    DeviceType  device_ = DeviceType::cpu;

  public:
    Storage() noexcept            = delete;
    Storage(const Storage& other) = delete;
    Storage& operator=(const Storage& other) = delete;
    
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