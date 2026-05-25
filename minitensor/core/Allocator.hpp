// minitensor/core/Allocator.hpp
#pragma once

#include <minitensor/minitensor.hpp>

namespace mt
{
class DefaultAllocator {
  public:
    static void* allocate(std::size_t n, std::size_t size);
    static void  deallocate(void* ptr);
};
} // namespace mt