#include "Allocator.hpp"

namespace mt
{

void* DefaultAllocator::allocate(std::size_t n, std::size_t size) {
    if (n == 0 || size == 0) {
        return nullptr;
    }
    return std::malloc(n * size);
}

void DefaultAllocator::deallocate(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}
} // namespace mt