#pragma once

#include <vector>

namespace mt
{
enum class DataType {
    f16,
    f32,
    f64,
    bf16,
};

enum class DeviceType {
    cpu,
    cuda,
};

using Shape = std::vector<std::size_t>;

template <class T>
concept Index = std::convertible_to<T, std::size_t>;
} // namespace mt