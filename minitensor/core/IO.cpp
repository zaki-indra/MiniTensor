// minitensor/core/IO.cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

namespace mt {

std::ostream& operator<<(std::ostream& os, EDataType dtype) {
    switch (dtype) {
    case EDataType::i32:
        os << "int32";
        break;
    case EDataType::i64:
        os << "int64";
        break;
    case EDataType::f32:
        os << "float32";
        break;
    case EDataType::f64:
        os << "float64";
        break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, EDeviceType device) {
    switch (device) {
    case EDeviceType::cpu:
        os << "CPU";
        break;
    case EDeviceType::cuda:
        os << "CUDA";
        break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i + 1 < shape.size()) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

} // namespace mt
