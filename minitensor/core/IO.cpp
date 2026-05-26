// minitensor/core/IO.cpp
#include <minitensor/minitensor.hpp>
#include <iostream>

namespace mt {

std::ostream& operator<<(std::ostream& os, DataType dtype) {
    switch (dtype) {
    case DataType::i32:
        os << "int32";
        break;
    case DataType::i64:
        os << "int64";
        break;
    case DataType::f32:
        os << "float32";
        break;
    case DataType::f64:
        os << "float64";
        break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, DeviceType device) {
    switch (device) {
    case DeviceType::cpu:
        os << "CPU";
        break;
    case DeviceType::cuda:
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
