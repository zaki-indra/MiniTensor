// minitensor/core/Macros.hpp
#pragma once

#include <cstdint>
#include <minitensor/minitensor.hpp>
#include <stdexcept>

// ---------------------------------------------------------
// Runtime Type Dispatcher
// ---------------------------------------------------------

#define MT_DISPATCH_ALL_TYPES(TYPE, TYPE_NAME, ...)                                                                    \
    [&]() {                                                                                                            \
        switch (TYPE) {                                                                                                \
        case mt::DataType::f32: {                                                                                      \
            using TYPE_NAME = float;                                                                                   \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case mt::DataType::f64: {                                                                                      \
            using TYPE_NAME = double;                                                                                  \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case mt::DataType::i32: {                                                                                      \
            using TYPE_NAME = int32_t;                                                                                 \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case mt::DataType::i64: {                                                                                      \
            using TYPE_NAME = int64_t;                                                                                 \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        default:                                                                                                       \
            throw std::runtime_error("Unsupported DataType in MT_DISPATCH_ALL_TYPES");                                 \
        }                                                                                                              \
    }()