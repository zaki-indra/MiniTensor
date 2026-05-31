// minitensor/core/Macros.hpp
#pragma once

#include <cstdint>
#include <minitensor/minitensor.hpp>
#include <stdexcept>

// ---------------------------------------------------------
// Runtime Type Dispatcher
// ---------------------------------------------------------

// `[[maybe_unused]]` on the type alias so dispatch sites whose body
// happens not to reference TYPE_NAME (rare but legal) don't emit
// -Wunused-local-typedef warnings.
#define MT_DISPATCH_ALL_TYPES(TYPE, TYPE_NAME, ...)                                                                    \
    [&]() {                                                                                                            \
        switch (TYPE) {                                                                                                \
        case mt::EDataType::f32: {                                                                                      \
            using TYPE_NAME [[maybe_unused]] = float;                                                                  \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case mt::EDataType::f64: {                                                                                      \
            using TYPE_NAME [[maybe_unused]] = double;                                                                 \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case mt::EDataType::i32: {                                                                                      \
            using TYPE_NAME [[maybe_unused]] = int32_t;                                                                \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case mt::EDataType::i64: {                                                                                      \
            using TYPE_NAME [[maybe_unused]] = int64_t;                                                                \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        default:                                                                                                       \
            throw mt::DispatchError("Unsupported DataType in MT_DISPATCH_ALL_TYPES");                                  \
        }                                                                                                              \
    }()

// ---------------------------------------------------------
// Combined Device + Type Dispatcher
// ---------------------------------------------------------
//
// Aliases `DEVICE_TAG` to a compile-time device tag (mt::CpuDevice /
// mt::CudaDevice) and `TYPE_NAME` to the C++ type matching the runtime
// DataType, then invokes the body lambda. Both aliases are in scope
// inside the body, so kernels can specialize per (device, dtype) via:
//
//     MT_DISPATCH(arr.device(), Dev, arr.dtype(), T, [&]() {
//         if constexpr (std::is_same_v<Dev, mt::CpuDevice>) { ... }
//         else                                              { ... }
//     });
//
// Reuses MT_DISPATCH_ALL_TYPES rather than duplicating its case table —
// a bug fixed once is fixed everywhere.
#define MT_DISPATCH(DEVICE, DEVICE_TAG, TYPE, TYPE_NAME, ...)                                                          \
    [&]() {                                                                                                            \
        switch (DEVICE) {                                                                                              \
        case mt::EDeviceType::cpu: {                                                                                    \
            using DEVICE_TAG [[maybe_unused]] = mt::CpuDevice;                                                         \
            return MT_DISPATCH_ALL_TYPES(TYPE, TYPE_NAME, __VA_ARGS__);                                                \
        }                                                                                                              \
        case mt::EDeviceType::cuda: {                                                                                   \
            using DEVICE_TAG [[maybe_unused]] = mt::CudaDevice;                                                        \
            return MT_DISPATCH_ALL_TYPES(TYPE, TYPE_NAME, __VA_ARGS__);                                                \
        }                                                                                                              \
        default:                                                                                                       \
            throw mt::DispatchError("Unsupported DeviceType in MT_DISPATCH");                                          \
        }                                                                                                              \
    }()