// minitensor/core/ErrorMacros.hpp
//
// Internal error-reporting machinery. Not part of the public API: do not
// install this header and do not include it from anything under include/.

#pragma once

#include <minitensor/minitensor/Error.hpp>
#include <sstream>
#include <string>

namespace mt::detail
{

template <MTErrorConcept E>
[[noreturn]] inline void throw_error(const char* file, int line, const std::string& msg) {
    std::ostringstream oss;
    oss << E::ename() << ": " << file << ":" << line << " -> " << msg;
    throw E(oss.str());
}

} // namespace mt::detail

// =========================================================
// Tier 1: Core Asserts
// =========================================================

#define MT_CHECK(condition, ExceptionType, ...)                                                                        \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::ostringstream _oss_err;                                                                               \
            _oss_err << __VA_ARGS__;                                                                                   \
            ::mt::detail::throw_error<ExceptionType>(__FILE__, __LINE__, _oss_err.str());                              \
        }                                                                                                              \
    } while (false)

#define MT_THROW(ExceptionType, ...)                                                                                   \
    do {                                                                                                               \
        std::ostringstream _oss_err;                                                                                   \
        _oss_err << __VA_ARGS__;                                                                                       \
        ::mt::detail::throw_error<ExceptionType>(__FILE__, __LINE__, _oss_err.str());                                  \
    } while (false)

// =========================================================
// Tier 2: Atomic Checks (Preconditions)
// =========================================================

#define MT_CHECK_DEFINED(a) MT_CHECK((a).defined(), mt::UndefinedError, "Tensor is not defined (has no storage).")

#define MT_CHECK_SAME_DEVICE(a, b)                                                                                     \
    MT_CHECK((a).device() == (b).device(), mt::DeviceError,                                                            \
             "Device mismatch! Tensor A is on " << (a).device() << " but Tensor B is on " << (b).device())

#define MT_CHECK_SAME_DTYPE(a, b)                                                                                      \
    MT_CHECK((a).dtype() == (b).dtype(), mt::DTypeError,                                                               \
             "DataType mismatch! Tensor A has dtype " << (a).dtype() << " but Tensor B has dtype " << (b).dtype())

#define MT_CHECK_SAME_SHAPE(a, b)                                                                                      \
    MT_CHECK((a).shape() == (b).shape(), mt::ShapeError,                                                               \
             "Shape mismatch! Tensor A has shape " << (a).shape() << " but Tensor B has shape " << (b).shape())

#define MT_CHECK_DIM(a, expected_dim)                                                                                  \
    MT_CHECK((a).ndim() == (expected_dim), mt::ShapeError,                                                             \
             "Dimension mismatch! Expected " << (expected_dim) << "D tensor, but got " << (a).ndim() << "D tensor")

#define MT_CHECK_INDEX(idx, bound, dim)                                                                                \
    MT_CHECK((idx) < (bound), mt::IndexError,                                                                          \
             "Index " << (idx) << " is out of bounds for dimension " << (dim) << " with size " << (bound))

// =========================================================
// Tier 3: Usable Macros for Operations
// =========================================================

#define MT_CHECK_BINARY_ELEMENTWISE(a, b)                                                                              \
    do {                                                                                                               \
        MT_CHECK_DEFINED(a);                                                                                           \
        MT_CHECK_DEFINED(b);                                                                                           \
        MT_CHECK_SAME_DEVICE(a, b);                                                                                    \
        MT_CHECK_SAME_DTYPE(a, b);                                                                                     \
        MT_CHECK_SAME_SHAPE(a, b);                                                                                     \
    } while (false)

#define MT_CHECK_BINARY_SCALAR(a)                                                                                      \
    do {                                                                                                               \
        MT_CHECK_DEFINED(a);                                                                                           \
    } while (false)

#define MT_CHECK_UNARY_OP(a)                                                                                           \
    do {                                                                                                               \
        MT_CHECK_DEFINED(a);                                                                                           \
    } while (false)
