// include/minitensor/error.hpp
#pragma once

#include <stdexcept>
#include <concepts>
#include <sstream>
#include <string>

namespace mt
{

template <typename Derived>
class MTError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
    static std::string ename() {
        return Derived::error_name;
    }
};

class UndefinedError final : public MTError<UndefinedError> {
  public:
    using MTError::MTError;
    static constexpr const char* error_name = "UndefinedError";
};

class ShapeError final : public MTError<ShapeError> {
  public:
    using MTError::MTError;
    static constexpr const char* error_name = "ShapeError";
};

class DeviceError final : public MTError<DeviceError> {
  public:
    using MTError::MTError;
    static constexpr const char* error_name = "DeviceError";
};

class TypeError final : public MTError<TypeError> {
  public:
    using MTError::MTError;
    static constexpr const char* error_name = "TypeError";
};

class DTypeError final : public MTError<DTypeError> {
  public:
    using MTError::MTError;
    static constexpr const char* error_name = "DtypeError";
};

class IndexError final : public MTError<IndexError> {
  public:
    using MTError::MTError;
    static constexpr const char* error_name = "IndexError";
};

class ArithmeticError final : public MTError<ArithmeticError> {
  public:
    using MTError::MTError;
    static constexpr const char* error_name = "ArithmeticError";
};

// =========================================================
// Concept & Helper
// =========================================================

template <typename T>
concept MTErrorConcept = std::derived_from<T, mt::MTError<T>>;

template <MTErrorConcept E>
inline void throw_error(const std::string& file, int line, const std::string& msg) {
    std::ostringstream oss;
    oss << E::ename() << ": " << file << ":" << line << " -> " << msg;
    throw E(oss.str());
}

// =========================================================
// Tier 1: Core Asserts
// =========================================================

#define MT_CHECK(condition, ExceptionType, ...)                                                                        \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::ostringstream _oss_err;                                                                               \
            _oss_err << __VA_ARGS__;                                                                                   \
            ::mt::throw_error<ExceptionType>(__FILE__, __LINE__, _oss_err.str());                                      \
        }                                                                                                              \
    } while (false)

#define MT_THROW(ExceptionType, ...)                                                                                   \
    do {                                                                                                               \
        std::ostringstream _oss_err;                                                                                   \
        _oss_err << __VA_ARGS__;                                                                                       \
        ::mt::throw_error<ExceptionType>(__FILE__, __LINE__, _oss_err.str());                                          \
    } while (false)

// =========================================================
// Tier 2: Atomic Checks (Preconditions)
// =========================================================

#define MT_CHECK_DEFINED(a)                                                                                            \
    MT_CHECK((a).defined(), mt::UndefinedError, "Tensor is not defined (has no storage).")

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

} // namespace mt