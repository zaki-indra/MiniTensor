// include/minitensor/minitensor/Error.hpp
//
// Public exception types thrown by MiniTensor.

#pragma once

#include <concepts>
#include <stdexcept>
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

// Concept used by both public templates and the internal throw helper.
template <typename T>
concept MTErrorConcept = std::derived_from<T, mt::MTError<T>>;

} // namespace mt
