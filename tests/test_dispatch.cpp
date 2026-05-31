// tests/test_dispatch.cpp
//
// Direct tests for the MT_DISPATCH macro. The dtype-only dispatcher
// (MT_DISPATCH_ALL_TYPES) is exercised indirectly by ArrayIterator tests;
// this file pins down the device-and-dtype combined dispatch: that both
// aliases (device tag + dtype type) are in scope inside the body, that
// the body's return value forwards, and that the default branches throw
// mt::DispatchError on unknown values.

#include "core/Macros.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>
#include <string>
#include <type_traits>

// =========================================================
// Device-tag binding
// =========================================================

TEST(MTDispatchTest, CpuDispatchAliasesCpuDeviceTag) {
    bool is_cpu = false;
    MT_DISPATCH(mt::EDeviceType::cpu, Dev, mt::EDataType::f32, T, [&]() { is_cpu = std::is_same_v<Dev, mt::CpuDevice>; });
    EXPECT_TRUE(is_cpu);
}

TEST(MTDispatchTest, CudaDispatchAliasesCudaDeviceTag) {
    bool is_cuda = false;
    MT_DISPATCH(mt::EDeviceType::cuda, Dev, mt::EDataType::f32, T,
                [&]() { is_cuda = std::is_same_v<Dev, mt::CudaDevice>; });
    EXPECT_TRUE(is_cuda);
}

// =========================================================
// Dtype binding inside each device branch
// =========================================================

TEST(MTDispatchTest, DTypeAliasesCorrectlyForEachSupportedType) {
    bool match_f32 = false;
    MT_DISPATCH(mt::EDeviceType::cpu, Dev, mt::EDataType::f32, T, [&]() { match_f32 = std::is_same_v<T, float>; });
    EXPECT_TRUE(match_f32);

    bool match_f64 = false;
    MT_DISPATCH(mt::EDeviceType::cpu, Dev, mt::EDataType::f64, T, [&]() { match_f64 = std::is_same_v<T, double>; });
    EXPECT_TRUE(match_f64);

    bool match_i32 = false;
    MT_DISPATCH(mt::EDeviceType::cpu, Dev, mt::EDataType::i32, T, [&]() { match_i32 = std::is_same_v<T, int32_t>; });
    EXPECT_TRUE(match_i32);

    bool match_i64 = false;
    MT_DISPATCH(mt::EDeviceType::cpu, Dev, mt::EDataType::i64, T, [&]() { match_i64 = std::is_same_v<T, int64_t>; });
    EXPECT_TRUE(match_i64);
}

TEST(MTDispatchTest, DTypeBindingWorksUnderCudaDeviceToo) {
    // Verifies the dtype inner-dispatch isn't skipped on the cuda branch.
    bool match_i64 = false;
    MT_DISPATCH(mt::EDeviceType::cuda, Dev, mt::EDataType::i64, T, [&]() { match_i64 = std::is_same_v<T, int64_t>; });
    EXPECT_TRUE(match_i64);
}

// =========================================================
// Body sees both aliases simultaneously
// =========================================================

TEST(MTDispatchTest, BodyCanBranchOnDeviceTagViaIfConstexpr) {
    std::string result;
    MT_DISPATCH(mt::EDeviceType::cpu, Dev, mt::EDataType::f32, T, [&]() {
        if constexpr (std::is_same_v<Dev, mt::CpuDevice>)
            result = "cpu";
        else
            result = "other";
    });
    EXPECT_EQ(result, "cpu");
}

TEST(MTDispatchTest, BodyCanReadBothAliasesInOneExpression) {
    // sizeof(T) requires T to be in scope; std::is_same_v<Dev, ...> requires
    // Dev to be in scope. Both must hold simultaneously in the body.
    std::size_t reported_size = 0;
    bool        reported_cpu  = false;
    MT_DISPATCH(mt::EDeviceType::cpu, Dev, mt::EDataType::i64, T, [&]() {
        reported_size = sizeof(T);
        reported_cpu  = std::is_same_v<Dev, mt::CpuDevice>;
    });
    EXPECT_EQ(reported_size, 8u);
    EXPECT_TRUE(reported_cpu);
}

// =========================================================
// Return values forward through both layers of dispatch
// =========================================================

TEST(MTDispatchTest, BodyReturnValueIsForwardedThroughBothLayers) {
    int byte_count =
      MT_DISPATCH(mt::EDeviceType::cpu, Dev, mt::EDataType::i32, T, [&]() -> int { return static_cast<int>(sizeof(T)); });
    EXPECT_EQ(byte_count, 4);
}

// =========================================================
// Default branches throw on unknown enum values
// =========================================================

TEST(MTDispatchTest, InvalidDeviceThrowsDispatchError) {
    auto bogus = static_cast<mt::EDeviceType>(99);
    EXPECT_THROW(MT_DISPATCH(bogus, Dev, mt::EDataType::f32, T, [&]() {}), mt::DispatchError);
}

TEST(MTDispatchTest, InvalidDTypeThrowsDispatchError) {
    auto bogus = static_cast<mt::EDataType>(99);
    EXPECT_THROW(MT_DISPATCH(mt::EDeviceType::cpu, Dev, bogus, T, [&]() {}), mt::DispatchError);
}

TEST(MTDispatchTest, InvalidDTypeUnderCudaAlsoThrows) {
    // Confirms the inner dispatch's default actually runs on both branches,
    // not just the cpu one.
    auto bogus = static_cast<mt::EDataType>(99);
    EXPECT_THROW(MT_DISPATCH(mt::EDeviceType::cuda, Dev, bogus, T, [&]() {}), mt::DispatchError);
}
