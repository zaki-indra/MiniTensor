// tests/test_array_core.cpp

#include "helper.hpp"

#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>
#include <stdexcept>
#include <vector>

// ─────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────
TEST(ArrayCoreTest, DefaultConstructorLeavesArrayUndefined) {
    mt::Array t;
    EXPECT_FALSE(t.defined());
    EXPECT_EQ(t.numel(), 0);
}

TEST(ArrayCoreTest, VectorConstructorInfersShapeAndType) {
    std::vector<float> data_float{1.0f, 2.0f, 3.0f};
    mt::Array          t(data_float);
    expect_defined(t, {3}, {1}, 3, mt::DataType::f32);

    std::vector<double> data_double{1.0, 2.0, 3.0, 4.0, 5.0};
    mt::Array           t_double(data_double);
    expect_defined(t_double, {5}, {1}, 5, mt::DataType::f64);

    std::vector<int> data_int{1, 2, 3, 4};
    mt::Array        t_int(data_int);
    expect_defined(t_int, {4}, {1}, 4, mt::DataType::i32);

    std::vector<long> data_long{10L, 20L, 30L};
    mt::Array         t_long(data_long);
#if defined(_WIN32)
    expect_defined(t_long, {3}, {1}, 3, mt::DataType::i32);
#else
    expect_defined(t_long, {3}, {1}, 3, mt::DataType::i64);
#endif

    std::vector<long long> data_long2{10L, 20L, 30L};
    mt::Array              t_long2(data_long2);
    expect_defined(t_long2, {3}, {1}, 3, mt::DataType::i64);
}

TEST(ArrayCoreTest, VectorConstructorPreservesValues) {
    std::vector<float> data{4.0f, 5.0f, 6.0f};
    mt::Array          t(data);
    EXPECT_FLOAT_EQ(t.at<float>({0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at<float>({1}), 5.0f);
    EXPECT_FLOAT_EQ(t.at<float>({2}), 6.0f);
}

TEST(ArrayCoreTest, DataAndShapeConstructorInitializesCorrectly) {
    std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
    mt::Array            t(std::span<const int32_t>(data), {2, 3});
    expect_defined(t, {2, 3}, {3, 1}, 6, mt::DataType::i32);
    EXPECT_EQ(t.at<int32_t>({0, 0}), 1);
    EXPECT_EQ(t.at<int32_t>({1, 2}), 6);
}

// ─────────────────────────────────────────────
// Copy, Move, and Memory Management
// ─────────────────────────────────────────────
TEST(ArrayCoreTest, CopyConstructorSharesStorage) {
    std::vector<float> data{1.0f, 2.0f, 3.0f};
    mt::Array          a(data);
    mt::Array          b = a; // Copy constructor

    EXPECT_EQ(b.numel(), a.numel());

    // Modifying one should modify the other since they share the raw Storage
    b.at<float>({0}) = 10.0f;
    EXPECT_FLOAT_EQ(a.at<float>({0}), 10.0f);
}

TEST(ArrayCoreTest, MoveConstructorLeavesSourceUndefined) {
    std::vector<float> data{1, 2, 3};
    mt::Array          a(data);
    mt::Array          b = std::move(a);

    EXPECT_TRUE(b.defined());
    EXPECT_EQ(b.numel(), 3);
    EXPECT_FALSE(a.defined());
}

TEST(ArrayCoreTest, CloneCreatesIndependentMemoryStorage) {
    std::vector<float> data{1.0f, 2.0f, 3.0f};
    mt::Array          a(data);
    mt::Array          b = a.clone();

    // Modifying the clone should NOT affect the original array
    b.at<float>({0}) = 10.0f;
    EXPECT_FLOAT_EQ(a.at<float>({0}), 1.0f);
    EXPECT_FLOAT_EQ(b.at<float>({0}), 10.0f);
}

TEST(ArrayCoreTest, AssignmentOperatorsShareOrMoveStorage) {
    std::vector<float> data1{1.0f, 2.0f, 3.0f};
    std::vector<float> data2{4.0f, 5.0f, 6.0f};

    mt::Array a(data1);
    mt::Array b(data2);

    // Copy assignment (shares memory)
    b                = a;
    b.at<float>({0}) = 10.0f;
    EXPECT_FLOAT_EQ(a.at<float>({0}), 10.0f);

    // Move assignment
    mt::Array c;
    c = std::move(b);
    EXPECT_TRUE(c.defined());
    EXPECT_FALSE(b.defined());
    EXPECT_FLOAT_EQ(c.at<float>({0}), 10.0f);
}

// ─────────────────────────────────────────────
// Factory Functions
// ─────────────────────────────────────────────
TEST(ArrayCoreTest, ZerosFactoryAllocatesCorrectMemory) {
    mt::Array t = mt::zeros({3, 3});
    expect_defined(t, {3, 3}, {3, 1}, 9);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(t.at<float>({i, j}), 0.0f);
}

TEST(ArrayCoreTest, OnesFactoryAllocatesCorrectMemory) {
    mt::Array t = mt::ones({2, 3, 4});
    expect_defined(t, {2, 3, 4}, {12, 4, 1}, 24);
    EXPECT_FLOAT_EQ(t.at<float>({1, 2, 3}), 1.0f);
}

TEST(ArrayCoreTest, FullFactoryAllocatesCorrectMemory) {
    mt::Array t = mt::full({2, 2}, 7);
    expect_defined(t, {2, 2}, {2, 1}, 4, mt::DataType::i32);
    EXPECT_EQ(t.at<int32_t>({0, 0}), 7);
    EXPECT_EQ(t.at<int32_t>({1, 1}), 7);
}

// ─────────────────────────────────────────────
// Cast and Device Management
// ─────────────────────────────────────────────
TEST(ArrayCoreTest, CastConvertsDataTypes) {
    std::vector<double> double_data{1.5, -2.7, 3.0};
    mt::Array           a(std::span<const double>(double_data), {3});

    // Cast from f64 to i32
    mt::Array b = a.cast(mt::DataType::i32);
    EXPECT_EQ(b.dtype(), mt::DataType::i32);
    EXPECT_EQ(b.at<int32_t>({0}), 1);
    EXPECT_EQ(b.at<int32_t>({1}), -2);
    EXPECT_EQ(b.at<int32_t>({2}), 3);

    // Cast from i32 to f32
    mt::Array c = b.cast(mt::DataType::f32);
    EXPECT_EQ(c.dtype(), mt::DataType::f32);
    EXPECT_FLOAT_EQ(c.at<float>({0}), 1.0f);
    EXPECT_FLOAT_EQ(c.at<float>({1}), -2.0f);
    EXPECT_FLOAT_EQ(c.at<float>({2}), 3.0f);
}

TEST(ArrayCoreTest, ToCudaThrowsUnsupportedException) {
    mt::Array a = mt::zeros({3, 3});

    // Identity transfer (CPU -> CPU) should return same device
    mt::Array b = a.to(mt::DeviceType::cpu);
    EXPECT_EQ(b.device(), mt::DeviceType::cpu);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    // Transfer to CUDA must throw exception as it's unsupported
    EXPECT_THROW(a.to(mt::DeviceType::cuda), std::runtime_error);
    EXPECT_EQ(a.device(), mt::DeviceType::cpu);
#pragma GCC diagnostic pop
}

// ─────────────────────────────────────────────
// Safeties & Edge Cases
// ─────────────────────────────────────────────
TEST(ArrayCoreTest, Handles0DAndEmptyTensors) {
    // 0-D array (scalar)
    std::vector<float> data{3.14f};
    mt::Array          scalar(std::span<const float>(data), {});
    EXPECT_TRUE(scalar.defined());
    EXPECT_EQ(scalar.numel(), 1);
    EXPECT_FLOAT_EQ(scalar.item<float>(), 3.14f);

    // Empty tensor (shape with size 0 dimension)
    std::vector<float> empty_data{};
    mt::Array          empty(std::span<const float>(empty_data), {0, 5});
    EXPECT_TRUE(empty.defined());
    EXPECT_EQ(empty.numel(), 0);
}

TEST(ArrayCoreTest, TypeAccessThrowsOnMismatch) {
    mt::Array a = mt::ones({2, 2}, mt::DataType::i32);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    EXPECT_THROW(a.at<float>({0, 0}), std::runtime_error);
    EXPECT_THROW(a.data<float>(), std::runtime_error);

    std::vector<int32_t> single{100};
    mt::Array            c(single);
    EXPECT_THROW(c.item<float>(), std::runtime_error);

    // item() on multi-element array should throw
    EXPECT_THROW(a.item<int32_t>(), std::runtime_error);
#pragma GCC diagnostic pop
}