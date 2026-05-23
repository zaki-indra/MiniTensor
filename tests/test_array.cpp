// tests/test_array.cpp

#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>
#include <vector>

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
static void expect_defined(const mt::Array& t, mt::Shape expected_shape, mt::Shape expected_strides,
                           std::size_t expected_numel, mt::DataType expected_dtype = mt::DataType::f32,
                           mt::DeviceType expected_device = mt::DeviceType::cpu) {
    EXPECT_TRUE(t.defined());
    EXPECT_EQ(t.shape(), expected_shape);
    EXPECT_EQ(t.strides(), expected_strides);
    EXPECT_EQ(t.numel(), expected_numel);
    EXPECT_EQ(t.dtype(), expected_dtype);
    EXPECT_EQ(t.device(), expected_device);
}

// ─────────────────────────────────────────────
// Default constructor
// ─────────────────────────────────────────────
TEST(ArrayConstructor, DefaultIsUndefined) {
    mt::Array t;
    EXPECT_FALSE(t.defined());
    EXPECT_EQ(t.numel(), 0);
}

// ─────────────────────────────────────────────
// Array(vector<T>) — Type Inferencing Constructor
// ─────────────────────────────────────────────
TEST(ArrayConstructor, VectorConstructorSetsShapeAndType) {
    std::vector<float> data_float{1.0f, 2.0f, 3.0f};
    mt::Array          t(data_float);
    expect_defined(t, {3}, {1}, 3, mt::DataType::f32);

    std::vector<int> data_int{1, 2, 3, 4};
    mt::Array       t_int(data_int);
    expect_defined(t_int, {4}, {1}, 4, mt::DataType::i32);

    std::vector<double> data_double{1.0, 2.0, 3.0, 4.0, 5.0};
    mt::Array          t_double(data_double);
    expect_defined(t_double, {5}, {1}, 5, mt::DataType::f64);

    std::vector<long> data_long{10L, 20L, 30L};
    mt::Array        t_long(data_long);
    expect_defined(t_long, {3}, {1}, 3, mt::DataType::i64);

    std::vector<long long> data_long2{10L, 20L, 30L};
    mt::Array        t_long2(data_long2);
    expect_defined(t_long2, {3}, {1}, 3, mt::DataType::i64);
}

TEST(ArrayConstructor, VectorConstructorPreservesValues) {
    std::vector<float> data{4.0f, 5.0f, 6.0f};
    mt::Array          t(data);
    EXPECT_FLOAT_EQ(t.at<float>({0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at<float>({1}), 5.0f);
    EXPECT_FLOAT_EQ(t.at<float>({2}), 6.0f);
}

// ─────────────────────────────────────────────
// Array(vector<T>, Shape) 
// ─────────────────────────────────────────────
TEST(ArrayConstructor, DataAndShapeConstructor) {
    std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
    mt::Array          t(std::span<const int32_t>(data), {2, 3});
    expect_defined(t, {2, 3}, {3, 1}, 6, mt::DataType::i32);
    EXPECT_EQ(t.at<int32_t>({0, 0}), 1);
    EXPECT_EQ(t.at<int32_t>({1, 2}), 6);
}

// ─────────────────────────────────────────────
// Strides
// ─────────────────────────────────────────────
TEST(ArrayStrides, RowMajor3D) {
    mt::Array t = mt::zeros({2, 3, 4});
    EXPECT_EQ(t.strides(), mt::Shape({12, 4, 1}));
}

// ─────────────────────────────────────────────
// Copy and move
// ─────────────────────────────────────────────
TEST(ArrayCopyMove, CopyConstructorIsIndependent) {
    std::vector<float> data{1, 2, 3};
    mt::Array a(data);
    mt::Array b = a;
    EXPECT_EQ(b.numel(), a.numel());
    EXPECT_FLOAT_EQ(b.at<float>({0}), 1.0f);
}

TEST(ArrayCopyMove, MoveConstructorLeavesSourceUndefined) {
    std::vector<float> data{1, 2, 3};
    mt::Array a(data);
    mt::Array b = std::move(a);
    EXPECT_TRUE(b.defined());
    EXPECT_EQ(b.numel(), 3);
    EXPECT_FALSE(a.defined());
}

// ─────────────────────────────────────────────
// Factory functions (zeros / ones)
// ─────────────────────────────────────────────
TEST(ArrayFactory, ZerosAllFields) {
    mt::Array t = mt::zeros({3, 3});
    expect_defined(t, {3, 3}, {3, 1}, 9);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(t.at<float>({i, j}), 0.0f);
}

TEST(ArrayFactory, Ones3D) {
    mt::Array t = mt::ones({2, 3, 4});
    expect_defined(t, {2, 3, 4}, {12, 4, 1}, 24);
    EXPECT_FLOAT_EQ(t.at<float>({1, 2, 3}), 1.0f);
}

// ─────────────────────────────────────────────
// Element-wise operations
// ─────────────────────────────────────────────
TEST(ArrayElementwise, Add) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 2});
    mt::Array c = a + b;
    expect_defined(c, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(c.at<float>({1, 1}), 2.0f);
}

TEST(ArrayElementwise, Subtract) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 2});
    mt::Array c = a - b;
    expect_defined(c, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(c.at<float>({1, 1}), 0.0f);
}

TEST(ArrayElementwise, Multiply) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 2});
    mt::Array c = a * b;
    expect_defined(c, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(c.at<float>({1, 1}), 1.0f);
}

// ---------------------------------------------------------
// Scalar operations
// ---------------------------------------------------------
TEST(ArrayScalarOps, Multiply) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = a * 4.0;
    expect_defined(b, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(b.at<float>({0, 0}), 4.0f);
}

// ─────────────────────────────────────────────
// Multi-DataType Support & Preserving Precision
// ─────────────────────────────────────────────
TEST(ArrayMultiDataType, DoublePrecision) {
    std::vector<double> data{1.5, 2.5, 3.5};
    mt::Array a(std::span<const double>(data), {3});
    
    EXPECT_EQ(a.dtype(), mt::DataType::f64);
    EXPECT_EQ(a.numel(), 3);

    // Test scalar operations preserving double accuracy
    mt::Array b = a + 2.0;
    EXPECT_EQ(b.dtype(), mt::DataType::f64);
    EXPECT_TRUE(b.defined());

    EXPECT_DOUBLE_EQ(b.at<double>({0}), 3.5);
    EXPECT_DOUBLE_EQ(b.at<double>({2}), 5.5);
}

TEST(ArrayMultiDataType, Int32PrecisionWithoutFloatDecay) {
    std::vector<int32_t> data{10, 20, 30};
    mt::Array a(data);
    
    EXPECT_EQ(a.dtype(), mt::DataType::i32);
    
    mt::Array b = a * 2.0; // Scaler passed as double, safely truncates inside dispatch
    EXPECT_EQ(b.dtype(), mt::DataType::i32);
    
    EXPECT_EQ(b.at<int32_t>({0}), 20);
    EXPECT_EQ(b.at<int32_t>({2}), 60);
    
    // Test native single item fetch
    std::vector<int32_t> single_data{42};
    mt::Array c(single_data);
    EXPECT_EQ(c.item<int32_t>(), 42);
}

// ─────────────────────────────────────────────
// Clone, Cast, and Memory Management
// ─────────────────────────────────────────────
TEST(ArrayMemoryManagement, CopySharesStorage) {
    std::vector<float> data{1.0f, 2.0f, 3.0f};
    mt::Array a(data);
    mt::Array b = a;

    // Modifying one should modify the other since they share the raw Storage
    b.at<float>({0}) = 10.0f;
    EXPECT_FLOAT_EQ(a.at<float>({0}), 10.0f);
}

TEST(ArrayMemoryManagement, CloneIsIndependent) {
    std::vector<float> data{1.0f, 2.0f, 3.0f};
    mt::Array a(data);
    mt::Array b = a.clone();

    // Modifying the clone should NOT affect the original array
    b.at<float>({0}) = 10.0f;
    EXPECT_FLOAT_EQ(a.at<float>({0}), 1.0f);
    EXPECT_FLOAT_EQ(b.at<float>({0}), 10.0f);
}

TEST(ArrayMemoryManagement, AssignmentOperators) {
    std::vector<float> data1{1.0f, 2.0f, 3.0f};
    std::vector<float> data2{4.0f, 5.0f, 6.0f};

    mt::Array a(data1);
    mt::Array b(data2);

    // Copy assignment
    b = a;
    EXPECT_FLOAT_EQ(b.at<float>({0}), 1.0f);
    
    // Sharing verification
    b.at<float>({0}) = 10.0f;
    EXPECT_FLOAT_EQ(a.at<float>({0}), 10.0f);

    // Move assignment
    mt::Array c;
    c = std::move(b);
    EXPECT_TRUE(c.defined());
    EXPECT_FALSE(b.defined());
    EXPECT_FLOAT_EQ(c.at<float>({0}), 10.0f);
}

TEST(ArrayOperations, Cast) {
    std::vector<double> double_data{1.5, -2.7, 3.0};
    mt::Array a(std::span<const double>(double_data), {3});

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