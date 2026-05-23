// tests/test_array.cpp

#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>

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
// Array(Shape)
// ─────────────────────────────────────────────
TEST(ArrayConstructor, ShapeConstructor1D) {
    mt::Array t(mt::Shape{5});
    expect_defined(t, {5}, {1}, 5);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_FLOAT_EQ(t.at({i}), 0.0f) << "index " << i;
}

TEST(ArrayConstructor, ShapeConstructor2D) {
    mt::Array t(mt::Shape{3, 4});
    expect_defined(t, {3, 4}, {4, 1}, 12);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_FLOAT_EQ(t.at({i, j}), 0.0f);
}

TEST(ArrayConstructor, ShapeConstructor3D) {
    mt::Array t(mt::Shape{2, 3, 4});
    expect_defined(t, {2, 3, 4}, {12, 4, 1}, 24);
}

TEST(ArrayConstructor, ShapeConstructorScalar) {
    // A 1-element shape — not the same as default-constructed
    mt::Array t(mt::Shape{1});
    expect_defined(t, {1}, {1}, 1);
    EXPECT_FLOAT_EQ(t.at({0}), 0.0f);
}

// ─────────────────────────────────────────────
// Array(vector<float>) — single-argument
// ─────────────────────────────────────────────
TEST(ArrayConstructor, VectorConstructorSetsShape) {
    std::vector<float> data{1.0f, 2.0f, 3.0f};
    mt::Array          t(data);
    expect_defined(t, {3}, {1}, 3);
}

TEST(ArrayConstructor, VectorConstructorPreservesValues) {
    std::vector<float> data{4.0f, 5.0f, 6.0f};
    mt::Array          t(data);
    EXPECT_FLOAT_EQ(t.at({0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1}), 5.0f);
    EXPECT_FLOAT_EQ(t.at({2}), 6.0f);
}

// Regression: bug where data could be moved-from before .size() was read,
// leaving shape as {0}. Verify shape == {N}, not {0}.
TEST(ArrayConstructor, VectorConstructorShapeMatchesDataSize) {
    std::vector<float> data(7, 1.0f);
    const auto         expected_size = data.size(); // capture before move
    mt::Array          t(std::move(data));
    EXPECT_EQ(t.numel(), expected_size);
    EXPECT_EQ(t.shape(), mt::Shape({expected_size}));
}

// explicit: should not compile
// TEST(ArrayConstructor, NoImplicitConversionFromVector) {
//     mt::Array t = std::vector<float>{1.0f, 2.0f};  // must NOT compile
// }

// ─────────────────────────────────────────────
// Array(vector<float>, Shape)
// ─────────────────────────────────────────────
TEST(ArrayConstructor, DataAndShapeConstructor) {
    std::vector<float> data{1, 2, 3, 4, 5, 6};
    mt::Array          t(data, {2, 3});
    expect_defined(t, {2, 3}, {3, 1}, 6);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

// Regression: old code called Array(data, shape) inside the body (a no-op
// temporary). Verify delegation actually sets dtype/device/defined.
TEST(ArrayConstructor, DataShapeConstructorDelegatesCorrectly) {
    mt::Array t(std::vector<float>(6, 0.f), mt::Shape{2, 3});
    EXPECT_EQ(t.dtype(), mt::DataType::f32);
    EXPECT_EQ(t.device(), mt::DeviceType::cpu);
    EXPECT_TRUE(t.defined());
}

// ─────────────────────────────────────────────
// Array(vector<float>, Shape, DataType, DeviceType) — canonical
// ─────────────────────────────────────────────
TEST(ArrayConstructor, CanonicalConstructorAllFields) {
    std::vector<float> data{1, 2, 3, 4};
    mt::Array          t(data, {2, 2}, mt::DataType::f32, mt::DeviceType::cpu);
    expect_defined(t, {2, 2}, {2, 1}, 4, mt::DataType::f32, mt::DeviceType::cpu);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 4.0f);
}

TEST(ArrayConstructor, CanonicalConstructorNumelMatchesData) {
    std::vector<float> data(12, 1.f);
    mt::Array          t(data, {3, 4}, mt::DataType::f32, mt::DeviceType::cpu);
    EXPECT_EQ(t.numel(), data.size());
}

// ─────────────────────────────────────────────
// Strides
// ─────────────────────────────────────────────
TEST(ArrayStrides, RowMajor1D) {
    mt::Array t(mt::Shape{8});
    EXPECT_EQ(t.strides(), mt::Shape({1}));
}

TEST(ArrayStrides, RowMajor2D) {
    mt::Array t(mt::Shape{4, 5});
    EXPECT_EQ(t.strides(), mt::Shape({5, 1}));
}

TEST(ArrayStrides, RowMajor3D) {
    mt::Array t(mt::Shape{2, 3, 4});
    EXPECT_EQ(t.strides(), mt::Shape({12, 4, 1}));
}

TEST(ArrayStrides, RowMajor4D) {
    mt::Array t(mt::Shape{2, 3, 4, 5});
    EXPECT_EQ(t.strides(), mt::Shape({60, 20, 5, 1}));
}

// ─────────────────────────────────────────────
// Copy and move
// ─────────────────────────────────────────────
TEST(ArrayCopyMove, CopyConstructorIsIndependent) {
    mt::Array a(std::vector<float>{1, 2, 3}, {3});
    mt::Array b = a; // copy
    EXPECT_EQ(b.numel(), a.numel());
    EXPECT_EQ(b.shape(), a.shape());
    EXPECT_EQ(b.strides(), a.strides());
    EXPECT_FLOAT_EQ(b.at({0}), 1.0f);
}

TEST(ArrayCopyMove, MoveConstructorLeavesSourceUndefined) {
    mt::Array a(std::vector<float>{1, 2, 3}, {3});
    mt::Array b = std::move(a);
    EXPECT_TRUE(b.defined());
    EXPECT_EQ(b.numel(), 3);
    EXPECT_FALSE(a.defined());
}

// ─────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────

TEST(ArrayEdgeCases, SingleElementArray) {
    mt::Array t(mt::Shape{1, 1, 1});
    expect_defined(t, {1, 1, 1}, {1, 1, 1}, 1);
    EXPECT_FLOAT_EQ(t.at({0, 0, 0}), 0.0f);
}

TEST(ArrayEdgeCases, LargeFlat) {
    constexpr std::size_t N = 1'000'000;
    mt::Array             t(mt::Shape{N});
    EXPECT_EQ(t.numel(), N);
    EXPECT_FLOAT_EQ(t.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({N - 1}), 0.0f);
}

// ─────────────────────────────────────────────
// Factory functions (zeros / ones)
// ─────────────────────────────────────────────
TEST(ArrayFactory, ZerosAllFields) {
    mt::Array t = mt::zeros({3, 3});
    expect_defined(t, {3, 3}, {3, 1}, 9);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(t.at({i, j}), 0.0f);
}

TEST(ArrayFactory, OnesAllFields) {
    mt::Array t = mt::ones({3, 3});
    expect_defined(t, {3, 3}, {3, 1}, 9);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(t.at({i, j}), 1.0f);
}

TEST(ArrayFactory, Zeros1D) {
    mt::Array t = mt::zeros({5});
    expect_defined(t, {5}, {1}, 5);
}

TEST(ArrayFactory, Ones3D) {
    mt::Array t = mt::ones({2, 3, 4});
    expect_defined(t, {2, 3, 4}, {12, 4, 1}, 24);
    EXPECT_FLOAT_EQ(t.at({1, 2, 3}), 1.0f);
}

// ─────────────────────────────────────────────
// Element-wise operations
// ─────────────────────────────────────────────
TEST(ArrayElementwise, Add) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 2});
    mt::Array c = a + b;
    expect_defined(c, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 2.0f);
}

TEST(ArrayElementwise, Subtract) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 2});
    mt::Array c = a - b;
    expect_defined(c, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 0.0f);
}

TEST(ArrayElementwise, Multiply) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 2});
    mt::Array c = a * b;
    expect_defined(c, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 1.0f);
}

TEST(ArrayElementwise, Divide) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 2});
    mt::Array c = a / b;
    expect_defined(c, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 1.0f);
}

TEST(ArrayElementwise, AddMismatchedShapes) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({3, 3});
    mt::Array c = a + b;
    EXPECT_FALSE(c.defined());
}

// ---------------------------------------------------------
// Scalar operations
// ---------------------------------------------------------
TEST(ArrayScalarOps, Add) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = a + 3.0f;
    expect_defined(b, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(b.at({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(b.at({1, 1}), 4.0f);
}

TEST(ArrayScalarOps, Subtract) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = a - 0.5f;
    expect_defined(b, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(b.at({0, 0}), 0.5f);
    EXPECT_FLOAT_EQ(b.at({1, 1}), 0.5f);
}

TEST(ArrayScalarOps, Multiply) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = a * 4.0f;
    expect_defined(b, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(b.at({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(b.at({1, 1}), 4.0f);
}

TEST(ArrayScalarOps, Divide) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = a / 0.5f;
    expect_defined(b, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(b.at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(b.at({1, 1}), 2.0f);
}

// ─────────────────────────────────────────────
// Multi-DataType Support
// ─────────────────────────────────────────────
TEST(ArrayMultiDataType, DoublePrecision) {
    std::vector<double> data{1.5f, 2.5f, 3.5f};
    mt::Array           a(data.data(), {1, 3}, mt::DataType::f64);
    EXPECT_EQ(a.dtype(), mt::DataType::f64);
    EXPECT_EQ(a.numel(), 3);

    // Test scalar operations and item() on double
    mt::Array b = a + 2.0f;
    EXPECT_EQ(b.dtype(), mt::DataType::f64);
    EXPECT_TRUE(b.defined());

    // Convert to a 1-element slice or clone to check items via item()
    mt::Array slice(std::vector<float>{b.get_item_as_float(0)}, {1}, mt::DataType::f32);
    EXPECT_FLOAT_EQ(slice.item(), 3.5f);
}

TEST(ArrayMultiDataType, CloneSupport) {
    std::vector<float> data{4.5f, 5.5f};
    mt::Array          a(data, {2}, mt::DataType::f64);
    mt::Array          b = a.clone();
    EXPECT_EQ(b.dtype(), mt::DataType::f64);
    EXPECT_EQ(b.numel(), 2);
    mt::Array slice(std::vector<float>{b.get_item_as_float(0)}, {1}, mt::DataType::f32);
    EXPECT_FLOAT_EQ(slice.item(), 4.5f);
}