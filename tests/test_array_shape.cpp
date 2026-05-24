// tests/test_array_shape.cpp

#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>
#include <stdexcept>
#include <vector>

// ─────────────────────────────────────────────
// Strides
// ─────────────────────────────────────────────
TEST(ArrayShapeTest, DefaultStridesAreRowMajor) {
    mt::Array t = mt::zeros({2, 3, 4});
    EXPECT_EQ(t.strides(), mt::Shape({12, 4, 1}));
}

// ─────────────────────────────────────────────
// Reshaping
// ─────────────────────────────────────────────
TEST(ArrayShapeTest, ReshapeMaintainsSharedMemoryView) {
    std::vector<float> data{1, 2, 3, 4, 5, 6};
    mt::Array          a(std::span<const float>(data), {2, 3});

    // Reshape to contiguous 1D
    mt::Array b = a.reshape({6});
    EXPECT_EQ(b.shape(), mt::Shape({6}));
    EXPECT_FLOAT_EQ(b.at<float>({0}), 1.0f);
    EXPECT_FLOAT_EQ(b.at<float>({5}), 6.0f);

    // Modify view and verify shared storage
    b.at<float>({0}) = 99.0f;
    EXPECT_FLOAT_EQ(a.at<float>({0, 0}), 99.0f);
}

TEST(ArrayShapeTest, ReshapeInfersNegativeOneDimension) {
    std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8};
    mt::Array            a(data);

    // Test -1 inference at start
    mt::Array b = a.reshape({static_cast<std::size_t>(-1), 4});
    EXPECT_EQ(b.shape(), mt::Shape({2, 4}));

    // Test -1 inference at end
    mt::Array c = a.reshape({4, static_cast<std::size_t>(-1)});
    EXPECT_EQ(c.shape(), mt::Shape({4, 2}));
}

TEST(ArrayShapeTest, ReshapeThrowsOnInvalidDimensions) {
    std::vector<float> data{1, 2, 3, 4};
    mt::Array          a(data);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

    // Mismatched size
    EXPECT_THROW(a.reshape({3}), std::runtime_error);

    // Multiple -1 elements
    EXPECT_THROW(a.reshape({static_cast<std::size_t>(-1), static_cast<std::size_t>(-1)}), std::runtime_error);

    // Undefined array reshape
    mt::Array undef;

    EXPECT_THROW(undef.reshape({1}), std::runtime_error);
#pragma GCC diagnostic pop
}

// ─────────────────────────────────────────────
// Flattening
// ─────────────────────────────────────────────
TEST(ArrayShapeTest, FlattenReducesDimensions) {
    mt::Array a = mt::ones({2, 3, 4});

    // Flatten all dims
    mt::Array f_all = a.flatten();
    EXPECT_EQ(f_all.shape(), mt::Shape({24}));

    // Flatten a range
    mt::Array f_range = a.flatten(0, 1);
    EXPECT_EQ(f_range.shape(), mt::Shape({6, 4}));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    // Out of range dimensions
    EXPECT_THROW(a.flatten(2, 4), std::runtime_error);
    EXPECT_THROW(a.flatten(2, 1), std::runtime_error);
#pragma GCC diagnostic pop
}

// ─────────────────────────────────────────────
// Boundary Checking
// ─────────────────────────────────────────────
TEST(ArrayShapeTest, AtMethodThrowsOnOutOfBoundsIndex) {
    mt::Array a = mt::ones({2, 3});

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    // Verify successful access
    EXPECT_NO_THROW(a.at<float>({0, 0}));
    EXPECT_NO_THROW(a.at<float>({1, 2}));

    // Dimension size mismatch
    EXPECT_THROW(a.at<float>({0}), std::out_of_range);
    EXPECT_THROW(a.at<float>({0, 0, 0}), std::out_of_range);

    // Index out of bounds
    EXPECT_THROW(a.at<float>({2, 0}), std::out_of_range);
    EXPECT_THROW(a.at<float>({0, 3}), std::out_of_range);
#pragma GCC diagnostic pop
}