// tests/test_array.cpp

#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>

TEST(ArrayTest, DefaultConstructor) {
    mt::Array t;
    EXPECT_FALSE(t.defined());
}

TEST(ArrayTest, TestZeros) {
    size_t     rows = 3, cols = 3;
    mt::Array t = mt::zeros({rows, cols});

    EXPECT_TRUE(t.defined());
    EXPECT_EQ(t.numel(), 9);
    EXPECT_EQ(t.shape(), mt::Shape({3, 3}));
    EXPECT_EQ(t.strides(), mt::Shape({3, 1}));
    EXPECT_EQ(t.dtype(), mt::DataType::f32);
    EXPECT_EQ(t.device(), mt::DeviceType::cpu);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            EXPECT_FLOAT_EQ(t.at({i, j}), 0.0f);
        }
    }
}

TEST(ArrayTest, TestOnes) {
    size_t     rows = 3, cols = 3;
    mt::Array t = mt::ones({rows, cols});

    EXPECT_TRUE(t.defined());
    EXPECT_EQ(t.numel(), 9);
    EXPECT_EQ(t.shape(), mt::Shape({3, 3}));
    EXPECT_EQ(t.strides(), mt::Shape({3, 1}));
    EXPECT_EQ(t.dtype(), mt::DataType::f32);
    EXPECT_EQ(t.device(), mt::DeviceType::cpu);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            EXPECT_FLOAT_EQ(t.at({i, j}), 1.0f);
        }
    }
}
