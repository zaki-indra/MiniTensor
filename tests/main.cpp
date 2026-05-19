// tests/main.cpp

#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>

TEST(TensorTest, DefaultConstructor) {
    mt::Tensor t;
    EXPECT_FALSE(t.defined());
}

TEST(TensorTest, ZeroInitializer) {
    size_t     rows = 3, cols = 3;
    mt::Tensor t = mt::zeros({rows, cols});

    EXPECT_TRUE(t.defined());
    EXPECT_EQ(t.numel(), 9);
    EXPECT_EQ(t.shape(), mt::Shape({3, 3}));
    EXPECT_EQ(t.dtype(), mt::DataType::f32);
    EXPECT_EQ(t.device(), mt::DeviceType::cpu);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            EXPECT_FLOAT_EQ(t.at({i, j}), 0.0f);
        }
    }
}
