// tests/main.cpp

#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>

TEST(TensorTest, DefaultConstructor) {
    mt::Tensor t;
    EXPECT_FALSE(t.defined());
}

TEST(TensorTest, ZeroInitializer) {
    mt::Tensor t = mt::zeros({3, 3});

    EXPECT_TRUE(t.defined());
    EXPECT_EQ(t.numel(), 9);
    EXPECT_EQ(t.shape(), mt::Shape({3, 3}));
    EXPECT_EQ(t.dtype(), mt::DataType::f32);
    EXPECT_EQ(t.device(), mt::DeviceType::cpu);
    EXPECT_FALSE(t.requires_grad());
}
