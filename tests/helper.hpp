// tests/helper.hpp

#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>

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