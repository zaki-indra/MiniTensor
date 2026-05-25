// tests/test_array_error_handling.cpp

#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>
#include <stdexcept>
#include <vector>

// Suppress unused result/value warnings for exception tests
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-value"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4834) // discarding return value of function with 'nodiscard' attribute
#pragma warning(disable: 4702) // unreachable code
#endif

// ─────────────────────────────────────────────
// 1. Undefined Tensor Usage Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, UndefinedTensorThrowsUndefinedError) {
    mt::Array undef;

    // Accessing data/items on undefined array
    EXPECT_THROW(undef.item<float>(), mt::UndefinedError);
    EXPECT_THROW(undef.reshape({1}), mt::UndefinedError);
    EXPECT_THROW(undef.flatten(), mt::UndefinedError);

    // Math unary ops on undefined array
    EXPECT_THROW(mt::sin(undef), mt::UndefinedError);
    EXPECT_THROW(mt::cos(undef), mt::UndefinedError);
    EXPECT_THROW(mt::tan(undef), mt::UndefinedError);
    EXPECT_THROW(mt::exp(undef), mt::UndefinedError);
    EXPECT_THROW(mt::log(undef), mt::UndefinedError);
    EXPECT_THROW(mt::sqrt(undef), mt::UndefinedError);

    // Math binary ops on undefined array
    mt::Array defined = mt::ones({2, 2});
    EXPECT_THROW(undef + defined, mt::UndefinedError);
    EXPECT_THROW(defined + undef, mt::UndefinedError);
    EXPECT_THROW(undef - defined, mt::UndefinedError);
    EXPECT_THROW(undef * defined, mt::UndefinedError);
    EXPECT_THROW(mt::add(undef, defined), mt::UndefinedError);
    EXPECT_THROW(mt::subtract(defined, undef), mt::UndefinedError);
    EXPECT_THROW(mt::multiply(undef, defined), mt::UndefinedError);
    EXPECT_THROW(mt::divide(defined, undef), mt::UndefinedError);

    // Scalar ops on undefined array
    EXPECT_THROW(undef + 5.0, mt::UndefinedError);
    EXPECT_THROW(undef - 5.0, mt::UndefinedError);
    EXPECT_THROW(undef * 5.0, mt::UndefinedError);
    EXPECT_THROW(undef / 5.0, mt::UndefinedError);
}

// ─────────────────────────────────────────────
// 2. Device Mismatch Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, DeviceMismatchThrowsDeviceError) {
    mt::Array a = mt::ones({2, 2});

    // Device CUDA unsupported transfer throws DeviceError
    EXPECT_THROW(a.to(mt::DeviceType::cuda), mt::DeviceError);
}

// ─────────────────────────────────────────────
// 3. Shape Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, ShapeMismatchThrowsShapeError) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 3});

    // Elementwise binary operations with mismatched shapes
    EXPECT_THROW(a + b, mt::ShapeError);
    EXPECT_THROW(a - b, mt::ShapeError);
    EXPECT_THROW(a * b, mt::ShapeError);
    EXPECT_THROW(mt::add(a, b), mt::ShapeError);
    EXPECT_THROW(mt::subtract(a, b), mt::ShapeError);
    EXPECT_THROW(mt::multiply(a, b), mt::ShapeError);
    EXPECT_THROW(mt::divide(a, b), mt::ShapeError);

    // item() on multi-element array
    EXPECT_THROW(a.item<float>(), mt::ShapeError);

    // at() with wrong number of dimensions/indices
    EXPECT_THROW(a.at<float>({0}), mt::ShapeError);
    EXPECT_THROW(a.at<float>({0, 0, 0}), mt::ShapeError);

    // Reshape shape error scenarios
    EXPECT_THROW(a.reshape({3}), mt::ShapeError); // incompatible size
    EXPECT_THROW(a.reshape({static_cast<std::size_t>(-1), static_cast<std::size_t>(-1)}), mt::ShapeError); // multiple -1
    EXPECT_THROW(a.reshape({static_cast<std::size_t>(-1), 3}), mt::ShapeError); // incompatible -1

    // Flatten shape error scenarios
    mt::Array c = mt::ones({2, 3, 4});
    EXPECT_THROW(c.flatten(2, 4), mt::ShapeError); // start/end dims out of bounds
    EXPECT_THROW(c.flatten(2, 1), mt::ShapeError); // start > end
}

// ─────────────────────────────────────────────
// 4. Type & DType Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, TypeMismatchThrowsTypeErrorAndDTypeError) {
    // 4.1 Type mismatch between requested dynamic template accessor type and actual DataType (mt::TypeError)
    mt::Array a = mt::ones({2, 2}, mt::DataType::i32);
    EXPECT_THROW(a.data<float>(), mt::TypeError);
    EXPECT_THROW(a.at<float>({0, 0}), mt::TypeError);

    mt::Array a_single = mt::full({1}, 5, mt::DataType::i32);
    EXPECT_THROW(a_single.item<float>(), mt::TypeError);

    // 4.2 DataType mismatch between two arrays in binary operations (mt::DTypeError)
    mt::Array b_float = mt::ones({2, 2}, mt::DataType::f32);
    mt::Array b_int = mt::ones({2, 2}, mt::DataType::i32);
    EXPECT_THROW(b_float + b_int, mt::DTypeError);
    EXPECT_THROW(b_float - b_int, mt::DTypeError);
    EXPECT_THROW(b_float * b_int, mt::DTypeError);
    EXPECT_THROW(mt::add(b_float, b_int), mt::DTypeError);
    EXPECT_THROW(mt::subtract(b_float, b_int), mt::DTypeError);
    EXPECT_THROW(mt::multiply(b_float, b_int), mt::DTypeError);
    EXPECT_THROW(mt::divide(b_float, b_int), mt::DTypeError);
}

// ─────────────────────────────────────────────
// 5. Index Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, IndexOutOfBoundsThrowsIndexError) {
    mt::Array a = mt::ones({2, 3});

    // Access out of bounds indices
    EXPECT_THROW(a.at<float>({2, 0}), mt::IndexError);
    EXPECT_THROW(a.at<float>({0, 3}), mt::IndexError);
    EXPECT_THROW(a.at<float>({2, 3}), mt::IndexError);
}

// ─────────────────────────────────────────────
// 6. Arithmetic Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, ArithmeticErrorsThrowArithmeticError) {
    // 6.1 Square root of negative integer
    std::vector<int32_t> neg_data{4, -9, 16};
    mt::Array neg_arr(neg_data);
    EXPECT_THROW(mt::sqrt(neg_arr), mt::ArithmeticError);

    // 6.2 Integer division by zero
    std::vector<int32_t> num{10, 20};
    std::vector<int32_t> den{2, 0};
    mt::Array n_arr(num);
    mt::Array d_arr(den);
    EXPECT_THROW(mt::divide(n_arr, d_arr), mt::ArithmeticError);
}

// ─────────────────────────────────────────────
// 7. Error Message Format verification
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, ErrorMessageFormattingContainsCorrectDetail) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 3});

    try {
        a + b;
        FAIL() << "Expected mt::ShapeError to be thrown";
    } catch (const mt::ShapeError& e) {
        std::string msg = e.what();
        // Check that it contains context information like file, line, and descriptive text
        EXPECT_TRUE(msg.find("ShapeError") != std::string::npos);
        EXPECT_TRUE(msg.find("Array.cpp") != std::string::npos);
        EXPECT_TRUE(msg.find("Shape mismatch") != std::string::npos);
        EXPECT_TRUE(msg.find("[2, 2]") != std::string::npos);
        EXPECT_TRUE(msg.find("[2, 3]") != std::string::npos);
    }
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
