// tests/test_array_error_handling.cpp

#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>
#include <stdexcept>
#include <vector>

using BinOp = std::function<mt::Array(const mt::Array&, const mt::Array&)>;

// ─────────────────────────────────────────────
// 1. Undefined Tensor Usage Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, UndefinedTensorThrowsUndefinedError) {
    mt::Array undef;

    // Accessing data/items on undefined array
    EXPECT_THROW({ auto _ = undef.item<float>(); }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = undef.reshape({1}); }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = undef.flatten(); }, mt::UndefinedError);

    // Math unary ops on undefined array
    EXPECT_THROW({ auto _ = mt::sin(undef); }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = mt::cos(undef); }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = mt::tan(undef); }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = mt::exp(undef); }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = mt::log(undef); }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = mt::sqrt(undef); }, mt::UndefinedError);

    // Scalar ops on undefined array
    EXPECT_THROW({ auto _ = undef + 5.0; }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = undef - 5.0; }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = undef * 5.0; }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = undef / 5.0; }, mt::UndefinedError);
}

class UndefinedBinaryTest : public ::testing::TestWithParam<BinOp> {};
TEST_P(UndefinedBinaryTest, ThrowsUndefinedError) {
    mt::Array undef;
    mt::Array def = mt::ones({2, 2});
    EXPECT_THROW({ auto _ = GetParam()(undef, def); }, mt::UndefinedError);
    EXPECT_THROW({ auto _ = GetParam()(def, undef); }, mt::UndefinedError);
}

// ─────────────────────────────────────────────
// 2. Device Mismatch Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, DeviceMismatchThrowsDeviceError) {
    mt::Array a = mt::ones({2, 2});

    // Device CUDA unsupported transfer throws DeviceError
    EXPECT_THROW({ auto _ = a.to(mt::DeviceType::cuda); }, mt::DeviceError);
}

// ─────────────────────────────────────────────
// 3. Shape Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, ShapeMismatchThrowsShapeError) {
    mt::Array a = mt::ones({2, 2});

    // item() on multi-element array
    EXPECT_THROW({ auto _ = a.item<float>(); }, mt::ShapeError);

    // at() with wrong number of dimensions/indices
    EXPECT_THROW({ auto _ = a.at<float>({0}); }, mt::ShapeError);
    EXPECT_THROW({ auto _ = a.at<float>({0, 0, 0}); }, mt::ShapeError);

    // Reshape shape error scenarios
    EXPECT_THROW({ auto _ = a.reshape({3}); }, mt::ShapeError); // incompatible size
    EXPECT_THROW({ auto _ = a.reshape({static_cast<std::size_t>(-1), static_cast<std::size_t>(-1)}); }, mt::ShapeError); // multiple -1
    EXPECT_THROW({ auto _ = a.reshape({static_cast<std::size_t>(-1), 3}); }, mt::ShapeError); // incompatible -1

    // Flatten shape error scenarios
    mt::Array c = mt::ones({2, 3, 4});
    EXPECT_THROW({ auto _ = c.flatten(2, 4); }, mt::ShapeError); // start/end dims out of bounds
    EXPECT_THROW({ auto _ = c.flatten(2, 1); }, mt::ShapeError); // start > end
}

class ShapeBinaryTest : public ::testing::TestWithParam<BinOp> {};
TEST_P(ShapeBinaryTest, ThrowsShapeError) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 3});
    EXPECT_THROW({ auto _ = GetParam()(a, b); }, mt::ShapeError);
}

// ─────────────────────────────────────────────
// 4. Type & DType Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, TypeMismatchThrowsTypeErrorAndDTypeError) {
    // 4.1 Type mismatch between requested dynamic template accessor type and actual DataType (mt::TypeError)
    mt::Array a = mt::ones({2, 2}, mt::DataType::i32);
    EXPECT_THROW({ auto _ = a.data<float>(); }, mt::TypeError);
    EXPECT_THROW({ auto _ = a.at<float>({0, 0}); }, mt::TypeError);

    mt::Array a_single = mt::full({1}, 5, mt::DataType::i32);
    EXPECT_THROW({ auto _ = a_single.item<float>(); }, mt::TypeError);
}

class DTypeBinaryTest : public ::testing::TestWithParam<BinOp> {};
TEST_P(DTypeBinaryTest, ThrowsDTypeError) {
    mt::Array b_float = mt::ones({2, 2}, mt::DataType::f32);
    mt::Array b_int = mt::ones({2, 2}, mt::DataType::i32);
    EXPECT_THROW({ auto _ = GetParam()(b_float, b_int); }, mt::DTypeError);
}

// ─────────────────────────────────────────────
// Parameterize all binary op tests
// ─────────────────────────────────────────────
INSTANTIATE_TEST_SUITE_P(AllBinaryOps, UndefinedBinaryTest, ::testing::Values(
    [](const mt::Array& a, const mt::Array& b){ return a + b; },
    [](const mt::Array& a, const mt::Array& b){ return a - b; },
    [](const mt::Array& a, const mt::Array& b){ return a * b; },
    [](const mt::Array& a, const mt::Array& b){ return mt::divide(a, b); }
));
INSTANTIATE_TEST_SUITE_P(AllBinaryOps, ShapeBinaryTest, ::testing::Values(
    [](const mt::Array& a, const mt::Array& b){ return a + b; },
    [](const mt::Array& a, const mt::Array& b){ return a - b; },
    [](const mt::Array& a, const mt::Array& b){ return a * b; },
    [](const mt::Array& a, const mt::Array& b){ return mt::divide(a, b); }
));
INSTANTIATE_TEST_SUITE_P(AllBinaryOps, DTypeBinaryTest, ::testing::Values(
    [](const mt::Array& a, const mt::Array& b){ return a + b; },
    [](const mt::Array& a, const mt::Array& b){ return a - b; },
    [](const mt::Array& a, const mt::Array& b){ return a * b; },
    [](const mt::Array& a, const mt::Array& b){ return mt::divide(a, b); }
));

// ─────────────────────────────────────────────
// 5. Index Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, IndexOutOfBoundsThrowsIndexError) {
    mt::Array a = mt::ones({2, 3});

    // Access out of bounds indices
    EXPECT_THROW({ auto _ = a.at<float>({2, 0}); }, mt::IndexError);
    EXPECT_THROW({ auto _ = a.at<float>({0, 3}); }, mt::IndexError);
    EXPECT_THROW({ auto _ = a.at<float>({2, 3}); }, mt::IndexError);
}

// ─────────────────────────────────────────────
// 6. Arithmetic Error Tests
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, ArithmeticErrorsThrowArithmeticError) {
    // 6.1 Square root of negative integer
    std::vector<int32_t> neg_data{4, -9, 16};
    mt::Array neg_arr(neg_data);
    EXPECT_THROW({ auto _ = mt::sqrt(neg_arr); }, mt::ArithmeticError);

    // 6.2 Integer division by zero
    // Since we remove `mt::divide`, we should use `/` if it exists. But wait, `Array::operator/` is only for scalar division in the current REFACTOR_NOTES.md (item 1). Wait, does Array have elementwise division operator/?
    // REFACTOR_NOTES.md item 1 says: "three binary operators (+, -, *), one binary helper (divide), four scalar operators (+, -, *, /)".
    // So there is NO binary `operator/` right now, just a `divide` helper function!
    // But wait, the task is to replace `divide`? Let's check `REFACTOR_NOTES.md`:
    // It says "three binary operators (+, -, *), one binary helper (divide)"
    // If we delete the free-functions, what happens to `divide`? Let's look at item 7:
    // "Either define them inline in Core.hpp (no extra TU cost) or delete them and tell users to call Array::zeros / a + b directly."
    // It doesn't mention deleting `divide`. But `divide` is a free function. Is it `mt::divide`?
    // Let's keep `mt::divide` in the tests for now, and see if it's there. Actually, let's look at `tests/test_array_error_handling.cpp` line 150: `EXPECT_THROW(mt::divide(n_arr, d_arr), mt::ArithmeticError);`
    // I will include `mt::divide(a, b)` in the parameterized tests too! Let's modify the values:
    
    std::vector<int32_t> num{10, 20};
    std::vector<int32_t> den{2, 0};
    mt::Array n_arr(num);
    mt::Array d_arr(den);
    EXPECT_THROW({ auto _ = mt::divide(n_arr, d_arr); }, mt::ArithmeticError);
}

// ─────────────────────────────────────────────
// 7. Error Message Format verification
// ─────────────────────────────────────────────
TEST(ArrayErrorHandlingTest, ErrorMessageFormattingContainsCorrectDetail) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 3});

    try {
        auto _ = a + b;
        FAIL() << "Expected mt::ShapeError to be thrown";
    } catch (const mt::ShapeError& e) {
        std::string msg = e.what();
        // Check that it contains context information like file, line, and descriptive text
        EXPECT_TRUE(msg.find("ShapeError") != std::string::npos);
        // Validation now lives in the ArrayIterator builder.
        EXPECT_TRUE(msg.find("ArrayIterator.cpp") != std::string::npos);
        EXPECT_TRUE(msg.find("Shape mismatch") != std::string::npos);
        EXPECT_TRUE(msg.find("[2, 2]") != std::string::npos);
        EXPECT_TRUE(msg.find("[2, 3]") != std::string::npos);
    }
}
