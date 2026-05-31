// tests/test_array_iterator.cpp
//
// Direct unit tests for the ArrayIterator infrastructure. Op-level tests
// already exercise the iterator indirectly; this file pins down the
// iterator's own contract — build() validation, kernel dispatch across
// dtypes, arity detection, kernel exceptions, aliasing, and the
// configuration seams (`check_same_*`).

#include "core/ArrayIterator.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>
#include <stdexcept>
#include <vector>

// =========================================================
// build() validation
// =========================================================

// In the build-failure tests we discard the (never-produced) iterator
// returned by build(); cast to void to silence [[nodiscard]].

TEST(ArrayIteratorTest, BuildThrowsWhenNoOutputConfigured) {
    mt::Array a = mt::ones({3});
    EXPECT_THROW({ (void)mt::ArrayIteratorConfig().add_input(a).build(); }, mt::ShapeError);
}

TEST(ArrayIteratorTest, BuildThrowsWhenNoInputsConfigured) {
    mt::Array out = mt::zeros({3});
    EXPECT_THROW({ (void)mt::ArrayIteratorConfig().add_output(out).build(); }, mt::ShapeError);
}

TEST(ArrayIteratorTest, BuildThrowsWhenTooManyInputs) {
    mt::Array out = mt::zeros({3});
    mt::Array a   = mt::ones({3});
    mt::Array b   = mt::ones({3});
    mt::Array c   = mt::ones({3});
    EXPECT_THROW({ (void)mt::ArrayIteratorConfig().add_output(out).add_input(a).add_input(b).add_input(c).build(); },
                 mt::ShapeError);
}

TEST(ArrayIteratorTest, BuildThrowsOnUndefinedOutput) {
    mt::Array undef;
    mt::Array a = mt::ones({3});
    EXPECT_THROW({ (void)mt::ArrayIteratorConfig().add_output(undef).add_input(a).build(); }, mt::UndefinedError);
}

TEST(ArrayIteratorTest, BuildThrowsOnUndefinedInput) {
    mt::Array out = mt::zeros({3});
    mt::Array undef;
    EXPECT_THROW({ (void)mt::ArrayIteratorConfig().add_output(out).add_input(undef).build(); }, mt::UndefinedError);
}

TEST(ArrayIteratorTest, BuildThrowsOnShapeMismatch) {
    mt::Array out = mt::zeros({2, 3});
    mt::Array a   = mt::ones({2, 3});
    mt::Array b   = mt::ones({3, 2});
    EXPECT_THROW({ (void)mt::ArrayIteratorConfig().add_output(out).add_input(a).add_input(b).build(); },
                 mt::ShapeError);
}

TEST(ArrayIteratorTest, BuildThrowsOnDtypeMismatch) {
    mt::Array out = mt::zeros({3}, mt::EDataType::f32);
    mt::Array a   = mt::ones({3}, mt::EDataType::f32);
    mt::Array b   = mt::ones({3}, mt::EDataType::i32);
    EXPECT_THROW({ (void)mt::ArrayIteratorConfig().add_output(out).add_input(a).add_input(b).build(); },
                 mt::DTypeError);
}

TEST(ArrayIteratorTest, BuildSucceedsWithSingleInput) {
    mt::Array out = mt::zeros({3});
    mt::Array a   = mt::ones({3});
    EXPECT_NO_THROW({
        [[maybe_unused]] auto iter = mt::ArrayIteratorConfig().add_output(out).add_input(a).build();
    });
}

TEST(ArrayIteratorTest, BuildSucceedsWithTwoInputs) {
    mt::Array out = mt::zeros({3});
    mt::Array a   = mt::ones({3});
    mt::Array b   = mt::ones({3});
    EXPECT_NO_THROW({
        [[maybe_unused]] auto iter =
          mt::ArrayIteratorConfig().add_output(out).add_input(a).add_input(b).build();
    });
}

// =========================================================
// Iterator accessors reflect configuration
// =========================================================

TEST(ArrayIteratorTest, AccessorsReflectCommonMetadata) {
    mt::Array out  = mt::zeros({4, 5}, mt::EDataType::i64);
    mt::Array a    = mt::ones({4, 5}, mt::EDataType::i64);
    mt::Array b    = mt::ones({4, 5}, mt::EDataType::i64);
    auto      iter = mt::ArrayIteratorConfig().add_output(out).add_input(a).add_input(b).build();

    EXPECT_EQ(iter.numel(), 20u);
    EXPECT_EQ(iter.common_dtype(), mt::EDataType::i64);
    EXPECT_EQ(iter.device(), mt::EDeviceType::cpu);
    EXPECT_EQ(iter.n_inputs(), 2u);
}

// =========================================================
// for_each: correctness across dtypes
// =========================================================

template <typename T>
static mt::Array make_array(const std::vector<T>& v) {
    return mt::Array(v);
}

TEST(ArrayIteratorTest, UnaryKernelRunsOverAllElements_f32) {
    mt::Array a   = make_array<float>({1.0f, 2.0f, 3.0f, 4.0f});
    mt::Array out = mt::zeros({4}, mt::EDataType::f32);
    mt::ArrayIteratorConfig().add_output(out).add_input(a).build().for_each(
      []<typename T>(T x) -> T { return x * static_cast<T>(2); });
    EXPECT_FLOAT_EQ(out.at<float>({0}), 2.0f);
    EXPECT_FLOAT_EQ(out.at<float>({3}), 8.0f);
}

TEST(ArrayIteratorTest, UnaryKernelWorksForI32) {
    mt::Array a   = make_array<int32_t>({1, 2, 3, 4});
    mt::Array out = mt::zeros({4}, mt::EDataType::i32);
    mt::ArrayIteratorConfig().add_output(out).add_input(a).build().for_each(
      []<typename T>(T x) -> T { return x + static_cast<T>(10); });
    EXPECT_EQ(out.at<int32_t>({0}), 11);
    EXPECT_EQ(out.at<int32_t>({3}), 14);
}

TEST(ArrayIteratorTest, UnaryKernelWorksForI64) {
    mt::Array a   = make_array<long long>({1LL, 2LL, 3LL});
    mt::Array out = mt::zeros({3}, mt::EDataType::i64);
    mt::ArrayIteratorConfig().add_output(out).add_input(a).build().for_each(
      []<typename T>(T x) -> T { return x * x; });
    EXPECT_EQ(out.at<int64_t>({2}), 9);
}

TEST(ArrayIteratorTest, UnaryKernelWorksForF64) {
    mt::Array a   = make_array<double>({1.5, 2.5});
    mt::Array out = mt::zeros({2}, mt::EDataType::f64);
    mt::ArrayIteratorConfig().add_output(out).add_input(a).build().for_each(
      []<typename T>(T x) -> T { return x + static_cast<T>(0.5); });
    EXPECT_DOUBLE_EQ(out.at<double>({0}), 2.0);
    EXPECT_DOUBLE_EQ(out.at<double>({1}), 3.0);
}

TEST(ArrayIteratorTest, BinaryKernelComputesElementwise) {
    mt::Array a   = make_array<float>({1.0f, 2.0f, 3.0f});
    mt::Array b   = make_array<float>({10.0f, 20.0f, 30.0f});
    mt::Array out = mt::zeros({3}, mt::EDataType::f32);
    mt::ArrayIteratorConfig().add_output(out).add_input(a).add_input(b).build().for_each(
      []<typename T>(T x, T y) -> T { return x + y; });
    EXPECT_FLOAT_EQ(out.at<float>({0}), 11.0f);
    EXPECT_FLOAT_EQ(out.at<float>({1}), 22.0f);
    EXPECT_FLOAT_EQ(out.at<float>({2}), 33.0f);
}

// =========================================================
// for_each: kernel-shape variants the iterator must accept
// =========================================================

TEST(ArrayIteratorTest, GenericLambdaKernelIsAccepted) {
    mt::Array a   = make_array<int32_t>({1, 2, 3});
    mt::Array b   = make_array<int32_t>({4, 5, 6});
    mt::Array out = mt::zeros({3}, mt::EDataType::i32);
    // Generic lambda (auto-deduced) instead of a template lambda.
    mt::ArrayIteratorConfig().add_output(out).add_input(a).add_input(b).build().for_each(
      [](auto x, auto y) { return x * y; });
    EXPECT_EQ(out.at<int32_t>({0}), 4);
    EXPECT_EQ(out.at<int32_t>({2}), 18);
}

TEST(ArrayIteratorTest, KernelCanBranchOnTypeViaIfConstexpr) {
    // Same kernel used across dtypes; integer path adds 1, float path adds 0.5.
    auto run_for = [](mt::Array a) {
        mt::Array out(a.shape(), a.dtype(), a.device());
        mt::ArrayIteratorConfig().add_output(out).add_input(a).build().for_each([]<typename T>(T x) -> T {
            if constexpr (std::is_integral_v<T>)
                return x + static_cast<T>(1);
            else
                return x + static_cast<T>(0.5);
        });
        return out;
    };

    mt::Array int_in = make_array<int32_t>({10, 20});
    mt::Array int_out = run_for(int_in);
    EXPECT_EQ(int_out.at<int32_t>({0}), 11);

    mt::Array flt_in  = make_array<float>({1.0f, 2.0f});
    mt::Array flt_out = run_for(flt_in);
    EXPECT_FLOAT_EQ(flt_out.at<float>({0}), 1.5f);
}

TEST(ArrayIteratorTest, ScalarValueViaCaptureWorks) {
    mt::Array a     = make_array<float>({1.0f, 2.0f, 3.0f});
    mt::Array out   = mt::zeros({3}, mt::EDataType::f32);
    double    scale = 3.0;
    mt::ArrayIteratorConfig().add_output(out).add_input(a).build().for_each(
      [scale]<typename T>(T x) -> T { return x * static_cast<T>(scale); });
    EXPECT_FLOAT_EQ(out.at<float>({2}), 9.0f);
}

// =========================================================
// for_each: short-circuits, exceptions, and aliasing
// =========================================================

TEST(ArrayIteratorTest, EmptyArrayIsNoOp) {
    mt::Array a   = mt::zeros({0});
    mt::Array out = mt::zeros({0});
    EXPECT_EQ(out.numel(), 0u);
    // Must not crash and must not invoke the kernel.
    bool kernel_invoked = false;
    mt::ArrayIteratorConfig().add_output(out).add_input(a).build().for_each(
      [&kernel_invoked]<typename T>(T x) -> T {
          kernel_invoked = true;
          return x;
      });
    EXPECT_FALSE(kernel_invoked);
}

TEST(ArrayIteratorTest, ExceptionThrownFromKernelPropagates) {
    mt::Array a   = make_array<float>({1.0f, 2.0f});
    mt::Array out = mt::zeros({2}, mt::EDataType::f32);
    EXPECT_THROW(
      {
          mt::ArrayIteratorConfig().add_output(out).add_input(a).build().for_each(
            []<typename T>(T x) -> T {
                if (x > static_cast<T>(1))
                    throw mt::ArithmeticError("intentional");
                return x;
            });
      },
      mt::ArithmeticError);
}

TEST(ArrayIteratorTest, OutputAliasedToInputIsValidInPlace) {
    // Common pattern: writing back into one of the inputs.
    mt::Array a = make_array<float>({1.0f, 2.0f, 3.0f});
    mt::ArrayIteratorConfig().add_output(a).add_input(a).build().for_each(
      []<typename T>(T x) -> T { return x + static_cast<T>(10); });
    EXPECT_FLOAT_EQ(a.at<float>({0}), 11.0f);
    EXPECT_FLOAT_EQ(a.at<float>({2}), 13.0f);
}

// =========================================================
// Arity mismatch: build()-validated count vs kernel signature
// =========================================================

TEST(ArrayIteratorTest, BinaryKernelWithOneInputThrowsLogicError) {
    mt::Array a   = make_array<float>({1.0f, 2.0f});
    mt::Array out = mt::zeros({2}, mt::EDataType::f32);
    auto      iter = mt::ArrayIteratorConfig().add_output(out).add_input(a).build();
    // Two-parameter kernel but only one input configured.
    EXPECT_THROW({ iter.for_each([]<typename T>(T x, T y) -> T { return x + y; }); }, std::logic_error);
}

TEST(ArrayIteratorTest, UnaryKernelWithTwoInputsThrowsLogicError) {
    mt::Array a   = make_array<float>({1.0f, 2.0f});
    mt::Array b   = make_array<float>({3.0f, 4.0f});
    mt::Array out = mt::zeros({2}, mt::EDataType::f32);
    auto      iter = mt::ArrayIteratorConfig().add_output(out).add_input(a).add_input(b).build();
    EXPECT_THROW({ iter.for_each([]<typename T>(T x) -> T { return x; }); }, std::logic_error);
}

// =========================================================
// Configuration seams (future-broadcasting plumbing)
// =========================================================

TEST(ArrayIteratorTest, DisablingShapeCheckAllowsBuildWithDifferentShapes) {
    // The shape-check flag is a future seam for broadcasting. We only verify
    // the plumbing — that build() accepts the mismatch when the check is
    // disabled. We do NOT exercise for_each in this state because v0's loop
    // would walk past the smaller buffer.
    mt::Array out = mt::zeros({2, 3});
    mt::Array a   = mt::ones({3, 2}); // same numel, different shape
    EXPECT_NO_THROW({
        [[maybe_unused]] auto iter =
          mt::ArrayIteratorConfig().add_output(out).add_input(a).check_same_shape(false).build();
    });
}

TEST(ArrayIteratorTest, DisablingDtypeCheckAllowsBuildWithDifferentDtypes) {
    mt::Array out = mt::zeros({3}, mt::EDataType::f32);
    mt::Array a   = mt::ones({3}, mt::EDataType::i32);
    EXPECT_NO_THROW({
        [[maybe_unused]] auto iter =
          mt::ArrayIteratorConfig().add_output(out).add_input(a).check_same_dtype(false).build();
    });
}
