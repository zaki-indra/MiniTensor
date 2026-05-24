// tests/test_array_math.cpp

#include "helper.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <minitensor/minitensor.hpp>
#include <stdexcept>
#include <vector>

static constexpr double PI = 3.14159265358979323846;

// ─────────────────────────────────────────────
// Element-wise arithmetic
// ─────────────────────────────────────────────
TEST(ArrayMathTest, ElementwiseAddition) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 2});
    mt::Array c = a + b;
    expect_defined(c, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 2.0f);
}

TEST(ArrayMathTest, ElementwiseSubtraction) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 2});
    mt::Array c = a - b;
    expect_defined(c, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 0.0f);
}

TEST(ArrayMathTest, ElementwiseMultiplication) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::ones({2, 2});
    mt::Array c = a * b;
    expect_defined(c, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 1.0f);
}

TEST(ArrayMathTest, BinaryArithmeticHelpersWork) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = mt::full({2, 2}, 3.0f);

    mt::Array c_add = mt::add(a, b);
    EXPECT_FLOAT_EQ(c_add.at<float>({0, 0}), 4.0f);

    mt::Array c_sub = mt::subtract(b, a);
    EXPECT_FLOAT_EQ(c_sub.at<float>({0, 0}), 2.0f);

    mt::Array c_mul = mt::multiply(a, b);
    EXPECT_FLOAT_EQ(c_mul.at<float>({0, 0}), 3.0f);
}

// ─────────────────────────────────────────────
// Scalar operations
// ─────────────────────────────────────────────
TEST(ArrayMathTest, ScalarMultiplication) {
    mt::Array a = mt::ones({2, 2});
    mt::Array b = a * 4.0;
    expect_defined(b, {2, 2}, {2, 1}, 4);
    EXPECT_FLOAT_EQ(b.at<float>({0, 0}), 4.0f);
}

// ─────────────────────────────────────────────
// Multi-DataType Support & Preserving Precision
// ─────────────────────────────────────────────
TEST(ArrayMathTest, MathOperationsPreserveDoublePrecision) {
    std::vector<double> data{1.5, 2.5, 3.5};
    mt::Array           a(std::span<const double>(data), {3});

    mt::Array b = a + 2.0;
    EXPECT_EQ(b.dtype(), mt::DataType::f64);
    EXPECT_DOUBLE_EQ(b.at<double>({0}), 3.5);
    EXPECT_DOUBLE_EQ(b.at<double>({2}), 5.5);
}

TEST(ArrayMathTest, MathOperationsPreserveIntegerPrecision) {
    std::vector<int32_t> data{10, 20, 30};
    mt::Array            a(data);

    // Scaler passed as double, safely truncates inside dispatch
    mt::Array b = a * 2.0;
    EXPECT_EQ(b.dtype(), mt::DataType::i32);
    EXPECT_EQ(b.at<int32_t>({0}), 20);
    EXPECT_EQ(b.at<int32_t>({2}), 60);
}

// ─────────────────────────────────────────────
// Trigonometric Operations
// ─────────────────────────────────────────────
TEST(ArrayMathTest, SineCalculatesCorrectly) {
    std::vector<double> data{0.0, PI / 2.0, PI};
    mt::Array           a(std::span<const double>(data), {3});
    mt::Array           s = mt::sin(a);

    EXPECT_EQ(s.dtype(), mt::DataType::f64);
    EXPECT_NEAR(s.at<double>({0}), 0.0, 1e-7);
    EXPECT_NEAR(s.at<double>({1}), 1.0, 1e-7);
    EXPECT_NEAR(s.at<double>({2}), 0.0, 1e-7);
}

TEST(ArrayMathTest, CosineCalculatesCorrectly) {
    std::vector<double> data{0.0, PI / 2.0, PI};
    mt::Array           a(std::span<const double>(data), {3});
    mt::Array           c = mt::cos(a);

    EXPECT_NEAR(c.at<double>({0}), 1.0, 1e-7);
    EXPECT_NEAR(c.at<double>({1}), 0.0, 1e-7);
    EXPECT_NEAR(c.at<double>({2}), -1.0, 1e-7);
}

TEST(ArrayMathTest, TangentCalculatesCorrectly) {
    std::vector<double> data{0.0, PI};
    mt::Array           a(std::span<const double>(data), {2});
    mt::Array           t = mt::tan(a);

    EXPECT_NEAR(t.at<double>({0}), 0.0, 1e-7);
    EXPECT_NEAR(t.at<double>({1}), 0.0, 1e-7);
}

// ─────────────────────────────────────────────
// Exponential and Roots
// ─────────────────────────────────────────────
TEST(ArrayMathTest, SquareRootCalculatesCorrectly) {
    std::vector<float> data{1.0f, 4.0f, 9.0f};
    mt::Array          a(data);
    mt::Array          sq = mt::sqrt(a);

    EXPECT_FLOAT_EQ(sq.at<float>({0}), 1.0f);
    EXPECT_FLOAT_EQ(sq.at<float>({1}), 2.0f);
    EXPECT_FLOAT_EQ(sq.at<float>({2}), 3.0f);
}

TEST(ArrayMathTest, SquareRootThrowsOnNegativeInput) {
    std::vector<int32_t> data{4, -9, 16};
    mt::Array            a(data);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    EXPECT_THROW(mt::sqrt(a), std::runtime_error);
#pragma GCC diagnostic pop
}

TEST(ArrayMathTest, ExponentialCalculatesCorrectly) {
    std::vector<double> data{1.0, 2.0};
    mt::Array           a(data);
    mt::Array           e = mt::exp(a);

    EXPECT_NEAR(e.at<double>({0}), std::exp(1.0), 1e-7);
    EXPECT_NEAR(e.at<double>({1}), std::exp(2.0), 1e-7);
}

TEST(ArrayMathTest, LogarithmCalculatesCorrectly) {
    std::vector<double> data{1.0, std::exp(1.0), std::exp(2.0)};
    mt::Array           a(std::span<const double>(data), {3});
    mt::Array           lg = mt::log(a);

    EXPECT_NEAR(lg.at<double>({0}), 0.0, 1e-7);
    EXPECT_NEAR(lg.at<double>({1}), 1.0, 1e-7);
    EXPECT_NEAR(lg.at<double>({2}), 2.0, 1e-7);
}

// ─────────────────────────────────────────────
// Division Safeties
// ─────────────────────────────────────────────
TEST(ArrayMathTest, FloatDivisionByZeroYieldsInf) {
    std::vector<float> f_num{1.0f, 2.0f};
    std::vector<float> f_den{0.0f, 2.0f};
    mt::Array          f_n(f_num);
    mt::Array          f_d(f_den);

    mt::Array f_res = mt::divide(f_n, f_d);
    EXPECT_TRUE(std::isinf(f_res.at<float>({0})));
    EXPECT_FLOAT_EQ(f_res.at<float>({1}), 1.0f);
}

TEST(ArrayMathTest, IntegerDivisionByZeroThrowsException) {
    std::vector<int32_t> i_num{10, 20};
    std::vector<int32_t> i_den{2, 0};
    mt::Array            i_n(i_num);
    mt::Array            i_d(i_den);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    EXPECT_THROW(mt::divide(i_n, i_d), std::runtime_error);
#pragma GCC diagnostic pop
}