/*
 * xtestrunner.h
 *
 *  Created on: Apr 1, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XTESTRUNNER_H_
#define XTESTRUNNER_H_

#include <cstdio>
#include <memory>
#include <map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xtimer.h"

/**
 * Base test class
 */
class CudaTest: public ::testing::Test
{
protected:
    CudaTest() {}
    virtual ~CudaTest() {}
    virtual void SetUp() { checkCudaErrors(cudaDeviceReset()); }
    virtual void TearDown() {}
};


/**
 * If T is integral type (char, unsigned char, short, ushort, int, uint, long, ulong)
 * provide member constant value equal true. For any other type value is false.
 */
template <typename T>
struct IsIntegralType
{ static const bool value = false; };

template <>
struct IsIntegralType<char>
{ static const bool value = true; };

template <>
struct IsIntegralType<unsigned char>
{ static const bool value = true; };

template <>
struct IsIntegralType<short>
{ static const bool value = true; };

template <>
struct IsIntegralType<ushort>
{ static const bool value = true; };

template <>
struct IsIntegralType<int>
{ static const bool value = true; };

template <>
struct IsIntegralType<uint>
{ static const bool value = true; };

template <>
struct IsIntegralType<long>
{ static const bool value = true; };

template <>
struct IsIntegralType<ulong>
{ static const bool value = true; };


/**
 * If T is floating-point type (float, double)
 * provide member constant value equal true. For any other type value is false.
 */
template <typename T>
struct IsFloatingPointType
{ static const bool value = false; };

template <>
struct IsFloatingPointType<float>
{ static const bool value = true; };

template <>
struct IsFloatingPointType<double>
{ static const bool value = true; };


/**
 * Int to Type map
 */
template <int N>
struct Int2Type
{ static const int value = N; };


/**
 * Asserts and Expects for arrays of integral and floating-point elements
 */

enum { IntegralExpect, FloatingPointExpect, UnknownExpect };


template <typename T>
void AssertRangeEq(const std::vector<T>& expected, const std::vector<T>& actual, const char* const file, const int line, Int2Type<IntegralExpect>)
{
    for (size_t i = 0; i < actual.size(); ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}

template <typename T>
void AssertRangeEq(const std::vector<T>& expected, const std::vector<T>& actual, const char* const file, const int line, Int2Type<FloatingPointExpect>)
{
    for (size_t i = 0; i < actual.size(); ++i)
    {
        ASSERT_NEAR(expected[i], actual[i], 0.0001) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}

template <typename T>
void ExpectRangeEq(const std::vector<T>& expected, const std::vector<T>& actual, const char* const file, const int line, Int2Type<IntegralExpect>)
{
    for (size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

template <typename T>
void ExpectRangeEq(const std::vector<T>& expected, const std::vector<T>& actual, const char* const file, const int line, Int2Type<FloatingPointExpect>)
{
    for (size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(expected[i], actual[i], 0.0001);
    }
}

template <typename T>
void ASSERT_RANGE_EQ(const std::vector<T>& expected, const std::vector<T>& actual, const char* const file, const int line)
{
    enum { ExpectType = IsFloatingPointType<T>::value ? FloatingPointExpect : (IsIntegralType<T>::value ? IntegralExpect : UnknownExpect) };
    AssertRangeEq(expected, actual, file, line, Int2Type<ExpectType>());
}

template <typename T>
void EXPECT_RANGE_EQ(const std::vector<T>& expected, const std::vector<T>& actual, const char* const file, const int line)
{
    enum { ExpectType = IsFloatingPointType<T>::value ? FloatingPointExpect : (IsIntegralType<T>::value ? IntegralExpect : UnknownExpect) };
    ExpectRangeEq(expected, actual, file, line, Int2Type<ExpectType>());
}


#define ASSERT_RANGE_EQ(expected, actual) ASSERT_RANGE_EQ((expected), (actual), __FILE__, __LINE__)
#define EXPECT_RANGE_EQ(expected, actual) EXPECT_RANGE_EQ((expected), (actual), __FILE__, __LINE__)


#endif /* XTESTRUNNER_H_ */
