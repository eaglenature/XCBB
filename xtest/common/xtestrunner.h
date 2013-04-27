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

class CudaTest: public ::testing::Test
{
protected:
    CudaTest() {}
    virtual ~CudaTest() {}
    virtual void SetUp() { checkCudaErrors(cudaDeviceReset()); }
    virtual void TearDown() {}
};


template <typename T>
bool eaqualRanges(const std::vector<T>& a, const std::vector<T>& b)
{
    bool equal = true;
    for (int i = 0; i < a.size(); ++i)
    {
        if (a[i] == b[i]) continue;
        equal = false;
        break;
    }
    return equal;
}

template <typename T> void EQUAL_RANGES(const std::vector<T>& a, const std::vector<T>& b)
{
    if (eaqualRanges(a, b)) {
        printf("Perfectly correct\n");
    } else {
        printf("Incorrect! >>>>>>>>>>>>\n");
    }
}
template <typename T> void EQUAL(const T& a, const T& b)
{
    if (a == b) {
        printf("Perfectly correct\n");
    } else {
        printf("Incorrect! >>>>>>>>>>>>\n");
    }
}


#endif /* XTESTRUNNER_H_ */
