/*
 * <xcbb/xerror.h>
 *
 *  Created on: Apr 1, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XERROR_H_
#define XERROR_H_

#include <cstdlib>
#include <cstdio>
#include <iostream>

#define __XCBB_ENABLE_KERNEL_SYNCHRONIZATION__ 1

template <typename ErrorType>
void check(ErrorType err, const char* const func, const char* const file, const int line)
{
    if (cudaSuccess != err) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << func << " " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void device_synchronize(const char* const message, const char* const file, const int line)
{
    cudaError_t err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        std::cerr << "CUDA synchronization error: " << file << ":" << line << std::endl;
        std::cerr << message << " " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

#if __XCBB_ENABLE_KERNEL_SYNCHRONIZATION__
#define synchronizeIfEnabled(msg) device_synchronize( (msg), __FILE__, __LINE__)
#else
#define synchronizeIfEnabled(msg)
#endif



#endif /* XERROR_H_ */
