/*
 * xtimer.h
 *
 *  Created on: Apr 1, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XTIMER_H_
#define XTIMER_H_

#include <xcbb/xerror.h>

class CudaDeviceTimer
{
public:

    CudaDeviceTimer() {
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
    }

    ~CudaDeviceTimer() {
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));
    }

    inline void Start() {
        checkCudaErrors(cudaEventRecord(start, 0));
    }

    inline void Stop() {
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));
    }

    inline float ElapsedTime() const {
        float totalTimeMsec = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&totalTimeMsec, start, stop));
        return totalTimeMsec;
    }

    inline void Reset() {
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
    }

private:
    cudaEvent_t start, stop;
};


#endif /* XTIMER_H_ */
