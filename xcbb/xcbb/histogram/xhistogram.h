/*
 * <xcbb/histogram/xhistogram.h>
 *
 *  Created on: May 3, 2013
 *      Author: eaglenature@gmail.com
 */
#ifndef XHISTOGRAM_H_
#define XHISTOGRAM_H_

#include <xcbb/xerror.h>
#include <xcbb/histogram/cuda/xhistogram.h>
#include <xcbb/histogram/xwork.h>


void Histogram(uint* const d_inKeys, uint* const d_histogram, int numElements, int numBins)
{
    const int numBlocks            = 128;
    const int numThreads           = 256;
    const int numElementsPerThread = 4;

    HistogramWorkDecomposition workload = {0};
    ComputeWorkload(workload, numBlocks, numThreads, numElementsPerThread, numElements);

    HistogramKernel<numThreads, numElementsPerThread><<<numBlocks, numThreads>>>(
            d_histogram,
            d_inKeys,
            workload);
    checkCudaErrors(cudaDeviceSynchronize());
}

#endif /* XHISTOGRAM_H_ */
