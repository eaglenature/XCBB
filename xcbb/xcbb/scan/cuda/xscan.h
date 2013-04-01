/*
 * <xcbb/scan/cuda/xscan.h>
 *
 *  Created on: Apr 1, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef SCAN_CUDA_XSCAN_H_
#define SCAN_CUDA_XSCAN_H_


#include <xcbb/scan/cuda/detail/xscan.h>


template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
__global__
void UpsweepReduceKernel(uint* partials, uint* data, WorkDecomposition workdecomp)
{
    const WorkDecomposition work = workdecomp;

    // Determine blockId
    const int block = blockIdx.x;

    // Reduce single block and store its reduction in partials
    BlockReduce<NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(partials, data, block, work);
}


template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
__global__
void SpineScanKernel(uint* partials, WorkDecomposition workdecomp)
{
    const WorkDecomposition work = workdecomp;

    const int block = blockIdx.x;

    // Get seed for this block
    const int seed = 0;

    // Scan single block and store its result in partials
    BlockScan<NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(partials, partials, block, work, seed);
}


template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
__global__
void DownsweepScanKernel(uint* partials, uint* data, WorkDecomposition workdecomp)
{
    const WorkDecomposition work = workdecomp;

    const int block = blockIdx.x;

    // Get seed for this block
    const int seed = partials[block];

    // Scan single block and store its result in partials
    BlockScan<NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(partials, data, block, work, seed);
}

#endif /* SCAN_CUDA_XSCAN_H_ */
