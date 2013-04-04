/*
 * <xcbb/scan/cuda/xreduce.h>
 *
 *  Created on: Apr 3, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef REDUCE_CUDA_XREDUCE_H_
#define REDUCE_CUDA_XREDUCE_H_

#include <xcbb/reduce/cuda/detail/xreduce.h>


template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
__global__
void ReduceKernel(uint* partials, uint* data, ReduceWorkDecomposition workdecomp)
{
    const ReduceWorkDecomposition work = workdecomp;

    // Determine blockId
    const int block = blockIdx.x;

    // Reduce single block and store its reduction in partials
    BlockReduce<NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(partials, data, block, work);
}


template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
__global__
void SpineReduceKernel(uint* partials, ReduceWorkDecomposition workdecomp)
{
    const ReduceWorkDecomposition work = workdecomp;

    // Determine blockId
    const int block = blockIdx.x;

    if (block > 0) return;

    SpineBlockReduce<NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(partials, partials, block, work);
}

#endif /* REDUCE_CUDA_XREDUCE_H_ */
