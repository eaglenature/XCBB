/*
 * <xcbb/histogram/cuda/xhistogram.h>
 *
 *  Created on: May 3, 2013
 *      Author: eaglenature@gmail.com
 */
#ifndef HISTOGRAM_CUDA_XHISTOGRAM_H_
#define HISTOGRAM_CUDA_XHISTOGRAM_H_

#include <xcbb/histogram/cuda/detail/xhistogram.h>


template <int NUM_THREADS,
          int NUM_ELEMENTS_PER_THREAD>
__global__
void HistogramKernel(
        uint* const d_histogram,
        uint* const d_values,
        HistogramWorkDecomposition workload)
{
    HistogramWorkDecomposition work = workload;

    const int block = blockIdx.x;
    const int BL    = work.numTilesPerBlock + 1;
    const int BN    = work.numTilesPerBlock;
    const int B     = (block < work.numLargeBlocks) ? BL : BN;

    const int precedingTiles  = (block < work.numLargeBlocks) ? (block * BL) : (block * BN + work.numLargeBlocks * (BL - BN));
    const int blockDataOffset = precedingTiles * blockDim.x * NUM_ELEMENTS_PER_THREAD;
    const int numElementsExtra = (blockIdx.x == gridDim.x - 1) ? work.numElementsExtra : 0;

    const int LANES = 4;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    __shared__ uint shared_histogram[LANES * NUM_THREADS];

    #pragma unroll
    for (int i = 0; i < (int)LANES; ++i)
    {
        shared_histogram[threadIdx.x + i * NUM_THREADS] = 0;
    }

    __syncthreads();

    uint4* d_valuesPtr = 0;
    uint4 datum = make_uint4(0, 0, 0, 0);

    for (int tile = 0; tile < B; ++tile)
    {
        d_valuesPtr = reinterpret_cast<uint4*>(d_values + blockDataOffset + tile * blockDim.x * NUM_ELEMENTS_PER_THREAD);
        datum = d_valuesPtr[threadIdx.x];

        atomicAdd(&shared_histogram[datum.x], 1);
        atomicAdd(&shared_histogram[datum.y], 1);
        atomicAdd(&shared_histogram[datum.z], 1);
        atomicAdd(&shared_histogram[datum.w], 1);
    }

    if (numElementsExtra)
    {
        int hitWarp = 0;

        if      (NUM_ELEMENTS_PER_THREAD == 8) hitWarp = numElementsExtra >> 8;
        else if (NUM_ELEMENTS_PER_THREAD == 4) hitWarp = numElementsExtra >> 7;
        else if (NUM_ELEMENTS_PER_THREAD == 2) hitWarp = numElementsExtra >> 6;
        else if (NUM_ELEMENTS_PER_THREAD == 1) hitWarp = numElementsExtra >> 5;

        if (warp < hitWarp)
        {
            d_valuesPtr = reinterpret_cast<uint4*>(d_values + work.numFullTiles * work.numElementsPerTile);
            datum = d_valuesPtr[threadIdx.x];

            atomicAdd(&shared_histogram[datum.x], 1);
            atomicAdd(&shared_histogram[datum.y], 1);
            atomicAdd(&shared_histogram[datum.z], 1);
            atomicAdd(&shared_histogram[datum.w], 1);
        }
        if (warp == hitWarp)
        {
            int numElementsLeft = 0;

            if      (NUM_ELEMENTS_PER_THREAD == 8) numElementsLeft = numElementsExtra & 255;
            else if (NUM_ELEMENTS_PER_THREAD == 4) numElementsLeft = numElementsExtra & 127;
            else if (NUM_ELEMENTS_PER_THREAD == 2) numElementsLeft = numElementsExtra & 63;
            else if (NUM_ELEMENTS_PER_THREAD == 1) numElementsLeft = numElementsExtra & 31;

            if (lane == 0)
            {
                uint* d_data = d_values + work.numFullTiles * work.numElementsPerTile + hitWarp * WARP_SIZE * NUM_ELEMENTS_PER_THREAD;
                for (int i = 0; i < numElementsLeft; ++i)
                {
                    atomicAdd(&shared_histogram[d_data[i]], 1);
                }
            }
        }
    }

    __syncthreads();

    uint4* histogram_ptr = reinterpret_cast<uint4*>(shared_histogram);
    datum = histogram_ptr[threadIdx.x];

    const int threadOffset = threadIdx.x * (NUM_ELEMENTS_PER_THREAD);

    atomicAdd(&d_histogram[threadOffset + 0], datum.x);
    atomicAdd(&d_histogram[threadOffset + 1], datum.y);
    atomicAdd(&d_histogram[threadOffset + 2], datum.z);
    atomicAdd(&d_histogram[threadOffset + 3], datum.w);
}



#endif /* HISTOGRAM_CUDA_XHISTOGRAM_H_ */
