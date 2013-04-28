/*
 * <xcbb/radixsort/cuda/xradixsort.h>
 *
 *  Created on: Apr 6, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef RADIXSORT_CUDA_XRADIXSORT_H_
#define RADIXSORT_CUDA_XRADIXSORT_H_


#include <xcbb/radixsort/cuda/detail/xradixsort.h>



template <int PASS,
          int RADIX_BITS,
          int CURRENT_BIT,
          int NUM_ELEMENTS_PER_THREAD,
          int NUM_WARPS>
__global__
void ReductionKernel(
        bool *d_swap,
        int  *d_spine,
        uint *d_outKeys,
        uint *d_inKeys,
        RadixsortWorkDecomposition workdecomp)
{
    RadixsortWorkDecomposition work = workdecomp;
    const int block = blockIdx.x;

    const bool swap = (PASS == 0) ? false : d_swap[PASS & 0x1];
    if (swap) d_inKeys = d_outKeys;

    BlockReduction<RADIX_BITS, CURRENT_BIT, NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(
            d_spine, d_inKeys, block, work);
}


template <int RADIX_BITS,
          int NUM_ELEMENTS_PER_THREAD,
          int NUM_WARPS>
__global__
void SpineKernel(
        int *d_spine,
        int  numSpineElements)
{

    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int maxOffset = numSpineElements - RADIX_DIGITS;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    __shared__ volatile uint shared_storage[NUM_WARPS][WARP_SIZE];
    __shared__ uint shared_totals[RADIX_DIGITS];

    if (blockIdx.x > 0) return;

    uint4 *data;
    uint4  x;
    int total;

    int tileOffset = 0;
    int tile = 0;

    while (tileOffset < maxOffset)
    {
        data = reinterpret_cast<uint4*>(d_spine + tileOffset + GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(warp));
        x = data[lane];

        total = x.w;

        shared_storage[warp][lane] = ReduceWord(x);

        __syncthreads();

        KoggeStoneWarpExclusiveScan(&shared_storage[warp][0], lane);

        ScanWord(x, shared_storage[warp][lane]);
        data[lane] = x;

        if (lane == WARP_SIZE - 1)
        {
            shared_totals[NUM_WARPS * tile + warp] = total + x.w;
        }

        tileOffset += NUM_WARPS * WARP_SIZE * NUM_ELEMENTS_PER_THREAD;
        ++tile;
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        ScanSegment<RADIX_DIGITS>(shared_totals);
    }

    __syncthreads();

    if (threadIdx.x < RADIX_DIGITS)
    {
        d_spine[maxOffset + threadIdx.x] = shared_totals[threadIdx.x];
    }
}


template <int PASS,
          int RADIX_BITS,
          int CURRENT_BIT,
          int NUM_ELEMENTS_PER_THREAD,
          int NUM_WARPS>
__global__
void ScanAndScatterKernel(
        bool *d_swap,
        int  *d_spine,
        uint *d_outKeys,
        uint *d_inKeys,
        RadixsortWorkDecomposition workload)
{
    RadixsortWorkDecomposition work = workload;

    const int block = blockIdx.x;

    BlockScanAndScatter<PASS, RADIX_BITS, CURRENT_BIT, NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(
            d_swap, d_spine, d_outKeys, d_inKeys, work, block);
}

#endif /* RADIXSORT_CUDA_XRADIXSORT_H_ */
