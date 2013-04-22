/*
 * <xcbb/radixsort/cuda/xradixsort.h>
 *
 *  Created on: Apr 6, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef RADIXSORT_CUDA_XRADIXSORT_H_
#define RADIXSORT_CUDA_XRADIXSORT_H_


#include <xcbb/radixsort/cuda/detail/xradixsort.h>



template <int PASS, int RADIX_BITS, int CURRENT_BIT, int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
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


template <int RADIX_BITS, int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
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
        ScanSegment(shared_totals, RADIX_DIGITS); // TODO change to template
    }

    __syncthreads();

    if (threadIdx.x < RADIX_DIGITS)
    {
        d_spine[maxOffset + threadIdx.x] = shared_totals[threadIdx.x];
    }
}



template <int PASS, int RADIX_BITS, int CURRENT_BIT, int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
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

    const int BL = work.numTilesPerBlock + 1;
    const int BN = work.numTilesPerBlock;
    const int B  = (block < work.numLargeBlocks) ? BL : BN;

    const int precedingTiles  = (block < work.numLargeBlocks) ? (block * BL) : (block * BN + work.numLargeBlocks * (BL - BN));
    const int blockDataOffset = GetBlockDataOffset<NUM_ELEMENTS_PER_THREAD>(precedingTiles);

    const int RADIX_DIGITS          = 1 << RADIX_BITS;
    const int RAKING_THREADS        = 32 * 2;
    const int RAKING_SEGMENT        = 16;
    const int WARPSYNC_SCAN_THREADS = 8;


    __shared__ int scan_storage[32 * 33];
    __shared__ volatile int warp_storage[8][2][WARPSYNC_SCAN_THREADS];
    __shared__ volatile int digit_scan[2][RADIX_DIGITS];
    __shared__ int digit_count[2][RADIX_DIGITS];

    __shared__ int digit_total[RADIX_DIGITS];
    __shared__ int block_total[RADIX_DIGITS];
    __shared__ int carry_total[RADIX_DIGITS];

    __shared__ bool trivial_pass;
    __shared__ bool swap;


    int  warp = threadIdx.x >> 5;
    int  lane = threadIdx.x & (WARP_SIZE - 1);
    int* const thread_base = scan_storage + 33 * warp + lane;
    int* raking_base = 0;

    if (threadIdx.x < RAKING_THREADS)
    {
        if (threadIdx.x < WARPSYNC_SCAN_THREADS)
        {
            for (int irow  = 0; irow < 8; ++irow)
            {
                warp_storage[irow][0][threadIdx.x] = 0;
            }
        }
        if (threadIdx.x < RADIX_DIGITS)
        {
            int digitTotal = d_spine[threadIdx.x * gridDim.x + block];
            int blockTotal = d_spine[RADIX_DIGITS * gridDim.x + threadIdx.x];

            digit_total[threadIdx.x] = digitTotal;
            block_total[threadIdx.x] = blockTotal;
            carry_total[threadIdx.x] = 0;

            digit_scan[0][threadIdx.x] = 0;
            digit_scan[1][threadIdx.x] = 0;

            // Determine where to read our input
            swap = (PASS == 0) ? false : d_swap[PASS & 0x1];

            // Deterimne if early exit
            trivial_pass = false;
            bool predicate = false;
            if (blockIdx.x  > 0) predicate = (digitTotal > 0);
            if (blockIdx.x == 0) predicate = (d_spine[threadIdx.x * gridDim.x + 1] > 0);
            trivial_pass = (__popc(__ballot(predicate)) == 1) ? true : false;
        }

        int row  = threadIdx.x >> 1;
        int col  = (threadIdx.x & 1) << 4;
        raking_base = scan_storage + 33 * row + col;

        // Set if need to swap for the next pass
        if (blockIdx.x == 0) d_swap[(PASS + 1) & 0x1] = !(swap ^ trivial_pass);
    }

    __syncthreads();

    // Exit early if trivial pass detected
    if (trivial_pass) return;

    if (swap)
    {
        uint* p = d_inKeys;
        d_inKeys = d_outKeys;
        d_outKeys = p;
    }


    // Process tiles
    for (int tile = 0; tile < B; ++tile)
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            thread_base[i * 33 * 4] = 0;
        }
        __syncthreads();

        uint2 keys[2];
        int2  digits[2];
        int2  flagoffsets[2];
        int2  ranks[2];

        uint2* d_inKeysPtr = reinterpret_cast<uint2*>(d_inKeys + blockDataOffset + GetTileDataOffset<NUM_ELEMENTS_PER_THREAD>(tile));

        keys[0] = d_inKeysPtr[threadIdx.x + 0 * NUM_THREADS];
        keys[1] = d_inKeysPtr[threadIdx.x + 1 * NUM_THREADS];

        // decode keys
        digits[0].x = (keys[0].x >> CURRENT_BIT) & 0xf;
        digits[0].y = (keys[0].y >> CURRENT_BIT) & 0xf;
        digits[1].x = (keys[1].x >> CURRENT_BIT) & 0xf;
        digits[1].y = (keys[1].y >> CURRENT_BIT) & 0xf;

        // compute offsets
        flagoffsets[0].x = (digits[0].x >> 2) * 4 * (4 * 33) + (digits[0].x & 0x3);
        flagoffsets[0].y = (digits[0].y >> 2) * 4 * (4 * 33) + (digits[0].y & 0x3);
        flagoffsets[1].x = (digits[1].x >> 2) * 4 * (4 * 33) + (digits[1].x & 0x3) + 4 * (16 * 33);
        flagoffsets[1].y = (digits[1].y >> 2) * 4 * (4 * 33) + (digits[1].y & 0x3) + 4 * (16 * 33);

        // place flags in scan storage
        unsigned char* const thread_base_byte = reinterpret_cast<unsigned char*>(thread_base);

        thread_base_byte[flagoffsets[0].x] = 1;
        thread_base_byte[flagoffsets[0].y] = 1 + (digits[0].x == digits[0].y);
        thread_base_byte[flagoffsets[1].x] = 1;
        thread_base_byte[flagoffsets[1].y] = 1 + (digits[1].x == digits[1].y);

        __syncthreads();

        if (threadIdx.x < RAKING_THREADS)
        {
            // Serial reduction
            int partialsum = SerialReduce<RAKING_SEGMENT>(raking_base);

            int lane = threadIdx.x >> 3;
            int widx = threadIdx.x & (WARPSYNC_SCAN_THREADS - 1);

            // Inclusive scan - warp sync - width: 8
            warp_storage[lane][1][widx] = partialsum;
            warp_storage[lane][1][widx] = partialsum = partialsum + warp_storage[lane][1][widx - 1];
            warp_storage[lane][1][widx] = partialsum = partialsum + warp_storage[lane][1][widx - 2];
            warp_storage[lane][1][widx] = partialsum = partialsum + warp_storage[lane][1][widx - 4];

            // Exclusive scan
            int seed = warp_storage[lane][1][widx - 1];

            // Serial exclusive scan
            ScanSegment<RAKING_SEGMENT>(raking_base, seed);
        }

        __syncthreads();

        ranks[0].x = thread_base_byte[flagoffsets[0].x];
        ranks[0].y = thread_base_byte[flagoffsets[0].y] + (digits[0].x == digits[0].y);
        ranks[1].x = thread_base_byte[flagoffsets[1].x];
        ranks[1].y = thread_base_byte[flagoffsets[1].y] + (digits[1].x == digits[1].y);


        int carry = 0;

        if (threadIdx.x < RADIX_DIGITS)
        {
            int counts[2];
            int lane  = threadIdx.x >> 2;
            int qbyte = threadIdx.x & 3;

            counts[0] = (warp_storage[lane + 0][1][WARPSYNC_SCAN_THREADS - 1] >> (qbyte << 3)) & 0xff;
            counts[1] = (warp_storage[lane + 4][1][WARPSYNC_SCAN_THREADS - 1] >> (qbyte << 3)) & 0xff;

            // (3)
            int total = counts[0] + counts[1];
            carry = total;

            counts[1] = counts[0];
            counts[0] = 0;

            digit_count[0][threadIdx.x] = counts[0];
            digit_count[1][threadIdx.x] = counts[1];

            // Inclusive scan - warp sync - width: 16
            digit_scan[1][threadIdx.x] = total;
            digit_scan[1][threadIdx.x] = total = total + digit_scan[1][threadIdx.x - 1];
            digit_scan[1][threadIdx.x] = total = total + digit_scan[1][threadIdx.x - 2];
            digit_scan[1][threadIdx.x] = total = total + digit_scan[1][threadIdx.x - 4];
            digit_scan[1][threadIdx.x] = total = total + digit_scan[1][threadIdx.x - 8];

            // Exclusive scan
            digit_scan[1][threadIdx.x] = total = digit_scan[1][threadIdx.x - 1];
        }

        __syncthreads();

        ranks[0].x += digit_count[0][digits[0].x] + digit_scan[1][digits[0].x];
        ranks[0].y += digit_count[0][digits[0].y] + digit_scan[1][digits[0].y];
        ranks[1].x += digit_count[1][digits[1].x] + digit_scan[1][digits[1].x];
        ranks[1].y += digit_count[1][digits[1].y] + digit_scan[1][digits[1].y];

        scan_storage[ranks[0].x] = keys[0].x;
        scan_storage[ranks[0].y] = keys[0].y;
        scan_storage[ranks[1].x] = keys[1].x;
        scan_storage[ranks[1].y] = keys[1].y;

        __syncthreads();

        keys[0].x = scan_storage[threadIdx.x + 0 * NUM_THREADS];
        keys[0].y = scan_storage[threadIdx.x + 1 * NUM_THREADS];
        keys[1].x = scan_storage[threadIdx.x + 2 * NUM_THREADS];
        keys[1].y = scan_storage[threadIdx.x + 3 * NUM_THREADS];

        int2 offsets[2];

        offsets[0].x = threadIdx.x + 0 * NUM_THREADS + carry_total[(keys[0].x >> CURRENT_BIT) & 0xf] + digit_total[(keys[0].x >> CURRENT_BIT) & 0xf] + block_total[(keys[0].x >> CURRENT_BIT) & 0xf] - digit_scan[1][(keys[0].x >> CURRENT_BIT) & 0xf];
        offsets[0].y = threadIdx.x + 1 * NUM_THREADS + carry_total[(keys[0].y >> CURRENT_BIT) & 0xf] + digit_total[(keys[0].y >> CURRENT_BIT) & 0xf] + block_total[(keys[0].y >> CURRENT_BIT) & 0xf] - digit_scan[1][(keys[0].y >> CURRENT_BIT) & 0xf];
        offsets[1].x = threadIdx.x + 2 * NUM_THREADS + carry_total[(keys[1].x >> CURRENT_BIT) & 0xf] + digit_total[(keys[1].x >> CURRENT_BIT) & 0xf] + block_total[(keys[1].x >> CURRENT_BIT) & 0xf] - digit_scan[1][(keys[1].x >> CURRENT_BIT) & 0xf];
        offsets[1].y = threadIdx.x + 3 * NUM_THREADS + carry_total[(keys[1].y >> CURRENT_BIT) & 0xf] + digit_total[(keys[1].y >> CURRENT_BIT) & 0xf] + block_total[(keys[1].y >> CURRENT_BIT) & 0xf] - digit_scan[1][(keys[1].y >> CURRENT_BIT) & 0xf];

        __syncthreads();

        d_outKeys[offsets[0].x] = keys[0].x;
        d_outKeys[offsets[0].y] = keys[0].y;
        d_outKeys[offsets[1].x] = keys[1].x;
        d_outKeys[offsets[1].y] = keys[1].y;

        if (threadIdx.x < RADIX_DIGITS)
        {
            carry_total[threadIdx.x] += carry;
        }

        __syncthreads();
    }
}


#endif /* RADIXSORT_CUDA_XRADIXSORT_H_ */
