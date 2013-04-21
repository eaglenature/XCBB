/*
 * <xcbb/radixsort/cuda/detail/xradixsort.h>
 *
 *  Created on: Apr 6, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef RADIXSORT_CUDA_DETAIL_XSCAN_H_
#define RADIXSORT_CUDA_DETAIL_XSCAN_H_

#include <xcbb/radixsort/xwork.h>
#include <xcbb/xutils.h>

#define NUM_THREADS 128
#define HISTOGRAM_LANES 4
#define RAKING_ELEMENT_PER_THREAD 4


template<int BITS> struct DigitMask;
template<typename T> struct UpdateHistogram;
template<typename T, int BITS, int OFFSET> struct ExtractMaskedValue;

template<> struct DigitMask<1> { static const uint value = 0x00000001; };
template<> struct DigitMask<2> { static const uint value = 0x00000003; };
template<> struct DigitMask<4> { static const uint value = 0x0000000F; };
template<> struct DigitMask<8> { static const uint value = 0x000000FF; };

template<> struct UpdateHistogram<uint> {
    __device__
    inline static void apply(uint* const histogram, uint element) {
        ++histogram[element];
    }
};
template<> struct UpdateHistogram<uint2> {
    __device__
    inline static void apply(uint* const histogram, const uint2& element) {
        ++histogram[element.x];
        ++histogram[element.y];
    }
};
template<> struct UpdateHistogram<uint4> {
    __device__
    inline static void apply(uint* const histogram, const uint4& element) {
        ++histogram[element.x];
        ++histogram[element.y];
        ++histogram[element.z];
        ++histogram[element.w];
    }
};

template<int BITS, int OFFSET> struct ExtractMaskedValue<uint,  BITS, OFFSET> {
    __device__
    inline static uint get(uint a) {
        return (a >> OFFSET) & DigitMask<BITS>::value;
    }
};
template<int BITS, int OFFSET> struct ExtractMaskedValue<uint2, BITS, OFFSET> {
    __device__
    inline static uint2 get(const uint2& a) {
        return make_uint2((a.x >> OFFSET) & DigitMask<BITS>::value,
                          (a.y >> OFFSET) & DigitMask<BITS>::value);
    }
};
template<int BITS, int OFFSET> struct ExtractMaskedValue<uint4, BITS, OFFSET> {
    __device__
    inline static uint4 get(const uint4& a) {
        return make_uint4((a.x >> OFFSET) & DigitMask<BITS>::value,
                          (a.y >> OFFSET) & DigitMask<BITS>::value,
                          (a.z >> OFFSET) & DigitMask<BITS>::value,
                          (a.w >> OFFSET) & DigitMask<BITS>::value);
    }
};


template <int LANES>
__device__ __forceinline__
void ResetStorage(uint shared_storage[LANES][NUM_THREADS])
{
    #pragma unroll
    for (int ilane = 0; ilane < (int)LANES; ilane++) {
        shared_storage[ilane][threadIdx.x] = 0;
    }
}

template <typename T, int NUM_ELEMENTS>
__device__ __forceinline__
void ResetSegment(T segment[])
{
    #pragma unroll
    for (int i = 0; i < (int)NUM_ELEMENTS; ++i) {
        segment[i] = 0;
    }
}


__device__ __forceinline__ uint IncQbyte(int qbyte) {
    return 1 << (qbyte << 3);
}

__device__ __forceinline__ uint IncQbyte(int inc, int qbyte) {
    return inc << (qbyte << 3);
}

__device__ __forceinline__ uint DecodeQbyte(uint qword, int qbyte) {
    return (qword >> (qbyte << 3)) & 0xff;
}

template <int CURRENT_BIT>
__device__ __forceinline__
void Bucket(uint key, uint shared_storage[HISTOGRAM_LANES][NUM_THREADS])
{
    int bucket = (key >> CURRENT_BIT) & 0xf; // mask-out four bits starting from CURRENT_BIT. in {0, ... ,15}
    int hlane  = bucket >> 2;                // bucket/4
    int qbyte  = bucket & 3;                 // bucket%4
    shared_storage[hlane][threadIdx.x] += IncQbyte(qbyte);
}

template <int CURRENT_BIT>
__device__ __forceinline__
void Bucket(uint4 datum, uint shared_storage[HISTOGRAM_LANES][NUM_THREADS])
{
    Bucket<CURRENT_BIT>(datum.x, shared_storage);
    Bucket<CURRENT_BIT>(datum.y, shared_storage);
    Bucket<CURRENT_BIT>(datum.z, shared_storage);
    Bucket<CURRENT_BIT>(datum.w, shared_storage);
}

template <bool FULL_TILE_LOAD, int CURRENT_BIT, int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
__device__ __forceinline__
void TileReduction(
        uint local_histogram[4],
        uint shared_storage[HISTOGRAM_LANES][NUM_THREADS],
        uint* d_data,
        const RadixsortWorkDecomposition& work,
        int tile = 0)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    if (FULL_TILE_LOAD)
    {
        uint4* data = reinterpret_cast<uint4*>(d_data + GetTileDataOffset<NUM_ELEMENTS_PER_THREAD>(tile));

        Bucket<CURRENT_BIT>(data[threadIdx.x], shared_storage);

        __syncthreads();

        int encoded;

        #pragma unroll
        for (int i = 0; i < (int)RAKING_ELEMENT_PER_THREAD; ++i)
        {
            encoded = shared_storage[warp][lane + i*WARP_SIZE];
            local_histogram[0] += DecodeQbyte(encoded, 0);
            local_histogram[1] += DecodeQbyte(encoded, 1);
            local_histogram[2] += DecodeQbyte(encoded, 2);
            local_histogram[3] += DecodeQbyte(encoded, 3);
        }

        __syncthreads();

        ResetStorage<HISTOGRAM_LANES>(shared_storage);

        __syncthreads();
    }
    else
    {
        const int numExtra = work.numElementsExtra;

        if (numExtra > 0)
        {
            int hitWarp = 0;

            if (NUM_ELEMENTS_PER_THREAD == 4)
                hitWarp = numExtra >> 7;

            else if (NUM_ELEMENTS_PER_THREAD == 2)
                hitWarp = numExtra >> 6;

            else if (NUM_ELEMENTS_PER_THREAD == 1)
                hitWarp = numExtra >> 5;


            if (warp < hitWarp)
            {
                uint4* data = reinterpret_cast<uint4*>(d_data);

                Bucket<CURRENT_BIT>(data[threadIdx.x], shared_storage);

            }
            if (warp == hitWarp)
            {
                uint* data = d_data + GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(hitWarp);

                if (lane == 0)
                {
                    int numElementsLeft = 0;

                    if (NUM_ELEMENTS_PER_THREAD == 4)
                        numElementsLeft =  numExtra & 127;

                    else if (NUM_ELEMENTS_PER_THREAD == 2)
                        numElementsLeft =  numExtra & 63;

                    else if (NUM_ELEMENTS_PER_THREAD == 1)
                        numElementsLeft =  numExtra & 31;

                    for (int left = 0; left < numElementsLeft; ++left) {

                        Bucket<CURRENT_BIT>(data[left], shared_storage);
                    }
                }
            }

            __syncthreads();

            int encoded;

            #pragma unroll
            for (int i = 0; i < (int)RAKING_ELEMENT_PER_THREAD; ++i)
            {
                encoded = shared_storage[warp][lane + i*WARP_SIZE];
                local_histogram[0] += DecodeQbyte(encoded, 0);
                local_histogram[1] += DecodeQbyte(encoded, 1);
                local_histogram[2] += DecodeQbyte(encoded, 2);
                local_histogram[3] += DecodeQbyte(encoded, 3);
            }
        }
    }
}

template <int RADIX_BITS, int CURRENT_BIT, int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
__device__ __forceinline__
void BlockReduction(
        int  *d_spine,
        uint *d_inputKey,
        int   block,
        const RadixsortWorkDecomposition& work)
{

    const int BL = work.numTilesPerBlock + 1;
    const int BN = work.numTilesPerBlock;
    const int B  = (block < work.numLargeBlocks) ? BL : BN;

    const int precedingTiles  = (block < work.numLargeBlocks) ? (block * BL) : (block * BN + work.numLargeBlocks * (BL - BN));
    const int blockDataOffset = GetBlockDataOffset<NUM_ELEMENTS_PER_THREAD>(precedingTiles);

    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    __shared__ uint shared_storage[HISTOGRAM_LANES][NUM_THREADS];

    ResetStorage<HISTOGRAM_LANES>(shared_storage);
    __syncthreads();

    uint local_histogram[4];
    local_histogram[0] = 0;
    local_histogram[1] = 0;
    local_histogram[2] = 0;
    local_histogram[3] = 0;

    uint* blockData = d_inputKey + blockDataOffset;

    for (int tile = 0; tile < B; ++tile)
    {
        TileReduction<true, CURRENT_BIT, NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(
                local_histogram, shared_storage, blockData, work, tile);
    }

    if (block == gridDim.x - 1)
    {
        blockData = d_inputKey + work.numFullTiles * work.numElementsPerTile;
        TileReduction<false, CURRENT_BIT, NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(
                local_histogram, shared_storage, blockData, work);
    }

    __syncthreads();


    shared_storage[warp][lane + 0] = local_histogram[0];
    KoggeStoneWarpReduce(&shared_storage[warp][0], lane);

    __syncthreads();

    shared_storage[warp][lane + 1] = local_histogram[1];
    KoggeStoneWarpReduce(&shared_storage[warp][1], lane);

    __syncthreads();

    shared_storage[warp][lane + 2] = local_histogram[2];
    KoggeStoneWarpReduce(&shared_storage[warp][2], lane);

    __syncthreads();

    shared_storage[warp][lane + 3] = local_histogram[3];
    KoggeStoneWarpReduce(&shared_storage[warp][3], lane);

    __syncthreads();


    if (threadIdx.x < RADIX_DIGITS)
    {
        int row = threadIdx.x >> 2;
        int col = threadIdx.x & 3;
        d_spine[(gridDim.x * threadIdx.x) + block] = shared_storage[row][col];
    }
}

#endif /* RADIXSORT_CUDA_DETAIL_XSCAN_H_ */
