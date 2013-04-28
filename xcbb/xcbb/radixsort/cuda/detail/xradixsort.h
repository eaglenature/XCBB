/*
 * <xcbb/radixsort/cuda/detail/xradixsort.h>
 *
 *  Created on: Apr 6, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef RADIXSORT_CUDA_DETAIL_XRADIXSORT_H_
#define RADIXSORT_CUDA_DETAIL_XRADIXSORT_H_

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

template<> struct UpdateHistogram<uint>
{
    __device__ __forceinline__
    static void apply(uint* const histogram, uint element)
    {
        ++histogram[element];
    }
};
template<> struct UpdateHistogram<uint2>
{
    __device__ __forceinline__
    static void apply(uint* const histogram, const uint2& element)
    {
        ++histogram[element.x];
        ++histogram[element.y];
    }
};
template<> struct UpdateHistogram<uint4>
{
    __device__ __forceinline__
    static void apply(uint* const histogram, const uint4& element)
    {
        ++histogram[element.x];
        ++histogram[element.y];
        ++histogram[element.z];
        ++histogram[element.w];
    }
};

template<int BITS, int OFFSET> struct ExtractMaskedValue<uint,  BITS, OFFSET>
{
    __device__ __forceinline__
    static uint get(uint a)
    {
        return (a >> OFFSET) & DigitMask<BITS>::value;
    }
};
template<int BITS, int OFFSET> struct ExtractMaskedValue<uint2, BITS, OFFSET>
{
    __device__ __forceinline__
    static uint2 get(const uint2& a)
    {
        return make_uint2((a.x >> OFFSET) & DigitMask<BITS>::value,
                          (a.y >> OFFSET) & DigitMask<BITS>::value);
    }
};
template<int BITS, int OFFSET> struct ExtractMaskedValue<uint4, BITS, OFFSET>
{
    __device__ __forceinline__
    static uint4 get(const uint4& a)
    {
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
    for (int ilane = 0; ilane < (int)LANES; ilane++)
    {
        shared_storage[ilane][threadIdx.x] = 0;
    }
}

template <typename T, int NUM_ELEMENTS>
__device__ __forceinline__
void ResetSegment(T segment[])
{
    #pragma unroll
    for (int i = 0; i < (int)NUM_ELEMENTS; ++i)
    {
        segment[i] = 0;
    }
}


__device__ __forceinline__
uint IncQbyte(int qbyte)
{
    return 1 << (qbyte << 3);
}
__device__ __forceinline__
uint IncQbyte(int inc, int qbyte)
{
    return inc << (qbyte << 3);
}

__device__ __forceinline__
uint DecodeQbyte(uint qword, int qbyte)
{
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



template <bool FULL_TILE_LOAD,
          int CURRENT_BIT,
          int NUM_ELEMENTS_PER_THREAD,
          int NUM_WARPS>
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
            encoded = shared_storage[warp][lane + i * WARP_SIZE];
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

                    for (int left = 0; left < numElementsLeft; ++left)
                    {
                        Bucket<CURRENT_BIT>(data[left], shared_storage);
                    }
                }
            }

            __syncthreads();

            int encoded;

            #pragma unroll
            for (int i = 0; i < (int)RAKING_ELEMENT_PER_THREAD; ++i)
            {
                encoded = shared_storage[warp][lane + i * WARP_SIZE];
                local_histogram[0] += DecodeQbyte(encoded, 0);
                local_histogram[1] += DecodeQbyte(encoded, 1);
                local_histogram[2] += DecodeQbyte(encoded, 2);
                local_histogram[3] += DecodeQbyte(encoded, 3);
            }
        }
    }
}


template <int RADIX_BITS,
          int CURRENT_BIT,
          int NUM_ELEMENTS_PER_THREAD,
          int NUM_WARPS>
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



template <bool FULL_TILE_LOAD,
          int  RADIX_DIGITS,
          int  CURRENT_BIT,
          int  NUM_ELEMENTS_PER_THREAD,
          int  RAKING_THREADS,
          int  RAKING_SEGMENT,
          int  WARPSYNC_SCAN_THREADS>
__device__ __forceinline__
void TileScanAndScatter(
        uint *d_outKeys,
        uint *d_inKeys,
        int scan_storage[32 * 33],
        volatile int warp_storage[8][2][WARPSYNC_SCAN_THREADS],
        volatile int digit_scan[2][RADIX_DIGITS],
        int digit_count[2][RADIX_DIGITS],
        int digit_total[RADIX_DIGITS],
        int block_total[RADIX_DIGITS],
        int carry_total[RADIX_DIGITS],
        int* const thread_base,
        int* raking_base,
        int blockDataOffset,
        int numElementsExtra,
        int tile)
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

    if (FULL_TILE_LOAD)
    {
        uint2* d_inKeysPtr = reinterpret_cast<uint2*>(d_inKeys + blockDataOffset + GetTileDataOffset<NUM_ELEMENTS_PER_THREAD>(tile));

        keys[0] = d_inKeysPtr[threadIdx.x + 0 * NUM_THREADS];
        keys[1] = d_inKeysPtr[threadIdx.x + 1 * NUM_THREADS];
    }
    else // Guarded loads
    {
        int offset;
        uint* d_inKeysPtr = d_inKeys + blockDataOffset;

        offset = (threadIdx.x << 1) + 0 * (NUM_THREADS << 1);
        keys[0].x = (offset + 0 - numElementsExtra < 0) ? d_inKeysPtr[offset + 0] : (uint) -1u;
        keys[0].y = (offset + 1 - numElementsExtra < 0) ? d_inKeysPtr[offset + 1] : (uint) -1u;

        offset = (threadIdx.x << 1) + 1 * (NUM_THREADS << 1);
        keys[1].x = (offset + 0 - numElementsExtra < 0) ? d_inKeysPtr[offset + 0] : (uint) -1u;
        keys[1].y = (offset + 1 - numElementsExtra < 0) ? d_inKeysPtr[offset + 1] : (uint) -1u;
    }

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

    if (FULL_TILE_LOAD)
    {
        d_outKeys[offsets[0].x] = keys[0].x;
        d_outKeys[offsets[0].y] = keys[0].y;
        d_outKeys[offsets[1].x] = keys[1].x;
        d_outKeys[offsets[1].y] = keys[1].y;
    }
    else // Guarded stores
    {
        if (threadIdx.x + 0 * NUM_THREADS < numElementsExtra) d_outKeys[offsets[0].x] = keys[0].x;
        if (threadIdx.x + 1 * NUM_THREADS < numElementsExtra) d_outKeys[offsets[0].y] = keys[0].y;
        if (threadIdx.x + 2 * NUM_THREADS < numElementsExtra) d_outKeys[offsets[1].x] = keys[1].x;
        if (threadIdx.x + 3 * NUM_THREADS < numElementsExtra) d_outKeys[offsets[1].y] = keys[1].y;
    }

    if (threadIdx.x < RADIX_DIGITS)
    {
        carry_total[threadIdx.x] += carry;
    }

    __syncthreads();
}


template <int PASS,
          int RADIX_BITS,
          int CURRENT_BIT,
          int NUM_ELEMENTS_PER_THREAD,
          int NUM_WARPS>
__device__ __forceinline__
void BlockScanAndScatter(
        bool *d_swap,
        int  *d_spine,
        uint *d_outKeys,
        uint *d_inKeys,
        const RadixsortWorkDecomposition& work,
        int   block)
{
    const int BL = work.numTilesPerBlock + 1;
    const int BN = work.numTilesPerBlock;
    const int B  = (block < work.numLargeBlocks) ? BL : BN;

    const int precedingTiles   = (block < work.numLargeBlocks) ? (block * BL) : (block * BN + work.numLargeBlocks * (BL - BN));
    const int blockDataOffset  = GetBlockDataOffset<NUM_ELEMENTS_PER_THREAD>(precedingTiles);
    const int numElementsExtra = (blockIdx.x == gridDim.x - 1) ? work.numElementsExtra : 0;

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

    // Swap global pointers if needed
    if (swap)
    {
        uint* tmp = d_inKeys;
        d_inKeys  = d_outKeys;
        d_outKeys = tmp;
    }

    // Process full tiles
    for (int tile = 0; tile < B; ++tile)
    {
        TileScanAndScatter<true, RADIX_DIGITS, CURRENT_BIT, NUM_ELEMENTS_PER_THREAD, RAKING_THREADS, RAKING_SEGMENT, WARPSYNC_SCAN_THREADS>(
                d_outKeys,
                d_inKeys,
                scan_storage,
                warp_storage,
                digit_scan,
                digit_count,
                digit_total,
                block_total,
                carry_total,
                thread_base,
                raking_base,
                blockDataOffset,
                numElementsExtra,
                tile);
    }
    // Last block process extra elements if any
    if (numElementsExtra)
    {
        TileScanAndScatter<false, RADIX_DIGITS, CURRENT_BIT, NUM_ELEMENTS_PER_THREAD, RAKING_THREADS, RAKING_SEGMENT, WARPSYNC_SCAN_THREADS>(
                d_outKeys,
                d_inKeys,
                scan_storage,
                warp_storage,
                digit_scan,
                digit_count,
                digit_total,
                block_total,
                carry_total,
                thread_base,
                raking_base,
                work.numFullTiles * work.numElementsPerTile,
                numElementsExtra,
                0);
    }
}

#endif /* RADIXSORT_CUDA_DETAIL_XRADIXSORT_H_ */
