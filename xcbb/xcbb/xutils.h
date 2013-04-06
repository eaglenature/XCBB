/*
 * <xcbb/xutils.h>
 *
 *  Created on: Apr 1, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XUTILS_H_
#define XUTILS_H_

#define WARP_SIZE 32

template <int n> struct VectorTypeTraits;
template <>      struct VectorTypeTraits<1> { typedef uint  type; };
template <>      struct VectorTypeTraits<2> { typedef uint2 type; };
template <>      struct VectorTypeTraits<4> { typedef uint4 type; };



__device__ inline uint2 operator+(const uint2& a, const uint2& b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
__device__ inline uint4 operator+(const uint4& a, const uint4& b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline uint2 operator+(const uint2& a, uint alpha)
{
    return make_uint2(a.x + alpha, a.y + alpha);
}

__device__ inline uint4 operator+(const uint4& a, uint alpha)
{
    return make_uint4(a.x + alpha, a.y + alpha, a.z + alpha, a.w + alpha);
}




template <int NUM_ELEMENTS_PER_THREAD>
__device__ inline int GetWarpDataOffset(int warp)
{
    return warp * WARP_SIZE * NUM_ELEMENTS_PER_THREAD;
}



template <int NUM_ELEMENTS_PER_THREAD>
__device__ inline int GetBlockDataOffset(int block, int tilesPerBlock)
{
    return block * tilesPerBlock * blockDim.x * NUM_ELEMENTS_PER_THREAD;
}



template <int NUM_ELEMENTS_PER_THREAD>
__device__ inline int GetBlockDataOffset(int precedingTiles)
{
    return precedingTiles * blockDim.x * NUM_ELEMENTS_PER_THREAD;
}



template <int NUM_ELEMENTS_PER_THREAD>
__device__ inline int GetTileDataOffset(int tile)
{
    return tile * blockDim.x * NUM_ELEMENTS_PER_THREAD;
}



template <int NUM_WARPS, int WARP_STORAGE_SIZE>
__device__ inline void LoadReduceAndStore(
        uint shared_storage[NUM_WARPS][WARP_STORAGE_SIZE],
        uint2* global_storage,
        int warp,
        int lane)
{
    uint2 word = global_storage[lane];
    shared_storage[warp][lane] = word.x + word.y;
}




template <int NUM_WARPS, int WARP_STORAGE_SIZE>
__device__ inline void LoadReduceAndStore(
        uint shared_storage[NUM_WARPS][WARP_STORAGE_SIZE],
        uint4* global_storage,
        int warp,
        int lane)
{
    uint4 word = global_storage[lane];
    shared_storage[warp][lane] = word.x + word.y + word.z + word.w;
}



__device__ inline uint LoadReduce(
        uint* global_storage,
        int lane)
{
    uint word = global_storage[lane];
    return word;
}


__device__ inline uint LoadReduce(
        uint2* global_storage,
        int lane)
{
    uint2 word = global_storage[lane];
    return word.x + word.y;
}


__device__ inline uint LoadReduce(
        uint4* global_storage,
        int lane)
{
    uint4 word = global_storage[lane];
    return word.x + word.y + word.z + word.w;
}

/*
 * Reduce word
 */
__device__ inline uint ReduceWord(const uint& word)
{
    return word;
}

__device__ inline uint ReduceWord(const uint2& word)
{
    return word.x + word.y;
}

__device__ inline uint ReduceWord(const uint4& word)
{
    return word.x + word.y + word.z + word.w;
}

/*
 * Scan word
 */
__device__ __host__ inline void ScanWord(uint& word)
{
    word = 0;
}

__device__ __host__ inline void ScanWord(uint2& word)
{
    word.y = word.x;
    word.x = 0;
}

__device__ __host__ inline void ScanWord(uint4& word)
{
    uint sum, x;
    sum = 0;
    x = word.x; word.x = sum; sum += x;
    x = word.y; word.y = sum; sum += x;
    x = word.z; word.z = sum; sum += x;
    x = word.w; word.w = sum;
}

__device__ __host__ inline void ScanWord(uint& word, int seed)
{
    word = seed;
}

__device__ __host__ inline void ScanWord(uint2& word, int seed)
{
    word.y = word.x + seed;
    word.x = seed;
}

__device__ __host__ inline void ScanWord(uint4& word, int seed)
{
    uint sum, x;
    sum = 0;
    x = word.x; word.x = sum + seed; sum += x;
    x = word.y; word.y = sum + seed; sum += x;
    x = word.z; word.z = sum + seed; sum += x;
    x = word.w; word.w = sum + seed;
}

/*
 * Scan segment
 */
template <int NUM_ELEMENTS>
__device__ __host__ inline void ScanSegment(uint segment[])
{
    uint sum, x;
    sum = 0;
    #pragma unroll
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        x = segment[i];
        segment[i] = sum;
        sum += x;
    }
}

template <int NUM_ELEMENTS>
__device__ __host__ inline void ScanSegment(uint segment[], int seed)
{
    uint sum, x;
    sum = 0;
    #pragma unroll
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        x = segment[i];
        segment[i] = sum + seed;
        sum += x;
    }
}

__device__ __host__ inline void ScanSegment(uint* segment, int num_elements)
{
    uint sum, x;
    sum = 0;
    for (int i = 0; i < num_elements; ++i) {
        x = segment[i];
        segment[i] = sum;
        sum += x;
    }
}

__device__ __host__ inline void ScanSegment(uint* segment, int num_elements, int seed)
{
    uint sum, x;
    sum = 0;
    for (int i = 0; i < num_elements; ++i) {
        x = segment[i];
        segment[i] = sum + seed;
        sum += x;
    }
}


template <int NUM_WARPS, int WARP_STORAGE_SIZE>
__device__ inline uint* GetWarpStorage(
        uint shared_storage[NUM_WARPS][WARP_STORAGE_SIZE],
        int warp)
{
    return &shared_storage[warp][0];
}

/*
 * TODO implement version with no conditionals that reqire WARP_SIZE + WARP_SIZE/2 shared memory
 */
__device__ inline
void KoggeStoneWarpExclusiveScan(volatile uint* shared_storage, int tid)
{
    uint x = shared_storage[tid];
    int sum = x;
    if (tid >= 1)  sum += shared_storage[tid - 1];
    shared_storage[tid] = sum;
    if (tid >= 2)  sum += shared_storage[tid - 2];
    shared_storage[tid] = sum;
    if (tid >= 4)  sum += shared_storage[tid - 4];
    shared_storage[tid] = sum;
    if (tid >= 8)  sum += shared_storage[tid - 8];
    shared_storage[tid] = sum;
    if (tid >= 16) sum += shared_storage[tid - 16];
    shared_storage[tid] = sum;
    shared_storage[tid] = sum - x;
}

/*
 * TODO implement version with no conditionals that reqire WARP_SIZE + WARP_SIZE/2 shared memory
 */
__device__ inline
void KoggeStoneWarpExclusiveScan(volatile uint* shared_storage, int tid, int seed)
{
    uint x = shared_storage[tid];
    int sum = x;
    if (tid >= 1)  sum += shared_storage[tid - 1];
    shared_storage[tid] = sum;
    if (tid >= 2)  sum += shared_storage[tid - 2];
    shared_storage[tid] = sum;
    if (tid >= 4)  sum += shared_storage[tid - 4];
    shared_storage[tid] = sum;
    if (tid >= 8)  sum += shared_storage[tid - 8];
    shared_storage[tid] = sum;
    if (tid >= 16) sum += shared_storage[tid - 16];
    shared_storage[tid] = sum;
    shared_storage[tid] = sum - x + seed;
}

/*
 * TODO implement version with no conditionals that reqire WARP_SIZE + WARP_SIZE/2 shared memory
 */
__device__ inline
void KoggeStoneWarpInclusiveScan(volatile uint* shared_storage, int tid)
{
    int sum = shared_storage[tid];
    if (tid >= 1)  sum += shared_storage[tid - 1];
    shared_storage[tid] = sum;
    if (tid >= 2)  sum += shared_storage[tid - 2];
    shared_storage[tid] = sum;
    if (tid >= 4)  sum += shared_storage[tid - 4];
    shared_storage[tid] = sum;
    if (tid >= 8)  sum += shared_storage[tid - 8];
    shared_storage[tid] = sum;
    if (tid >= 16) sum += shared_storage[tid - 16];
    shared_storage[tid] = sum;
}

/*
 * TODO implement version with no conditionals that reqire WARP_SIZE + WARP_SIZE/2 shared memory
 */
__device__ inline
void KoggeStoneWarpInclusiveScan(volatile uint* shared_storage, int tid, int seed)
{
    int sum = shared_storage[tid];
    if (tid >= 1)  sum += shared_storage[tid - 1];
    shared_storage[tid] = sum;
    if (tid >= 2)  sum += shared_storage[tid - 2];
    shared_storage[tid] = sum;
    if (tid >= 4)  sum += shared_storage[tid - 4];
    shared_storage[tid] = sum;
    if (tid >= 8)  sum += shared_storage[tid - 8];
    shared_storage[tid] = sum;
    if (tid >= 16) sum += shared_storage[tid - 16];
    shared_storage[tid] = sum + seed;
}

/*
 * TODO implement version with no conditionals that reqire WARP_SIZE + WARP_SIZE/2 shared memory
 */
__device__ inline
void KoggeStoneWarpReduceTODO(volatile uint* shared_storage, int tid)
{
    //uint x = shared_storage[tid];
    int sum = shared_storage[tid];
    sum += shared_storage[tid + 1];
    shared_storage[tid] = sum;
    sum += shared_storage[tid + 2];
    shared_storage[tid] = sum;
    sum += shared_storage[tid + 4];
    shared_storage[tid] = sum;
    sum += shared_storage[tid + 8];
    shared_storage[tid] = sum;
    sum += shared_storage[tid + 16];
    shared_storage[tid] = sum;
}
__device__ inline
void KoggeStoneWarpReduce(volatile uint* shared_storage, int tid)
{
    //uint x = shared_storage[tid];
    int sum = shared_storage[tid];
    if (tid < WARP_SIZE - 1)  sum += shared_storage[tid + 1];
    shared_storage[tid] = sum;
    if (tid < WARP_SIZE - 2)  sum += shared_storage[tid + 2];
    shared_storage[tid] = sum;
    if (tid < WARP_SIZE - 4)  sum += shared_storage[tid + 4];
    shared_storage[tid] = sum;
    if (tid < WARP_SIZE - 8)  sum += shared_storage[tid + 8];
    shared_storage[tid] = sum;
    if (tid < WARP_SIZE - 16) sum += shared_storage[tid + 16];
    shared_storage[tid] = sum;
}


/*
 * NUM_ELEMENTS = 64  =>  Active Threads = 32
 * NUM_ELEMENTS = 32  =>  Active Threads = 16
 * NUM_ELEMENTS = 16  =>  Active Threads = 8
 * NUM_ELEMENTS = 8   =>  Active Threads = 4
 * NUM_ELEMENTS = 4   =>  Active Threads = 2
 * NUM_ELEMENTS = 2   =>  Active Threads = 1
 */
template <int NUM_ELEMENTS>
__device__ inline
void WarpReduce(volatile uint* shared_storage, int tid)
{
    if (tid < (NUM_ELEMENTS >> 1))
    {
        if (NUM_ELEMENTS > 32) shared_storage[tid] += shared_storage[tid + 32];
        if (NUM_ELEMENTS > 16) shared_storage[tid] += shared_storage[tid + 16];
        if (NUM_ELEMENTS >  8) shared_storage[tid] += shared_storage[tid +  8];
        if (NUM_ELEMENTS >  4) shared_storage[tid] += shared_storage[tid +  4];
        if (NUM_ELEMENTS >  2) shared_storage[tid] += shared_storage[tid +  2];
        if (NUM_ELEMENTS >  1) shared_storage[tid] += shared_storage[tid +  1];
    }
}

template <int NUM_ELEMENTS>
__device__ inline int SerialReduce(uint* segment) {
    uint reduce = 0;
    #pragma unroll
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        reduce += segment[i];
    }
    return reduce;
}

__device__ inline int SerialReduce(uint* segment, int num_elements) {
    uint reduce = 0;
    for (int i = 0; i < num_elements; ++i) {
        reduce += segment[i];
    }
    return reduce;
}

// TODO do it more generic for raking segment other than 4
__device__ inline uint* GetRakingThreadDataSegment(
        uint shared_storage[][WARP_SIZE + 1],
        int lane)
{
    /*
     * Get offset of lane in [4][33] input array, used only by single warp
     * row = lane/8, col = (lane%8)*4,
     */
    return &shared_storage[lane >> 3][(lane & 7) << 2];
}




__device__ inline void WarpRakingReduce(
        uint* shared_totals,
        uint shared_storage[][WARP_SIZE + 1],
        int lane)
{
    /*
     * Single warp perform serial reduction in shared memory
     */
    shared_totals[lane] = SerialReduce<4>(GetRakingThreadDataSegment(shared_storage, lane));
    /*
     * Warp synchronous reduction Kogge-Stone
     */
    KoggeStoneWarpReduce(shared_totals, lane);
}



__device__ inline uint WarpRakingScan(
        uint* shared_totals,
        uint shared_storage[][WARP_SIZE + 1],
        int lane)
{
    /*
     * Single warp perform serial reduction in shared memory
     */
    uint x = shared_totals[lane] = SerialReduce<4>(GetRakingThreadDataSegment(shared_storage, lane));

    /*
     * Warp synchronous scan Kogge-Stone
     */
    KoggeStoneWarpExclusiveScan(shared_totals, lane);
    x += shared_totals[lane];

    /*
     * Single warp perform serial scan and seed in shared memory
     */
    ScanSegment<4>(GetRakingThreadDataSegment(shared_storage, lane), shared_totals[lane]);

    /*
     * last lane returns total of current tile
     */
    return x;
}


#endif /* XUTILS_H_ */
