/*
 * <xcbb/scan/cuda/detail/xreduce.h>
 *
 *  Created on: Apr 3, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef REDUCE_CUDA_DETAIL_XREDUCE_H_
#define REDUCE_CUDA_DETAIL_XREDUCE_H_

#include <xcbb/reduce/xwork.h>
#include <xcbb/xutils.h>


template <bool FULL_TILE_LOAD, int NUM_ELEMENTS_PER_THREAD>
__device__ __forceinline__
uint ReduceTile(
        uint* data,
        const ReduceWorkDecomposition& work,
        int blockDataOffset,
        int warp,
        int lane,
        int tile = 0)
{
    typedef typename VectorTypeTraits<NUM_ELEMENTS_PER_THREAD>::type  VectorType;

    uint reduce = 0;

    if (FULL_TILE_LOAD)
    {
        VectorType* segment = reinterpret_cast<VectorType*>(
                data +
                blockDataOffset +
                GetTileDataOffset<NUM_ELEMENTS_PER_THREAD>(tile) +
                GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(warp));

        // Each thread reduces 4 elements in registers and accumulate result in register
        reduce += LoadReduce(segment, lane);
    }
    else
    {
        const int numExtra = work.numElementsExtra;

        if (numExtra > 0)
        {
            // which warp was hit - check if we can proceed some warps in full speed
            // notice warp is capturing 128 elements (4 per load), thats why division by 128
            int hitWarp;

            if (NUM_ELEMENTS_PER_THREAD == 4)
                hitWarp = numExtra >> 7;

            else if (NUM_ELEMENTS_PER_THREAD == 2)
                hitWarp = numExtra >> 6;

            else if (NUM_ELEMENTS_PER_THREAD == 1)
                hitWarp = numExtra >> 5;

            if (warp < hitWarp)
            {
                // get data offset assigned for degenerated tile in global memory
                uint* degeneratedTile = data + work.numFullTiles * work.numElementsPerTile;

                // Process full 4 element per load and register reduction
                VectorType* segment = reinterpret_cast<VectorType*>(degeneratedTile + GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(warp));
                reduce += LoadReduce(segment, lane);
            }

            __syncthreads();

            if (warp == hitWarp)
            {
                // process guarded reduction within warp
                if (lane == 0)
                {
                    // get data offset assigned for degenerated tile in global memory
                    uint* segment = data +
                            work.numFullTiles * work.numElementsPerTile +
                            GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(warp);

                    if (NUM_ELEMENTS_PER_THREAD == 4)
                        reduce += SerialReduce(segment, numExtra & 127);

                    else if (NUM_ELEMENTS_PER_THREAD == 2)
                        reduce += SerialReduce(segment, numExtra & 63);

                    else if (NUM_ELEMENTS_PER_THREAD == 1)
                        reduce += SerialReduce(segment, numExtra & 31);

                }
            }
        }
    }
    return reduce;
}



template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
__device__ __forceinline__
void BlockReduce(
        uint* partials,
        uint* data,
        int block,
        const ReduceWorkDecomposition& work)
{

    // determine if large or normal blocks
    const int BL = work.numTilesPerBlock + 1;
    const int BN = work.numTilesPerBlock;
    const int B  = (block < work.numLargeBlocks) ? BL : BN;

    const int precedingTiles  = (block < work.numLargeBlocks) ? (block * BL) : (block * BN + work.numLargeBlocks * (BL - BN));
    const int blockDataOffset = GetBlockDataOffset<NUM_ELEMENTS_PER_THREAD>(precedingTiles);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    uint reduce = 0;

    // Reduce all tiles at full speed
    for (int tile = 0; tile < B; ++tile)
    {
        reduce += ReduceTile<true, NUM_ELEMENTS_PER_THREAD>(data, work, blockDataOffset, warp, lane, tile);
    }

    // Last CTA perfrom cleanup work
    if (block == gridDim.x - 1)
    {
        reduce += ReduceTile<false, NUM_ELEMENTS_PER_THREAD>(data, work, blockDataOffset, warp, lane);
    }

    __shared__ uint shared_storage[NUM_WARPS][WARP_SIZE + 1];
    __shared__ uint shared_totals[WARP_SIZE];

    shared_storage[warp][lane] = reduce;

    __syncthreads();

    // Single warp raking reduce in shared and the warp sync reduction of raking totals
    if (warp == 0)
    {
        WarpRakingReduce(shared_totals, shared_storage, lane);
        if (lane == 0) partials[block] = shared_totals[0];
    }
}


template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
__device__ __forceinline__
void SpineBlockReduce(
        uint* partials,
        uint* data,
        int block,
        const ReduceWorkDecomposition& work)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const int blockDataOffset = GetBlockDataOffset<NUM_ELEMENTS_PER_THREAD>(0);

    __shared__ uint shared_totals[WARP_SIZE];

    if (warp == 0)
    {
        /*
         * Load and reduce in registers and store result in shared memory
         */
        shared_totals[lane] = ReduceTile<true, NUM_ELEMENTS_PER_THREAD>(data, work, blockDataOffset, warp, lane);

        /*
         * Reduce in warp synchronous way
         */
        KoggeStoneWarpReduce(shared_totals, lane);

        /*
         * Store result in global memorr
         */
        partials[lane] = shared_totals[lane];
    }
}


#endif /* REDUCE_CUDA_DETAIL_XREDUCE_H_ */
