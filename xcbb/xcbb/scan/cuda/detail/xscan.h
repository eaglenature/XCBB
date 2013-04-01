/*
 * <xcbb/scan/cuda/detail/xscan.h>
 *
 *  Created on: Apr 1, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef SCAN_CUDA_DETAIL_XSCAN_H_
#define SCAN_CUDA_DETAIL_XSCAN_H_

#include <xcbb/scan/xwork.h>
#include <xcbb/xutils.h>


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


template
<
    bool FULL_TILE_LOAD,
    int NUM_ELEMENTS_PER_THREAD
>
__device__ inline
uint ReduceTile(
        uint* data,
        const WorkDecomposition& work,
        int blockDataOffset,
        int warp,
        int lane,
        int tile = 0)
{
    typedef typename VectorTypeTraits<NUM_ELEMENTS_PER_THREAD>::type  VectorType;

    uint reduce = 0;

    if (FULL_TILE_LOAD) {

        VectorType* segment = reinterpret_cast<VectorType*>(
                data +
                blockDataOffset +
                GetTileDataOffset<NUM_ELEMENTS_PER_THREAD>(tile) +
                GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(warp));

        // Each thread reduces 4 elements in registers and accumulate result in register
        reduce = LoadReduce(segment, lane);

    } else {

        const int numExtra = work.numElementsExtra;

        if (numExtra > 0) {

            // which warp was hit - check if we can proceed some warps in full speed
            // notice warp is capturing 128 elements (4 per load), thats why division by 128
            int hitWarp;

            if (NUM_ELEMENTS_PER_THREAD == 4)
                hitWarp = numExtra >> 7; // 3 + 4

            else if (NUM_ELEMENTS_PER_THREAD == 2)
                hitWarp = numExtra >> 6; //2 + 4

            else if (NUM_ELEMENTS_PER_THREAD == 1)
                hitWarp = numExtra >> 5;

            if (warp < hitWarp) {

                // get data offset assigned for degenerated tile in global memory
                uint* degeneratedTile = data + work.numFullTiles * work.numElementsPerTile;

                // Process full 4 element per load and register reduction
                VectorType* segment = reinterpret_cast<VectorType*>(degeneratedTile + GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(warp));
                reduce = LoadReduce(segment, lane);

            }

            if (warp == hitWarp) {

                // process guarded reduction within warp
                if (lane == 0) {

                    // get data offset assigned for degenerated tile in global memory
                    uint* segment = data +
                            work.numFullTiles * work.numElementsPerTile +
                            GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(warp);

                    if (NUM_ELEMENTS_PER_THREAD == 4)
                        reduce = SerialReduce(segment, numExtra & 127);//(NUM_ELEMENTS_PER_THREAD * WARP_SIZE - 1)

                    else if (NUM_ELEMENTS_PER_THREAD == 2)
                        reduce = SerialReduce(segment, numExtra & 63);

                    else if (NUM_ELEMENTS_PER_THREAD == 1)
                        reduce = SerialReduce(segment, numExtra & 31);

                }
            }
        }
    }
    return reduce;
}


template
<
    int NUM_ELEMENTS_PER_THREAD,
    int NUM_WARPS
>
__device__
void BlockReduce(
        uint* partials,
        uint* data,
        int block,
        const WorkDecomposition& work)
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
    for (int tile = 0; tile < B; ++tile) {

        reduce += ReduceTile<true, NUM_ELEMENTS_PER_THREAD>(data, work, blockDataOffset, warp, lane, tile);
    }

    // Last CTA perfrom cleanup work
    if (block == gridDim.x - 1) {

        reduce += ReduceTile<false, NUM_ELEMENTS_PER_THREAD>(data, work, blockDataOffset, warp, lane);
    }

    __shared__ uint shared_storage[NUM_WARPS][WARP_SIZE + 1];
    __shared__ uint shared_totals[WARP_SIZE];
    //__shared__ uint shared_totals[WARP_SIZE + WARP_SIZE/2];
    //if (lane < WARP_SIZE >> 1) shared_totals[lane + WARP_SIZE] = 0;

    shared_storage[warp][lane] = reduce;

    __syncthreads();

    // Single warp raking reduce in shared and the warp sync reduction of raking totals
    if (warp == 0) {

        WarpRakingReduce(shared_totals, shared_storage, lane);
        if (lane == 0) partials[block] = shared_totals[0];
    }
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


template
<
    bool FULL_TILE_LOAD,
    int NUM_ELEMENTS_PER_THREAD,
    int NUM_WARPS
>
__device__ inline
void ScanTile(
        uint* data,
        const WorkDecomposition& work,
        int blockDataOffset,
        int warp,
        int lane,
        volatile uint* tile_total,
        int blockSeed,
        int tile = 0)
{

    typedef typename VectorTypeTraits<NUM_ELEMENTS_PER_THREAD>::type  VectorType;

    __shared__ uint shared_storage[NUM_WARPS][WARP_SIZE + 1];

    shared_storage[warp][lane] = 0;
    uint tileSeed = tile_total[0];


    if (FULL_TILE_LOAD) { // FULL TILE PATH

        __shared__ uint shared_totals[WARP_SIZE];

        VectorType* segment = reinterpret_cast<VectorType*>(
                data +
                blockDataOffset +
                GetTileDataOffset<NUM_ELEMENTS_PER_THREAD>(tile) +
                GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(warp));

        VectorType x = segment[lane];
        shared_storage[warp][lane] = ReduceWord(x);

        __syncthreads();

        if (warp == 0) {

            int total = WarpRakingScan(shared_totals, shared_storage, lane);
            if (lane == WARP_SIZE - 1) tile_total[0] += total;
        }

        __syncthreads();

        ScanWord(x, shared_storage[warp][lane]);
        segment[lane] = x + tileSeed + blockSeed;


    } else {  // DEGENERATED TILE PATH

        const int numExtra = work.numElementsExtra;

        if (numExtra > 0) {

            int hitWarp = 0;

            if (NUM_ELEMENTS_PER_THREAD == 4)
                hitWarp = numExtra >> 7;

            else if (NUM_ELEMENTS_PER_THREAD == 2)
                hitWarp = numExtra >> 6;

            else if (NUM_ELEMENTS_PER_THREAD == 1)
                hitWarp = numExtra >> 5;

            VectorType  vec;
            VectorType* segment;
            uint* lastsegment;

            if (warp < hitWarp) {

                uint* degeneratedTile = data + work.numFullTiles * work.numElementsPerTile;
                segment = reinterpret_cast<VectorType*>(degeneratedTile + GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(warp));

                vec = segment[lane];
                shared_storage[warp][lane] = ReduceWord(vec);
            }

            if (warp == hitWarp) {

                lastsegment = data +
                        work.numFullTiles * work.numElementsPerTile +
                        GetWarpDataOffset<NUM_ELEMENTS_PER_THREAD>(warp);
            }

            __syncthreads();

            if (warp == 0) {

                __shared__ uint shared_totals[WARP_SIZE];
                shared_totals[lane] = 0;

                if (lane < (hitWarp << 3)) {

                    shared_totals[lane] = SerialReduce<4>(GetRakingThreadDataSegment(shared_storage, lane));
                }

                KoggeStoneWarpExclusiveScan(shared_totals, lane);

                if (lane == (hitWarp << 3)) {

                    tile_total[1] = shared_totals[lane];
                }

                if (lane < (hitWarp << 3)) {

                    ScanSegment<4>(GetRakingThreadDataSegment(shared_storage, lane), shared_totals[lane]);
                }
            }

            __syncthreads();

            if (warp < hitWarp) {

                ScanWord(vec, shared_storage[warp][lane]);
                segment[lane] = vec + tileSeed + blockSeed;
            }

            if (warp == hitWarp) {

                // process guarded scan within warp
                if (lane == 0) {

                    if (NUM_ELEMENTS_PER_THREAD == 4)
                        ScanSegment(lastsegment, numExtra & 127, tile_total[1] + tileSeed + blockSeed);

                    else if (NUM_ELEMENTS_PER_THREAD == 2)
                        ScanSegment(lastsegment, numExtra & 63,  tile_total[1] + tileSeed + blockSeed);

                    else if (NUM_ELEMENTS_PER_THREAD == 1)
                        ScanSegment(lastsegment, numExtra & 31,  tile_total[1] + tileSeed + blockSeed);

                }
            }
        }
    }
}


template
<
    int NUM_ELEMENTS_PER_THREAD,
    int NUM_WARPS
>
__device__
void BlockScan(
        uint* partials,
        uint* data,
        int block,
        const WorkDecomposition& work,
        int blockSeed)
{

    // determine if large or normal blocks
    const int BL = work.numTilesPerBlock + 1;
    const int BN = work.numTilesPerBlock;
    const int B  = (block < work.numLargeBlocks) ? BL : BN;

    const int precedingTiles  = (block < work.numLargeBlocks) ? (block * BL) : (block * BN + work.numLargeBlocks * (BL - BN));
    const int blockDataOffset = GetBlockDataOffset<NUM_ELEMENTS_PER_THREAD>(precedingTiles);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & (WARP_SIZE - 1);


    volatile __shared__ uint tile_total[2];

    if (threadIdx.x < 2) tile_total[threadIdx.x] = 0;
    __syncthreads();


    // Scan all tiles at full speed, each tile returns its total in tile_total to seed next tile
    for (int tile = 0; tile < B; ++tile) {

        ScanTile<true,  NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(data, work, blockDataOffset, warp, lane, tile_total, blockSeed, tile);
    }


    // Last CTA perfrom cleanup work
    if (block == gridDim.x - 1) {

        ScanTile<false, NUM_ELEMENTS_PER_THREAD, NUM_WARPS>(data, work, blockDataOffset, warp, lane, tile_total, blockSeed);
    }
}


#endif /* SCAN_CUDA_DETAIL_XSCAN_H_ */
