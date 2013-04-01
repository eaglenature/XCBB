/*
 * <xcbb/scan/xwork.h>
 *
 *  Created on: Apr 1, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef SCAN_XWORK_H_
#define SCAN_XWORK_H_


struct WorkDecomposition
{
    int numElements;
    int numElementsPerTile;
    int numElementsExtra;
    int numFullTiles;
    int numTilesPerBlock;
    int numLargeBlocks;
};

void ComputeWorkload(WorkDecomposition& work, int numBlocks, int numThreads, int numElementsPerThread, int numElements)
{
    printf("==================================\n");
    printf("Work decomposition:\n");
    printf("==================================\n");
    printf("numElements            %d\n", numElements);
    printf("numBlocks              %d\n", numBlocks);
    printf("numThreads             %d\n", numThreads);
    printf("numElementsPerThread   %d\n", numElementsPerThread);

    work.numElements          = numElements;
    work.numElementsPerTile   = numThreads * numElementsPerThread;
    work.numFullTiles         = work.numElements / work.numElementsPerTile;
    work.numElementsExtra     = work.numElements - (work.numFullTiles * work.numElementsPerTile);
    work.numTilesPerBlock     = work.numFullTiles / numBlocks;
    work.numLargeBlocks       = work.numFullTiles - (work.numTilesPerBlock * numBlocks);

    printf("==================================\n");
    printf("numElementsPerTile     %d\n", work.numElementsPerTile);
    printf("numFullTiles           %d\n", work.numFullTiles);
    printf("numElementsExtra       %d\n", work.numElementsExtra);
    printf("numTilesPerBlock       %d\n", work.numTilesPerBlock);
    printf("numLargeBlocks         %d\n", work.numLargeBlocks);

    int numNormalBlocks       = numBlocks - work.numLargeBlocks;
    int numTilesPerBigBlock   = work.numTilesPerBlock + 1;

    printf("==================================\n");
    printf("numNormalBlocks        %d\n", numNormalBlocks);
    printf("numTilesPerBigBlock    %d\n", numTilesPerBigBlock);
    printf("==================================\n");
}

#endif /* SCAN_XWORK_H_ */
