/*
 * <xcbb/histogram/xwork.h>
 *
 *  Created on: May 3, 2013
 *      Author: eaglenature@gmail.com
 */
#ifndef HISTOGRAM_XWORK_H_
#define HISTOGRAM_XWORK_H_


struct HistogramWorkDecomposition
{
    int numElements;
    int numElementsPerTile;
    int numElementsExtra;
    int numFullTiles;
    int numTilesPerBlock;
    int numLargeBlocks;
};

void ComputeWorkload(HistogramWorkDecomposition& work, int numBlocks, int numThreads, int numElementsPerThread, int numElements)
{
    printf("==================================\n");
    printf("Histogram Work decomposition:\n");
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

#endif /* HISTOGRAM_XWORK_H_ */
