#include "common/xtestrunner.h"
#include "common/xtimer.h"
#include <xcbb/xcbb.h>

#include <algorithm>


CUDATEST(ParallelRadixsort, 0)
{
    std::srand(time(0));

    checkCudaErrors(cudaDeviceReset());

    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    // 2 PASSES
    // T->T OK
    //for (int i = 0; i < numElements; ++i) data[i] = 0;

    // T->N OK
    //for (int i = 0; i < numElements; ++i) data[i] = (rand() % RADIX_DIGITS) << 4;

    // N->T OK
    //for (int i = 0; i < numElements; ++i) data[i] = (rand() % RADIX_DIGITS) << 0;

    // N->N  OK
    //for (int i = 0; i < numElements; ++i) data[i] = (rand() % RADIX_DIGITS) << 4;
    //for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 0;


    // 3 PASSES
    // T->T->T
    //for (int i = 0; i < numElements; ++i) data[i] = 0;

    // T->N->T
    //for (int i = 0; i < numElements; ++i) data[i] = (rand() % RADIX_DIGITS) << 4;

    // N->T->N
    //for (int i = 0; i < numElements; ++i) data[i]  = (rand() % RADIX_DIGITS) << 8;
    //for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 0;

    // N->N-<N
    //for (int i = 0; i < numElements; ++i) data[i]  = (rand() % RADIX_DIGITS) << 8;
    //for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 4;
    //for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 0;

    // 8 PASSES N-> ... ->N
    for (int i = 0; i < numElements; ++i) data[i]  = (rand() % RADIX_DIGITS) << 0;
    for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 4;
    for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 8;
    for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 12;
    for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 16;
    for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 20;
    for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 24;
    for (int i = 0; i < numElements; ++i) data[i] += (rand() % RADIX_DIGITS) << 28;

    // Rand
    //for (int i = 0; i < numElements; ++i) data[i] = (rand() % 1000000);

    // Push array of keys to device
    uint* d_data;
    checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(data.begin(), data.end());

    // Initialize scan storage
    RadixsortStorage<uint> storage(numElements);
    storage.InitDeviceStorage(d_data);

    // Create sort enactor
    RadixsortEnactor<uint> sorter(numElements);

    // Perform radix sort algorithm
    CudaDeviceTimer timer;
    timer.Start();
    sorter.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(result.data(), d_data, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_data));

    // Compare with reference solution
    EQUAL_RANGES(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}



struct RadixsortTestList
{
    void Create(int argc, char** argv, std::vector<unsigned int>& list)
    {
        list.push_back(0);
    }
};

int main(int argc, char** argv)
{
    TestSuite<RadixsortTestList> suite(argc, argv);
    TestRunner::GetInstance().RunSuite(suite);
}
