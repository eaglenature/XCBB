#include "common/xtestrunner.h"
#include <xcbb/xcbb.h>

#include <algorithm>

class RadixSortTest: public CudaTest
{
protected:
    RadixSortTest() {}
    virtual ~RadixSortTest() {}
    virtual void SetUp()
    {
        CudaTest::SetUp();
        std::srand(time(0));
    }
    virtual void TearDown() {}

protected:

    template <int RADIX_DIGITS>
    void CreateSample0(std::vector<uint>& data)
    {
        for (int i = 0; i < data.size(); ++i) data[i] = 0;
    }

    template <int RADIX_DIGITS>
    void CreateSample1(std::vector<uint>& data)
    {
        for (int i = 0; i < data.size(); ++i) data[i] = (rand() % RADIX_DIGITS) << 4;
    }

    template <int RADIX_DIGITS>
    void CreateSample2(std::vector<uint>& data)
    {
        for (int i = 0; i < data.size(); ++i) data[i] = (rand() % RADIX_DIGITS) << 0;
    }

    template <int RADIX_DIGITS>
    void CreateSample3(std::vector<uint>& data)
    {
        for (int i = 0; i < data.size(); ++i) data[i]  = (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 0;
    }

    template <int RADIX_DIGITS>
    void CreateSample4(std::vector<uint>& data)
    {
        for (int i = 0; i < data.size(); ++i) data[i]  = (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 0;
    }

    template <int RADIX_DIGITS>
    void CreateSample5(std::vector<uint>& data)
    {
        for (int i = 0; i < data.size(); ++i) data[i]  = (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 0;
    }

    template <int RADIX_DIGITS>
    void CreateSample6(std::vector<uint>& data)
    {
        for (int i = 0; i < data.size(); ++i) data[i]  = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 12;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 16;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 20;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 24;
        for (int i = 0; i < data.size(); ++i) data[i] += (rand() % RADIX_DIGITS) << 28;
    }
};


TEST_F(RadixSortTest, RadixSort0)
{
    //std::srand(time(0));

    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    CreateSample0<RADIX_DIGITS>(data);

    // Push array of keys to device
    uint* d_data;
    checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(data.begin(), data.end());

    // Initialize sort storage
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


TEST_F(RadixSortTest, RadixSort1)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    CreateSample1<RADIX_DIGITS>(data);

    // Push array of keys to device
    uint* d_data;
    checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(data.begin(), data.end());

    // Initialize sort storage
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


TEST_F(RadixSortTest, RadixSort2)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    CreateSample2<RADIX_DIGITS>(data);

    // Push array of keys to device
    uint* d_data;
    checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(data.begin(), data.end());

    // Initialize sort storage
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


TEST_F(RadixSortTest, RadixSort3)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    CreateSample3<RADIX_DIGITS>(data);

    // Push array of keys to device
    uint* d_data;
    checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(data.begin(), data.end());

    // Initialize sort storage
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

TEST_F(RadixSortTest, RadixSort4)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    CreateSample4<RADIX_DIGITS>(data);

    // Push array of keys to device
    uint* d_data;
    checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(data.begin(), data.end());

    // Initialize sort storage
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

TEST_F(RadixSortTest, RadixSort5)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    CreateSample5<RADIX_DIGITS>(data);

    // Push array of keys to device
    uint* d_data;
    checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(data.begin(), data.end());

    // Initialize sort storage
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

TEST_F(RadixSortTest, RadixSort6)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    CreateSample6<RADIX_DIGITS>(data);

    // Push array of keys to device
    uint* d_data;
    checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(data.begin(), data.end());

    // Initialize sort storage
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


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
