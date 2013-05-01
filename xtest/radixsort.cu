#include "common/xtestrunner.h"
#include <algorithm>

#include <xcbb/xcbb.h>


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

    template <int RADIX_DIGITS>
    void CreateSample7(std::vector<uint>& key)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 8;
        key[(rand() % 220460)] += (rand() % RADIX_DIGITS) << 12;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 16;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 20;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 24;
    }

    template <int RADIX_DIGITS>
    void CreateSample8(std::vector<uint>& key)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 12;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 16;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 20;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 24;
        key[(rand() % 220460)] += (rand() % RADIX_DIGITS) << 28;
    }
};

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, RadixSort0)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    EXPECT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, RadixSort1)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    EXPECT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, RadixSort2)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    EXPECT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, RadixSort3)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    EXPECT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, RadixSort4)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    EXPECT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, RadixSort5)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    EXPECT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, RadixSort6)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    EXPECT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, WithExtraElements0)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 10;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 4*512 + 200;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    ASSERT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, WithExtraElements1)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 10;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 7*512 + 256;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    ASSERT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, WithExtraElements2)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 10;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 4*512 + 11;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    ASSERT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, WithExtraElements3)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 10;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 4*512 + 23;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    ASSERT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, WithExtraElements4)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 10;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 4*512 + 511;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    ASSERT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, WithExtraElements5)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 10;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 4*512 + 1;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    ASSERT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, WithExtraElements6)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 200;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 7*512 + 317;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

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
    ASSERT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, WithExtraElements7)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 8;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 7*512 + 317;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

    CreateSample7<RADIX_DIGITS>(data);

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
    ASSERT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortTest, WithExtraElements8)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 5;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 7*512 + 317;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    ASSERT_EQ(numElements, data.size());
    ASSERT_EQ(numElements, result.size());

    CreateSample8<RADIX_DIGITS>(data);

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
    ASSERT_RANGE_EQ(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/*********************************************************************************************************************
 *
 *
 *
 *********************************************************************************************************************/
class RadixSortKeyValueTest: public CudaTest
{
protected:
    RadixSortKeyValueTest() {}
    virtual ~RadixSortKeyValueTest() {}
    virtual void SetUp()
    {
        CudaTest::SetUp();
        std::srand(time(0));
    }
    virtual void TearDown() {}

protected:

    template <int RADIX_DIGITS>
    void CreateSample0(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i] = 0;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample1(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i] = (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample2(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i] = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample3(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 8;
        key[(rand() % (key.size() - 2))] += (rand() % RADIX_DIGITS) << 12;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 16;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 20;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 28;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample4(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 8;
        key[(rand() % (key.size() - 2))] += (rand() % RADIX_DIGITS) << 12;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 16;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 20;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 24;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 28;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample5(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample6(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 12;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 16;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 20;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 24;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 28;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample7(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 12;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 16;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 20;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 24;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample8(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 12;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 16;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 20;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 24;
        key[(rand() % (key.size() - 2))] += (rand() % RADIX_DIGITS) << 28;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }
};


/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortKeyValueTest, KeyValue0)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 5;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES;

    std::vector<uint> keys(numElements);
    std::vector<uint> values(numElements);
    std::vector<uint> resultkeys(numElements);
    std::vector<uint> resultvalues(numElements);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numElements, values.size());
    ASSERT_EQ(numElements, resultkeys.size());
    ASSERT_EQ(numElements, resultvalues.size());

    CreateSample0<RADIX_DIGITS>(keys, values);

    uint* d_keys;
    checkCudaErrors(cudaMalloc((void**) &d_keys,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    uint* d_values;
    checkCudaErrors(cudaMalloc((void**) &d_values,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_values, values.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(keys.begin(), keys.end());

    // Initialize sort storage
    RadixsortStorage<uint, uint> storage(numElements);
    storage.InitDeviceStorage(d_keys, d_values);

    // Create sort enactor
    RadixsortEnactor<uint, uint> sorter(numElements);

    // Perform radix sort algorithm
    CudaDeviceTimer timer;
    timer.Start();
    sorter.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(resultkeys.data(), d_keys, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(resultvalues.data(), d_values, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_values));

    // Compare with reference solution
    EXPECT_RANGE_EQ(keys, resultkeys);
    EXPECT_RANGE_EQ(keys, resultvalues);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortKeyValueTest, KeyValue1)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 3;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 100;

    std::vector<uint> keys(numElements);
    std::vector<uint> values(numElements);
    std::vector<uint> resultkeys(numElements);
    std::vector<uint> resultvalues(numElements);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numElements, values.size());
    ASSERT_EQ(numElements, resultkeys.size());
    ASSERT_EQ(numElements, resultvalues.size());

    CreateSample1<RADIX_DIGITS>(keys, values);

    uint* d_keys;
    checkCudaErrors(cudaMalloc((void**) &d_keys,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    uint* d_values;
    checkCudaErrors(cudaMalloc((void**) &d_values,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_values, values.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(keys.begin(), keys.end());

    // Initialize sort storage
    RadixsortStorage<uint, uint> storage(numElements);
    storage.InitDeviceStorage(d_keys, d_values);

    // Create sort enactor
    RadixsortEnactor<uint, uint> sorter(numElements);

    // Perform radix sort algorithm
    CudaDeviceTimer timer;
    timer.Start();
    sorter.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(resultkeys.data(), d_keys, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(resultvalues.data(), d_values, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_values));

    // Compare with reference solution
    EXPECT_RANGE_EQ(keys, resultkeys);
    EXPECT_RANGE_EQ(keys, resultvalues);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortKeyValueTest, KeyValue2)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 8;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 512 + 100;

    std::vector<uint> keys(numElements);
    std::vector<uint> values(numElements);
    std::vector<uint> resultkeys(numElements);
    std::vector<uint> resultvalues(numElements);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numElements, values.size());
    ASSERT_EQ(numElements, resultkeys.size());
    ASSERT_EQ(numElements, resultvalues.size());

    CreateSample2<RADIX_DIGITS>(keys, values);

    uint* d_keys;
    checkCudaErrors(cudaMalloc((void**) &d_keys,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    uint* d_values;
    checkCudaErrors(cudaMalloc((void**) &d_values,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_values, values.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(keys.begin(), keys.end());

    // Initialize sort storage
    RadixsortStorage<uint, uint> storage(numElements);
    storage.InitDeviceStorage(d_keys, d_values);

    // Create sort enactor
    RadixsortEnactor<uint, uint> sorter(numElements);

    // Perform radix sort algorithm
    CudaDeviceTimer timer;
    timer.Start();
    sorter.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(resultkeys.data(), d_keys, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(resultvalues.data(), d_values, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_values));

    // Compare with reference solution
    EXPECT_RANGE_EQ(keys, resultkeys);
    EXPECT_RANGE_EQ(keys, resultvalues);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortKeyValueTest, KeyValue3)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 7;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 2*512 + 32;

    std::vector<uint> keys(numElements);
    std::vector<uint> values(numElements);
    std::vector<uint> resultkeys(numElements);
    std::vector<uint> resultvalues(numElements);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numElements, values.size());
    ASSERT_EQ(numElements, resultkeys.size());
    ASSERT_EQ(numElements, resultvalues.size());

    CreateSample3<RADIX_DIGITS>(keys, values);

    uint* d_keys;
    checkCudaErrors(cudaMalloc((void**) &d_keys,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    uint* d_values;
    checkCudaErrors(cudaMalloc((void**) &d_values,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_values, values.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(keys.begin(), keys.end());

    // Initialize sort storage
    RadixsortStorage<uint, uint> storage(numElements);
    storage.InitDeviceStorage(d_keys, d_values);

    // Create sort enactor
    RadixsortEnactor<uint, uint> sorter(numElements);

    // Perform radix sort algorithm
    CudaDeviceTimer timer;
    timer.Start();
    sorter.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(resultkeys.data(), d_keys, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(resultvalues.data(), d_values, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_values));

    // Compare with reference solution
    EXPECT_RANGE_EQ(keys, resultkeys);
    EXPECT_RANGE_EQ(keys, resultvalues);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortKeyValueTest, KeyValue4)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 7;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 10*512 + 300;

    std::vector<uint> keys(numElements);
    std::vector<uint> values(numElements);
    std::vector<uint> resultkeys(numElements);
    std::vector<uint> resultvalues(numElements);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numElements, values.size());
    ASSERT_EQ(numElements, resultkeys.size());
    ASSERT_EQ(numElements, resultvalues.size());

    CreateSample4<RADIX_DIGITS>(keys, values);

    uint* d_keys;
    checkCudaErrors(cudaMalloc((void**) &d_keys,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    uint* d_values;
    checkCudaErrors(cudaMalloc((void**) &d_values,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_values, values.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(keys.begin(), keys.end());

    // Initialize sort storage
    RadixsortStorage<uint, uint> storage(numElements);
    storage.InitDeviceStorage(d_keys, d_values);

    // Create sort enactor
    RadixsortEnactor<uint, uint> sorter(numElements);

    // Perform radix sort algorithm
    CudaDeviceTimer timer;
    timer.Start();
    sorter.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(resultkeys.data(), d_keys, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(resultvalues.data(), d_values, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_values));

    // Compare with reference solution
    EXPECT_RANGE_EQ(keys, resultkeys);
    EXPECT_RANGE_EQ(keys, resultvalues);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortKeyValueTest, KeyValue5)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 5;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 7*512 + 500;

    std::vector<uint> keys(numElements);
    std::vector<uint> values(numElements);
    std::vector<uint> resultkeys(numElements);
    std::vector<uint> resultvalues(numElements);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numElements, values.size());
    ASSERT_EQ(numElements, resultkeys.size());
    ASSERT_EQ(numElements, resultvalues.size());

    CreateSample5<RADIX_DIGITS>(keys, values);

    uint* d_keys;
    checkCudaErrors(cudaMalloc((void**) &d_keys,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    uint* d_values;
    checkCudaErrors(cudaMalloc((void**) &d_values,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_values, values.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(keys.begin(), keys.end());

    // Initialize sort storage
    RadixsortStorage<uint, uint> storage(numElements);
    storage.InitDeviceStorage(d_keys, d_values);

    // Create sort enactor
    RadixsortEnactor<uint, uint> sorter(numElements);

    // Perform radix sort algorithm
    CudaDeviceTimer timer;
    timer.Start();
    sorter.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(resultkeys.data(), d_keys, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(resultvalues.data(), d_values, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_values));

    // Compare with reference solution
    EXPECT_RANGE_EQ(keys, resultkeys);
    EXPECT_RANGE_EQ(keys, resultvalues);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortKeyValueTest, KeyValue6)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 10;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 512 + 7;

    std::vector<uint> keys(numElements);
    std::vector<uint> values(numElements);
    std::vector<uint> resultkeys(numElements);
    std::vector<uint> resultvalues(numElements);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numElements, values.size());
    ASSERT_EQ(numElements, resultkeys.size());
    ASSERT_EQ(numElements, resultvalues.size());

    CreateSample6<RADIX_DIGITS>(keys, values);

    uint* d_keys;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    uint* d_values;
    checkCudaErrors(cudaMalloc((void**) &d_values, sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_values, values.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(keys.begin(), keys.end());

    // Initialize sort storage
    RadixsortStorage<uint, uint> storage(numElements);
    storage.InitDeviceStorage(d_keys, d_values);

    // Create sort enactor
    RadixsortEnactor<uint, uint> sorter(numElements);

    // Perform radix sort algorithm
    CudaDeviceTimer timer;
    timer.Start();
    sorter.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(resultkeys.data(), d_keys, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(resultvalues.data(), d_values, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_values));

    // Compare with reference solution
    EXPECT_RANGE_EQ(keys, resultkeys);
    EXPECT_RANGE_EQ(keys, resultvalues);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortKeyValueTest, KeyValue7)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 10;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 512 + 7;

    std::vector<uint> keys(numElements);
    std::vector<uint> values(numElements);
    std::vector<uint> resultkeys(numElements);
    std::vector<uint> resultvalues(numElements);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numElements, values.size());
    ASSERT_EQ(numElements, resultkeys.size());
    ASSERT_EQ(numElements, resultvalues.size());

    CreateSample7<RADIX_DIGITS>(keys, values);

    uint* d_keys;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    uint* d_values;
    checkCudaErrors(cudaMalloc((void**) &d_values, sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_values, values.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(keys.begin(), keys.end());

    // Initialize sort storage
    RadixsortStorage<uint, uint> storage(numElements);
    storage.InitDeviceStorage(d_keys, d_values);

    // Create sort enactor
    RadixsortEnactor<uint, uint> sorter(numElements);

    // Perform radix sort algorithm
    CudaDeviceTimer timer;
    timer.Start();
    sorter.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(resultkeys.data(), d_keys, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(resultvalues.data(), d_values, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_values));

    // Compare with reference solution
    EXPECT_RANGE_EQ(keys, resultkeys);
    EXPECT_RANGE_EQ(keys, resultvalues);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


/***********************************************************************************
 *
 ***********************************************************************************/
TEST_F(RadixSortKeyValueTest, KeyValue8)
{
    const int RADIX_BITS   = 4;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int TILES        = 10;

    const int numBlocks    = 128;
    const int numElements  = numBlocks * 512 * TILES + 512 + 7;

    std::vector<uint> keys(numElements);
    std::vector<uint> values(numElements);
    std::vector<uint> resultkeys(numElements);
    std::vector<uint> resultvalues(numElements);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numElements, values.size());
    ASSERT_EQ(numElements, resultkeys.size());
    ASSERT_EQ(numElements, resultvalues.size());

    CreateSample8<RADIX_DIGITS>(keys, values);

    uint* d_keys;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    uint* d_values;
    checkCudaErrors(cudaMalloc((void**) &d_values, sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_values, values.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial sort
    std::sort(keys.begin(), keys.end());

    // Initialize sort storage
    RadixsortStorage<uint, uint> storage(numElements);
    storage.InitDeviceStorage(d_keys, d_values);

    // Create sort enactor
    RadixsortEnactor<uint, uint> sorter(numElements);

    // Perform radix sort algorithm
    CudaDeviceTimer timer;
    timer.Start();
    sorter.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(resultkeys.data(), d_keys, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(resultvalues.data(), d_values, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_values));

    // Compare with reference solution
    EXPECT_RANGE_EQ(keys, resultkeys);
    EXPECT_RANGE_EQ(keys, resultvalues);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
