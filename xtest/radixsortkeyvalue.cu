#include "common/xtestrunner.h"
#include <algorithm>

#include <xcbb/xcbb.h>

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
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample4(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 0;
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



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
