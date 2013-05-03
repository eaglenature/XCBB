#include "common/xtestrunner.h"
#include <xcbb/xcbb.h>


class HistogramTest: public CudaTest
{
protected:
    HistogramTest() {}
    virtual ~HistogramTest() {}
    virtual void SetUp()
    {
        CudaTest::SetUp();
        std::srand(time(0));
    }
    virtual void TearDown() {}

protected:

    void CreateSample0(std::vector<uint>& keys)
    {
        for (uint i = 0; i < keys.size(); ++i) keys[i] = rand() % 1024;
    }

    void CreateSample1(std::vector<uint>& keys)
    {
        ASSERT_GT(keys.size(), 100);
        for (uint i = 0;  i < 10;          ++i) keys[i] = 1000;
        for (uint i = 10; i < 30;          ++i) keys[i] = 1001;
        for (uint i = 30; i < 60;          ++i) keys[i] = 1002;
        for (uint i = 60; i < keys.size(); ++i) keys[i] = rand() % 1000;
    }

    void CreateSample2(std::vector<uint>& keys)
    {
        const uint constkey = rand() % 1024;
        for (uint i = 0; i < keys.size(); ++i) keys[i] = constkey;
    }

    void CreateSample3(std::vector<uint>& keys)
    {
        const uint constkey1 = rand() % 1024;
        const uint constkey2 = rand() % 1023 + 1;
        const uint range = rand() % 2000 + 1;
        const uint n = keys.size();
        ASSERT_TRUE(range < n);
        for (uint i = 0; i < range; ++i) keys[i] = constkey1;
        for (uint i = range; i < n; ++i) keys[i] = constkey2;
    }

    void CreateSample4(std::vector<uint>& keys)
    {
        for (uint i = 0; i < keys.size(); ++i) keys[i] = 0;
    }

    void SerialHistogram(const std::vector<uint>& keys, std::vector<uint>& histogram)
    {
        for (uint k = 0; k < keys.size(); ++k)
        {
            uint key = keys[k];
            ASSERT_LT(key, histogram.size());
            ASSERT_GE(key, 0);
            ++histogram[key];
        }
    }
};


TEST_F(HistogramTest, Test0)
{
    const int numElements  = 10240000;
    const int numBins      = 1024;

    std::vector<uint> keys(numElements);
    std::vector<uint> expected(numBins);
    std::vector<uint> histogram(numBins);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numBins, histogram.size());

    CreateSample0(keys);

    uint* d_keys;
    uint* d_histogram;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMalloc((void**) &d_histogram, sizeof(uint) * numBins));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(uint) * numBins));

    SerialHistogram(keys, expected);

    CudaDeviceTimer timer;
    timer.Start();
    Histogram(d_keys, d_histogram, numElements, numBins);
    timer.Stop();

    checkCudaErrors(cudaMemcpy(histogram.data(), d_histogram, sizeof(uint) * numBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_histogram));

    ASSERT_RANGE_EQ(expected, histogram);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


TEST_F(HistogramTest, Test1)
{
    const int numElements  = 10240000 + 5 * 128;
    const int numBins      = 1024;

    std::vector<uint> keys(numElements);
    std::vector<uint> expected(numBins);
    std::vector<uint> histogram(numBins);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numBins, histogram.size());

    CreateSample0(keys);

    uint* d_keys;
    uint* d_histogram;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMalloc((void**) &d_histogram, sizeof(uint) * numBins));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(uint) * numBins));

    SerialHistogram(keys, expected);

    CudaDeviceTimer timer;
    timer.Start();
    Histogram(d_keys, d_histogram, numElements, numBins);
    timer.Stop();

    checkCudaErrors(cudaMemcpy(histogram.data(), d_histogram, sizeof(uint) * numBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_histogram));

    ASSERT_RANGE_EQ(expected, histogram);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


TEST_F(HistogramTest, Test2)
{
    const int numElements  = 10240000 + 3 * 128 + 63;
    const int numBins      = 1024;

    std::vector<uint> keys(numElements);
    std::vector<uint> expected(numBins);
    std::vector<uint> histogram(numBins);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numBins, histogram.size());

    CreateSample0(keys);

    uint* d_keys;
    uint* d_histogram;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMalloc((void**) &d_histogram, sizeof(uint) * numBins));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(uint) * numBins));

    SerialHistogram(keys, expected);

    CudaDeviceTimer timer;
    timer.Start();
    Histogram(d_keys, d_histogram, numElements, numBins);
    timer.Stop();

    checkCudaErrors(cudaMemcpy(histogram.data(), d_histogram, sizeof(uint) * numBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_histogram));

    ASSERT_RANGE_EQ(expected, histogram);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}

TEST_F(HistogramTest, Test3)
{
    const int numElements  = 10240000 + 7 * 128 + 1;
    const int numBins      = 1024;

    std::vector<uint> keys(numElements);
    std::vector<uint> expected(numBins);
    std::vector<uint> histogram(numBins);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numBins, histogram.size());

    CreateSample0(keys);

    uint* d_keys;
    uint* d_histogram;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMalloc((void**) &d_histogram, sizeof(uint) * numBins));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(uint) * numBins));

    SerialHistogram(keys, expected);

    CudaDeviceTimer timer;
    timer.Start();
    Histogram(d_keys, d_histogram, numElements, numBins);
    timer.Stop();

    checkCudaErrors(cudaMemcpy(histogram.data(), d_histogram, sizeof(uint) * numBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_histogram));

    ASSERT_RANGE_EQ(expected, histogram);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


TEST_F(HistogramTest, Test4)
{
    const int numElements  = 10240000 + 1024 + 6 * 128 + 127;
    const int numBins      = 1024;

    std::vector<uint> keys(numElements);
    std::vector<uint> expected(numBins);
    std::vector<uint> histogram(numBins);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numBins, histogram.size());

    CreateSample0(keys);

    uint* d_keys;
    uint* d_histogram;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMalloc((void**) &d_histogram, sizeof(uint) * numBins));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(uint) * numBins));

    SerialHistogram(keys, expected);

    CudaDeviceTimer timer;
    timer.Start();
    Histogram(d_keys, d_histogram, numElements, numBins);
    timer.Stop();

    checkCudaErrors(cudaMemcpy(histogram.data(), d_histogram, sizeof(uint) * numBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_histogram));

    ASSERT_RANGE_EQ(expected, histogram);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}



TEST_F(HistogramTest, Test5)
{
    const int numElements  = 220480;
    const int numBins      = 1024;

    std::vector<uint> keys(numElements);
    std::vector<uint> expected(numBins);
    std::vector<uint> histogram(numBins);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numBins, histogram.size());

    CreateSample1(keys);

    uint* d_keys;
    uint* d_histogram;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMalloc((void**) &d_histogram, sizeof(uint) * numBins));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(uint) * numBins));

    SerialHistogram(keys, expected);

    CudaDeviceTimer timer;
    timer.Start();
    Histogram(d_keys, d_histogram, numElements, numBins);
    timer.Stop();

    checkCudaErrors(cudaMemcpy(histogram.data(), d_histogram, sizeof(uint) * numBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_histogram));

    ASSERT_RANGE_EQ(expected, histogram);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


TEST_F(HistogramTest, Test6)
{
    const int numElements  = 512 * 128 + 7 * 512 + 71;
    const int numBins      = 1024;

    std::vector<uint> keys(numElements);
    std::vector<uint> expected(numBins);
    std::vector<uint> histogram(numBins);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numBins, histogram.size());

    CreateSample2(keys);

    uint* d_keys;
    uint* d_histogram;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMalloc((void**) &d_histogram, sizeof(uint) * numBins));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(uint) * numBins));

    SerialHistogram(keys, expected);

    CudaDeviceTimer timer;
    timer.Start();
    Histogram(d_keys, d_histogram, numElements, numBins);
    timer.Stop();

    checkCudaErrors(cudaMemcpy(histogram.data(), d_histogram, sizeof(uint) * numBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_histogram));

    ASSERT_RANGE_EQ(expected, histogram);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


TEST_F(HistogramTest, Test7)
{
    const int numElements  = 512 * 128 + 78 * 512 + 45;
    const int numBins      = 1024;

    std::vector<uint> keys(numElements);
    std::vector<uint> expected(numBins);
    std::vector<uint> histogram(numBins);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numBins, histogram.size());

    CreateSample3(keys);

    uint* d_keys;
    uint* d_histogram;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMalloc((void**) &d_histogram, sizeof(uint) * numBins));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(uint) * numBins));

    SerialHistogram(keys, expected);

    CudaDeviceTimer timer;
    timer.Start();
    Histogram(d_keys, d_histogram, numElements, numBins);
    timer.Stop();

    checkCudaErrors(cudaMemcpy(histogram.data(), d_histogram, sizeof(uint) * numBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_histogram));

    ASSERT_RANGE_EQ(expected, histogram);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


TEST_F(HistogramTest, Test8)
{
    const int numElements  = 512 * 128 + 78 * 512 + 45;
    const int numBins      = 1024;

    std::vector<uint> keys(numElements);
    std::vector<uint> expected(numBins);
    std::vector<uint> histogram(numBins);

    ASSERT_EQ(numElements, keys.size());
    ASSERT_EQ(numBins, histogram.size());

    CreateSample4(keys);

    uint* d_keys;
    uint* d_histogram;
    checkCudaErrors(cudaMalloc((void**) &d_keys, sizeof(uint) * numElements));
    checkCudaErrors(cudaMalloc((void**) &d_histogram, sizeof(uint) * numBins));
    checkCudaErrors(cudaMemcpy(d_keys, keys.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(uint) * numBins));

    SerialHistogram(keys, expected);

    CudaDeviceTimer timer;
    timer.Start();
    Histogram(d_keys, d_histogram, numElements, numBins);
    timer.Stop();

    checkCudaErrors(cudaMemcpy(histogram.data(), d_histogram, sizeof(uint) * numBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_keys));
    checkCudaErrors(cudaFree(d_histogram));

    ASSERT_RANGE_EQ(expected, histogram);

    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
