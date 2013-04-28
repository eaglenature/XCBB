#include "common/xtestrunner.h"
#include <xcbb/xcbb.h>

class ReduceTest: public CudaTest
{
protected:
    ReduceTest() {}
    virtual ~ReduceTest() {}
    virtual void SetUp()
    {
        CudaTest::SetUp();
        std::srand(time(0));
    }
    virtual void TearDown() {}

protected:
    uint SerialReduce(const std::vector<uint>& segment)
    {
        uint reduce = 0;
        for (int i = 0;  i < segment.size(); ++i)
        {
            reduce += segment[i];
        }
        return reduce;
    }
};



TEST_F(ReduceTest, ParallelReduceSingle)
{
    std::srand(time(0));
    const int numElements = 16384111;

    std::vector<uint> data(numElements);

    for (int i = 0; i < numElements; ++i)
        data[i] = 1; //rand() % 100;

    // Push scanned array to device
    uint* d_data;
    checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial version
    uint serialReduce = SerialReduce(data);

    // Initialize reduce storage
    ReduceStorage<uint> storage(numElements);
    storage.InitDeviceStorage(d_data);

    // Create reduce enactor
    ReduceEnactor<uint> reduce(numElements);

    // Perform scan algorithm
    CudaDeviceTimer timer;
    timer.Start();
    uint deviceResult = reduce.Enact(storage);
    timer.Stop();

    checkCudaErrors(cudaFree(d_data));

    EXPECT_EQ(serialReduce, deviceResult);
    printf("Results:    %d  %d\n", serialReduce, deviceResult);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}



TEST_F(ReduceTest, ParallelReduceMany)
{
    const int n[] = { 128*512,
                      128*512*4,
                      128*512*100,
                      128*512 + 1*128,
                      128*512 + 2*128,
                      128*512 + 3*128,
                      128*512 + 33,
                      128*512 + 127,
                      128*512 + 128 + 45,
                      128*512 + 512 + 17,
                      128*512 + 512 + 2*128,
                      128*512 + 60*512 + 3*128 + 65};

    int numElements = 0;

    for (int isize = 0; isize < sizeof(n)/sizeof(n[0]); isize++)
    {
        numElements = n[isize];

        std::vector<uint> data(numElements);

        for (int i = 0; i < numElements; ++i)
            data[i] = rand() % 100;

        // Push scanned array to device
        uint* d_data;
        checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
        checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

        // Reference serial version
        uint serialReduce = SerialReduce(data);

        // Initialize reduce storage
        ReduceStorage<uint> storage(numElements);
        storage.InitDeviceStorage(d_data);

        // Create reduce enactor
        ReduceEnactor<uint> reduce(numElements);

        // Perform scan algorithm
        CudaDeviceTimer timer;
        timer.Start();
        uint deviceResult = reduce.Enact(storage);
        timer.Stop();

        checkCudaErrors(cudaFree(d_data));

        EXPECT_EQ(serialReduce, deviceResult);
        printf("====================================================================================\n");
        printf("Results:    %d  %d\n", serialReduce, deviceResult);
        printf("Problem:    %d\n", numElements);
        printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
        printf("====================================================================================\n");
    }
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
