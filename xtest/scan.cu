#include "common/xtestrunner.h"
#include "common/xtimer.h"
#include <xcbb/xcbb.h>


void SerialExclusiveScan(std::vector<uint>& out, const std::vector<uint>& in)
{
    uint sum = 0;
    for (int i = 0;  i < in.size(); ++i) {
        uint x = in[i];
        out[i] = sum;
        sum += x;
    }
}


TEST_F(CudaTest, ParallelExclusiveScanSingle)
{
    std::srand(time(0));
    const int numElements = 16384111;

    std::vector<uint> data(numElements);
    std::vector<uint> result(numElements);

    for (int i = 0; i < numElements; ++i)
        data[i] = rand() % 100;

    // Push scanned array to device
    uint* d_data;
    checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
    checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

    // Reference serial version
    SerialExclusiveScan(data, data);

    // Initialize scan storage
    ExclusiveScanStorage<uint> storage(numElements);
    storage.InitDeviceStorage(d_data);

    // Create scan enactor
    ExclusiveScanEnactor<uint> scanner(numElements);

    // Perform scan algorithm
    CudaDeviceTimer timer;
    timer.Start();
    scanner.Enact(storage);
    timer.Stop();

    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(result.data(), d_data, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_data));

    EQUAL_RANGES(data, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


TEST_F(CudaTest, ParallelExclusiveScanMany)
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

    std::srand(time(0));
    int numElements = 0;

    for (int isize = 0; isize < sizeof(n)/sizeof(n[0]); isize++)
    {
        numElements = n[isize];

        std::vector<uint> data(numElements);
        std::vector<uint> result(numElements);

        for (int i = 0; i < numElements; ++i)
            data[i] = rand() % 100;

        // Push scanned array to device
        uint* d_data;
        checkCudaErrors(cudaMalloc((void**) &d_data,    sizeof(uint) * numElements));
        checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(uint) * numElements, cudaMemcpyHostToDevice));

        // Reference serial version
        SerialExclusiveScan(data, data);

        // Initialize scan storage
        ExclusiveScanStorage<uint> storage(numElements);
        storage.InitDeviceStorage(d_data);

        // Create scan enactor
        ExclusiveScanEnactor<uint> scanner(numElements);

        // Perform scan algorithm
        CudaDeviceTimer timer;
        timer.Start();
        scanner.Enact(storage);
        timer.Stop();

        // Get scanned array back to host
        checkCudaErrors(cudaMemcpy(result.data(), d_data, sizeof(uint) * numElements, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_data));

        EQUAL_RANGES(data, result);
        printf("====================================================================================\n");
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

