#include "common/xtestrunner.h"
#include "common/xtimer.h"
#include <xcbb/xcbb.h>


uint SerialReduce(const std::vector<uint>& in)
{
    uint reduce = 0;
    for (int i = 0;  i < in.size(); ++i) {
        reduce += in[i];
    }
    return reduce;
}


CUDATEST(ParallelReduce, 0)
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
    uint reduce = SerialReduce(data);

    // Initialize reduce storage
    ReduceStorage<uint> storage(numElements);
    storage.InitDeviceStorage(d_data);

    // Create reduce enactor
    ReduceEnactor<uint> red(numElements);

    // Perform scan algorithm
    CudaDeviceTimer timer;
    timer.Start();
    red.Enact(storage);
    timer.Stop();

    uint result = 0;
    // Get scanned array back to host
    checkCudaErrors(cudaMemcpy(&result, d_data, sizeof(uint) * 1, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_data));

    EQUAL(reduce, result);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


struct ArrayScanTestList
{
    void Create(int argc, char** argv, std::vector<unsigned int>& list)
    {
        list.push_back(0);
    }
};

int main(int argc, char** argv)
{
    TestSuite<ArrayScanTestList> suite(argc, argv);
    TestRunner::GetInstance().RunSuite(suite);
}
