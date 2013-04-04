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

    EQUAL(serialReduce, deviceResult);
    printf("Results:    %d  %d\n", serialReduce, deviceResult);
    printf("Problem:    %d\n", numElements);
    printf("Time:       %.3f [ms]\n", timer.ElapsedTime());
}


struct ReduceTestList
{
    void Create(int argc, char** argv, std::vector<unsigned int>& list)
    {
        list.push_back(0);
    }
};

int main(int argc, char** argv)
{
    TestSuite<ReduceTestList> suite(argc, argv);
    TestRunner::GetInstance().RunSuite(suite);
}
