/*
 * <xcbb/scan/xreduce.h>
 *
 *  Created on: Apr 3, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XREDUCE_H_
#define XREDUCE_H_

#include <xcbb/xerror.h>
#include <xcbb/reduce/cuda/xreduce.h>
#include <xcbb/reduce/xwork.h>


template <typename Key>
struct ReduceStorage
{
    Key*   d_data;
    uint*  d_spine;
    uint   numElements;
    uint   numSpineElements;

    inline void InitDeviceStorage(Key* inData)
    {
        d_data = inData;
        checkCudaErrors(cudaMalloc((void**) &d_spine, sizeof(uint) * numSpineElements));
    }

    inline void ReleaseDeviceStorage()
    {
        if (d_spine) checkCudaErrors(cudaFree(d_spine));
    }

    explicit ReduceStorage(uint size)
    : d_data(0)
    , d_spine(0)
    , numElements(size)
    , numSpineElements(128)
    {
    }

    ~ReduceStorage()
    {
        ReleaseDeviceStorage();
    }
};


template <typename Key>
class ReduceEnactor
{
private:

    int  _numElements;
    int  _numBlocks;
    int  _numThreads;
    int  _numElementsPerThread;

    ReduceWorkDecomposition _regularWorkload;
    ReduceWorkDecomposition _spineWorkload;

    template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
    inline cudaError_t ReducePass(ReduceStorage<Key>& storage);

public:

    explicit ReduceEnactor(int numElements);
    inline cudaError_t Enact(ReduceStorage<Key>& storage);
};


template <typename Key>
ReduceEnactor<Key>::ReduceEnactor(int numElements)
: _numElements(numElements)
, _numBlocks(128)
, _numThreads(128)
, _numElementsPerThread(4)
{
    ComputeWorkload(_regularWorkload, _numBlocks, _numThreads, _numElementsPerThread, _numElements);
    ComputeWorkload(_spineWorkload, _numBlocks, _numThreads, _numElementsPerThread, _numBlocks);
}


template <typename Key>
cudaError_t
ReduceEnactor<Key>::Enact(ReduceStorage<Key>& storage)
{
    ReducePass<4, 4>(storage);
    return cudaSuccess;
}

template <typename Key>
template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
cudaError_t
ReduceEnactor<Key>::ReducePass(ReduceStorage<Key>& storage)
{
    ReduceKernel<NUM_ELEMENTS_PER_THREAD, NUM_WARPS><<<_numBlocks, _numThreads>>>(
            storage.d_spine,
            storage.d_data,
            _regularWorkload);
    synchronizeIfEnabled("ReduceKernel");

    SpineReduceKernel<NUM_ELEMENTS_PER_THREAD, NUM_WARPS><<<_numBlocks, _numThreads>>>(
            storage.d_spine,
            _spineWorkload);
    synchronizeIfEnabled("SpineReduceKernel");

    return cudaSuccess;
}

#endif /* XREDUCE_H_ */
