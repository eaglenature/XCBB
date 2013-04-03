/*
 * <xcbb/scan/xscan.h>
 *
 *  Created on: Apr 1, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XSCAN_H_
#define XSCAN_H_

#include <xcbb/xerror.h>
#include <xcbb/scan/cuda/xscan.h>
#include <xcbb/scan/xwork.h>


template <typename Key>
struct ExclusiveScanStorage
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

    explicit ExclusiveScanStorage(uint size)
    : d_data(0)
    , d_spine(0)
    , numElements(size)
    , numSpineElements(128)
    {
    }

    ~ExclusiveScanStorage()
    {
        ReleaseDeviceStorage();
    }
};


template <typename Key>
class ExclusiveScanEnactor
{
private:

    int  _numElements;
    int  _numBlocks;
    int  _numThreads;
    int  _numElementsPerThread;

    ScanWorkDecomposition _regularWorkload;
    ScanWorkDecomposition _spineWorkload;

    template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
    inline cudaError_t ReduceThenScanPass(ExclusiveScanStorage<Key>& storage);

public:

    explicit ExclusiveScanEnactor(int numElements);
    inline cudaError_t Enact(ExclusiveScanStorage<Key>& storage);
};


template <typename Key>
ExclusiveScanEnactor<Key>::ExclusiveScanEnactor(int numElements)
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
ExclusiveScanEnactor<Key>::Enact(ExclusiveScanStorage<Key>& storage)
{
    ReduceThenScanPass<4, 4>(storage);
    return cudaSuccess;
}

template <typename Key>
template <int NUM_ELEMENTS_PER_THREAD, int NUM_WARPS>
cudaError_t
ExclusiveScanEnactor<Key>::ReduceThenScanPass(ExclusiveScanStorage<Key>& storage)
{

    UpsweepReduceKernel<NUM_ELEMENTS_PER_THREAD, NUM_WARPS><<<_numBlocks, _numThreads>>>(
            storage.d_spine,
            storage.d_data,
            _regularWorkload);
    synchronizeIfEnabled("UpsweepReduceKernel");

    SpineScanKernel<NUM_ELEMENTS_PER_THREAD, NUM_WARPS><<<_numBlocks, _numThreads>>>(
            storage.d_spine,
            _spineWorkload);
    synchronizeIfEnabled("SpineScanKernel");

    DownsweepScanKernel<NUM_ELEMENTS_PER_THREAD, NUM_WARPS><<<_numBlocks, _numThreads>>>(
            storage.d_spine,
            storage.d_data,
            _regularWorkload);
    synchronizeIfEnabled("DownsweepScanKernel");

    return cudaSuccess;
}


#endif /* XSCAN_H_ */
