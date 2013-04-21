/*
 * <xcbb/radixsort/xradixsort.h>
 *
 *  Created on: Apr 6, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XRADIXSORT_H_
#define XRADIXSORT_H_

#include <xcbb/xerror.h>
#include <xcbb/radixsort/cuda/xradixsort.h>
#include <xcbb/radixsort/xwork.h>



template <typename Key>
struct RadixsortStorage
{
    Key   *d_inKeys;
    Key   *d_outKeys;
    int   *d_spine;
    bool  *d_swap;
    int    numElements;

    inline void InitDeviceStorage(Key* inKeys)
    {
        d_inKeys = inKeys;
        checkCudaErrors(cudaMalloc((void**) &d_outKeys, sizeof(Key) * numElements));
        checkCudaErrors(cudaMalloc((void**) &d_swap, sizeof(bool) * 2));
    }

    inline void ReleaseDeviceStorage()
    {
        if (d_outKeys) checkCudaErrors(cudaFree(d_outKeys));
        if (d_spine) checkCudaErrors(cudaFree(d_spine));
        if (d_swap) checkCudaErrors(cudaFree(d_swap));
    }

    explicit RadixsortStorage(int numElements)
        : d_inKeys(0)
        , d_outKeys(0)
        , d_spine(0)
        , d_swap(0)
        , numElements(numElements)
    {
    }

    ~RadixsortStorage()
    {
        ReleaseDeviceStorage();
    }
};


template <typename Key>
class RadixsortEnactor
{
private:

    int  _numElements;
    int  _numBlocks;
    int  _numThreads;
    int  _numElementsPerThread;
    int  _numSpineElements;

    RadixsortWorkDecomposition _regularWorkload;

    template<int PASS, int RADIX_BITS, int CURRENT_BIT>
    inline cudaError_t DistributionSortPass(RadixsortStorage<Key>& storage);

public:

    explicit RadixsortEnactor(int numElements);
    inline cudaError_t Enact(RadixsortStorage<Key>& storage);
};


template <typename Key>
RadixsortEnactor<Key>::RadixsortEnactor(int numElements)
    : _numElements(numElements)
    , _numBlocks(128)
    , _numThreads(128)
    , _numElementsPerThread(4)
{
    ComputeWorkload(_regularWorkload, _numBlocks, _numThreads, _numElementsPerThread, _numElements);
    _numSpineElements = _numBlocks * (1 << 4) + (1 << 4);
}


template <typename Key>
cudaError_t RadixsortEnactor<Key>::Enact(RadixsortStorage<Key>& storage)
{
    const int PASSES = 8;
    checkCudaErrors(cudaMalloc((void**) &storage.d_spine, sizeof(int) * _numSpineElements));

    DistributionSortPass< 0, 4,  0 >(storage);
    DistributionSortPass< 1, 4,  4 >(storage);
    DistributionSortPass< 2, 4,  8 >(storage);
    DistributionSortPass< 3, 4, 12 >(storage);
    DistributionSortPass< 4, 4, 16 >(storage);
    DistributionSortPass< 5, 4, 20 >(storage);
    DistributionSortPass< 6, 4, 24 >(storage);
    DistributionSortPass< 7, 4, 28 >(storage);

    bool needSwap = false;
    checkCudaErrors(cudaMemcpy(&needSwap, &storage.d_swap[PASSES & 0x1], sizeof(bool), cudaMemcpyDeviceToHost));

    if (needSwap)
    {
        checkCudaErrors(cudaMemcpy(storage.d_inKeys, storage.d_outKeys, sizeof(Key) * _numElements, cudaMemcpyDeviceToDevice));
    }
    return cudaSuccess;
}

template <typename Key>
template <int PASS, int RADIX_BITS, int CURRENT_BIT>
cudaError_t RadixsortEnactor<Key>::DistributionSortPass(RadixsortStorage<Key>& storage)
{

    ReductionKernel<PASS, RADIX_BITS, CURRENT_BIT, 4, 4><<<_numBlocks, _numThreads>>>(
            storage.d_swap,
            storage.d_spine,
            storage.d_outKeys,
            storage.d_inKeys,
            _regularWorkload);
    synchronizeIfEnabled("ReductionKernel");


    SpineKernel<RADIX_BITS, 4, 4><<<_numBlocks, _numThreads>>>(
            storage.d_spine,
            _numSpineElements);
    synchronizeIfEnabled("SpineKernel");


    ScanAndScatterKernel<PASS, RADIX_BITS, CURRENT_BIT, 4, 4><<<_numBlocks, _numThreads>>>(
            storage.d_swap,
            storage.d_spine,
            storage.d_outKeys,
            storage.d_inKeys,
            _regularWorkload);
    synchronizeIfEnabled("ScanAndScatterKernel");

    return cudaSuccess;
}

#endif /* XRADIXSORT_H_ */
