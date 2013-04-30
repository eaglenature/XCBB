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



template <typename Key, typename Value = NoValue>
struct RadixsortStorage
{
    Key   *d_inKeys;
    Key   *d_outKeys;
    Value *d_inValues;
    Value *d_outValues;
    int   *d_spine;
    bool  *d_swap;
    int    numElements;

    __forceinline__ void InitDeviceStorage(Key* inKeys, Value* inValues = 0)
    {
        d_inKeys = inKeys;
        d_inValues = inValues;
        checkCudaErrors(cudaMalloc((void**) &d_outKeys, sizeof(Key) * numElements));
        if (!IsKeyOnly<Value>::value) checkCudaErrors(cudaMalloc((void**) &d_outValues, sizeof(Value) * numElements));
        checkCudaErrors(cudaMalloc((void**) &d_swap, sizeof(bool) * 2));
    }

    __forceinline__ void ReleaseDeviceStorage()
    {
        if (d_outKeys) checkCudaErrors(cudaFree(d_outKeys));
        if (d_outValues) checkCudaErrors(cudaFree(d_outValues));
        if (d_spine) checkCudaErrors(cudaFree(d_spine));
        if (d_swap) checkCudaErrors(cudaFree(d_swap));
    }

    explicit RadixsortStorage(int numElements)
        : d_inKeys(0)
        , d_outKeys(0)
        , d_inValues(0)
        , d_outValues(0)
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


template <typename Key, typename Value = NoValue>
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
    __forceinline__ cudaError_t DistributionSortPass(RadixsortStorage<Key, Value>& storage);

public:

    explicit RadixsortEnactor(int numElements);
    __forceinline__ cudaError_t Enact(RadixsortStorage<Key, Value>& storage);
};


template <typename Key, typename Value>
RadixsortEnactor<Key, Value>::RadixsortEnactor(int numElements)
    : _numElements(numElements)
    , _numBlocks(128)
    , _numThreads(128)
    , _numElementsPerThread(4)
{
    ComputeWorkload(_regularWorkload, _numBlocks, _numThreads, _numElementsPerThread, _numElements);
    _numSpineElements = _numBlocks * (1 << 4) + (1 << 4);
}


template <typename Key, typename Value>
cudaError_t RadixsortEnactor<Key, Value>::Enact(RadixsortStorage<Key, Value>& storage)
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
        if (!IsKeyOnly<Value>::value)
        {
            checkCudaErrors(cudaMemcpy(storage.d_inValues, storage.d_outValues, sizeof(Value) * _numElements, cudaMemcpyDeviceToDevice));
        }
    }
    return cudaSuccess;
}

template <typename Key, typename Value>
template <int PASS, int RADIX_BITS, int CURRENT_BIT>
cudaError_t RadixsortEnactor<Key, Value>::DistributionSortPass(RadixsortStorage<Key, Value>& storage)
{

    ReductionKernel<Key, PASS, RADIX_BITS, CURRENT_BIT, 4, 4><<<_numBlocks, _numThreads>>>(
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


    ScanAndScatterKernel<Key, Value, PASS, RADIX_BITS, CURRENT_BIT, 4, 4><<<_numBlocks, _numThreads>>>(
            storage.d_swap,
            storage.d_spine,
            storage.d_outKeys,
            storage.d_inKeys,
            storage.d_outValues,
            storage.d_inValues,
            _regularWorkload);
    synchronizeIfEnabled("ScanAndScatterKernel");

    return cudaSuccess;
}

#endif /* XRADIXSORT_H_ */
