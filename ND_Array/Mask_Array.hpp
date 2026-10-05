#ifndef MASK_ARRAY_HPP
#define MASK_ARRAY_HPP

#include "helper.hpp"


#ifdef __CUDACC__

namespace ND::CUDA
{
template<typename T, int N>
__global__ void SetMemoryMasked_K(T* data_, T val, bool* mask)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    if(mask[idx])
        data_[idx] =  val;
}
template<typename E, int N>
__global__ void CollapseExpressionMasked_K(typename E::value_type* data_, const E expr, bool* mask, const size_t shift = 0)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    if(mask[idx])
        data_[idx] = expr.get_element(shift + idx);
}
}

#endif


namespace ND {


// Mask Array
template <ComputeBackend Backend, typename T, size_t firstDim, size_t... RestDims>
class Mask_Array
{
    template<ComputeBackend Backend_, typename T_, size_t f_Dim, size_t... R_dims>
    friend class Array;

protected:

    static constexpr size_t length = firstDim * (RestDims * ...);
    using value_type = T;

    Array<Backend, T, firstDim, RestDims...>& arr_;
    // TODO : Replace mask_ by an Array_Expression with value_type == bool.
    // Can evaluate evaluate mask in expression collapse.
    const Array<Backend, bool, firstDim, RestDims...>& mask_;

#ifdef __CUDACC__
    static constexpr int GRID_SIZE = (length + BLOCK_DIM_1D - 1) >> BLOCK_DIM_LOG2_1D;
#endif

    Mask_Array(Array<Backend, T, firstDim, RestDims...>& arr, const Array<Backend, bool, firstDim, RestDims...>& mask)
    : arr_(arr), mask_(mask)
    {}

public:
    template<class E>
    requires(is_Array_Expression<E>::value)
    void operator=(const E& expr)
    {
        if constexpr (Backend == ComputeBackend::CPU)
        {
            PARALLEL_FOR(length)
            for (size_t i = 0; i < length; ++i)
                if(mask_.get_element(i))    arr_.data_[i] = expr.get_element(i);
        }
        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_FUNC((CUDA::CollapseExpressionMasked_K<E, length><<<GRID_SIZE, BLOCK_DIM_1D>>>(arr_.data_, expr, mask_.data_)));
    }
    void operator=(const value_type& val)
    {
        if constexpr (Backend == ComputeBackend::CPU)
        {
            PARALLEL_FOR(length)
            for (size_t i = 0; i < length; ++i)
                if(mask_.get_element(i))    arr_.data_[i] = val;
        }
        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_FUNC((CUDA::SetMemoryMasked_K<value_type, length><<<GRID_SIZE, BLOCK_DIM_1D>>>(arr_.data_, val, mask_.data_)));
    }
    template<class E>
    void operator+=(const E& RHS)
    {
        *this = arr_ + RHS;
    }
    template<class E>
    void operator-=(const E& RHS)
    {
        *this = arr_ - RHS;
    }
    template<class E>
    void operator*=(const E& RHS)
    {
        *this = arr_ * RHS;
    }
    template<class E>
    void operator/=(const E& RHS)
    {
        *this = arr_  / RHS;
    }
};


}


#endif