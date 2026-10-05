#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <cstddef>
#include <iostream>
#include <type_traits>

#include "Array_Expression.hpp"
#include "Unary_Expression.hpp"
#include "Binary_Expression.hpp"
#include "Mask_Array.hpp"

#ifdef __CUDACC__
#define NON_CPU_ComputeBackend

#define BLOCK_DIM_1D 1024
#define BLOCK_DIM_LOG2_1D 10


namespace ND::CUDA
{
template<typename T, int N>
__global__ void SetMemory_K(T* data_, T val)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    data_[idx] = val;
}
template<typename T, int N>
__global__ void SetMemory_K(T* data, T* newData)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    data[idx] = newData[idx];
}
template<typename T, int N, int Sub_N>
__global__ void SetMemoryRepeat_K(T* data, T* subData)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    data[idx] = subData[idx%Sub_N];
}
template<typename E, int N>
__global__ void CollapseExpression_K(typename E::value_type* data_, const E expr, const size_t shift = 0)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    data_[idx] = expr.get_element(shift + idx);
}

struct LogicalAnd
{
    __device__ bool operator()(bool a, bool b) const
    {
        return a && b;
    }
};
}

#endif


namespace ND {

template <ComputeBackend Backend, typename T, size_t firstDim, size_t... RestDims>
class Array : public Array_Expression<Array<Backend, T, firstDim, RestDims...>>
{
    template<ComputeBackend Backend_, typename T_, size_t f_Dim, size_t... R_dims>
    friend class Array;

    template<ComputeBackend Backend_, typename T_, size_t f_Dim, size_t... R_dims>
    friend class Mask_Array;

public:
    
    static constexpr size_t N = sizeof...(RestDims) + 1;
    static constexpr size_t length = firstDim * (RestDims * ...);
    static constexpr size_t Dims[N] = {firstDim, RestDims...};
    static constexpr ComputeBackend computeBackend = Backend;
    
    typedef typename base_traits<Array>::terminal_type terminal_type;
    typedef typename base_traits<Array>::terminal_sub_type terminal_sub_type;
    typedef typename base_traits<Array>::value_type  value_type;


    template<typename any_type>
    using generic_terminal_type = typename base_traits<Array>::template generic_terminal_type<any_type>;

    template<typename any_type>
    using generic_terminal_sub_type = typename base_traits<Array>::template generic_terminal_sub_type<any_type>;

protected:

    value_type* data_;
    bool is_original;

#ifdef __CUDACC__
    static constexpr int GRID_SIZE = (length + BLOCK_DIM_1D - 1) >> BLOCK_DIM_LOG2_1D;
#endif

public:

    HOST_DEVICE
    inline const value_type get_element(const size_t i) const     {   return data_[i];    }

    // Base constructor
    Array()
    : is_original(true)
    {
        _AllocateMemory();
    }
    Array(const value_type& val)
    : is_original(true)
    {
        _AllocateMemory();

        _SetMemory(val);
    }

    // copy constructor
    Array(const Array<Backend, T, firstDim, RestDims...>& other)
    : is_original(true)
    {
        _AllocateMemory();

        _SetMemory(other.data_);
    }
    
protected:
    // Constructor from pointer
    Array(value_type* p, bool is_or)
    {
        data_ = p;
        is_original = is_or;
    }
public:
    // explicit shallow copy
    Array shallowCopy() const
    {
        return Array(data_, false);
    }
    // copy assigment operator
    const Array<Backend, T, firstDim, RestDims...>& operator=(const Array<Backend, T, firstDim, RestDims...>& other)
    {
        _SetMemory(other.data_);

        return *this;
    }
    const Array<Backend, T, firstDim, RestDims...>& operator=(const value_type& val)
    {
        _SetMemory(val);
        
        return *this;
    }
    // constructor from N-1 dimensional array
    Array(const Array<Backend, T, RestDims...>& slice)
    : is_original(true)
    {
        _AllocateMemory();

        _SetMemoryRepeat(slice.data_);
    }

    // construct from Array_expressions
    // Shift can be used if N<expr.N (e.g. operator[] on expressions)
    template <typename E>
    requires(is_Array_Expression<E>::value)
    Array(const E& expr, size_t shift = 0)
    : is_original(true)
    {
        _AllocateMemory();
        
        _CollapseExpression(expr, shift);
    }
    template <typename E>
    requires(is_Array_Expression<E>::value)
    const Array& operator=(const E& expr)
    {
        _CollapseExpression(expr);

        return *this;
    }

    // destructor
    ~Array()
    {
        if(is_original)
        {
            if constexpr(Backend == ComputeBackend::CPU)
                delete[] data_;

            else if constexpr(Backend == ComputeBackend::CUDA)
                CUDA_CALL(cudaFree(data_));
        }
    }



    // access element
    template <typename... ind_type>
    requires (sizeof...(ind_type) == N)
    inline value_type& operator()(ind_type... indices)
    {
        size_t offset = 0;
        size_t temp[N] = {static_cast<size_t>(indices)...};

        for (size_t i = 0; i < N; ++i) {
            offset = offset * Dims[i] + temp[i];
        }
        return data_[offset];
    }
    template <typename... ind_type>
    requires (sizeof...(ind_type) == N)
    inline const value_type operator()(ind_type... indices) const
    {
        size_t offset = 0;
        size_t temp[N] = {static_cast<size_t>(indices)...};

        for (size_t i = 0; i < N; ++i) {
            offset = offset * Dims[i] + temp[i];
        }
        return data_[offset];
    }


    // access element
    inline Array<Backend, T, RestDims...> operator[](size_t index)
    {
        return Array<Backend, T, RestDims...>(data_ + index * (RestDims * ...), false);  // Guaranteed copy elision
    }
    inline const Array<Backend, T, RestDims...> operator[](size_t index) const
    {
        return Array<Backend, T, RestDims...>(data_ + index * (RestDims * ...), false);  // Guaranteed copy elision
    }

    Mask_Array<Backend, T, firstDim, RestDims...> operator[](const Array<Backend, bool, firstDim, RestDims...>& mask)
    {
        return Mask_Array(*this, mask);
    }


    const Array<Backend, T, firstDim, RestDims...>& fill(const value_type& val)
    {
        _SetMemory(val);

        return *this;
    }
    const Array<Backend, T, firstDim, RestDims...>& fill(const Array<Backend, T, firstDim, RestDims...>& other)
    {
        if(data_ != other.data_)
        {
            _SetMemory(other.data_);
        }
        
        return *this;
    }
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, firstDim, RestDims...>& fill(const E& expr)
    {
        _CollapseExpression(expr);

        return *this;
    }


    inline const size_t size(const size_t index = 0) const
    {
        return Dims[index];
    }
    




    // Arithmetic operations

    // += operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, firstDim, RestDims...>& operator+=(const E& expr)
    {
        *this = *this + expr;

        return *this;
    }
    // scalar
    const Array<Backend, T, firstDim, RestDims...>& operator+=(const value_type& val)
    {
        *this = *this + val;

        return *this;
    }

    // -= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, firstDim, RestDims...>& operator-=(const E& expr)
    {
        *this = *this - expr;

        return *this;
    }
    // scalar
    const Array<Backend, T, firstDim, RestDims...>& operator-=(const value_type& val)
    {
        *this = *this - val;

        return *this;
    }

    // *= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, firstDim, RestDims...>& operator*=(const E& expr)
    {
        *this = *this * expr;

        return *this;
    }
    // scalar
    const Array<Backend, T, firstDim, RestDims...>& operator*=(const value_type& val)
    {
        *this = *this * val;

        return *this;
    }

    // /= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, firstDim, RestDims...>& operator/=(const E& expr)
    {
        *this = *this / expr;

        return *this;
    }
    // scalar
    const Array<Backend, T, firstDim, RestDims...>& operator/=(const value_type& val)
    {
        value_type inv_val = 1.0/val;

        *this = *this * inv_val;

        return *this;
    }


#ifdef __CUDACC__
    const bool all()
    {
        bool* d_result;
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        bool init = true;
        
        CUDA_CALL(cudaMalloc(&d_result, sizeof(bool)));

        CUDA_CALL(cub::DeviceReduce::Reduce(
            d_temp, temp_bytes,
            data_, d_result, length,
            CUDA::LogicalAnd{}, init));

        CUDA_CALL(cudaMalloc(&d_temp, temp_bytes));

        CUDA_CALL(cub::DeviceReduce::Reduce(
            d_temp, temp_bytes,
            data_, d_result, length,
            CUDA::LogicalAnd{}, init));

        bool result;
        cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

        CUDA_CALL(cudaFree(d_temp));
        CUDA_CALL(cudaFree(d_result));

        return result;
    }
    const value_type min()
    {
        value_type* d_result;
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        
        CUDA_CALL(cudaMalloc(&d_result, sizeof(value_type)));

        CUDA_CALL(cub::DeviceReduce::Min(
            d_temp, temp_bytes,
            data_, d_result, length));

        CUDA_CALL(cudaMalloc(&d_temp, temp_bytes));

        CUDA_CALL(cub::DeviceReduce::Min(
            d_temp, temp_bytes,
            data_, d_result, length));

        value_type result;
        cudaMemcpy(&result, d_result, sizeof(value_type), cudaMemcpyDeviceToHost);

        CUDA_CALL(cudaFree(d_temp));
        CUDA_CALL(cudaFree(d_result));

        return result;
    }
    const value_type max()
    {
        value_type* d_result;
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        
        CUDA_CALL(cudaMalloc(&d_result, sizeof(value_type)));

        CUDA_CALL(cub::DeviceReduce::Max(
            d_temp, temp_bytes,
            data_, d_result, length));

        CUDA_CALL(cudaMalloc(&d_temp, temp_bytes));

        CUDA_CALL(cub::DeviceReduce::Max(
            d_temp, temp_bytes,
            data_, d_result, length));

        value_type result;
        cudaMemcpy(&result, d_result, sizeof(value_type), cudaMemcpyDeviceToHost);

        CUDA_CALL(cudaFree(d_temp));
        CUDA_CALL(cudaFree(d_result));

        return result;
    }
    const value_type sum()
    {
        value_type* d_result;
        CUDA_CALL(cudaMalloc(&d_result, sizeof(value_type)));

        CUDA_CALL(cub::DeviceReduce::Sum(data_, d_result, length));

        value_type result;
        cudaMemcpy(&result, d_result, sizeof(value_type), cudaMemcpyDeviceToHost);

        CUDA_CALL(cudaFree(d_result));

        return result;
    }
#endif

protected:

    void _AllocateMemory()
    {
        if constexpr(Backend == ComputeBackend::CPU)
            data_ = new value_type[length];

        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_CALL(cudaMalloc(&data_, length * sizeof(value_type)));
    }

    void _SetMemory(const value_type& val)
    {
        if constexpr(Backend == ComputeBackend::CPU)
        {
            PARALLEL_FOR(length)
            for (size_t i = 0; i < length; ++i)
            {
                data_[i] = val;
            }
        }

        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_FUNC((CUDA::SetMemory_K<value_type, length><<<GRID_SIZE, BLOCK_DIM_1D>>>(data_, val)));
    }

    void _SetMemory(value_type* newData)
    {
        if constexpr(Backend == ComputeBackend::CPU)
        {
            PARALLEL_FOR(length)
            for (size_t i = 0; i < length; ++i)
            {
                data_[i] = newData[i];
            }
        }

        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_FUNC((CUDA::SetMemory_K<value_type, length><<<GRID_SIZE, BLOCK_DIM_1D>>>(data_, newData)));
    }

    void _SetMemoryRepeat(value_type* subData)
    {
        if constexpr(Backend == ComputeBackend::CPU)
        {
            PARALLEL_FOR(length)
            for (size_t i = 0; i < length; ++i)
            {
                data_[i] = subData[i%(RestDims*...)];
            }
        }

        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_FUNC((CUDA::SetMemoryRepeat_K<value_type, length, (RestDims*...)><<<GRID_SIZE, BLOCK_DIM_1D>>>(data_, subData)));
    }

    template<typename E>
    requires(is_Array_Expression<E>::value)
    void _CollapseExpression(const E& expr, const size_t shift = 0)
    {
        if constexpr(Backend == ComputeBackend::CPU)
        {
            PARALLEL_FOR(length)
            for (size_t i = 0; i < length; ++i)
            {
                data_[i] = expr.get_element(i);
            }
        }

        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_FUNC((CUDA::CollapseExpression_K<E, length><<<GRID_SIZE, BLOCK_DIM_1D>>>(data_, expr, shift)));
    }
};



// ostream
template <ComputeBackend Backend, typename T, size_t firstDim, size_t... RestDims>
    requires(Backend == ComputeBackend::CPU)
std::ostream& operator<<(std::ostream& output, const Array<Backend, T, firstDim, RestDims...>& other)
{
    if (sizeof...(RestDims) > 0)
    {
        for(size_t i = 0; i<firstDim; ++i)
            output<<other[i]<<std::endl;
    }
    else
    {
        for(size_t i = 0; i<firstDim; ++i)
            output<<other[i]<<"\t";
    }

    return output;
}




template <ComputeBackend Backend, typename T, size_t Dim>
class Array<Backend, T, Dim> : public Array_Expression<Array<Backend, T, Dim>>
{
    template<ComputeBackend Backend_, typename T_, size_t f_Dim, size_t... R_dims>
    friend class Array;

    template<ComputeBackend Backend_, typename T_, size_t f_Dim, size_t... R_dims>
    friend class Mask_Array;
    
    
public:

    static constexpr size_t N = 1;
    static constexpr size_t length = Dim;
    static constexpr size_t Dims[N] = {Dim};
    static constexpr ComputeBackend computeBackend = Backend;

    typedef typename base_traits<Array>::terminal_type terminal_type;
    typedef typename base_traits<Array>::terminal_sub_type terminal_sub_type;
    typedef typename base_traits<Array>::value_type value_type;


    template<typename any_type>
    using generic_terminal_type = typename base_traits<Array>::template generic_terminal_type<any_type>;

    template<typename any_type>
    using generic_terminal_sub_type = typename base_traits<Array>::template generic_terminal_sub_type<any_type>;

protected:

    value_type* data_;
    bool is_original;

#ifdef __CUDACC__
    static constexpr int GRID_SIZE = (length + BLOCK_DIM_1D - 1) >> BLOCK_DIM_LOG2_1D;
#endif
    
public:

    HOST_DEVICE
    inline const value_type get_element(size_t i) const     {   return data_[i];    }

    // Base constructor
    Array()
    : is_original(true)
    {
        _AllocateMemory();
    }
    Array(const value_type& val)
    : is_original(true)
    {
        _AllocateMemory();
        
        _SetMemory(val);
    }

    // copy constructor
    Array(const Array<Backend, T, Dim>& other)
    : is_original(true)
    {
        _AllocateMemory();

        _SetMemory(other.data_);
    }

protected:
    // Constructor from pointer
    Array(value_type* p, bool is_or)
    {
        data_ = p;
        is_original = is_or;
    }
public:
    // explicit shallow copy
    Array shallowCopy() const
    {
        return Array(data_, false);
    }
    // copy assigment operator
    const Array<Backend, T, Dim>& operator=(const Array<Backend, T, Dim>& other)
    {
        _SetMemory(other.data_);

        return *this;
    }
    const Array<Backend, T, Dim>& operator=(const value_type& val)
    {
        _SetMemory(val);
        
        return *this;
    }
    
    // construct from Array_expressions
    // Shift can be used if N<expr.N (e.g. operator[] on expressions)
    template <typename E>
    requires(is_Array_Expression<E>::value)
    Array(const E& expr, size_t shift = 0)
    : is_original(true)
    {
        _AllocateMemory();
        
        _CollapseExpression(expr, shift);
    }
    template <typename E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, Dim>& operator=(const E& expr)
    {
        _CollapseExpression(expr);

        return *this;
    }

    // destructor
    ~Array()
    {
        if(is_original)
        {
            if constexpr(Backend == ComputeBackend::CPU)
                delete[] data_;

            else if constexpr(Backend == ComputeBackend::CUDA)
                CUDA_CALL(cudaFree(data_));
        }
    }



    // access element
    inline value_type& operator()(size_t index)                 {   return data_[index];    }
    inline const value_type operator()(size_t index) const      {   return data_[index];    }
    inline value_type& operator[](size_t index)                 {   return data_[index];    }
    inline const value_type operator[](size_t index) const      {   return data_[index];    }

    Mask_Array<Backend, T, Dim> operator[](const Array<Backend, bool, Dim> mask)  {   return Mask_Array(*this, mask); }




    const Array<Backend, T, Dim>& fill(const value_type& val)
    {
        _SetMemory(val);

        return *this;
    }
    const Array<Backend, T, Dim>& fill(const Array<Backend, T, Dim>& other)
    {
        if(data_ != other.data_)
        {
            _SetMemory(other.data_);
        }
        
        return *this;
    }
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, Dim>& fill(const E& expr)
    {
        _CollapseExpression(expr);

        return *this;
    }


    inline const size_t size(const size_t index = 0) const
    {
        return Dims[index];
    }






    // Arithmetic operations


    // += operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, Dim>& operator+=(const E& expr)
    {
        *this = *this + expr;

        return *this;
    }
    // scalar
    const Array<Backend, T, Dim>& operator+=(const value_type& val)
    {
        *this = *this + val;

        return *this;
    }

    // -= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, Dim>& operator-=(const E& expr)
    {
        *this = *this - expr;

        return *this;
    }
    // scalar
    const Array<Backend, T, Dim>& operator-=(const value_type& val)
    {
        *this = *this - val;

        return *this;
    }

    // *= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, Dim>& operator*=(const E& expr)
    {
        *this = *this * expr;

        return *this;
    }
    // scalar
    const Array<Backend, T, Dim>& operator*=(const value_type& val)
    {
        *this = *this * val;

        return *this;
    }

    // /= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<Backend, T, Dim>& operator/=(const E& expr)
    {
        *this = *this / expr;

        return *this;
    }
    // scalar
    const Array<Backend, T, Dim>& operator/=(const value_type& val)
    {
        value_type inv_val = 1.0/val;

        *this = *this * inv_val;

        return *this;
    }


#ifdef __CUDACC__
    const bool all()
    {
        bool* d_result;
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        bool init = true;
        
        CUDA_CALL(cudaMalloc(&d_result, sizeof(bool)));

        CUDA_CALL(cub::DeviceReduce::Reduce(
            d_temp, temp_bytes,
            data_, d_result, length,
            CUDA::LogicalAnd{}, init));

        CUDA_CALL(cudaMalloc(&d_temp, temp_bytes));

        CUDA_CALL(cub::DeviceReduce::Reduce(
            d_temp, temp_bytes,
            data_, d_result, length,
            CUDA::LogicalAnd{}, init));

        bool result;
        cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

        CUDA_CALL(cudaFree(d_temp));
        CUDA_CALL(cudaFree(d_result));

        return result;
    }
    const value_type min()
    {
        value_type* d_result;
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        
        CUDA_CALL(cudaMalloc(&d_result, sizeof(value_type)));

        CUDA_CALL(cub::DeviceReduce::Min(
            d_temp, temp_bytes,
            data_, d_result, length));

        CUDA_CALL(cudaMalloc(&d_temp, temp_bytes));

        CUDA_CALL(cub::DeviceReduce::Min(
            d_temp, temp_bytes,
            data_, d_result, length));

        value_type result;
        cudaMemcpy(&result, d_result, sizeof(value_type), cudaMemcpyDeviceToHost);

        CUDA_CALL(cudaFree(d_temp));
        CUDA_CALL(cudaFree(d_result));

        return result;
    }
    const value_type max()
    {
        value_type* d_result;
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        
        CUDA_CALL(cudaMalloc(&d_result, sizeof(value_type)));

        CUDA_CALL(cub::DeviceReduce::Max(
            d_temp, temp_bytes,
            data_, d_result, length));

        CUDA_CALL(cudaMalloc(&d_temp, temp_bytes));

        CUDA_CALL(cub::DeviceReduce::Max(
            d_temp, temp_bytes,
            data_, d_result, length));

        value_type result;
        cudaMemcpy(&result, d_result, sizeof(value_type), cudaMemcpyDeviceToHost);

        CUDA_CALL(cudaFree(d_temp));
        CUDA_CALL(cudaFree(d_result));

        return result;
    }
    const value_type sum()
    {
        value_type* d_result;
        CUDA_CALL(cudaMalloc(&d_result, sizeof(value_type)));

        CUDA_CALL(cub::DeviceReduce::Sum(data_, d_result, length));

        value_type result;
        cudaMemcpy(&result, d_result, sizeof(value_type), cudaMemcpyDeviceToHost);

        CUDA_CALL(cudaFree(d_result));

        return result;
    }
#endif

protected:

    void _AllocateMemory()
    {
        if constexpr(Backend == ComputeBackend::CPU)
            data_ = new value_type[length];

        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_CALL(cudaMalloc(&data_, length * sizeof(value_type)));
    }

    void _SetMemory(const value_type& val)
    {
        if constexpr(Backend == ComputeBackend::CPU)
        {
            PARALLEL_FOR(length)
            for (size_t i = 0; i < length; ++i)
            {
                data_[i] = val;
            }
        }

        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_FUNC((CUDA::SetMemory_K<value_type, length><<<GRID_SIZE, BLOCK_DIM_1D>>>(data_, val)));
    }

    void _SetMemory(value_type* newData)
    {
        if constexpr(Backend == ComputeBackend::CPU)
        {
            PARALLEL_FOR(length)
            for (size_t i = 0; i < length; ++i)
            {
                data_[i] = newData[i];
            }
        }

        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_FUNC((CUDA::SetMemory_K<value_type, length><<<GRID_SIZE, BLOCK_DIM_1D>>>(data_, newData)));
    }

    template<typename E>
    requires(is_Array_Expression<E>::value)
    void _CollapseExpression(const E& expr, const size_t shift = 0)
    {
        if constexpr(Backend == ComputeBackend::CPU)
        {
            PARALLEL_FOR(length)
            for (size_t i = 0; i < length; ++i)
            {
                data_[i] = expr.get_element(i);
            }
        }

        else if constexpr(Backend == ComputeBackend::CUDA)
            CUDA_FUNC((CUDA::CollapseExpression_K<E, length><<<GRID_SIZE, BLOCK_DIM_1D>>>(data_, expr, shift)));
    }
};


template<typename T, size_t firstDim, size_t... RestDims>
using Array_CPU = Array<ComputeBackend::CPU, T, firstDim, RestDims...>;

#ifdef __CUDACC__
template<typename T, size_t firstDim, size_t... RestDims>
using Array_CUDA = Array<ComputeBackend::CUDA, T, firstDim, RestDims...>;
#endif

}

#endif
