#ifndef HELPER_HPP
#define HELPER_HPP

#include <stddef.h>
#include <type_traits>


#define PAR_SIZE 1024


#ifdef _OPENMP
    #include <omp.h>
    #define STRINGIFY(a) #a
    #define PARALLEL_FOR(n) _Pragma(STRINGIFY(omp parallel for simd if(n>PAR_SIZE)))
    #define PARALLEL_FOR_REDUCE(op,n,var) _Pragma(STRINGIFY(omp parallel for simd reduction(op:var) if(n>PAR_SIZE)))
#else
    #define PARALLEL_FOR(n)
    #define PARALLEL_FOR_REDUCE(op,n,var)
#endif

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#define CUDA_FUNC(x) x
#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

#define BLOCK_DIM_1D 1024
#define BLOCK_DIM_LOG2_1D 10

#else
#define HOST_DEVICE
#define CUDA_FUNC(x)
#define CUDA_CALL( call )
#endif


enum ComputeBackend
{
    CPU,
    CUDA
};

namespace ND {


// Base traits
template <class E> 
struct base_traits
{
    typedef E value_type;
};



// Array expression
template <class E>
class Array_Expression;


template<typename E>
struct is_Array_Expression : std::false_type {};

template<typename E>
struct is_Array_Expression<Array_Expression<E>> : std::true_type {};

template<typename E>
requires(std::is_base_of<Array_Expression<E>,E>::value)
struct is_Array_Expression<E> : std::true_type {};


// Unary operation
template <class OP, class E>
requires (ND::is_Array_Expression<E>::value)
class Unary_Op;


template <class OP, class E>
struct base_traits<Unary_Op<OP,E>>
{
    typename base_traits<E>::value_type arg_value_type;
    typedef decltype(OP::apply(arg_value_type)) value_type;


    template<typename any_type>
    using generic_terminal_type = typename base_traits<E>::template generic_terminal_type<any_type>;

    template<typename any_type>
    using generic_terminal_sub_type = typename base_traits<E>::template generic_terminal_sub_type<any_type>;
    

    typedef generic_terminal_type<value_type> terminal_type;
    typedef generic_terminal_sub_type<value_type> terminal_sub_type;

};



// Binary operation
template <class E1, class OP, class E2>
requires (  (ND::is_Array_Expression<E1>::value && ND::is_Array_Expression<E2>::value) ||
            (ND::is_Array_Expression<E1>::value && std::is_convertible<typename base_traits<E2>::value_type, typename E1::value_type>::value) ||
            (ND::is_Array_Expression<E2>::value && std::is_convertible<typename base_traits<E1>::value_type, typename E2::value_type>::value))
class Binary_Op;


template <class E1, class OP, class E2>
struct base_traits<Binary_Op<E1,OP,E2>>
{
    typename base_traits<E1>::value_type arg1_value_type;
    typename base_traits<E2>::value_type arg2_value_type;

    typedef decltype(OP::apply(arg1_value_type,arg2_value_type)) value_type;


    template<typename any_type>
    using generic_terminal_type = typename std::conditional< ND::is_Array_Expression<E1>::value,
                                                    base_traits<E1>,
                                                    base_traits<E2>
                                                    >::type::template generic_terminal_type<any_type>;

    template<typename any_type>
    using generic_terminal_sub_type = typename std::conditional< ND::is_Array_Expression<E1>::value,
                                                        base_traits<E1>,
                                                        base_traits<E2>
                                                        >::type::template generic_terminal_sub_type<any_type>;


    typedef generic_terminal_type<value_type> terminal_type;
    typedef generic_terminal_sub_type<value_type> terminal_sub_type;
};



// ND Array
template<ComputeBackend B, typename T, size_t firstDim, size_t... RestDims>
class Array;


template<ComputeBackend B, typename T, size_t firstDim, size_t... RestDims>
struct base_traits<Array<B, T, firstDim, RestDims...>>
{
    typedef T value_type;


    template<typename any_type>
    using generic_terminal_type = Array<B, any_type, firstDim, RestDims...>;

    template<typename any_type>
    using generic_terminal_sub_type = Array<B, any_type, RestDims...>;


    typedef generic_terminal_type<value_type> terminal_type;
    typedef generic_terminal_sub_type<value_type> terminal_sub_type;
};


template<ComputeBackend B, typename T, size_t Dim>
struct base_traits<Array<B, T, Dim>>
{
    typedef T value_type;


    template<typename any_type>
    using generic_terminal_type = Array<B, any_type, Dim>;

    template<typename any_type>
    using generic_terminal_sub_type = any_type;


    typedef generic_terminal_type<value_type> terminal_type;
    typedef generic_terminal_sub_type<value_type> terminal_sub_type;
};



// Mask Array
template <ComputeBackend B, typename T, size_t firstDim, size_t... RestDims>
class Mask_Array;


}


#endif