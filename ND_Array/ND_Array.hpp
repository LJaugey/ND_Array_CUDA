#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <cstddef>
#include <iostream>
#include <type_traits>

#include "Array_Expression.hpp"
#include "Unary_Expression.hpp"
#include "Binary_Expression.hpp"
#include "Mask_Array.hpp"



namespace ND {


template <typename T, size_t firstDim, size_t... RestDims>
class Array : public Array_Expression<Array<T, firstDim, RestDims...>>
{
    template<typename T_, size_t f_Dim, size_t... R_dims>
    friend class Array;

    template<typename T_, size_t f_Dim, size_t... R_dims>
    friend class Mask_Array;

public:
    
    static constexpr size_t N = sizeof...(RestDims) + 1;
    static constexpr size_t length = firstDim * (RestDims * ...);
    static constexpr size_t Dims[N] = {firstDim, RestDims...};

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


public:

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
    Array(const Array<T, firstDim, RestDims...>& other)
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
    const Array<T, firstDim, RestDims...>& operator=(const Array<T, firstDim, RestDims...>& other)
    {
        _SetMemory(other.data_);

        return *this;
    }
    const Array<T, firstDim, RestDims...>& operator=(const value_type& val)
    {
        _SetMemory(val);
        
        return *this;
    }
    // constructor from N-1 dimensional array
    Array(const Array<T, RestDims...>& slice)
    : is_original(true)
    {
        _AllocateMemory();

        PARALLEL_FOR(length)
        for (size_t i = 0; i < length; ++i)
        {
            data_[i] = slice.data_[i%(RestDims*...)];
        }
    }

    // construct from Array_expressions
    // Shift can be used if N<expr.N (e.g. operator[] on expressions)
    template <typename E>
    requires(is_Array_Expression<E>::value)
    Array(const E& expr, size_t shift = 0)
    : is_original(true)
    {
        _AllocateMemory();
        
        _CollapseExpression(expr);
    }
    template <typename E>
    requires(is_Array_Expression<E>::value)
    const Array& operator=(const E& expr)
    {
        _CollapseExpression(expr);

        return *this;
    }

    // destructor
    ~Array() {  if(is_original)    delete[] data_;   }



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
    inline Array<T, RestDims...> operator[](size_t index)
    {
        return Array<T, RestDims...>(data_ + index * (RestDims * ...), false);  // Guaranteed copy elision
    }
    inline const Array<T, RestDims...> operator[](size_t index) const
    {
        return Array<T, RestDims...>(data_ + index * (RestDims * ...), false);  // Guaranteed copy elision
    }

    Mask_Array<T, firstDim, RestDims...> operator[](const Array<bool, firstDim, RestDims...>& mask)
    {
        return Mask_Array(*this, mask);
    }


    const Array<T, firstDim, RestDims...>& fill(const value_type& val)
    {
        _SetMemory(val);

        return *this;
    }
    const Array<T, firstDim, RestDims...>& fill(const Array<T, firstDim, RestDims...>& other)
    {
        if(data_ != other.data_)
        {
            _SetMemory(other.data_);
        }
        
        return *this;
    }
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<T, firstDim, RestDims...>& fill(const E& expr)
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
    const Array<T, firstDim, RestDims...>& operator+=(const E& expr)
    {
        *this = *this + expr;

        return *this;
    }
    // scalar
    const Array<T, firstDim, RestDims...>& operator+=(const value_type& val)
    {
        *this = *this + val;

        return *this;
    }

    // -= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<T, firstDim, RestDims...>& operator-=(const E& expr)
    {
        *this = *this - expr;

        return *this;
    }
    // scalar
    const Array<T, firstDim, RestDims...>& operator-=(const value_type& val)
    {
        *this = *this - val;

        return *this;
    }

    // *= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<T, firstDim, RestDims...>& operator*=(const E& expr)
    {
        *this = *this * expr;

        return *this;
    }
    // scalar
    const Array<T, firstDim, RestDims...>& operator*=(const value_type& val)
    {
        *this = *this * val;

        return *this;
    }

    // /= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<T, firstDim, RestDims...>& operator/=(const E& expr)
    {
        *this = *this / expr;

        return *this;
    }
    // scalar
    const Array<T, firstDim, RestDims...>& operator/=(const value_type& val)
    {
        value_type inv_val = 1.0/val;

        *this = *this * inv_val;

        return *this;
    }
protected:

    void _AllocateMemory()
    {
        data_ = new value_type[length];
    }
    void _SetMemory(const value_type& val)
    {
        PARALLEL_FOR(length)
        for (size_t i = 0; i < length; ++i)
        {
            data_[i] = val;
        }
    }
    void _SetMemory(value_type* newData)
    {
        PARALLEL_FOR(length)
        for (size_t i = 0; i < length; ++i)
        {
            data_[i] = newData[i];
        }
    }
    template<typename E>
    requires(is_Array_Expression<E>::value)
    void _CollapseExpression(const E& expr, const size_t shift = 0)
    {
        PARALLEL_FOR(length)
        for (size_t i = 0; i < length; ++i)
        {
            data_[i] = expr.get_element(i);
        }
    }
};



// ostream
template <typename T, size_t firstDim, size_t... RestDims>
std::ostream& operator<<(std::ostream& output, const Array<T, firstDim, RestDims...>& other)
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



template <typename T, size_t Dim>
class Array<T, Dim> : public Array_Expression<Array<T, Dim>>
{
    template<typename T_, size_t f_Dim, size_t... R_dims>
    friend class Array;

    template<typename T_, size_t f_Dim, size_t... R_dims>
    friend class Mask_Array;
    
public:

    static constexpr size_t N = 1;
    static constexpr size_t length = Dim;
    static constexpr size_t Dims[N] = {Dim};

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

    
public:

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
    Array(const Array<T, Dim>& other)
    : is_original(true)
    {
        _AllocateMemory();

        _SetMemory(other.data);
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
    const Array<T, Dim>& operator=(const Array<T, Dim>& other)
    {
        _SetMemory(other.data_);

        return *this;
    }
    const Array<T, Dim>& operator=(const value_type& val)
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
    const Array<T, Dim>& operator=(const E& expr)
    {
        _CollapseExpression(expr);

        return *this;
    }

    // destructor
    ~Array() {  if(is_original)    delete[] data_;   }



    // access element
    inline value_type& operator()(size_t index)                 {   return data_[index];    }
    inline const value_type operator()(size_t index) const      {   return data_[index];    }
    inline value_type& operator[](size_t index)                 {   return data_[index];    }
    inline const value_type operator[](size_t index) const      {   return data_[index];    }

    Mask_Array<T, Dim> operator[](const Array<bool, Dim> mask)  {   return Mask_Array(*this, mask); }




    const Array<T, Dim>& fill(const value_type& val)
    {
        _SetMemory(val);

        return *this;
    }
    const Array<T, Dim>& fill(const Array<T, Dim>& other)
    {
        if(data_ != other.data_)
        {
            _SetMemory(other.data_);
        }
        
        return *this;
    }
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<T, Dim>& fill(const E& expr)
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
    const Array<T, Dim>& operator+=(const E& expr)
    {
        *this = *this + expr;

        return *this;
    }
    // scalar
    const Array<T, Dim>& operator+=(const value_type& val)
    {
        *this = *this + val;

        return *this;
    }

    // -= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<T, Dim>& operator-=(const E& expr)
    {
        *this = *this - expr;

        return *this;
    }
    // scalar
    const Array<T, Dim>& operator-=(const value_type& val)
    {
        *this = *this - val;

        return *this;
    }

    // *= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<T, Dim>& operator*=(const E& expr)
    {
        *this = *this * expr;

        return *this;
    }
    // scalar
    const Array<T, Dim>& operator*=(const value_type& val)
    {
        *this = *this * val;

        return *this;
    }

    // /= operator
    template<class E>
    requires(is_Array_Expression<E>::value)
    const Array<T, Dim>& operator/=(const E& expr)
    {
        *this = *this / expr;

        return *this;
    }
    // scalar
    const Array<T, Dim>& operator/=(const value_type& val)
    {
        value_type inv_val = 1.0/val;

        *this = *this * inv_val;

        return *this;
    }

protected:

    void _AllocateMemory()
    {
        data_ = new value_type[length];
    }
    void _SetMemory(const value_type& val)
    {
        PARALLEL_FOR(length)
        for (size_t i = 0; i < length; ++i)
        {
            data_[i] = val;
        }
    }
    void _SetMemory(value_type* newData)
    {
        PARALLEL_FOR(length)
        for (size_t i = 0; i < length; ++i)
        {
            data_[i] = newData[i];
        }
    }
    template<typename E>
    requires(is_Array_Expression<E>::value)
    void _CollapseExpression(const E& expr, const size_t shift = 0)
    {
        PARALLEL_FOR(length)
        for (size_t i = 0; i < length; ++i)
        {
            data_[i] = expr.get_element(i);
        }
    }
};



}

#endif
