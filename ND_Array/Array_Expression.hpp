#ifndef ARRAY_EXPRESSION_HPP
#define ARRAY_EXPRESSION_HPP

#include <iostream>
#include <cmath>

#include "helper.hpp"




namespace ND {

template <class E>
class Array_Expression
{
public:
    
    typedef typename base_traits<E>::terminal_type terminal_type;
    typedef typename base_traits<E>::terminal_sub_type terminal_sub_type;
    typedef typename base_traits<E>::value_type value_type;

    HOST_DEVICE
    inline const value_type get_element(const size_t i) const   {  return static_cast<const E&>(*this).get_element(i);   }

    Array_Expression shallowCopy() const
    {
        return static_cast<const E&>(*this).shallowCopy();
    }

    template <typename... ind_type>
    requires (sizeof...(ind_type) == terminal_type::N)
    inline const value_type operator()(const ind_type... indices) const   {   return static_cast<const E&>(*this)(indices...);    }


    inline const terminal_type eval() const
    {
        return terminal_type(static_cast<const E&>(*this));    // Guaranteed copy elision
    }

    // operator[] only collapses sub-array
    inline const terminal_sub_type operator[](size_t index) const
    {
        if constexpr (terminal_type::N==1)
        {
            return get_element(index);
        }
        else
        {
            return terminal_sub_type(static_cast<const E&>(*this), index*terminal_sub_type::length);  // Guaranteed copy elision
        }
    }

    



    // min
    const value_type min() const
    {
        value_type res = get_element(0);
        
        PARALLEL_FOR_REDUCE(min,terminal_type::length,res)
        for (size_t i = 1; i < terminal_type::length; i++)
        {
            res = std::min(res,get_element(i));
        }

        return res;
    }
    // max
    const value_type max() const
    {
        value_type res = get_element(0);
        
        PARALLEL_FOR_REDUCE(max,terminal_type::length,res)
        for (size_t i = 1; i < terminal_type::length; i++)
        {
            res = std::max(res,get_element(i));
        }

        return res;
    }
    // sum
    const value_type sum() const
    {
        value_type res = get_element(0);
        
        PARALLEL_FOR_REDUCE(+,terminal_type::length,res)
        for (size_t i = 1; i < terminal_type::length; i++)
        {
            res += get_element(i);
        }
        
        return res;
    }

    // Check if all values are true
    const bool all() const
    {
        for (size_t i = 0; i < terminal_type::length; i++)
        {
            if(get_element(i) == false) return false;
        }
        
        return true;
    }
};


template<class E>
inline const auto min(const Array_Expression<E>& expr)   {   return expr.min();  }
template<class E>
inline const auto max(const Array_Expression<E>& expr)   {   return expr.max();  }
template<class E>
inline const auto sum(const Array_Expression<E>& expr)   {   return expr.sum();  }

template<class E>
inline std::ostream& operator<<(std::ostream& output, const Array_Expression<E>& expr)
{
    return output<<expr.eval();
}

}

#endif