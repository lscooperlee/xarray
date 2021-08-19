#ifndef COMMON_H
#define COMMON_H

#include <type_traits>


template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

// template <typename S, typename T, typename I>
// class XarrayBase;
template <typename U>
//concept XBaseType = std::derived_from<U, XarrayBase<typename U::sub_type, typename U::value_type, typename U::imp_type>>;
concept XBaseType = std::is_class_v<U>;

#endif
