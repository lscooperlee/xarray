#ifndef XARRAY_COMMON_H
#define XARRAY_COMMON_H

#include <type_traits>


template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template <typename A, int N, typename I>
class XarrayBase;

template <typename A, int N>
class XBase;

template <typename U>
// concept XBaseType = std::is_convertible_v<U, XarrayBase<typename U::value_type, U::shape_size, typename U::imp_type>>;
// concept XBaseType = std::is_base_of_v<XBase<typename U::value_type, U::shape_size>, U>;
// concept XBaseType = std::derived_from<U, XBase<typename U::value_type, U::shape_size>>;
// concept XBaseType = std::derived_from<U, XarrayBase<typename U::value_type, U::shape_size, typename U::imp_type>>;
// concept XBaseType = std::is_class_v<U>;
concept XBaseType = requires {
	typename U::value_type;
	typename U::imp_type;
	U::shape_size;
};

#endif
