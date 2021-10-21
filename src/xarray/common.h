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

#if defined (__GNUC__) && __GNUC__ < 10 && !defined(__clang__)

namespace std {

namespace detail {

template <class T, std::size_t N, std::size_t... I>
constexpr std::array<std::remove_cv_t<T>, N>
    to_array_impl(T (&a)[N], std::index_sequence<I...>)
{
    return { {a[I]...} };
}

}
 

template <class T, std::size_t N>
constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&a)[N]){
    return detail::to_array_impl(a, std::make_index_sequence<N>{});
}

template< class Derived, class Base >
concept derived_from = std::is_base_of_v<Base, Derived> && std::is_convertible_v<const volatile Derived*, const volatile Base*>;
}

#else

#include <concepts>

#endif



#endif
