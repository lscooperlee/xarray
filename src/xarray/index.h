#ifndef INDEX_H
#define INDEX_H

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <type_traits>

#include "xarray/common.h"

namespace xa {

template <int N>
class Index1D {
public:
    constexpr static int size = N;
    constexpr Index1D(const int (&n)[N]) noexcept
        : value(std::to_array(n)) {};

    template <int K>
    constexpr int get() const noexcept requires(K >= 0 && K < N)
    {
        return value[K];
    }

    constexpr int step() const noexcept
    {
        if constexpr (N > 2) {
            return get<2>();
        } else {
            return 1;
        }
    }

    constexpr int start(int size) const noexcept
    {
        if constexpr (N == 1) {
            auto tmp = get<0>();
            return tmp < 0 ? tmp + size : tmp;
        } else {
            auto tmp = get<0>();
            return std::clamp(tmp < 0 ? tmp + size : tmp, 0, size - 1);
        }
    }

    constexpr int stop(int size) const noexcept
    {
        if constexpr (N == 1) {
            return start(size) + 1;
        } else {
            auto tmp = get<1>();
            return std::clamp(tmp < 0 ? tmp + size : tmp, -1, size);
        }
    }

private:
    std::array<int, N> value = {};
};

namespace {
    template <int K, int... M>
    static constexpr bool all_are_one()
    {
        if constexpr (sizeof...(M) == 0) {
            return K == 1;
        } else {
            return (all_are_one<M...>() && (K == 1));
        }
    }

}

template <int... N>
requires(sizeof...(N) >= 1) class Index {

public:
    constexpr Index(const int (&... n)[N]) noexcept requires(!all_are_one<N...>())
        : data(n...) {};

    constexpr Index(int a) noexcept
        : data({ { a } }) {};

    constexpr Index(int a, int b) noexcept
        : data({ { a } }, { { b } }) {};

    template <int K, int J>
    constexpr int get() const noexcept requires(K < sizeof...(N))
    {
        return std::get<K>(data).template get<J>();
    }

    std::tuple<Index1D<N>...> data = {};
};

template <int... N>
Index(int a) -> Index<1>;

template <int... N>
Index(int a, int b) -> Index<1, 1>;

}

#endif
