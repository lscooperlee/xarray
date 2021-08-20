#ifndef INDEX_H
#define INDEX_H

#include <array>
#include <concepts>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <type_traits>

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

    constexpr int start() const noexcept
    {
        return get<0>();
    }

    constexpr int stop() const noexcept
    {
        if constexpr (N > 1) {
            return get<1>();
        } else {
            // return std::numeric_limits<int>::max();
            return start() + 1;
        }
    }

    constexpr int step() const noexcept
    {
        if constexpr (N > 2) {
            return get<2>();
        } else {
            return 1;
        }
    }

private:
    std::array<int, N> value = {};
};

template <int... N>
class Index {

public:
    constexpr Index(const int (&... n)[N]) noexcept
        : data(n...) {};

    template <int K>
    constexpr int start() const noexcept requires(K < sizeof...(N))
    {
        return std::get<K>(data).template get<0>();
    }

    template <int K>
    constexpr int stop() const noexcept requires(K < sizeof...(N))
    {
        const auto& o = std::get<K>(data);
        if constexpr (o.template size > 1) {
            return o.template get<1>();
        } else {
            return -1;
        }
    }

    template <int K>
    constexpr int step() const noexcept requires(K < sizeof...(N))
    {
        const auto& o = std::get<K>(data);
        if constexpr (o.template size > 2) {
            return o.template get<2>();
        } else {
            return 1;
        }
    }

    std::tuple<Index1D<N>...> data = {};
};

class Idx {
public:
};

}

#endif