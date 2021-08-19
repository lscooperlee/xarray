#ifndef SHAPE_H
#define SHAPE_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include <opencv2/core.hpp>

#include "common.h"

template <int N>
class Shape {
public:
    static constexpr int _size = N;

    constexpr Shape() noexcept requires(N >= 0) = default;

    constexpr Shape(const int (&d)[N]) noexcept requires(N > 0)
        : dim(std::to_array(d)) {};

    explicit constexpr Shape(int d1) noexcept requires(N == 1)
        : dim({ d1 })
    {
    }

    explicit constexpr Shape(int d1, int d2) noexcept requires(N == 2)
        : dim({ d1, d2 })
    {
    }

    explicit constexpr Shape(int d1, int d2, int d3) noexcept requires(N == 3)
        : dim({ d1, d2, d3 })
    {
    }

    constexpr int operator[](int idx) const
    {
        return dim[idx];
    }
    constexpr int& operator[](int idx)
    {
        return dim[idx];
    }

    template <int M>
    constexpr bool operator==(Shape<M> shape) const
    {
        if constexpr (M != N) {
            return false;
        } else {
            return this->dim == shape.dim;
        }
    }

    constexpr int total() const
    {
        return std::accumulate(std::begin(dim), std::begin(dim) + size(), 1, std::multiplies<int>());
    }

    constexpr int size() const
    {
        return _size;
    }

    std::array<int, N> dim = {};
};

template <int N>
std::ostream& operator<<(std::ostream& stream, const Shape<N>& s)
{
    stream << "(" << s[0];

    for (auto&& d : s.dim | std::views::drop(1)) {
        stream << ", " << d;
    }

    stream << ")";

    return stream;
}

Shape(int d1)->Shape<1>;
Shape(int d1, int d2)->Shape<2>;
Shape(int d1, int d2, int d3)->Shape<3>;

#endif